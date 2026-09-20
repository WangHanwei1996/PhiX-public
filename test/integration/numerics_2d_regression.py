#!/usr/bin/env python3
"""Deterministic 2D flagship comparison. All generated files stay under --work.

Requires compiled GP_binary and GFA_WBM in --source/applications/solvers/<name>/.
Optional --baseline-binaries contains the corresponding pre-refactor executables.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess


def jsonc(path):
    text = path.read_text()
    result, quoted, escape, i = [], False, False, 0
    while i < len(text):
        char = text[i]
        if not quoted and text[i:i + 2] == "//":
            i = text.find("\n", i)
            if i == -1:
                break
            continue
        result.append(char)
        if char == '"' and not escape:
            quoted = not quoted
        escape = quoted and char == "\\" and not escape
        i += 1
    return json.loads("".join(result))


def write_field(path, name, nx, ny, values):
    header = f"# PhiX ScalarField\nname    {name}\nnx {nx}  ny {ny}  nz 1\nghost   1\n---\n"
    path.write_bytes(header.encode() + struct.pack("<" + "d" * len(values), *values))


def values(path):
    raw = path.read_bytes().split(b"---\n", 1)[1]
    return raw, struct.unpack("<" + "d" * (len(raw) // 8), raw)


def compare(reference, actual):
    result = {}
    for path in sorted(reference.glob("*_20.field")):
        a, x = values(path)
        b, y = values(actual / path.name)
        if len(x) != len(y):
            raise AssertionError("layout differs: " + path.name)
        if not x or not all(math.isfinite(v) for v in (*x, *y)):
            raise AssertionError("empty or nonfinite field: " + path.name)
        error = max(abs(u - v) for u, v in zip(x, y))
        if not math.isfinite(error) or error > 1e-11:
            raise AssertionError(f"{path.name}: error {error} exceeds 1e-11")
        result[path.name] = {"max_abs": error, "bitwise": a == b}
    if not result:
        raise AssertionError("no output fields to compare")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--baseline-binaries", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    args.work.mkdir(parents=True, exist_ok=False)
    (args.work / ".phix-numerics-regression").write_text("isolated deterministic cases\n")
    report = {}
    for app in ["GP_binary", "GFA_WBM"]:
        variants = ["isotropic"] if app == "GP_binary" else [
            "isotropic", "kinetic", "energy", "random", "clamp", "nucleation", "linear_ghost"]
        for variant in variants:
            modes = ["off", "auto"]
            if args.baseline_binaries and variant != "linear_ghost":
                modes.insert(0, "baseline")
            for mode in modes:
                case = args.work / app / variant / mode
                initial = case / "settings/initial_field"
                initial.mkdir(parents=True)
                cfg = jsonc(source / f"applications/solvers/{app}/settings/settings.jsonc")
                nx, ny = 32, 24
                cfg["mesh"].update(nx=nx, ny=ny)
                cfg["initialize"].update(nSteps=20, nan_check_every=1)
                cfg["output"].update(print_interval=20, write_interval=20, format="BINARY")
                cfg["execution"] = {"fusion": "off" if mode == "baseline" else mode}
                if app == "GP_binary":
                    cfg["initialize"]["c_init"] = ""
                else:
                    cfg["diagnostics"] = {"diag_phi": False, "diag_eta": False}
                    for key, path in cfg["tables"].items():
                        candidate = source / "data/material_properties/Fe-B" / Path(path).name
                        if candidate.exists():
                            cfg["tables"][key] = str(candidate)
                    if variant == "kinetic":
                        cfg["anisotropy"]["delta_phi"] = .02
                    if variant == "energy":
                        cfg["anisotropy"]["delta_E"] = .02
                    if variant == "random":
                        cfg["anisotropy"].update(orientation="random", delta_phi=.02)
                    if variant == "clamp":
                        cfg["constants"]["c_clamp"] = True
                    if variant == "nucleation":
                        cfg["nucleation"] = dict(enable=True, check_interval=2, lambda_I=0,
                            eta_mode="plant", eta_T_on=900, eta_n=1, eta_r=2,
                            buffer_cells=1, thresh=.9, rng=123)
                (case / "settings/settings.jsonc").write_text(json.dumps(cfg, indent=2))
                schemes = jsonc(source / f"applications/solvers/{app}/settings/schemes.jsonc")
                # Older binaries do not know the face-family configuration keys.
                if mode == "baseline":
                    for key in ["interpolation", "snGrad", "faceGrad", "fluxDiv"]:
                        schemes.pop(key, None)
                elif variant == "linear_ghost":
                    schemes["interpolation"] = {"default": "LinearGhost"}
                (case / "settings/schemes.jsonc").write_text(json.dumps(schemes, indent=2))
                for name in (["c"] if app == "GP_binary" else ["c", "phi", "eta"]):
                    data = []
                    for j in range(ny):
                        for i in range(nx):
                            if name == "c":
                                v = (.5 if app == "GP_binary" else .28) + .005 * math.cos(
                                    2 * math.pi * (i + .5) / nx) * math.cos(2 * math.pi * (j + .5) / ny)
                                if variant == "clamp" and i == 0 and j == 0:
                                    v = -.1
                            else:
                                v = (.8 if name == "phi" else .05) * math.exp(
                                    -((i - nx * .5) ** 2 + (j - ny * .5) ** 2) / 12)
                            data.append(v)
                    write_field(initial / (name + ".field"), name, nx, ny, data)
                binary = (args.baseline_binaries / app if mode == "baseline" else
                          source / f"applications/solvers/{app}/{app}")
                with (case / "run.log").open("w") as log:
                    subprocess.run([str(binary.resolve()), "settings/settings.jsonc"], cwd=case,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
            reference_mode = modes[0]
            for mode in modes[1:]:
                key = f"{app}/{variant}/{reference_mode}->{mode}"
                report[key] = compare(args.work / app / variant / reference_mode / "output",
                                      args.work / app / variant / mode / "output")
                print(key, "passed", flush=True)
    (args.work / "comparison.json").write_text(json.dumps(report, indent=2))
    print("All comparisons passed.")


if __name__ == "__main__":
    main()
