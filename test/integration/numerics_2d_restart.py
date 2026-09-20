#!/usr/bin/env python3
"""Run the numerics2d tutorial continuously and through a file restart.

No cases or outputs are written outside the newly created --work directory.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from numerics_2d_regression import jsonc, values


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    report, outputs = {}, {}
    for backend, fusion in [("cpu", "off"), ("cuda", "off"), ("cuda", "auto")]:
        name = backend + "_" + fusion
        cfg = jsonc(args.settings / "settings.jsonc")
        cfg["execution"] = dict(backend=backend, fusion=fusion)
        cfg["output"].update(format="BINARY+VTI", write_interval=50)
        cfg["initialize"].update(nSteps=100, start_from="initial_field")
        for run in ["full", "split"]:
            case = args.work / name / run
            (case / "settings").mkdir(parents=True)
            shutil.copy2(args.settings / "schemes.jsonc", case / "settings/schemes.jsonc")
            def execute(steps, start, logname):
                cfg["initialize"].update(nSteps=steps, start_from=start)
                config = case / "settings/settings.jsonc"
                config.write_text(json.dumps(cfg, indent=2))
                with (case / logname).open("w") as log:
                    subprocess.run([str(args.binary.resolve()), "settings/settings.jsonc"],
                                   cwd=case, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=60, check=True)
            if run == "full":
                execute(100, "initial_field", "run.log")
            else:
                execute(50, "initial_field", "first.log")
                execute(100, "50", "restart.log")
        errors = {}
        for field in ["c", "mu"]:
            filename = field + "_100.field"
            a, x = values(args.work / name / "full/output" / filename)
            b, y = values(args.work / name / "split/output" / filename)
            if a != b:
                raise AssertionError(name + "/" + field + ": restart differs from continuous run")
            if not all(math.isfinite(v) for v in x):
                raise AssertionError("nonfinite output")
            errors[field] = dict(restart_bitwise=True)
            outputs[name, field] = x
        for run in ["full", "split"]:
            series = ET.parse(args.work / name / run / "output/c.pvd")
            entries = series.findall("./Collection/DataSet")
            times = [float(e.attrib["timestep"]) for e in entries]
            if len(times) != 3 or any(abs(t - v) > 1e-12 for t, v in
                                      zip(times, [0, .0005, .001])):
                raise AssertionError("incorrect output clock or restart PVD series")
        errors["output_clock"] = "0, 0.0005, 0.001; restart retains all entries"
        report[name] = errors
        print(name + ": continuous/restart bitwise equal (c and current mu)", flush=True)
    for field in ["c", "mu"]:
        baseline = outputs["cpu_off", field]
        for name in ["cuda_off", "cuda_auto"]:
            error = max(abs(x-y) for x, y in zip(baseline, outputs[name, field]))
            if error > args.tolerance:
                raise AssertionError(name + "/" + field + ": cross-backend error " + str(error))
            report[name][field]["cpu_max_abs"] = error
    (args.work / "comparison.json").write_text(json.dumps(report, indent=2))
    print("All restart and backend comparisons passed.")


if __name__ == "__main__":
    main()
