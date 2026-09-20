"""
tutorials/quickstart/postProcess.py — plot the .field snapshots of a run.

    python postProcess.py                 # every snapshot -> output/png/
    python postProcess.py --step 5000     # one snapshot
    python postProcess.py --show          # also display interactively

Reads the BINARY snapshots output/phi_<step>.field.  For ParaView open
output/phi.pvd instead — it indexes the .vti files by physical time.
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def read_field(path):
    """Read a PhiX .field file -> (data[nz, ny, nx], meta).

    Layout: a short text header ("name", "nx ny nz", "ghost") closed by a
    "---" line, then the physical cells as raw float64, x fastest.  Ghost
    cells are never written.
    """
    with open(path, "rb") as f:
        meta = {}
        for raw in f:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line == "---":
                break
            if line.startswith("#") or not line.strip():
                continue
            tokens = line.split()
            if tokens[0] == "name":
                meta["name"] = tokens[1] if len(tokens) > 1 else ""
            elif tokens[0] == "ghost":
                meta["ghost"] = int(tokens[1])
            else:
                for i, tok in enumerate(tokens):
                    if tok in ("nx", "ny", "nz") and i + 1 < len(tokens):
                        meta[tok] = int(tokens[i + 1])
        nx, ny, nz = meta.get("nx", 1), meta.get("ny", 1), meta.get("nz", 1)
        data = np.frombuffer(f.read(), dtype=np.float64)
    return data.reshape((nz, ny, nx)), meta


def _step_from_path(path):
    m = re.search(r"_(\d+)\.field$", path)
    return int(m.group(1)) if m else 0


def plot_snapshot(path, out_dir, cmap, show=False):
    data, meta = read_field(path)
    arr = np.squeeze(data)                      # (ny, nx) for a 2D field
    step = _step_from_path(path)
    name = meta.get("name", "phi")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=-1.0, vmax=1.0,
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(name)
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    ax.set_title(f"{name}   step = {step}")
    ax.set_xlabel("x  [cells]")
    ax.set_ylabel("y  [cells]")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{name}_{step:07d}.png")
    fig.savefig(save_path, dpi=150)
    print(f"  saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot PhiX quickstart snapshots")
    p.add_argument("--step", type=int, default=None, help="plot a single step (default: all)")
    p.add_argument("--input-dir", default="output", help="directory with .field files")
    p.add_argument("--output-dir", default="output/png", help="where PNG files go")
    p.add_argument("--cmap", default="RdBu_r", help="matplotlib colormap for phi in [-1, 1]")
    p.add_argument("--show", action="store_true", help="display each figure interactively")
    args = p.parse_args()

    pattern = os.path.join(args.input_dir,
                           f"*_{args.step}.field" if args.step is not None else "*.field")
    files = sorted(glob.glob(pattern), key=_step_from_path)
    if not files:
        print(f"No .field files found matching: {pattern}")
        return
    print(f"Found {len(files)} snapshot(s). Plotting...")
    for path in files:
        plot_snapshot(path, args.output_dir, args.cmap, show=args.show)
    print("Done.")


if __name__ == "__main__":
    main()
