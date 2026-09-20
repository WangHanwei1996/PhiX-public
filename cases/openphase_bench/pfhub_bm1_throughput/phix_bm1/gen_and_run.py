#!/usr/bin/env python3
"""Block B GPU side (2026-08-25: added "c_init": "" - the solver default became
random:0.45:0.55 on 2026-07-23; without it the cosine IC is silently ignored) — PFHub BM1a on the PhiX CH solver, 200^2 and 512^2.

Same numerics as the OpenPhase-idiom CPU driver (double, CD2, explicit Euler,
dt=1e-3, 5e4 steps, BM1a cosine IC); wall time from the run wrapper. Must run
on an idle GPU (queued after the W6 ladder).

Usage: python3 w9_pfhub_bench/phix_bm1/gen_and_run.py
Out  : w9_pfhub_bench/phix_bm1/timings_gpu.csv (+ pfhub F(t) CSVs)
"""
import math
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = ("$PHIX_DIR/applications/solvers/Cahn-Hillard_double-well/2D/"
          "Cahn-Hillard_double-well")
NSTEPS = 50000
DT = 1.0e-3

SETTINGS = """{{
    "mesh": {{ "nx": {n}, "ny": {n}, "dx": 1.0, "dy": 1.0, "x0": 0.0, "y0": 0.0 }},
    "initialize": {{ "start_from": "initial_field", "c_init": "", "dt": {dt}, "nSteps": {nsteps} }},
    "boundary_conditions": {{ "x_min": "Periodic", "x_max": "Periodic",
                              "y_min": "Periodic", "y_max": "Periodic" }},
    "constants": {{ "rho": 5.0, "ca": 0.3, "cb": 0.7, "kappa": 2.0, "M": 5.0 }},
    "output": {{ "print_interval": {nsteps}, "write_interval": {nsteps},
                 "format": "DAT" }},
    "pfhub": {{ "energy_interval": 10000, "csv": "output/bm1_energy.csv",
                "benchmark": "1a" }}
}}
"""


def bm1a_ic(n):
    rows = []
    for j in range(n):
        for i in range(n):
            x, y = float(i), float(j)
            v = 0.5 + 0.01 * (math.cos(0.105 * x) * math.cos(0.11 * y)
                              + math.cos(0.13 * x) ** 2 * math.cos(0.087 * y) ** 2
                              + math.cos(0.025 * x - 0.15 * y)
                              * math.cos(0.07 * x - 0.02 * y))
            rows.append((i, j, v))
    return rows


results = []
for n in (200, 512):
    d = os.path.join(HERE, f"run{n}")
    os.makedirs(os.path.join(d, "settings", "initial_field"), exist_ok=True)
    os.makedirs(os.path.join(d, "output"), exist_ok=True)
    with open(os.path.join(d, "settings", "settings.jsonc"), "w") as f:
        f.write(SETTINGS.format(n=n, dt=DT, nsteps=NSTEPS))
    with open(os.path.join(d, "settings", "initial_field", "c.field"), "w") as f:
        f.write(f"# PhiX ScalarField - DAT\n# name: c\n# nx {n}  ny {n}  nz 1\n"
                "# x y z value\n")
        for i, j, v in bm1a_ic(n):
            f.write(f"{i+0.5:.6e}  {j+0.5:.6e}  0.000000e+00  {v:.10e}\n")
    # solver's initial_field mode also reads the scratch mu field -> zeros
    with open(os.path.join(d, "settings", "initial_field", "mu.field"), "w") as f:
        f.write(f"# PhiX ScalarField - DAT\n# name: mu\n# nx {n}  ny {n}  nz 1\n"
                "# x y z value\n")
        for j in range(n):
            for i in range(n):
                f.write(f"{i+0.5:.6e}  {j+0.5:.6e}  0.000000e+00  0.0\n")
    t0 = time.time()
    r = subprocess.run([SOLVER, "settings/settings.jsonc"], cwd=d,
                       capture_output=True, text=True)
    wall = time.time() - t0
    ok = r.returncode == 0
    cups = n * n * NSTEPS / wall
    results.append((n, NSTEPS, DT, wall, cups, ok))
    print(f"PhiX BM1a {n}^2: wall={wall:.2f} s  {cups:.3e} cell-updates/s  "
          f"exit={'OK' if ok else r.returncode}", flush=True)
    if not ok:
        print(r.stdout[-800:], r.stderr[-400:])

with open(os.path.join(HERE, "timings_gpu.csv"), "w") as f:
    f.write("# PhiX BM1a GPU timings — RTX 5080 (WSL2), incl. solver startup\n"
            "grid,steps,dt,wall_s,cell_updates_per_s,ok\n")
    for n, ns, dt, w, c, ok in results:
        f.write(f"{n},{ns},{dt},{w:.2f},{c:.4e},{ok}\n")
print("GPU timings done.", flush=True)
