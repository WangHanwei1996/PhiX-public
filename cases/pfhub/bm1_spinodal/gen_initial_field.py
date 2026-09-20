#!/usr/bin/env python3
"""
PFHub Benchmark 1 (spinodal decomposition) 初始场生成器。

规格 (pages.nist.gov/pfhub, benchmark1):
  c(x,y) = c0 + eps*[ cos(0.105x)cos(0.11y)
                    + (cos(0.13x)cos(0.087y))^2
                    + cos(0.025x-0.15y)cos(0.07x-0.02y) ]
  c0 = 0.5, eps = 0.01,  域 200x200

用法（在 bm1_spinodal/ 目录下运行）:
  python3 gen_initial_field.py

输出: {a_periodic,b_noflux}/settings/initial_field/c.field  (DAT 格式)
"""

import math
import os

NX, NY, NZ = 200, 200, 1
DX = DY = 1.0
X0 = Y0 = 0.0
C0, EPS = 0.5, 0.01

HEADER = (
    "# PhiX ScalarField - DAT\n"
    "# name: c\n"
    f"# nx {NX}  ny {NY}  nz {NZ}\n"
    "# x y z value\n"
)

def ic(x, y):
    return C0 + EPS * (
        math.cos(0.105 * x) * math.cos(0.11 * y)
        + (math.cos(0.13 * x) * math.cos(0.087 * y)) ** 2
        + math.cos(0.025 * x - 0.15 * y) * math.cos(0.07 * x - 0.02 * y)
    )

here = os.path.dirname(os.path.abspath(__file__))
for case in ("a_periodic", "b_noflux"):
    out_dir = os.path.join(here, case, "settings", "initial_field")
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in (("c", ic), ("mu", lambda x, y: 0.0)):
        path = os.path.join(out_dir, f"{name}.field")
        with open(path, "w") as fh:
            fh.write(HEADER.replace("name: c", f"name: {name}"))
            for k in range(NZ):
                for j in range(NY):
                    y = Y0 + (j + 0.5) * DY
                    for i in range(NX):
                        x = X0 + (i + 0.5) * DX
                        fh.write(f"{x:.12e}  {y:.12e}  0.0  {fn(x, y):.12e}\n")
        print(f"wrote {path}")
