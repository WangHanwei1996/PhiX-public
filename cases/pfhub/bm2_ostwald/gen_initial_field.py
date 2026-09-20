#!/usr/bin/env python3
"""
PFHub Benchmark 2 (Ostwald ripening) 初始场生成器。

规格 (pages.nist.gov/pfhub, benchmark2; Jokisaari et al. CMS 2017):
  c(x,y)     = c0 + eps*[ cos(0.105x)cos(0.11y)
                        + (cos(0.13x)cos(0.087y))^2
                        + cos(0.025x-0.15y)cos(0.07x-0.02y) ]
  eta_i(x,y) = eps_eta * { cos((0.01i)x - 4)cos((0.007+0.01i)y)
                         + cos((0.11+0.01i)x)cos((0.11+0.01i)y)
                         + psi*[ cos((0.046+0.001i)x + (0.0405+0.001i)y)
                                *cos((0.031+0.001i)x - (0.004+0.001i)y) ]^2 }^2
  c0=0.5, eps=0.05, eps_eta=0.1, psi=1.5, i=1..4,  域 200x200

用法（在 bm2_ostwald/ 目录下运行）:
  python3 gen_initial_field.py

输出: a_periodic/settings/initial_field/{c,eta1..4}.field  (DAT 格式)
"""

import math
import os

NX, NY, NZ = 200, 200, 1
DX = DY = 1.0
X0 = Y0 = 0.0
C0, EPS = 0.5, 0.05
EPS_ETA, PSI = 0.1, 1.5

HEADER_TMPL = (
    "# PhiX ScalarField - DAT\n"
    "# name: {name}\n"
    f"# nx {NX}  ny {NY}  nz {NZ}\n"
    "# x y z value\n"
)

def ic_c(x, y):
    return C0 + EPS * (
        math.cos(0.105 * x) * math.cos(0.11 * y)
        + (math.cos(0.13 * x) * math.cos(0.087 * y)) ** 2
        + math.cos(0.025 * x - 0.15 * y) * math.cos(0.07 * x - 0.02 * y)
    )

def ic_eta(i, x, y):
    inner = (
        math.cos(0.01 * i * x - 4.0) * math.cos((0.007 + 0.01 * i) * y)
        + math.cos((0.11 + 0.01 * i) * x) * math.cos((0.11 + 0.01 * i) * y)
        + PSI * (math.cos((0.046 + 0.001 * i) * x + (0.0405 + 0.001 * i) * y)
                 * math.cos((0.031 + 0.001 * i) * x
                            - (0.004 + 0.001 * i) * y)) ** 2
    )
    return EPS_ETA * inner * inner

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "a_periodic", "settings", "initial_field")
os.makedirs(out_dir, exist_ok=True)

fields = {"c": ic_c}
for i in range(1, 5):
    fields[f"eta{i}"] = (lambda ii: lambda x, y: ic_eta(ii, x, y))(i)

for name, fn in fields.items():
    path = os.path.join(out_dir, f"{name}.field")
    with open(path, "w") as fh:
        fh.write(HEADER_TMPL.format(name=name))
        for k in range(NZ):
            for j in range(NY):
                y = Y0 + (j + 0.5) * DY
                for i in range(NX):
                    x = X0 + (i + 0.5) * DX
                    fh.write(f"{x:.12e}  {y:.12e}  0.0  {fn(x, y):.12e}\n")
    print(f"wrote {path}")
