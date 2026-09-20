#!/usr/bin/env python3
"""
PFHub Benchmark 6a (电化学 CH+Poisson) 初始场生成器。

c(x,y) = c0 + c1*{ cos(0.2x)cos(0.11y) + [cos(0.13x)cos(0.087y)]^2
                 + cos(0.025x-0.15y)cos(0.07x-0.02y) },  c0=0.5, c1=0.04
域 [0,100]^2。用法: python3 gen_initial_field.py
输出: a_square/settings/initial_field/{c,mu}.field  (DAT 格式)
"""
import math, os

NX = NY = 128
NZ = 1
DX = DY = 100.0 / 128
X0 = Y0 = 0.0
C0, C1 = 0.5, 0.04

HEADER = ("# PhiX ScalarField - DAT\n# name: {name}\n"
          f"# nx {NX}  ny {NY}  nz {NZ}\n# x y z value\n")

def ic(x, y):
    return C0 + C1 * (
        math.cos(0.2 * x) * math.cos(0.11 * y)
        + (math.cos(0.13 * x) * math.cos(0.087 * y)) ** 2
        + math.cos(0.025 * x - 0.15 * y) * math.cos(0.07 * x - 0.02 * y))

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "a_square", "settings", "initial_field")
os.makedirs(out, exist_ok=True)
for name, fn in (("c", ic), ("mu", lambda x, y: 0.0)):
    path = os.path.join(out, f"{name}.field")
    with open(path, "w") as fh:
        fh.write(HEADER.format(name=name))
        for j in range(NY):
            y = Y0 + (j + 0.5) * DY
            for i in range(NX):
                x = X0 + (i + 0.5) * DX
                fh.write(f"{x:.12e}  {y:.12e}  0.0  {fn(x, y):.12e}\n")
    print("wrote", path)
