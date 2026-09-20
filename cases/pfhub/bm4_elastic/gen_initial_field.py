#!/usr/bin/env python3
"""
PFHub Benchmark 4a (elastic precipitate) 初始场生成器。

圆形析出相 r=20 nm 居中，基体 eta_m=0.0065，析出相 eta=1，
tanh 界面（0.05<eta<0.95 宽 5 nm → delta=5/(2*atanh(0.9))≈1.70 nm）。
域 400x400 nm，dx=1。

用法（在 bm4_elastic/ 目录下运行）: python3 gen_initial_field.py
输出: a_circle/settings/initial_field/eta.field  (二进制 .field)
"""

import math
import os
import numpy as np

NX, NY, NZ = 400, 400, 1
DX = DY = 1.0
X0 = Y0 = 0.0
R, ETA_M = 20.0, 0.0065
DELTA = 5.0 / (2.0 * math.atanh(0.9))   # ≈ 1.6989

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "a_circle", "settings", "initial_field")
os.makedirs(out_dir, exist_ok=True)

x = X0 + (np.arange(NX) + 0.5) * DX
y = Y0 + (np.arange(NY) + 0.5) * DY
X, Y = np.meshgrid(x, y, indexing="ij")
cx, cy = X0 + NX * DX / 2, Y0 + NY * DY / 2
r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

prof = 0.5 * (1.0 - np.tanh((r - R) / DELTA))          # 1 内 / 0 外
eta = ETA_M + (1.0 - ETA_M) * prof

path = os.path.join(out_dir, "eta.field")
with open(path, "wb") as f:
    header = ("# PhiX ScalarField\n"
              "name    eta\n"
              f"nx {NX}  ny {NY}  nz {NZ}\n"
              "ghost   1\n---\n")
    f.write(header.encode("ascii"))
    f.write(eta.T.reshape(NZ, NY, NX).astype(np.float64).tobytes())
print(f"wrote {path}  eta:[{eta.min():.4f},{eta.max():.4f}]")
