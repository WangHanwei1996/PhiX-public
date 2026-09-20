#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成温度相关的 CH 成分迁移率表 M_c_L(T)、M_c_S(T)（供 GFA_FeB 从表插值）。
物理依据 = calibration-5.0/doc/Mc_diffusivity_relation.md + calibration-5.0/img/D_liquid_solid_vs_temperature.png：

  固相：D_S(T)=D0*exp(-Q/RT)，D0=8.7e-5 m²/s，Q=190 kJ/mol（硼化文献，Keddam/Campos-Silva…）
        M_c_S(T)=D_S(T)/χ_S，  χ_S=2*rho_s^2=1.8e11 (rho_s=3e5)
  液相：D_L(T)=4e-9*exp(-Q_L/R*(1/T-1/1586))，Q_L=50 kJ/mol（液态金属自扩散典型值，⚠假设，可调）
        M_c_L(T)=D_L(T)/χ_L^eff，χ_L^eff=2.94e10（doc §8/§10 实测有效曲率）

校核：M_c_L(1586)=1.36e-19、M_c_S(1586)=2.7e-22（与旧常数一致）。
⚠ 若改 rho_s（→χ_S 变）或想改 D_L 的 Q_L，重跑本脚本。
表格式：nc=2（两行相同，避开 nc=1 的 dc 除零）, nT=64, T∈[500,2000] K, CSV。
用法：python3 gen_Mc_tables.py
"""
import os, math

R = 8.314
HERE = os.path.dirname(os.path.abspath(__file__))
NC, NT = 2, 64
CMIN, CMAX, TMIN, TMAX = 0.0, 1.0, 500.0, 2000.0
Tgrid = [TMIN + (TMAX - TMIN) * i / (NT - 1) for i in range(NT)]

# --- physical diffusivities ---
def D_S(T): return 8.7e-5 * math.exp(-190e3 / (R * T))
def D_L(T): return 4e-9 * math.exp(-50e3 / R * (1.0 / T - 1.0 / 1586.0))
CHI_S = 1.8e11      # = 2*rho_s^2, rho_s=3e5
CHI_L = 2.94e10     # χ_L^eff (doc)

def M_c_S(T): return D_S(T) / CHI_S
def M_c_L(T): return D_L(T) / CHI_L

def write_table(name, fn, vals, extra):
    path = os.path.join(HERE, fn)
    with open(path, "w") as f:
        f.write("# ============================================================\n")
        f.write(f"# 成分迁移率表 — {name}  [m^5/(J·s)]\n")
        f.write("#\n")
        for ln in extra: f.write(f"# {ln}\n")
        f.write(f"# nc={NC}(两行相同,c 无关) nT={NT}  T∈[{TMIN},{TMAX}] K\n")
        f.write("# 第一非注释行: nc,nT,c_min,c_max,T_min,T_max ; 之后 nc 行、每行 nT 个逗号值\n")
        f.write("# ============================================================\n")
        f.write(f"{NC}, {NT}, {CMIN}, {CMAX}, {TMIN}, {TMAX}\n")
        row = ",".join(f"{v:.6e}" for v in vals)
        for _ in range(NC):
            f.write(row + "\n")
    return path

pL = write_table("M_c_L(T) 液相", "M_c_L_table.csv", [M_c_L(T) for T in Tgrid],
                 ["D_L(T)=4e-9*exp(-Q_L/R*(1/T-1/1586)), Q_L=50 kJ/mol (⚠液态金属典型值,假设可调)",
                  "M_c_L=D_L/χ_L^eff, χ_L^eff=2.94e10 (doc §8/§10)"])
pS = write_table("M_c_S(T) 固相 Fe2B", "M_c_S_table.csv", [M_c_S(T) for T in Tgrid],
                 ["D_S(T)=8.7e-5*exp(-190e3/RT) (硼化文献 Q=190 kJ/mol, 实测 1123-1273K 外推)",
                  "M_c_S=D_S/χ_S, χ_S=2*rho_s^2=1.8e11 (rho_s=3e5 ⇒ 改 rho_s 需重生成)"])

print("written:")
for p in (pL, pS): print("  ", p)
print(f"sanity @1586K:  M_c_L={M_c_L(1586):.3e} (old 1.36e-19)   M_c_S={M_c_S(1586):.3e} (old 2.7e-22)")
print(f"        @700K:  M_c_L={M_c_L(700):.3e}   M_c_S={M_c_S(700):.3e}   (冻结)")
print(f"       @2000K:  M_c_L={M_c_L(2000):.3e}   M_c_S={M_c_S(2000):.3e}")
