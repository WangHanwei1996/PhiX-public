#!/usr/bin/env python3
"""
gen_ic.py — Generate initial condition binary .field files for the
             dendrite growth solver (PFHub Benchmark 3, Karma-Rappel 1998).

Run from  develop/pfhub/bm3_dendrite/:
    python3 settings/gen_ic.py

Output:
    settings/initial_field/phi.field   (solid seed: tanh profile)
    settings/initial_field/U.field     (uniform undercooling U_0 = -0.3)

Binary format (.field):
    Text header lines followed by "---", then binary doubles (float64)
    in row-major order with i (x) varying fastest:
        [k=0,j=0,i=0], [k=0,j=0,i=1], ..., [k=0,j=ny-1,i=nx-1]
    (physical cells only, no ghost)
"""

import os
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Parameters (must match settings/settings.jsonc)
# ---------------------------------------------------------------------------
NX, NY, NZ = 1200, 1200, 1
DX, DY      = 0.8, 0.8
X0, Y0      = 0.0, 0.0

W_0    = 1.0    # interface width  [W_0 units]
GHOST  = 1      # ghost layer count (PhiX default)

DELTA  = 0.3    # initial undercooling |U_0| = delta
R_SEED = 8.0    # solid seed radius [W_0]

OUT_DIR = "settings/initial_field"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_field(path: str, name: str, data_kji: np.ndarray, ghost: int = 1):
    """Write a PhiX binary .field file.

    data_kji: ndarray of shape (nz, ny, nx), dtype float64
              layout matches the C++ loop: k outer, j middle, i inner
    """
    nz, ny, nx = data_kji.shape
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        # ── Text header ──────────────────────────────────────────────
        header  = "# PhiX ScalarField\n"
        header += f"name    {name}\n"
        header += f"nx {nx}  ny {ny}  nz {nz}\n"
        header += f"ghost   {ghost}\n"
        header += "---\n"
        f.write(header.encode("ascii"))

        # ── Binary body: C-contiguous, i fastest ─────────────────────
        buf = data_kji.astype(np.float64)   # already (nz, ny, nx) C-order
        f.write(buf.tobytes())

    print(f"  wrote  {path}  [{nx}×{ny}×{nz}, {nx*ny*nz*8} bytes]")


# ---------------------------------------------------------------------------
# Build grid
# ---------------------------------------------------------------------------
# Cell-centre coordinates (x = i-axis, y = j-axis)
x_c = X0 + (np.arange(NX, dtype=np.float64) + 0.5) * DX  # shape (NX,)
y_c = Y0 + (np.arange(NY, dtype=np.float64) + 0.5) * DY  # shape (NY,)

# Meshgrid, indexing='ij': X[i,j], Y[i,j]  →  shape (NX, NY)
X, Y = np.meshgrid(x_c, y_c, indexing="ij")

# Seed at corner (0, 0) — matching notebook: R = sqrt(X^2 + Y^2)
r = np.sqrt(X**2 + Y**2)   # shape (NX, NY)

# ---------------------------------------------------------------------------
# phi  — tanh nucleation seed (+1 solid / -1 liquid)
# ---------------------------------------------------------------------------
# phi = -tanh((r - R_SEED) / (sqrt(2) * W_0))  [matches notebook sign convention]
# Inside seed (r<R): phi → +1 ; outside (r>R): phi → -1
phi_ij = -np.tanh((r - R_SEED) / (np.sqrt(2.0) * W_0))  # shape (NX, NY)

# Reorder from (NX, NY) to (NZ, NY, NX) = (1, NY, NX)  [k=0,j,i]
phi_kji = phi_ij.T.reshape(NZ, NY, NX)

print(f"phi:  min={phi_kji.min():.4f}  max={phi_kji.max():.4f}  "
      f"seed cells={np.sum(phi_kji > 0)}")

# ---------------------------------------------------------------------------
# U  — uniform initial undercooling
# ---------------------------------------------------------------------------
U_kji = np.full((NZ, NY, NX), -DELTA, dtype=np.float64)

print(f"U  :  uniform = {-DELTA}")

# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------
write_field(f"{OUT_DIR}/phi.field", "phi", phi_kji, ghost=GHOST)
write_field(f"{OUT_DIR}/U.field",   "U",   U_kji,   ghost=GHOST)

print("\nDone.  Run the solver from develop/pfhub/bm3_dendrite/ with:")
print("  ../../../applications/solvers/dendrite_growth/2D/dendrite_growth")
