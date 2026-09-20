#pragma once

#include "scheme/Scheme.h"
#include "scheme/CentralDifference.h"

namespace PhiX::scheme {

// ---------------------------------------------------------------------------
// Iso9  —  9-point isotropic stencil (Patra-Karttunen, 2D only)
//
// Gradient (direction x, 2D):
//   df/dx[i,j] ≈ [4(f[i+1,j]-f[i-1,j])
//                +  (f[i+1,j+1]-f[i-1,j+1])
//                +  (f[i+1,j-1]-f[i-1,j-1])] / (12*dx)
//
// The Laplacian is NOT the direct sum of separable d² terms.
// Isotropic 9-point Laplacian (Patra-Karttunen):
//   ∇²f[i,j] ≈ [ 4·(f[i+1,j]+f[i-1,j]+f[i,j+1]+f[i,j-1])
//               +  (f[i+1,j+1]+f[i-1,j+1]+f[i+1,j-1]+f[i-1,j-1])
//               − 20·f[i,j] ] / (6*dx²)           (assumes dx==dy)
//
// For 1D or 3D meshes, both laplacian() and gradient() automatically fall
// back to standard CD2.  Stencil width is still 1 in all directions.
//
// RESTRICTION: the isotropic weights assume uniform spacing (dx == dy).
//              For non-square meshes, set the scheme explicitly to CD2.
// ---------------------------------------------------------------------------
struct Iso9 {
    static constexpr int ghostRequired() { return 1; }
    static constexpr int order() { return 2; }   // effective order on smooth solutions
    static constexpr const char* name() { return "Iso9"; }

    // Per-axis d1 fallback (CD2); used when dim != 2 or axis >= 2.
    __host__ __device__
    static Real d1(const Real* s, int c, int stride, Real inv_d) {
        return CD2::d1(s, c, stride, inv_d);
    }

    // Per-axis d2 fallback (CD2).
    __host__ __device__
    static Real d2(const Real* s, int c, int stride, Real inv_d2) {
        return CD2::d2(s, c, stride, inv_d2);
    }

    // -------------------------------------------------------------------------
    // Isotropic 9-point Laplacian (2D).  Falls back to CD2 for 1D/3D.
    // inv_dx2 must equal inv_dy2 (square mesh); the check is omitted for
    // performance — callers are responsible.
    // -------------------------------------------------------------------------
    __host__ __device__
    static Real laplacian(const Real* s, int c,
                            int sx, int /*sy*/, int dim,
                            Real inv_dx2, Real inv_dy2, Real inv_dz2) {
        if (dim == 2) {
            // Isotropic 9-point (Mehrstellen / Patra-Karttunen) weights:
            //   [4·(face sum) + (corner sum) − 20·center] / (6·dx²)
            // Consistency check on f = x²: 4·(4x²+2dx²)+(4x²+4dx²)−20x²
            //   = 12dx² → /(6dx²) = 2 = ∇²x².  (inv_dx2 assumed == inv_dy2.)
            // NOTE: weights fixed in v2.11.1 — the previous
            //   (face/2 + corner/4 − 3c)·(2/(3dx²))
            // form converged to (2/3)·∇² (zeroth-order inconsistent), caught
            // by the convergence suite.
            Real center  = s[c];
            Real face    = s[c + 1]      + s[c - 1]
                           + s[c + sx]     + s[c - sx];
            Real corner  = s[c + 1 + sx] + s[c - 1 + sx]
                           + s[c + 1 - sx] + s[c - 1 - sx];
            return (Real(4) * face + corner - Real(20) * center) / Real(6) * inv_dx2;
        }
        // fallback
        return CD2::laplacian(s, c, sx, /*sy will be unused in 1D*/ 1, dim,
                              inv_dx2, inv_dy2, inv_dz2);
    }

    // -------------------------------------------------------------------------
    // Isotropic 9-point gradient along `axis` (2D only).  Falls back for 1D/3D.
    // -------------------------------------------------------------------------
    __host__ __device__
    static Real gradient(const Real* s, int c, int axis,
                           int sx, int sy, int dim,
                           Real inv_dx, Real inv_dy, Real inv_dz) {
        if (dim == 2 && axis <= 1) {
            if (axis == 0) {
                Real inv_12dx = inv_dx / Real(12);
                int xp_jm = c + 1 - sx;
                int xp_j  = c + 1;
                int xp_jp = c + 1 + sx;
                int xm_jm = c - 1 - sx;
                int xm_j  = c - 1;
                int xm_jp = c - 1 + sx;
                return (Real(4) * (s[xp_j] - s[xm_j])
                           + (s[xp_jp] - s[xm_jp])
                           + (s[xp_jm] - s[xm_jm])) * inv_12dx;
            } else {
                Real inv_12dy = inv_dy / Real(12);
                int xm_yp = c - 1 + sx;
                int x_yp  = c     + sx;
                int xp_yp = c + 1 + sx;
                int xm_ym = c - 1 - sx;
                int x_ym  = c     - sx;
                int xp_ym = c + 1 - sx;
                return (Real(4) * (s[x_yp] - s[x_ym])
                           + (s[xp_yp] - s[xp_ym])
                           + (s[xm_yp] - s[xm_ym])) * inv_12dy;
            }
        }
        return CD2::gradient(s, c, axis, sx, sy, dim, inv_dx, inv_dy, inv_dz);
    }
};

// ---------------------------------------------------------------------------
// Iso27 — 27-point isotropic 3D Laplacian (O(h²), lattice-isotropic error)
//
//   ∇²f ≈ [ 7/15·(face sum) + 1/10·(edge sum) + 1/30·(corner sum)
//           − 64/15·f ] / h²          (requires dx == dy == dz)
//
// Consistency (per-axis second moment): 2·7/15 + 8·1/10 + 8·1/30 = 2  ✓
// Falls back to CD2 in 1D/2D and for gradient().
// ---------------------------------------------------------------------------
struct Iso27 {
    static constexpr int ghostRequired() { return 1; }
    static constexpr int order() { return 2; }
    static constexpr const char* name() { return "Iso27"; }

    __host__ __device__
    static Real d1(const Real* s, int c, int stride, Real inv_d) {
        return CD2::d1(s, c, stride, inv_d);
    }
    __host__ __device__
    static Real d2(const Real* s, int c, int stride, Real inv_d2) {
        return CD2::d2(s, c, stride, inv_d2);
    }

    __host__ __device__
    static Real laplacian(const Real* s, int c,
                          int sx, int sy, int dim,
                          Real inv_dx2, Real inv_dy2, Real inv_dz2) {
        if (dim == 3) {
            const int sz = sx * sy;
            const Real face =
                s[c+1] + s[c-1] + s[c+sx] + s[c-sx] + s[c+sz] + s[c-sz];
            const Real edge =
                s[c+1+sx] + s[c-1+sx] + s[c+1-sx] + s[c-1-sx]
              + s[c+1+sz] + s[c-1+sz] + s[c+1-sz] + s[c-1-sz]
              + s[c+sx+sz] + s[c-sx+sz] + s[c+sx-sz] + s[c-sx-sz];
            const Real corner =
                s[c+1+sx+sz] + s[c-1+sx+sz] + s[c+1-sx+sz] + s[c-1-sx+sz]
              + s[c+1+sx-sz] + s[c-1+sx-sz] + s[c+1-sx-sz] + s[c-1-sx-sz];
            return (Real(7.0/15.0) * face + Real(0.1) * edge
                    + Real(1.0/30.0) * corner
                    - Real(64.0/15.0) * s[c]) * inv_dx2;
        }
        return CD2::laplacian(s, c, sx, sy, dim, inv_dx2, inv_dy2, inv_dz2);
    }

    __host__ __device__
    static Real gradient(const Real* s, int c, int axis,
                         int sx, int sy, int dim,
                         Real inv_dx, Real inv_dy, Real inv_dz) {
        return CD2::gradient(s, c, axis, sx, sy, dim, inv_dx, inv_dy, inv_dz);
    }
};

} // namespace PhiX::scheme
