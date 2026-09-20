#pragma once

#include "scheme/Scheme.h"

namespace PhiX::scheme {

struct CD2 {
    static constexpr int ghostRequired() { return 1; }
    static constexpr int order() { return 2; }
    static constexpr const char* name() { return "CD2"; }

    // --- Separable per-axis primitives (stride-based) ---------------------
    __host__ __device__
    static Real d1(const Real* s, int c, int stride, Real inv_d) {
        return (s[c + stride] - s[c - stride]) * Real(0.5) * inv_d;
    }

    __host__ __device__
    static Real d2(const Real* s, int c, int stride, Real inv_d2) {
        return (s[c + stride] - Real(2) * s[c] + s[c - stride]) * inv_d2;
    }

    // --- Full-operator stencils (used by operator factories) -------------
    // Standard separable Laplacian: sum of second derivatives over axes.
    __host__ __device__
    static Real laplacian(const Real* s, int c,
                            int sx, int sy, int dim,
                            Real inv_dx2, Real inv_dy2, Real inv_dz2) {
        Real val = d2(s, c, 1, inv_dx2);
        if (dim >= 2) val += d2(s, c, sx, inv_dy2);
        if (dim >= 3) val += d2(s, c, sx * sy, inv_dz2);
        return val;
    }

    // Standard 3-point central gradient along `axis`.
    __host__ __device__
    static Real gradient(const Real* s, int c, int axis,
                           int sx, int sy, int /*dim*/,
                           Real inv_dx, Real inv_dy, Real inv_dz) {
        int    stride = (axis == 0) ? 1 : (axis == 1) ? sx : sx * sy;
        Real inv_d  = (axis == 0) ? inv_dx : (axis == 1) ? inv_dy : inv_dz;
        return d1(s, c, stride, inv_d);
    }
};

// ---------------------------------------------------------------------------
// CD4 — 4th-order central differences (5-point per axis, ghost >= 2)
//
//   d1: (f[i-2] - 8f[i-1] + 8f[i+1] - f[i+2]) / (12*dx)
//   d2: (-f[i-2] + 16f[i-1] - 30f[i] + 16f[i+1] - f[i+2]) / (12*dx²)
//
// Laplacian is the separable sum of per-axis d2 (works in 1D/2D/3D on
// non-square meshes).  Requires TWO ghost layers; setRHS validates this
// against the field's ghost width and throws on mismatch.
// ---------------------------------------------------------------------------
struct CD4 {
    static constexpr int ghostRequired() { return 2; }
    static constexpr int order() { return 4; }
    static constexpr const char* name() { return "CD4"; }

    __host__ __device__
    static Real d1(const Real* s, int c, int stride, Real inv_d) {
        return (s[c - 2*stride] - Real(8)*s[c - stride]
              + Real(8)*s[c + stride] - s[c + 2*stride]) * inv_d / Real(12);
    }

    __host__ __device__
    static Real d2(const Real* s, int c, int stride, Real inv_d2) {
        return (-s[c - 2*stride] + Real(16)*s[c - stride] - Real(30)*s[c]
              + Real(16)*s[c + stride] - s[c + 2*stride]) * inv_d2 / Real(12);
    }

    __host__ __device__
    static Real laplacian(const Real* s, int c,
                            int sx, int sy, int dim,
                            Real inv_dx2, Real inv_dy2, Real inv_dz2) {
        Real val = d2(s, c, 1, inv_dx2);
        if (dim >= 2) val += d2(s, c, sx, inv_dy2);
        if (dim >= 3) val += d2(s, c, sx * sy, inv_dz2);
        return val;
    }

    __host__ __device__
    static Real gradient(const Real* s, int c, int axis,
                           int sx, int sy, int /*dim*/,
                           Real inv_dx, Real inv_dy, Real inv_dz) {
        int    stride = (axis == 0) ? 1 : (axis == 1) ? sx : sx * sy;
        Real inv_d  = (axis == 0) ? inv_dx : (axis == 1) ? inv_dy : inv_dz;
        return d1(s, c, stride, inv_d);
    }
};

// ---------------------------------------------------------------------------
// CD6 — 6th-order central differences (7-point per axis, ghost >= 3)
//
//   d1: (−f[i−3] + 9f[i−2] − 45f[i−1] + 45f[i+1] − 9f[i+2] + f[i+3]) / (60·dx)
//   d2: (2f[i−3] − 27f[i−2] + 270f[i−1] − 490f[i]
//        + 270f[i+1] − 27f[i+2] + 2f[i+3]) / (180·dx²)
// ---------------------------------------------------------------------------
struct CD6 {
    static constexpr int ghostRequired() { return 3; }
    static constexpr int order() { return 6; }
    static constexpr const char* name() { return "CD6"; }

    __host__ __device__
    static Real d1(const Real* s, int c, int stride, Real inv_d) {
        return (-s[c - 3*stride] + Real(9)*s[c - 2*stride]
                - Real(45)*s[c - stride] + Real(45)*s[c + stride]
                - Real(9)*s[c + 2*stride] + s[c + 3*stride])
               * inv_d / Real(60);
    }

    __host__ __device__
    static Real d2(const Real* s, int c, int stride, Real inv_d2) {
        return (Real(2)*s[c - 3*stride] - Real(27)*s[c - 2*stride]
                + Real(270)*s[c - stride] - Real(490)*s[c]
                + Real(270)*s[c + stride] - Real(27)*s[c + 2*stride]
                + Real(2)*s[c + 3*stride]) * inv_d2 / Real(180);
    }

    __host__ __device__
    static Real laplacian(const Real* s, int c,
                          int sx, int sy, int dim,
                          Real inv_dx2, Real inv_dy2, Real inv_dz2) {
        Real val = d2(s, c, 1, inv_dx2);
        if (dim >= 2) val += d2(s, c, sx, inv_dy2);
        if (dim >= 3) val += d2(s, c, sx * sy, inv_dz2);
        return val;
    }

    __host__ __device__
    static Real gradient(const Real* s, int c, int axis,
                         int sx, int sy, int /*dim*/,
                         Real inv_dx, Real inv_dy, Real inv_dz) {
        int  stride = (axis == 0) ? 1 : (axis == 1) ? sx : sx * sy;
        Real inv_d  = (axis == 0) ? inv_dx : (axis == 1) ? inv_dy : inv_dz;
        return d1(s, c, stride, inv_d);
    }
};

} // namespace PhiX::scheme
