#pragma once

// ---------------------------------------------------------------------------
// Interpolants.h — the standard phase-field interpolation / double-well
// polynomials, shared by the GP and WBM/GFA solver families.
//
//   h_cubic   (x) = x²(3−2x)            h_cubic_prime   = 6x(1−x)     (KKS)
//   h_quintic (x) = x³(6x²−15x+10)      h_quintic_prime = 30x²(1−x)²  (PFHub)
//   g_dw      (x) = x²(1−x)²            g_dw_prime      = 2x(1−x)(1−2x)
//
// All functions are host+device inline (usable inside PHIX_FN functors and
// plain host code alike).  h_interp/h_interp_prime select cubic vs quintic
// at runtime — the pattern used by the grand-potential solver's "interp"
// config key.
//
// These definitions are the single source of truth for NEW solvers; the
// pre-v3.3 solvers keep their own frozen copies (regression oracles).
// ---------------------------------------------------------------------------

#ifdef __CUDACC__
#define PHIX_MAT_HD __host__ __device__
#else
#define PHIX_MAT_HD
#endif

namespace PhiX {
namespace Material {

// --- cubic (KKS) interpolant ------------------------------------------------
PHIX_MAT_HD inline double h_cubic(double x)       { return x * x * (3.0 - 2.0 * x); }
PHIX_MAT_HD inline double h_cubic_prime(double x) { return 6.0 * x * (1.0 - x); }

// --- quintic (PFHub / stage-6 GFA) interpolant ------------------------------
PHIX_MAT_HD inline double h_quintic(double x) {
    return x * x * x * (6.0 * x * x - 15.0 * x + 10.0);
}
PHIX_MAT_HD inline double h_quintic_prime(double x) {
    return 30.0 * x * x * (1.0 - x) * (1.0 - x);
}

// --- runtime selection (quintic ? h_quintic : h_cubic) ----------------------
PHIX_MAT_HD inline double h_interp(double x, bool quintic) {
    return quintic ? h_quintic(x) : h_cubic(x);
}
PHIX_MAT_HD inline double h_interp_prime(double x, bool quintic) {
    return quintic ? h_quintic_prime(x) : h_cubic_prime(x);
}

// --- standard double well ---------------------------------------------------
PHIX_MAT_HD inline double g_dw(double x) {
    return x * x * (1.0 - x) * (1.0 - x);
}
PHIX_MAT_HD inline double g_dw_prime(double x) {
    return 2.0 * x * (1.0 - x) * (1.0 - 2.0 * x);
}

} // namespace Material
} // namespace PhiX
