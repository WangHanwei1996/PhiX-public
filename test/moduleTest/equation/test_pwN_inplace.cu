// ---------------------------------------------------------------------------
// module_pwN_inplace — v3.7.0 equation-layer capability uplift.
//
// Covers the two halves of the release:
//
//  A. pw() with 5..8 fields.  The bicrystal directional-solidification
//     solver needed (psi, U, thT, p, psi_x, psi_y) in one functor and had
//     to materialise ~15 ferry fields under the old 4-field cap.
//     Verifies for N = 5..8:
//       1. GPU (Equation::computeRHS) matches an independent host reference.
//       2. CPU (computeRHSCPU) matches the same reference.
//       3. coeff scaling and multi-term accumulation (pwN + lap) work.
//
//  B. In-place computeRHS safety.  computeRHS zeroes its target before the
//     terms accumulate; before v3.7.0 a term READING the target therefore
//     saw zeros.  This silently pinned a solute field at zero for a whole
//     production run (the functor was a clamp: clamp(0) = 0 looked alive).
//     Verifies:
//       4. eq.setRHS(pw(f, clamp)); eq.computeRHS(f) equals clamp(f_old),
//          on GPU and CPU (the exact pattern that failed).
//       5. advanceSteady with the unknown in its own RHS (grain-index
//          update pattern: pw(p, S, phi)) equals the out-of-place result.
//       6. Aliased multi-term RHS (pw(f,g) + lap(f)) into f equals the
//          out-of-place evaluation into a fresh field.
// ---------------------------------------------------------------------------
#include "equation/Equation.h"
#include "equation/Term.h"
#include "field/ScalarField.h"
#include "operators/Laplacian.h"
#include "boundary/PeriodicBC.h"
#include "mesh/Mesh.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace PhiX;

static int failures = 0;
static void require(bool cond, const std::string& msg) {
    if (!cond) { ++failures; std::printf("  FAIL: %s\n", msg.c_str()); }
    else       {             std::printf("  ok  : %s\n", msg.c_str()); }
}

static double hashv(int i, int j, int salt) {
    double v = std::sin(12.9898 * (i + 1) + 78.233 * (j + 1) + 37.7 * salt)
             * 43758.5453;
    return v - std::floor(v) - 0.5;
}

static void fill(ScalarField& f, int salt) {
    for (int j = 0; j < f.mesh.n[1]; ++j)
        for (int i = 0; i < f.mesh.n[0]; ++i)
            f.curr[f.index(i, j, 0)] = static_cast<Real>(hashv(i, j, salt));
    f.uploadAllToDevice();
}

static double maxRelDiff(const ScalarField& a, const std::vector<double>& ref) {
    double m = 0.0; std::size_t k = 0;
    for (int j = 0; j < a.mesh.n[1]; ++j)
        for (int i = 0; i < a.mesh.n[0]; ++i, ++k) {
            const double av = a.curr[a.index(i, j, 0)];
            const double d  = std::fabs(av - ref[k]);
            const double s  = std::max(1.0, std::fabs(ref[k]));
            m = std::max(m, d / s);
        }
    return m;
}

int main() {
try {
    const int N = 64;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    N, 0.5, 0.0, N, 0.5, 0.0);
    auto mk = [&](const char* n, int salt) {
        ScalarField f(mesh, n, 1);
        f.fill(0.0); f.allocDevice(); fill(f, salt);
        return f;
    };
    ScalarField f1 = mk("f1", 1), f2 = mk("f2", 2), f3 = mk("f3", 3),
                f4 = mk("f4", 4), f5 = mk("f5", 5), f6 = mk("f6", 6),
                f7 = mk("f7", 7), f8 = mk("f8", 8);
    ScalarField out(mesh, "out", 1);
    out.fill(0.0); out.allocDevice(); out.uploadAllToDevice();

    // ------------------------------------------------------------------ A
    std::printf("A. pw with 5..8 fields, GPU + CPU vs host reference\n");

    auto func5 = PHIX_FN (double a, double b, double c, double d, double e) {
        return a * b - c * d + 0.5 * e; };
    auto func6 = PHIX_FN (double a, double b, double c, double d, double e,
                          double g) { return a * (b - c) + d * e - g; };
    auto func7 = PHIX_FN (double a, double b, double c, double d, double e,
                          double g, double h) {
        return a + b * c - d * e + g * h; };
    auto func8 = PHIX_FN (double a, double b, double c, double d, double e,
                          double g, double h, double q) {
        return (a - b) * (c + d) + (e - g) * (h + q); };

    std::vector<double> ref(static_cast<std::size_t>(N) * N);
    auto refAt = [&](int i, int j) {
        const double a = hashv(i, j, 1), b = hashv(i, j, 2),
                     c = hashv(i, j, 3), d = hashv(i, j, 4),
                     e = hashv(i, j, 5), g = hashv(i, j, 6),
                     h = hashv(i, j, 7), q = hashv(i, j, 8);
        struct R { double v5, v6, v7, v8; } r;
        r.v5 = a * b - c * d + 0.5 * e;
        r.v6 = a * (b - c) + d * e - g;
        r.v7 = a + b * c - d * e + g * h;
        r.v8 = (a - b) * (c + d) + (e - g) * (h + q);
        return r;
    };

    { // N = 5, with a coeff
        Equation eq(out, "pw5");
        eq.setRHS(pw(f1, f2, f3, f4, f5, func5, 2.0));
        eq.computeRHS(out); cudaDeviceSynchronize();
        out.downloadCurrFromDevice();
        std::size_t k = 0;
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i, ++k)
            ref[k] = 2.0 * refAt(i, j).v5;
        require(maxRelDiff(out, ref) < 1e-13, "pw5 GPU (coeff 2.0)");
        eq.computeRHSCPU(out);
        require(maxRelDiff(out, ref) < 1e-13, "pw5 CPU");
    }
    { // N = 6
        Equation eq(out, "pw6");
        eq.setRHS(pw(f1, f2, f3, f4, f5, f6, func6));
        eq.computeRHS(out); cudaDeviceSynchronize();
        out.downloadCurrFromDevice();
        std::size_t k = 0;
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i, ++k)
            ref[k] = refAt(i, j).v6;
        require(maxRelDiff(out, ref) < 1e-13, "pw6 GPU");
        eq.computeRHSCPU(out);
        require(maxRelDiff(out, ref) < 1e-13, "pw6 CPU");
    }
    { // N = 7
        Equation eq(out, "pw7");
        eq.setRHS(pw(f1, f2, f3, f4, f5, f6, f7, func7));
        eq.computeRHS(out); cudaDeviceSynchronize();
        out.downloadCurrFromDevice();
        std::size_t k = 0;
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i, ++k)
            ref[k] = refAt(i, j).v7;
        require(maxRelDiff(out, ref) < 1e-13, "pw7 GPU");
        eq.computeRHSCPU(out);
        require(maxRelDiff(out, ref) < 1e-13, "pw7 CPU");
    }
    { // N = 8, plus accumulation with a second term
        Equation eq(out, "pw8");
        eq.setRHS(pw(f1, f2, f3, f4, f5, f6, f7, f8, func8)
                + pw(f1, PHIX_FN (double a) { return 3.0 * a; }));
        eq.computeRHS(out); cudaDeviceSynchronize();
        out.downloadCurrFromDevice();
        std::size_t k = 0;
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i, ++k)
            ref[k] = refAt(i, j).v8 + 3.0 * hashv(i, j, 1);
        require(maxRelDiff(out, ref) < 1e-13, "pw8 + pw1 accumulation GPU");
        eq.computeRHSCPU(out);
        require(maxRelDiff(out, ref) < 1e-13, "pw8 + pw1 accumulation CPU");
    }

    // ------------------------------------------------------------------ B
    std::printf("B. in-place computeRHS safety\n");

    { // 4. the clamp pattern that silently failed before v3.7.0
        ScalarField u = mk("u", 11);
        std::vector<double> expect(static_cast<std::size_t>(N) * N);
        std::size_t k = 0;
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i, ++k)
            expect[k] = std::fmin(std::fmax(hashv(i, j, 11), -0.25), 0.25);

        Equation eq(u, "clamp");
        eq.setRHS(pw(u, PHIX_FN (double v) {
            return fmin(fmax(v, -0.25), 0.25); }));
        eq.computeRHS(u); cudaDeviceSynchronize();
        u.downloadCurrFromDevice();
        require(maxRelDiff(u, expect) < 1e-14,
                "in-place clamp GPU == clamp(old value), not clamp(0)");

        fill(u, 11);                       // reset and repeat on CPU
        u.downloadCurrFromDevice();        // sync host copy from fill upload
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i)
            u.curr[u.index(i, j, 0)] = static_cast<Real>(hashv(i, j, 11));
        eq.computeRHSCPU(u);
        require(maxRelDiff(u, expect) < 1e-14, "in-place clamp CPU");
    }

    { // 5. advanceSteady with unknown in its own RHS (grain-index pattern)
        ScalarField p = mk("p", 12), S = mk("S", 13), phi = mk("phi", 14);
        auto claim = PHIX_FN (double pp, double s, double ph) {
            if (pp > 0.5 || pp < -0.5) return pp;
            if (ph < 0.0) return 0.0;
            return (s > 0.0) ? 1.0 : -1.0; };
        std::vector<double> expect(static_cast<std::size_t>(N) * N);
        std::size_t k = 0;
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i, ++k) {
            const double pp = hashv(i, j, 12), s = hashv(i, j, 13),
                         ph = hashv(i, j, 14);
            expect[k] = (pp > 0.5 || pp < -0.5) ? pp
                      : (ph < 0.0 ? 0.0 : (s > 0.0 ? 1.0 : -1.0));
        }
        Equation eq(p, "claim");
        eq.setRHS(pw(p, S, phi, claim));
        eq.advanceSteady({}, nullptr);     // p := RHS(p, S, phi)
        cudaDeviceSynchronize();
        p.downloadCurrFromDevice();
        require(maxRelDiff(p, expect) < 1e-14,
                "advanceSteady with self-reading RHS");
    }

    { // 6. aliased multi-term (pointwise + stencil on the target)
        ScalarField a = mk("a", 21), b = mk("b", 22);
        PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
        PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
        bx.applyOnGPU(a); by.applyOnGPU(a);

        ScalarField outref(mesh, "outref", 1);
        outref.fill(0.0); outref.allocDevice(); outref.uploadAllToDevice();

        Equation eqRef(outref, "ref");
        eqRef.setRHS(pw(a, b, PHIX_FN (double x, double y) { return x * y; })
                   + lap(a, 0.7));
        eqRef.computeRHS(outref); cudaDeviceSynchronize();
        outref.downloadCurrFromDevice();
        std::vector<double> expect(static_cast<std::size_t>(N) * N);
        std::size_t k = 0;
        for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i, ++k)
            expect[k] = outref.curr[outref.index(i, j, 0)];

        Equation eqAl(a, "aliased");
        eqAl.setRHS(pw(a, b, PHIX_FN (double x, double y) { return x * y; })
                  + lap(a, 0.7));
        eqAl.computeRHS(a); cudaDeviceSynchronize();
        a.downloadCurrFromDevice();
        require(maxRelDiff(a, expect) < 1e-14,
                "aliased pw+lap == out-of-place result");
    }

    std::printf(failures ? "module_pwN_inplace: %d FAILURE(S)\n"
                         : "module_pwN_inplace: all passed\n", failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 2;
}
}
