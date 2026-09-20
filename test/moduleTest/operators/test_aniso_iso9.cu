// ---------------------------------------------------------------------------
// module_aniso_iso9 — scheme-selectable anisotropic divergence (AnisoScheme).
//
// The Iso9 face-normal gradient (transverse weights 1/12, 5/6, 1/12) removes
// the fourfold O(dx²) grid anisotropy of the 5-point form while KEEPING the
// face-flux structure: conservative and checkerboard-damped, unlike the
// two-pass nodal Iso9 composite (checkerboard symbol 0 → blow-up) that this
// scheme deliberately replaces.
//
// Checks:
//   1. EXACT LIMIT: eps = 0, Iso9 → anisoDiv == W0²·lap(f, "Iso9")
//      (9-point Patra–Karttunen), GPU and CPU (1e-11 rel).
//   2. CHECKERBOARD DAMPING: on f = (−1)^{i+j}, eps = 0, the Iso9 scheme
//      returns exactly −(16/3)/dx²·W0²·f per cell (1e-12) — the composite
//      two-pass variant would return 0 here.
//   3. GPU == CPU for Iso9 at eps = 0.05 (both overloads, 1e-13).
//   4. CONSERVATION: periodic sum(anisoDiv) ~ 0 with Iso9 + bicrystal
//      theta0Field (telescoping preserved).
//   5. REDUCTION: uniform theta0Field == scalar theta0 under Iso9 (1e-14).
//   6. anisoFactor Iso9: exact a(θ) on a linear (uniform-gradient) field.
//   7. anisoSchemeFromString: "CD2"/"Iso9" map, unknown name throws.
//   8. DEFAULT UNCHANGED: AnisoParams{}.scheme == CD2 and Iso9 differs from
//      CD2 on a generic field (the knob is real, not a no-op).
// ---------------------------------------------------------------------------
#include "equation/Equation.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "operators/Anisotropy.h"
#include "operators/Laplacian.h"
#include "boundary/PeriodicBC.h"
#include "boundary/BCBatch.h"
#include "mesh/Mesh.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PhiX;

static int failures = 0;
static void require(bool cond, const std::string& msg) {
    if (!cond) { ++failures; std::printf("  FAIL: %s\n", msg.c_str()); }
    else       {             std::printf("  ok  : %s\n", msg.c_str()); }
}

int main() {
try {
    const int N = 64;
    const double dx = 0.8;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    N, dx, 0.0, N, dx, 0.0);
    PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
    std::vector<BoundaryCondition*> bcs = {&bx, &by};

    ScalarField phi(mesh, "phi", 1);
    phi.initialize([&](double x, double y, double) {
        const double L = N * dx;
        return std::sin(2.0 * M_PI * x / L) * std::cos(4.0 * M_PI * y / L)
             + 0.3 * std::cos(6.0 * M_PI * (x + y) / L);
    });
    phi.allocDevice(); phi.uploadAllToDevice();
    BCBatch bPhi; bPhi.build(phi, bcs);   // corner ghosts needed (diagonals)
    bPhi.applyOnGPU(phi);
    phi.downloadCurrFromDevice();

    auto mkOut = [&](const char* n) {
        ScalarField f(mesh, n, 1);
        f.fill(0.0); f.allocDevice(); f.uploadAllToDevice();
        return f;
    };
    auto maxRel = [&](const ScalarField& a, const ScalarField& b) {
        double m = 0.0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const double av = a.curr[a.index(i, j, 0)];
                const double bv = b.curr[b.index(i, j, 0)];
                m = std::max(m, std::fabs(av - bv)
                                / std::max(1.0, std::fabs(bv)));
            }
        return m;
    };

    // === 1. eps = 0, Iso9 == W0²·9-point Patra–Karttunen ==================
    {
        AnisoParams p0;
        p0.W0 = 1.7; p0.eps = 0.0; p0.m = 4;
        p0.scheme = AnisoScheme::Iso9;

        ScalarField oA = mkOut("oA"), oL = mkOut("oL");
        Equation eA(oA, "aniso"); eA.setRHS(anisoDiv(phi, p0));
        eA.computeRHS(oA); cudaDeviceSynchronize(); oA.downloadCurrFromDevice();

        Equation eL(oL, "lap");
        eL.setRHS(lap(phi, "Iso9", p0.W0 * p0.W0));
        eL.computeRHS(oL); cudaDeviceSynchronize(); oL.downloadCurrFromDevice();

        require(maxRel(oA, oL) < 1e-11,
                "eps=0 Iso9 == W0^2 * PK 9-point laplacian (GPU)");

        eA.computeRHSCPU(oA);
        eL.computeRHSCPU(oL);
        require(maxRel(oA, oL) < 1e-11,
                "eps=0 Iso9 == W0^2 * PK 9-point laplacian (CPU)");
    }

    // === 2. checkerboard: symbol −16/3 exactly =============================
    {
        ScalarField cb(mesh, "cb", 1);
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                cb.curr[cb.index(i, j, 0)] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        cb.allocDevice(); cb.uploadAllToDevice();
        BCBatch bCb; bCb.build(cb, bcs);   // N even → periodic-compatible
        bCb.applyOnGPU(cb);
        cb.downloadCurrFromDevice();

        AnisoParams p0;
        p0.W0 = 1.0; p0.eps = 0.0; p0.m = 4;
        p0.scheme = AnisoScheme::Iso9;

        ScalarField o = mkOut("ocb");
        Equation e(o, "cb"); e.setRHS(anisoDiv(cb, p0));
        e.computeRHS(o); cudaDeviceSynchronize(); o.downloadCurrFromDevice();

        const double lam = -(16.0 / 3.0) / (dx * dx);
        double dev = 0.0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const double want = lam * cb.curr[cb.index(i, j, 0)];
                dev = std::max(dev, std::fabs(o.curr[o.index(i, j, 0)] - want)
                                    / std::fabs(lam));
            }
        require(dev < 1e-12, "checkerboard symbol == -16/3 (nonzero damping)");
    }

    // === 3. GPU == CPU for Iso9, eps = 0.05 ================================
    AnisoParams p;
    p.W0 = 1.0; p.eps = 0.05; p.m = 4; p.theta0 = 0.35;
    p.scheme = AnisoScheme::Iso9;
    {
        ScalarField oG = mkOut("oG"), oC = mkOut("oC");
        Equation e(oG, "iso9"); e.setRHS(anisoDiv(phi, p));
        e.computeRHS(oG); cudaDeviceSynchronize(); oG.downloadCurrFromDevice();
        e.computeRHSCPU(oC);
        require(maxRel(oG, oC) < 1e-13, "Iso9 scalar-theta0: GPU == CPU");
    }

    // bicrystal orientation field for 4/5
    ScalarField thF(mesh, "th", 1);
    for (std::size_t k = 0; k < thF.storedSize; ++k)
        thF.curr[k] = 0.0;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            thF.curr[thF.index(i, j, 0)] = (i < N / 2) ? 0.15 : 0.55;
    thF.allocDevice(); thF.uploadAllToDevice();
    BCBatch bTh; bTh.build(thF, bcs);
    bTh.applyOnGPU(thF);
    thF.downloadCurrFromDevice();

    {
        ScalarField oG = mkOut("tG"), oC = mkOut("tC");
        Equation e(oG, "iso9th"); e.setRHS(anisoDiv(phi, thF, p));
        e.computeRHS(oG); cudaDeviceSynchronize(); oG.downloadCurrFromDevice();
        e.computeRHSCPU(oC);
        require(maxRel(oG, oC) < 1e-13, "Iso9 theta0Field: GPU == CPU");

        // === 4. conservation with Iso9 + bicrystal =========================
        const double s  = reduce::fieldSum(oG);
        double sAbs = 0.0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                sAbs += std::fabs(oG.curr[oG.index(i, j, 0)]);
        require(std::fabs(s) <= 1e-10 * std::max(1.0, sAbs),
                "Iso9 bicrystal conservation: sum(anisoDiv) ~ 0");
    }

    // === 5. uniform theta0Field == scalar under Iso9 =======================
    {
        ScalarField thU(mesh, "thU", 1);
        for (std::size_t k = 0; k < thU.storedSize; ++k)
            thU.curr[k] = static_cast<Real>(p.theta0);
        thU.allocDevice(); thU.uploadAllToDevice();

        ScalarField oS = mkOut("uS"), oF = mkOut("uF");
        Equation eS(oS, "s"); eS.setRHS(anisoDiv(phi, p));
        eS.computeRHS(oS); cudaDeviceSynchronize(); oS.downloadCurrFromDevice();
        Equation eF(oF, "f"); eF.setRHS(anisoDiv(phi, thU, p));
        eF.computeRHS(oF); cudaDeviceSynchronize(); oF.downloadCurrFromDevice();
        require(maxRel(oF, oS) < 1e-14,
                "Iso9: uniform theta0Field == scalar theta0");
    }

    // === 6. anisoFactor Iso9: exact on a uniform-gradient field ============
    {
        ScalarField lin(mesh, "lin", 1);
        const double ax = 0.7, ay = -0.4;
        lin.initialize([&](double x, double y, double) {
            return ax * x + ay * y;
        });
        // linear field: fill ghosts analytically (periodic wrap would break
        // linearity) — both CD2 and Iso9 gradients only read +-1 cells
        for (int j = -1; j <= N; ++j)
            for (int i = -1; i <= N; ++i)
                lin.curr[lin.index(i, j, 0)] =
                    ax * mesh.coord(0, i) + ay * mesh.coord(1, j);
        lin.allocDevice(); lin.uploadAllToDevice();

        AnisoParams pf = p;   // eps = 0.05, m = 4, theta0 = 0.35, Iso9
        ScalarField aF = mkOut("aF");
        anisoFactorOnGPU(lin, aF, pf);
        cudaDeviceSynchronize(); aF.downloadCurrFromDevice();

        const double theta = std::atan2(ay, ax);
        const double want = 1.0 + pf.eps
            * std::cos(pf.m * (theta - pf.theta0));
        double dev = 0.0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                dev = std::max(dev,
                    std::fabs(aF.curr[aF.index(i, j, 0)] - want));
        require(dev < 1e-13, "anisoFactor Iso9 exact on linear field");
    }

    // === 7. string helper ==================================================
    {
        require(anisoSchemeFromString("CD2") == AnisoScheme::CD2
                && anisoSchemeFromString("Iso9") == AnisoScheme::Iso9,
                "anisoSchemeFromString maps CD2/Iso9");
        bool threw = false;
        try { anisoSchemeFromString("iso9"); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "anisoSchemeFromString throws on unknown name");
    }

    // === 8. default is CD2; Iso9 actually differs ==========================
    {
        require(AnisoParams{}.scheme == AnisoScheme::CD2,
                "default scheme is CD2 (legacy path untouched)");

        AnisoParams pc = p; pc.scheme = AnisoScheme::CD2;
        ScalarField oI = mkOut("dI"), oC = mkOut("dC");
        Equation eI(oI, "i"); eI.setRHS(anisoDiv(phi, p));
        eI.computeRHS(oI); cudaDeviceSynchronize(); oI.downloadCurrFromDevice();
        Equation eC(oC, "c"); eC.setRHS(anisoDiv(phi, pc));
        eC.computeRHS(oC); cudaDeviceSynchronize(); oC.downloadCurrFromDevice();
        require(maxRel(oI, oC) > 1e-6, "Iso9 differs from CD2 (knob is real)");
    }

    if (failures) { std::printf("module_aniso_iso9: %d FAILURES\n", failures); return 1; }
    std::printf("module_aniso_iso9: all passed\n");
    return 0;
} catch (const std::exception& ex) {
    std::printf("EXCEPTION: %s\n", ex.what());
    return 2;
}
}
