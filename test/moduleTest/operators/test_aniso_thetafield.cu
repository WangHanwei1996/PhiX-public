// ---------------------------------------------------------------------------
// module_aniso_thetafield — per-cell-orientation anisotropy (v3.7.0).
//
// anisoDiv(phi, theta0Field, p) and anisoFactorOnGPU/CPU(phi, theta0Field,
// aOut, p) let the preferred orientation vary per cell — the entry ticket
// for bicrystal/polycrystal solvers (orientation-field and frozen-GB
// methods), which previously had to hand-roll the anisotropy in pw
// functors.
//
// Checks:
//   1. REDUCTION: uniform theta0Field == scalar-theta0 operator (1e-14),
//      GPU and CPU, at theta0 = 0.35, for both anisoDiv and anisoFactor.
//   2. GPU == CPU for a smoothly varying theta0Field (1e-13).
//   3. CONSERVATION: under periodic BCs the fused divergence telescopes,
//      so sum over cells of anisoDiv must vanish (<= 1e-10 * sum|term|)
//      even with a BICRYSTAL orientation field — both adjacent cells build
//      each shared face flux from identical inputs, orientation included.
//   4. BICRYSTAL PATCH: in grain interiors (away from the orientation
//      seam), the field-theta result equals the corresponding scalar-theta
//      result of that grain (1e-13).
//   5. Regularisation path: eps above the convexity limit with
//      regularize=true runs and stays finite with a bicrystal field.
// ---------------------------------------------------------------------------
#include "equation/Equation.h"
#include "field/ScalarField.h"
#include "operators/Anisotropy.h"
#include "boundary/PeriodicBC.h"
#include "boundary/BCBatch.h"
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

int main() {
try {
    const int N = 64;
    const double dx = 0.8;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    N, dx, 0.0, N, dx, 0.0);
    PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
    std::vector<BoundaryCondition*> bcs = {&bx, &by};

    // smooth multi-mode phi (periodic-compatible)
    ScalarField phi(mesh, "phi", 1);
    phi.initialize([&](double x, double y, double) {
        const double L = N * dx;
        return std::sin(2.0 * M_PI * x / L) * std::cos(4.0 * M_PI * y / L)
             + 0.3 * std::cos(6.0 * M_PI * (x + y) / L);
    });
    phi.allocDevice(); phi.uploadAllToDevice();
    // anisoDiv reads DIAGONAL neighbours (the header is explicit): the
    // corner ghosts matter, so the refresh must go through BCBatch, whose
    // second pass fills them.  Plain per-BC applyOnGPU leaves the corners
    // stale and visibly breaks the telescoping test at the periodic wrap.
    BCBatch bPhi; bPhi.build(phi, bcs);
    bPhi.applyOnGPU(phi);
    phi.downloadCurrFromDevice();          // host mirror incl. ghosts for CPU

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

    AnisoParams p;
    p.W0 = 1.0; p.eps = 0.03; p.m = 4;

    // === 1. reduction: uniform field == scalar ============================
    {
        const double th0 = 0.35;
        ScalarField thF(mesh, "th", 1);
        thF.initialize([&](double, double, double) { return th0; });
        // fill the FULL stored array (ghosts included) with th0
        for (std::size_t k = 0; k < thF.storedSize; ++k)
            thF.curr[k] = static_cast<Real>(th0);
        thF.allocDevice(); thF.uploadAllToDevice();

        AnisoParams ps = p; ps.theta0 = th0;
        ScalarField oS = mkOut("oS"), oF = mkOut("oF");

        Equation eS(oS, "scalar"); eS.setRHS(anisoDiv(phi, ps));
        eS.computeRHS(oS); cudaDeviceSynchronize(); oS.downloadCurrFromDevice();

        Equation eF(oF, "field");  eF.setRHS(anisoDiv(phi, thF, p));
        eF.computeRHS(oF); cudaDeviceSynchronize(); oF.downloadCurrFromDevice();

        require(maxRel(oF, oS) < 1e-14, "anisoDiv: uniform field == scalar GPU");

        eF.computeRHSCPU(oF);
        eS.computeRHSCPU(oS);
        require(maxRel(oF, oS) < 1e-14, "anisoDiv: uniform field == scalar CPU");

        ScalarField aS = mkOut("aS"), aF = mkOut("aF");
        anisoFactorOnGPU(phi, aS, ps);
        anisoFactorOnGPU(phi, thF, aF, p);
        cudaDeviceSynchronize();
        aS.downloadCurrFromDevice(); aF.downloadCurrFromDevice();
        require(maxRel(aF, aS) < 1e-14, "anisoFactor: uniform field == scalar");
    }

    // === 2. GPU == CPU for smoothly varying theta ========================
    ScalarField thV(mesh, "thV", 1);
    thV.initialize([&](double x, double y, double) {
        const double L = N * dx;
        return 0.5 * std::sin(2.0 * M_PI * x / L)
                   * std::sin(2.0 * M_PI * y / L);
    });
    thV.allocDevice(); thV.uploadAllToDevice();
    BCBatch bThV; bThV.build(thV, bcs);
    bThV.applyOnGPU(thV);
    thV.downloadCurrFromDevice();
    {
        ScalarField oG = mkOut("oG"), oC = mkOut("oC");
        Equation e(oG, "varth"); e.setRHS(anisoDiv(phi, thV, p));
        e.computeRHS(oG); cudaDeviceSynchronize(); oG.downloadCurrFromDevice();
        e.computeRHSCPU(oC);
        require(maxRel(oG, oC) < 1e-13, "anisoDiv(thetaField): GPU == CPU");
    }

    // === 3+4. bicrystal: conservation + grain-interior reduction =========
    {
        const double thA = -0.30, thB = +0.45;
        const int jGB = N / 2;
        ScalarField thBi(mesh, "thBi", 1);
        for (std::size_t k = 0; k < thBi.storedSize; ++k)
            thBi.curr[k] = 0.0;
        for (int j = -1; j <= N; ++j)          // physical rows + y-ghosts
            for (int i = -1; i <= N; ++i) {
                const std::size_t k = static_cast<std::size_t>(
                    thBi.index(std::min(std::max(i, -1), N),
                               std::min(std::max(j, -1), N), 0));
                const int jw = (j + N) % N;    // periodic wrap for ghosts
                thBi.curr[k] = (jw < jGB) ? thA : thB;
            }
        thBi.allocDevice(); thBi.uploadAllToDevice();

        ScalarField oBi = mkOut("oBi");
        Equation e(oBi, "bi"); e.setRHS(anisoDiv(phi, thBi, p));
        e.computeRHS(oBi); cudaDeviceSynchronize(); oBi.downloadCurrFromDevice();

        double sum = 0.0, sumAbs = 0.0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const double v = oBi.curr[oBi.index(i, j, 0)];
                sum += v; sumAbs += std::fabs(v);
            }
        require(std::fabs(sum) <= 1e-10 * sumAbs,
                "bicrystal conservation: sum(anisoDiv) ~ 0 (telescoping)");

        AnisoParams pa = p; pa.theta0 = thA;
        AnisoParams pb = p; pb.theta0 = thB;
        ScalarField oA = mkOut("oA"), oB = mkOut("oB");
        Equation eA(oA, "gA"); eA.setRHS(anisoDiv(phi, pa));
        eA.computeRHS(oA); cudaDeviceSynchronize(); oA.downloadCurrFromDevice();
        Equation eB(oB, "gB"); eB.setRHS(anisoDiv(phi, pb));
        eB.computeRHS(oB); cudaDeviceSynchronize(); oB.downloadCurrFromDevice();

        double mA = 0.0, mB = 0.0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                if (j >= 2 && j < jGB - 2) {           // grain-A interior
                    const double d = std::fabs(oBi.curr[oBi.index(i, j, 0)]
                                             - oA.curr[oA.index(i, j, 0)]);
                    mA = std::max(mA, d);
                } else if (j >= jGB + 2 && j < N - 2) { // grain-B interior
                    const double d = std::fabs(oBi.curr[oBi.index(i, j, 0)]
                                             - oB.curr[oB.index(i, j, 0)]);
                    mB = std::max(mB, d);
                }
            }
        require(mA < 1e-13, "grain-A interior == scalar theta_A");
        require(mB < 1e-13, "grain-B interior == scalar theta_B");
    }

    // === 5. regularisation path with a bicrystal field ===================
    {
        AnisoParams pr = p; pr.eps = 0.08; pr.regularize = true;  // > 1/15
        ScalarField oR = mkOut("oR");
        Equation e(oR, "reg"); e.setRHS(anisoDiv(phi, thV, pr));
        e.computeRHS(oR); cudaDeviceSynchronize(); oR.downloadCurrFromDevice();
        bool finite = true;
        for (int j = 0; j < N && finite; ++j)
            for (int i = 0; i < N; ++i)
                if (!std::isfinite((double)oR.curr[oR.index(i, j, 0)])) {
                    finite = false; break;
                }
        require(finite, "regularised strong-eps run stays finite");
    }

    // === 6. Equal phase values must produce one shared face orientation =====
    // psi=y, theta jumps in x. The row sum equals the external Jx difference;
    // the former receiver-relative >= tie doubled that difference.
    {
        const int n=4;auto m=Mesh::makeUniform2D(CoordSys::CARTESIAN,n,1,0,n,1,0);
        ScalarField f(m,"linear",1),th(m,"angles",1),out(m,"out",1);
        f.fill(0);th.fill(0);out.fill(0);const double angle=25*M_PI/180;
        for(int j=-1;j<=n;++j)for(int i=-1;i<=n;++i){
            f.curr[f.index(i,j)]=m.coord(1,j);th.curr[th.index(i,j)]=i<2?0:angle;
        }
        for(auto* a:{&f,&th,&out}){a->allocDevice();a->uploadAllToDevice();}
        const double expected=(1+.007*std::cos(4*angle))*(-.028*std::sin(4*angle));
        for(auto scheme:{AnisoScheme::CD2,AnisoScheme::Iso9}){
            AnisoParams q;q.eps=.007;q.scheme=scheme;
            Equation e(out,"shared-face tie");e.setRHS(anisoDiv(f,th,q));
            for(bool gpu:{false,true}){
                if(gpu){e.computeRHS(out);out.downloadCurrFromDevice();}else e.computeRHSCPU(out);
                double sum=0;for(int i=0;i<n;++i)sum+=out.curr[out.index(i,1)];
                require(std::abs(sum-expected)<1e-13,"equal-phase shared-face flux remains unique");
            }
        }
    }

    std::printf(failures ? "module_aniso_thetafield: %d FAILURE(S)\n"
                         : "module_aniso_thetafield: all passed\n", failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 2;
}
}
