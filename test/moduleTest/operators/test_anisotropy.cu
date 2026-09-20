// ---------------------------------------------------------------------------
// module_anisotropy — fused m-fold anisotropic divergence (operators/Anisotropy.h)
//
// 1. ε = 0 must reduce exactly to W0²·∇²φ (CD2).
// 2. Equivalence with the dendrite solver's face chain (faceGrad + interp of
//    cell-centre gradients + facePW + divFace) — the fused kernel implements
//    the identical discretisation, so results must agree to fp-reassociation
//    tolerance on a dendrite-like blob.
// 3. Conservation: divergence form + periodic ghosts + centred blob ⇒ the
//    domain integral of the term vanishes.
// 4. GPU Term path == CPU Term path; anisoFactor GPU == CPU.
// 5. a(θ) symmetry: m-fold periodicity, ε = 0 ⇒ a ≡ 1.
// ---------------------------------------------------------------------------

#include "operators/Anisotropy.h"
#include "operators/FaceOps.h"
#include "operators/Laplacian.h"
#include "equation/Equation.h"
#include "solver/Solver.h"
#include "boundary/PeriodicBC.h"
#include "boundary/BCBatch.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// Dendrite-like blob with fourfold ripple, smooth, centred
static double blob(double x, double y) {
    const double r  = std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
    const double th = std::atan2(y - 0.5, x - 0.5);
    const double R  = 0.22 * (1.0 + 0.08 * std::cos(4.0 * th));
    return 0.5 * (1.0 - std::tanh((r - R) / 0.04));
}

template<typename Fn>
static void fillAll(ScalarField& f, Fn fn) {
    const int g = f.ghost;
    for (int j = -g; j < f.mesh.n[1] + g; ++j)
    for (int i = -g; i < f.mesh.n[0] + g; ++i)
        f.curr[static_cast<std::size_t>(f.index(i, j))] =
            fn(f.mesh.coord(0, i), f.mesh.coord(1, j));
    if (!f.deviceAllocated()) f.allocDevice();
    f.uploadCurrToDevice();
}

static double maxDiffPhys(ScalarField& a, ScalarField& b) {
    a.downloadCurrFromDevice();
    b.downloadCurrFromDevice();
    double m = 0.0;
    for (int j = 0; j < a.mesh.n[1]; ++j)
    for (int i = 0; i < a.mesh.n[0]; ++i) {
        const std::size_t idx = static_cast<std::size_t>(a.index(i, j));
        m = std::max(m, std::fabs(
            static_cast<double>(a.curr[idx]) - b.curr[idx]));
    }
    return m;
}

int main() {
    const int    N  = 96;
    const double dx = 1.0 / N;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    N, dx, 0.0, N, dx, 0.0);

    // =======================================================================
    // 1. ε = 0  ≡  W0²·lap (CD2)
    // =======================================================================
    {
        ScalarField phi(mesh, "phi", 1);
        fillAll(phi, [](double x, double y) {
            return std::sin(6.28 * x) * std::cos(6.28 * y) + 0.3 * x;
        });
        phi.allocDevice();
        phi.uploadAllToDevice();

        AnisoParams p;
        p.W0 = 1.7;
        p.eps = 0.0;

        ScalarField rA(mesh, "rA", 1), rL(mesh, "rL", 1);
        rA.allocDevice();
        rL.allocDevice();

        Equation eA(phi, "a");
        eA.setRHS(anisoDiv(phi, p));
        eA.computeRHS(rA);

        Equation eL(phi, "l");
        eL.setRHS(lap(phi, p.W0 * p.W0));
        eL.computeRHS(rL);

        const double dev = maxDiffPhys(rA, rL);
        require(dev < 1e-10, "eps=0 does not reduce to W0^2 * laplacian: "
                             + std::to_string(dev));
    }

    // =======================================================================
    // 2. Equivalence with the dendrite face chain
    // =======================================================================
    {
        const double eps = 0.05, th0 = 0.3;
        const int    m   = 4;
        const double W0  = 1.3, W0sq = W0 * W0;

        ScalarField phi(mesh, "phi", 2);   // ghost 2: host cc-grads reach -1/N
        fillAll(phi, blob);
        phi.allocDevice();
        phi.uploadAllToDevice();

        // Host cell-centre gradients incl. one ghost ring (chain's interp
        // inputs), exactly the CD2 central differences the app produces.
        ScalarField gx(mesh, "gx", 2), gy(mesh, "gy", 2);
        for (int j = -1; j <= N; ++j)
        for (int i = -1; i <= N; ++i) {
            const std::size_t idx = static_cast<std::size_t>(gx.index(i, j));
            gx.curr[idx] = (phi.curr[static_cast<std::size_t>(phi.index(i + 1, j))]
                          - phi.curr[static_cast<std::size_t>(phi.index(i - 1, j))])
                          / (2.0 * dx);
            gy.curr[idx] = (phi.curr[static_cast<std::size_t>(phi.index(i, j + 1))]
                          - phi.curr[static_cast<std::size_t>(phi.index(i, j - 1))])
                          / (2.0 * dx);
        }
        gx.allocDevice(); gx.uploadAllToDevice();
        gy.allocDevice(); gy.uploadAllToDevice();

        // ---- reference: the app's chain --------------------------------
        FaceField phi_x_xf(mesh, 0, "pxxf", 2), phi_y_xf(mesh, 0, "pyxf", 2);
        FaceField phi_y_yf(mesh, 1, "pyyf", 2), phi_x_yf(mesh, 1, "pxyf", 2);
        FaceField jx(mesh, 0, "jx", 2), jy(mesh, 1, "jy", 2);
        for (FaceField* ff : {&phi_x_xf, &phi_y_xf, &phi_y_yf,
                              &phi_x_yf, &jx, &jy})
            ff->allocDevice();

        faceGradGPU(phi, 0, phi_x_xf);
        interpGPU(gy, 0, phi_y_xf);
        facePWGPU(jx, phi_x_xf, phi_y_xf, PHIX_FN (Real px, Real py) {
            const Real theta = atan2(py, px);
            const Real a  = Real(1) + Real(0.05) * cos(Real(4) * (theta - Real(0.3)));
            const Real st = Real(0.05) * Real(4) * sin(Real(4) * (theta - Real(0.3)));
            return Real(1.69) * a * (a * px + st * py);
        });
        faceGradGPU(phi, 1, phi_y_yf);
        interpGPU(gx, 1, phi_x_yf);
        facePWGPU(jy, phi_y_yf, phi_x_yf, PHIX_FN (Real py, Real px) {
            const Real theta = atan2(py, px);
            const Real a  = Real(1) + Real(0.05) * cos(Real(4) * (theta - Real(0.3)));
            const Real st = Real(0.05) * Real(4) * sin(Real(4) * (theta - Real(0.3)));
            return Real(1.69) * a * (a * py - st * px);
        });

        ScalarField rChain(mesh, "rChain", 2), rFused(mesh, "rFused", 2);
        rChain.allocDevice();
        rFused.allocDevice();

        Equation eC(phi, "chain");
        eC.setRHS(divFace(jx, jy));
        eC.computeRHS(rChain);

        AnisoParams p;
        p.W0 = W0; p.eps = eps; p.m = m; p.theta0 = th0;
        Equation eF(phi, "fused");
        eF.setRHS(anisoDiv(phi, p));
        eF.computeRHS(rFused);

        // Interior cells only: at boundary faces the chain's interpGPU uses
        // one-sided values (it does not read the ghost), while the fused
        // kernel averages with the ghost cell — a boundary approximation
        // difference of the OLD chain, not an error of the fused kernel.
        rChain.downloadCurrFromDevice();
        rFused.downloadCurrFromDevice();
        double dev = 0.0, devAll = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const std::size_t idx =
                static_cast<std::size_t>(rChain.index(i, j));
            const double d = std::fabs(
                static_cast<double>(rChain.curr[idx]) - rFused.curr[idx]);
            devAll = std::max(devAll, d);
            if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
                dev = std::max(dev, d);
        }
        std::printf("  chain-vs-fused max dev: interior %.2e"
                    " (incl. boundary %.2e)\n", dev, devAll);
        require(dev < 1e-10, "fused anisoDiv differs from the face chain in "
                             "the interior: " + std::to_string(dev));
    }

    // =======================================================================
    // 3. Conservation under periodic ghosts (centred blob)
    // =======================================================================
    {
        ScalarField phi(mesh, "phi", 1);
        fillAll(phi, blob);
        phi.allocDevice();
        phi.uploadAllToDevice();
        PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
        PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
        BCBatch batch;
        batch.build(phi, {&bx, &by});
        batch.applyOnGPU(phi);

        AnisoParams p;
        p.W0 = 1.0; p.eps = 0.06; p.m = 6; p.theta0 = 0.1;

        ScalarField rhs(mesh, "rhs", 1);
        rhs.allocDevice();
        Equation e(phi, "cons");
        e.setRHS(anisoDiv(phi, p));
        e.computeRHS(rhs);
        rhs.downloadCurrFromDevice();

        double sum = 0.0, scale = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double v = rhs.curr[static_cast<std::size_t>(rhs.index(i, j))];
            sum += v;
            scale = std::max(scale, std::fabs(v));
        }
        require(std::fabs(sum) < 1e-9 * std::max(scale, 1.0) * N,
                "anisotropic divergence is not conservative: sum = "
                + std::to_string(sum));
    }

    // =======================================================================
    // 4. GPU == CPU (Term paths + factor field)
    // =======================================================================
    {
        ScalarField phi(mesh, "phi", 1);
        fillAll(phi, blob);
        phi.allocDevice();
        phi.uploadAllToDevice();

        AnisoParams p;
        p.W0 = 1.1; p.eps = 0.04; p.m = 4; p.theta0 = 0.2;

        ScalarField rG(mesh, "rG", 1), rC(mesh, "rC", 1);
        rG.allocDevice();
        Equation e(phi, "x");
        e.setRHS(anisoDiv(phi, p));
        e.computeRHS(rG);
        rC.fillCurr(0.0);
        e.computeRHSCPU(rC);
        rG.downloadCurrFromDevice();
        double dev = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const std::size_t idx = static_cast<std::size_t>(rG.index(i, j));
            dev = std::max(dev, std::fabs(
                static_cast<double>(rG.curr[idx]) - rC.curr[idx]));
        }
        require(dev < 1e-11, "anisoDiv GPU != CPU: " + std::to_string(dev));

        ScalarField aG(mesh, "aG", 1), aC(mesh, "aC", 1);
        aG.allocDevice();
        anisoFactorOnGPU(phi, aG, p);
        anisoFactorOnCPU(phi, aC, p);
        aG.downloadCurrFromDevice();
        dev = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const std::size_t idx = static_cast<std::size_t>(aG.index(i, j));
            dev = std::max(dev, std::fabs(
                static_cast<double>(aG.curr[idx]) - aC.curr[idx]));
        }
        require(dev < 1e-13, "anisoFactor GPU != CPU");
    }

    // =======================================================================
    // 5. a(θ) properties
    // =======================================================================
    {
        for (double th = -3.0; th <= 3.0; th += 0.37) {
            const double a4  = aniso::factor(th, 0.05, 4, 0.0);
            const double a4s = aniso::factor(th + M_PI / 2.0, 0.05, 4, 0.0);
            require(std::fabs(a4 - a4s) < 1e-12, "m=4 periodicity violated");
            require(std::fabs(aniso::factor(th, 0.0, 4, 0.0) - 1.0) < 1e-15,
                    "eps=0 factor != 1");
        }
        bool threw = false;
        AnisoParams bad;
        bad.eps = 1.5;
        try { bad.validate(); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "eps >= 1 did not throw");
    }

    // =======================================================================
    // 6. Eggleston convexification (strong anisotropy)
    // =======================================================================
    {
        // (a) matching conditions: γ̃ and γ̃' continuous at θ_m; γ̃+γ̃'' ≡ 0
        const double eps = 0.15;
        const int    m   = 4;                 // limit 1/15 ≈ 0.067 < 0.15
        const AnisoReg r = anisoComputeRegularization(eps, m);
        require(r.thetaM > 0.0, "supercritical eps produced thetaM == 0");
        const double g  = 1.0 + eps * std::cos(m * r.thetaM);
        const double gp = -eps * m * std::sin(m * r.thetaM);
        require(std::fabs(r.A * std::cos(r.thetaM) - g) < 1e-12,
                "Eggleston C0 matching failed");
        require(std::fabs(-r.A * std::sin(r.thetaM) - gp) < 1e-12,
                "Eggleston C1 matching failed");

        // sub-critical: no-op
        const AnisoReg r0 = anisoComputeRegularization(0.03, 4);
        require(r0.thetaM == 0.0, "sub-critical eps produced a cone");

        // (b) regularize=true below the limit gives identical results
        ScalarField phi(mesh, "phi", 1);
        fillAll(phi, blob);
        AnisoParams pOff;
        pOff.eps = 0.04;
        AnisoParams pOn = pOff;
        pOn.regularize = true;
        ScalarField rOff(mesh, "roff", 1), rOn(mesh, "ron", 1);
        rOff.allocDevice();
        rOn.allocDevice();
        Equation eOff(phi, "off");
        eOff.setRHS(anisoDiv(phi, pOff));
        eOff.computeRHS(rOff);
        Equation eOn(phi, "on");
        eOn.setRHS(anisoDiv(phi, pOn));
        eOn.computeRHS(rOn);
        require(maxDiffPhys(rOff, rOn) == 0.0,
                "sub-critical regularize flag changed results");

        // (c) strong-eps evolution smoke: regularized AC blob stays finite
        //     and bounded over an explicit run at eps = 0.15
        ScalarField ph2(mesh, "ph2", 1);
        fillAll(ph2, blob);
        AnisoParams strong;
        strong.W0 = 1.0;
        strong.eps = 0.15;
        strong.m = 4;
        strong.regularize = true;

        Equation eq(ph2, "acReg");
        eq.setRHS(anisoDiv(ph2, strong)
                  + pw(ph2, PHIX_FN (Real v) {
                        return -Real(50) * Real(2) * v * (Real(1) - v)
                               * (Real(1) - Real(2) * v);
                    }));
        PeriodicBC bx2(mesh.facePatch(Axis::X, Side::LOW));
        PeriodicBC by2(mesh.facePatch(Axis::Y, Side::LOW));
        Solver solver(eq, {&bx2, &by2},
                      0.2 * (1.0 / 96) * (1.0 / 96), TimeScheme::EULER);
        solver.run(400);
        require(!reduce::fieldHasNonFinite(ph2),
                "regularized strong-eps run produced NaN/Inf");
        const double mx = reduce::fieldMaxAbs(ph2);
        require(mx < 2.0, "regularized strong-eps run unbounded: max = "
                          + std::to_string(mx));
    }

    return 0;
}
