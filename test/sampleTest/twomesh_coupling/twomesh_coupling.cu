// ---------------------------------------------------------------------------
// twomesh_coupling — end-to-end multi-mesh sample:
//
//   coarse mesh (64×48)  : T   — periodic decaying Fourier mode (analytic)
//   fine   mesh (256×192): phi — local relaxation  dphi/dt = −phi + T(x,t)
//                          c   — conserved diffusion dc/dt = D∇²c
//
// Per step: advance T on the coarse mesh, MeshMap-interpolate it to the
// fine mesh (with ghost refresh), advance phi/c on the fine mesh.  Every
// K steps, conservative-map c down to the coarse mesh and close the
// integral.  Validates:
//   (1) phi tracks the ANALYTIC solution of the forced ODE
//         phi(x,t) = T0(x)·(e^{−λt} − e^{−t})/(1 − λ),  λ = α k²
//       (bilinear O(dx²) transfer error + Euler O(dt) bounded tolerance);
//   (2) ∫c dV identical on both meshes through the conservative map;
//   (3) two Equations on two meshes coexist with per-mesh BCs.
// ---------------------------------------------------------------------------

#include "core/RunGuard.h"
#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "field/MeshMap.h"
#include "field/Reduce.h"
#include "boundary/PeriodicBC.h"
#include "equation/Equation.h"

#include <cmath>
#include <cstdio>

static int phixMain()
{
    using namespace PhiX;

    const double Lx = 12.8, Ly = 9.6;
    Mesh coarse = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                      64, Lx / 64, 0.0, 48, Ly / 48, 0.0);
    Mesh fine   = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                      256, Lx / 256, 0.0, 192, Ly / 192, 0.0);
    MeshMap map(coarse, fine);

    const double alpha = 0.7;                       // T diffusivity
    const double kx    = 2.0 * M_PI / Lx;           // one periodic mode
    const double lam   = alpha * kx * kx;           // decay rate
    const double D     = 0.3;                       // c diffusivity
    const double dt    = 2e-4;
    const int    nSteps = 4000, checkEvery = 500;

    // --- fields -------------------------------------------------------------
    ScalarField T (coarse, "T",  1);
    ScalarField Tf(fine,   "Tf", 1);                // T mapped onto fine
    ScalarField phi(fine,  "phi", 1);
    ScalarField c (fine,   "c",  1);
    ScalarField cMon(coarse, "cMon", 1);            // conservative monitor

    T.initialize([kx](double x, double, double) { return std::sin(kx * x); });
    Tf.fill(0.0);
    phi.fill(0.0);
    c.initialize([Lx, Ly](double x, double y, double) {
        return 0.5 + 0.2 * std::sin(4.0 * M_PI * x / Lx)
                   * std::cos(2.0 * M_PI * y / Ly);
    });
    cMon.fill(0.0);
    for (ScalarField* f : {&T, &Tf, &phi, &c, &cMon}) {
        f->allocDevice();
        f->uploadAllToDevice();
    }

    // --- per-mesh BCs -------------------------------------------------------
    PeriodicBC bcTx(coarse.patch("xmin")), bcTy(coarse.patch("ymin"));
    PeriodicBC bcFx(fine.patch("xmin")),   bcFy(fine.patch("ymin"));
    std::vector<BoundaryCondition*> bcsT{&bcTx, &bcTy};
    std::vector<BoundaryCondition*> bcsF{&bcFx, &bcFy};

    // --- equations (each on its own mesh) -----------------------------------
    Equation eqT(T, "T_diffusion");
    eqT.setRHS(alpha * lap(T));

    Equation eqPhi(phi, "phi_relax");               // dphi/dt = −phi + Tf
    eqPhi.setRHS(pw(phi, Tf, PHIX_FN (double p, double tv) { return tv - p; }));

    Equation eqC(c, "c_diffusion");
    eqC.setRHS(D * lap(c));

    const double dVf = fine.d[0] * fine.d[1];
    const double dVc = coarse.d[0] * coarse.d[1];
    const double I0  = reduce::fieldSum(c) * dVf;

    double maxConsErr = 0.0;
    for (int s = 0; s < nSteps; ++s) {
        // 1. coarse mesh: advance T
        eqT.advanceTransient(bcsT, dt);
        // 2. cross-mesh: T -> fine (with fine-side ghost refresh)
        map.interpolate(T, Tf, bcsF);
        // 3. fine mesh: advance phi (reads Tf) and c
        eqPhi.advanceTransient(bcsF, dt);
        eqC.advanceTransient(bcsF, dt);

        if ((s + 1) % checkEvery == 0) {
            map.conserve(c, cMon);
            const double If = reduce::fieldSum(c)    * dVf;
            const double Ic = reduce::fieldSum(cMon) * dVc;
            maxConsErr = std::max(maxConsErr,
                                  std::fabs(If - Ic) / std::fabs(I0));
        }
    }

    // --- validation ---------------------------------------------------------
    const double tEnd = nSteps * dt;
    phi.downloadCurrFromDevice();
    const double factor =
        (std::exp(-lam * tEnd) - std::exp(-tEnd)) / (1.0 - lam);
    double maxPhiErr = 0.0;
    for (int j = 8; j < 192 - 8; ++j)
        for (int i = 8; i < 256 - 8; ++i) {
            const double x = fine.coord(0, i);
            const double ref = std::sin(kx * x) * factor;
            maxPhiErr = std::max(maxPhiErr,
                                 std::fabs(phi.curr[phi.index(i, j)] - ref));
        }

    const double If = reduce::fieldSum(c) * dVf;
    const double drift = std::fabs(If - I0) / std::fabs(I0);

    std::printf("twomesh_coupling: t=%.3f\n", tEnd);
    std::printf("  phi vs analytic (interior max err) : %.3e  (tol 2e-3)\n",
                maxPhiErr);
    std::printf("  cross-mesh conservation closure    : %.3e  (tol 1e-12)\n",
                maxConsErr);
    std::printf("  fine-mesh c drift over run         : %.3e  (tol 1e-12)\n",
                drift);

    const bool ok = maxPhiErr < 2e-3 && maxConsErr < 1e-12 && drift < 1e-12;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

int main() {
    return PhiX::runGuarded(phixMain);
}
