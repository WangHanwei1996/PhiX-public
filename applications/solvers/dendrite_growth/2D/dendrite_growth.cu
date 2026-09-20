/***********************************************************************\
 *
 *  Dendrite Growth Solver (2D)  — v2: staggered face-centred flux scheme
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *  Anisotropic Allen-Cahn (phi) + dimensionless thermal diffusion (U)
 *  for simulating solidification and dendritic growth.
 *
 *  Based on: Karma & Rappel, Phys. Rev. E 57, 4323 (1998)
 *  Reference: NIST PFHub Benchmark 3
 *
 *  Variables:
 *    phi  -- phase-field order parameter  (+1: solid, -1: liquid)
 *    U    -- dimensionless undercooling   (U = (T - T_m) / (L/c_p))
 *
 *  Evolution equations:
 *    tau(n)*dphi/dt = d/dx[W^2*phi_x + A*phi_y]
 *                  + d/dy[W^2*phi_y - A*phi_x]
 *                  + (1-phi^2)*(phi - lambda*U*(1-phi^2))
 *    dU/dt = D * lap(U) + 0.5 * dphi/dt
 *
 *  Anisotropy (m-fold symmetry):
 *    a(n)   = 1 + epsilon_m * cos(m*(theta - theta_0))
 *    W(n)   = W_0 * a(n)
 *    tau(n) = tau_0 * a(n)^2
 *    A      = W * W_0 * epsilon_m * m * sin(m*(theta - theta_0))
 *    theta  = atan2(phi_y, phi_x)
 *
 *  Coupling constant (thin-interface limit):
 *    lambda = D * tau_0 / (0.6267 * W_0^2)
 *
 *  Discrete scheme: staggered face-centred flux divergence (no checkerboard).
 *    jx on x-faces: W^2 * phi_x_xf + A * phi_y_xf
 *    jy on y-faces: W^2 * phi_y_yf - A * phi_x_yf
 *    div(J) = (jx[i+1] - jx[i])/dx + (jy[j+1] - jy[j])/dy
 *
 *  Solver pipeline (7 steps, Forward Euler):
 *   A1. BC(phi) -> phi_x_cc = grad(phi,0)  [CD2, cell-centre, for tau/interp]
 *   A2. BC(phi) -> phi_y_cc = grad(phi,1)  [CD2, cell-centre, for tau/interp]
 *   B1. faceGradGPU(phi,0)    -> phi_x_xf  [phi gradient on x-faces]
 *   B2. interpGPU(phi_y_cc,0) -> phi_y_xf  [phi_y_cc interpolated to x-faces]
 *   B3. facePWGPU              -> jx        [Jx anisotropic flux on x-faces]
 *   C1. faceGradGPU(phi,1)    -> phi_y_yf  [phi gradient on y-faces]
 *   C2. interpGPU(phi_x_cc,1) -> phi_x_yf  [phi_x_cc interpolated to y-faces]
 *   C3. facePWGPU              -> jy        [Jy anisotropic flux on y-faces]
 *    D. a_cc = pw(phi_x_cc, phi_y_cc)      [cell-centre a(n) for tau]
 *    E. dphi = inv_tau * (N + divFace(jx,jy))
 *    F. phi += dt * dphi
 *    G. U   += dt * (D*lap(U) + 0.5*dphi)
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "core/RunGuard.h"
#include "field/ScalarField.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "operators/Gradient.h"
#include "operators/Laplacian.h"
#include "operators/FaceOps.h"      // faceGradGPU, interpGPU, facePWGPU, divFace
                                    // also pulls in FaceField.h via FaceOps.h
#include "IO/ConfigFile.h"
#include "scheme/Schemes.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"
#include "IO/PFHubWriter.h"
#include "field/Reduce.h"
#include "field/ReducePW.h"
#include "diagnostics/Interface.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>

// === DEBUG helper: print min/max/mean over physical cells ===================
static void printStats(const std::string& tag, PhiX::ScalarField& f) {
    f.downloadAllFromDevice();
    double mn = 1e300, mx = -1e300, sum = 0.0;
    int n = 0;
    for (int k = 0; k < f.mesh.n[2]; ++k)
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i) {
        double v = f.curr[f.index(i, j, k)];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v; ++n;
    }
    std::cout << "  " << tag
              << ": min=" << std::scientific << std::setprecision(3) << mn
              << " max=" << mx
              << " mean=" << sum / n
              << std::defaultfloat << "\n";
}

static int phixMain(int argc, char* argv[])
{
    using namespace PhiX;

    IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);
    // Numerical schemes from settings/schemes.jsonc (missing file → the
    // defaults below, i.e. what this solver used before v3.8.8).
    Schemes sch = Schemes::load(cfg, {});

    // === 1. Mesh =============================================================
    const int    nx = cfg["mesh"]["nx"];
    const double dx = cfg["mesh"]["dx"];
    const double x0 = cfg["mesh"]["x0"];
    const int    ny = cfg["mesh"]["ny"];
    const double dy = cfg["mesh"]["dy"];
    const double y0 = cfg["mesh"]["y0"];

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, dx, x0, ny, dy, y0);
    mesh.print();

    // === 2. Time parameters ==================================================
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];

    // === 3. Physical constants ===============================================
    const double D         = cfg["constants"]["D"];
    const double tau_0     = cfg["constants"]["tau_0"];
    const double W_0       = cfg["constants"]["W_0"];
    const double epsilon_m = cfg["constants"]["epsilon_m"];
    const double m_order   = cfg["constants"]["m"];
    const double theta_0   = cfg["constants"]["theta_0"];

    const double W0_sq      = W_0 * W_0;
    const double lambda_val = D * tau_0 / (0.6267 * W0_sq);

    std::cout << "  lambda = " << lambda_val << "\n";

    // === 4. Cell-centred scalar fields =======================================
    ScalarField phi     (mesh, "phi",      1);
    ScalarField U       (mesh, "U",        1);
    // Auxiliary: cell-centre phi gradients (CD2) used for tau and face interp
    ScalarField phi_x_cc(mesh, "phi_x_cc", 1);
    ScalarField phi_y_cc(mesh, "phi_y_cc", 1);
    // Auxiliary: cell-centre anisotropy a(n) for tau
    ScalarField a_cc    (mesh, "a_cc",     1);
    // Auxiliary: dphi/dt (reused by U equation)
    ScalarField dphi    (mesh, "dphi",     1);

    phi.fill(0);  U.fill(0);
    phi_x_cc.fill(0); phi_y_cc.fill(0); a_cc.fill(1.0); dphi.fill(0);

    // === 5. Face fields ======================================================
    // x-faces (normalAxis = 0)
    FaceField phi_x_xf(mesh, 0, "phi_x_xf");  // phi gradient on x-faces
    FaceField phi_y_xf(mesh, 0, "phi_y_xf");  // phi_y_cc interpolated to x-faces
    FaceField jx      (mesh, 0, "jx");         // Jx anisotropic flux on x-faces

    // y-faces (normalAxis = 1)
    FaceField phi_y_yf(mesh, 1, "phi_y_yf");  // phi gradient on y-faces
    FaceField phi_x_yf(mesh, 1, "phi_x_yf");  // phi_x_cc interpolated to y-faces
    FaceField jy      (mesh, 1, "jy");         // Jy anisotropic flux on y-faces

    // === 6. Initialise fields =================================================
    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(phi, start_step);
    IO::initField(U,   start_step);

    auto allocUp = [](ScalarField& f){ f.allocDevice(); f.uploadAllToDevice(); };
    allocUp(phi);  allocUp(U);
    allocUp(phi_x_cc); allocUp(phi_y_cc); allocUp(a_cc); allocUp(dphi);

    auto allocUpFace = [](FaceField& f){ f.fill(0.0); f.allocDevice(); f.uploadToDevice(); };
    allocUpFace(phi_x_xf); allocUpFace(phi_y_xf); allocUpFace(jx);
    allocUpFace(phi_y_yf); allocUpFace(phi_x_yf); allocUpFace(jy);

    // === 7. Boundary conditions ==============================================
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // === 8. Equations ========================================================

    // A1. phi_x_cc = d(phi)/dx  (CD2, cell-centre)
    Equation eq_phi_x_cc(phi_x_cc, "phi_x_cc");
    eq_phi_x_cc.setRHS(grad(phi, 0, sch, 1.0));

    // A2. phi_y_cc = d(phi)/dy  (CD2, cell-centre)
    Equation eq_phi_y_cc(phi_y_cc, "phi_y_cc");
    eq_phi_y_cc.setRHS(grad(phi, 1, sch, 1.0));

    // D. a_cc = 1 + epsilon_m * cos(m*(theta - theta_0))
    Equation eq_a_cc(a_cc, "a_cc");
    eq_a_cc.setRHS(
        pw(phi_x_cc, phi_y_cc, PHIX_FN (double px, double py) {
            double theta = atan2(py, px);
            return 1.0 + epsilon_m * cos(m_order * (theta - theta_0));
        })
    );

    // E. dphi/dt = (1 / (tau_0 * a^2)) * (N(phi,U) + div(J))
    //    N(phi,U) = (1 - phi^2) * (phi - lambda*U*(1 - phi^2))
    auto N_bulk = pw(phi, U, PHIX_FN (double p, double u) {
        return (1.0 - p*p) * (p - lambda_val * u * (1.0 - p*p));
    });
    auto inv_tau = pw(a_cc, PHIX_FN (double a) {
        return 1.0 / (tau_0 * a * a);
    });

    Equation eq_dphi(dphi, "dphi_dt");
    eq_dphi.setRHS(
        inv_tau * (N_bulk + divFace(jx, jy))
    );

    // F. phi += dt * dphi
    Equation eq_phi(phi, "AC_phi");
    eq_phi.setRHS(1.0 * dphi);

    // G. U += dt * (D * lap(U) + 0.5 * dphi)
    Equation eq_U(U, "diffusion_U");
    eq_U.setRHS(lap(U, sch, D) + 0.5 * dphi);

    // === 9. Output & time loop ===============================================
    eq_U.step = start_step;
    eq_U.time = start_step * dt;

    IO::OutputWriter writer(cfg["output"]);

    // Optional PFHub BM3 CSV (cfg["pfhub"] section):
    //   time, solid_fraction, free_energy, tip_x
    //   F = Sum[ 1/2 W(n)^2 |grad phi|^2 - phi^2/2 + phi^4/4
    //            + lambda*U*phi*(1 - 2phi^2/3 + phi^4/5) ] dV
    //   tip_x: phi = 0 crossing along the bottom row (dendrite arm on +x axis)
    std::unique_ptr<IO::PFHubWriter> pfhub;
    int pfhubEvery = 0;
    const double dV = dx * dy;
    auto pfhubSample = [&](double time) {
        eq_phi_x_cc.advanceSteady(bcs, &phi);    // gradients of CURRENT phi
        eq_phi_y_cc.advanceSteady(bcs, &phi);
        const double solidFrac = reduce::fieldSumPW(phi, PHIX_FN (Real p) {
            return (p + Real(1)) * Real(0.5);
        }) * dV;
        double F = reduce::fieldSumPW(phi_x_cc, phi_y_cc,
            PHIX_FN (Real px, Real py) {
                const Real theta = atan2(py, px);
                const Real a = Real(1) + Real(epsilon_m)
                             * cos(Real(m_order) * (theta - Real(theta_0)));
                return Real(0.5) * Real(W0_sq) * a * a * (px * px + py * py);
            });
        F += reduce::fieldSumPW(phi, U, PHIX_FN (Real p, Real u) {
            return -Real(0.5) * p * p + Real(0.25) * p * p * p * p
                   + Real(lambda_val) * u * p
                     * (Real(1) - Real(2.0 / 3.0) * p * p
                        + Real(0.2) * p * p * p * p);
        });
        const double tipX = interfacePosition(phi, 0, 0, 0, 0.0, true);
        pfhub->addRow({time, solidFrac, F * dV, tipX});
    };
    if (cfg.has("pfhub")) {
        pfhubEvery = cfg["pfhub"]["energy_interval"];
        const std::string csvPath = cfg["pfhub"]["csv"];
        pfhub = std::make_unique<IO::PFHubWriter>(
            csvPath, std::vector<std::string>{
                "time", "solid_fraction", "free_energy", "tip_x"});
        if (start_step == 0) pfhubSample(0.0);
    }

    if (start_step == 0) {
        writer.writeFields(phi, 0, eq_U.time);
        writer.writeFields(U,   0, eq_U.time);
        std::cout << "Starting dendrite growth simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming dendrite growth simulation from step " << start_step
                  << " (t=" << start_step * dt << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt << "\n";
    }

    writer.resetTimer();
    sch.warnUnused();   // per-term keys nobody looked up = misspelled field names

    for (int s = start_step; s < nSteps; ++s) {

        // ── A: cell-centre phi gradients (BC(phi) applied) ───────────────────
        eq_phi_x_cc.advanceSteady(bcs, &phi);   // A1: phi_x_cc = grad(phi,0)
        eq_phi_y_cc.advanceSteady(bcs, &phi);   // A2: phi_y_cc = grad(phi,1)

        // ── B: assemble Jx on x-faces ────────────────────────────────────────
        //  phi ghost cells valid from step A
        faceGradGPU(phi, 0, phi_x_xf);          // B1: phi gradient on x-faces
        interpGPU(phi_y_cc, 0, phi_y_xf);       // B2: phi_y_cc -> x-faces
        facePWGPU(jx, phi_x_xf, phi_y_xf,       // B3: Jx = W^2*phi_x + A*phi_y
                  PHIX_FN (double px, double py) {
                      double theta    = atan2(py, px);
                      double a        = 1.0 + epsilon_m * cos(m_order * (theta - theta_0));
                      double sin_term = epsilon_m * m_order * sin(m_order * (theta - theta_0));
                      return W0_sq * a * (a * px + sin_term * py);
                  });

        // ── C: assemble Jy on y-faces ────────────────────────────────────────
        faceGradGPU(phi, 1, phi_y_yf);          // C1: phi gradient on y-faces
        interpGPU(phi_x_cc, 1, phi_x_yf);       // C2: phi_x_cc -> y-faces
        facePWGPU(jy, phi_y_yf, phi_x_yf,       // C3: Jy = W^2*phi_y - A*phi_x
                  PHIX_FN (double py, double px) {
                      double theta    = atan2(py, px);
                      double a        = 1.0 + epsilon_m * cos(m_order * (theta - theta_0));
                      double sin_term = epsilon_m * m_order * sin(m_order * (theta - theta_0));
                      return W0_sq * a * (a * py - sin_term * px);
                  });

        // ── D: cell-centre anisotropy for tau ─────────────────────────────────
        eq_a_cc.advanceSteady(bcs, nullptr);     // a_cc = a(phi_x_cc, phi_y_cc)

        // ── E: dphi/dt = (N + div(J)) / tau ──────────────────────────────────
        eq_dphi.advanceSteady(bcs, nullptr);

        // ── F: phi += dt * dphi ───────────────────────────────────────────────
        eq_phi.advanceTransient(bcs, dt, &phi);

        // ── G: U += dt * (D*lap(U) + 0.5*dphi) ──────────────────────────────
        eq_U.advanceTransient(bcs, dt, &U);

        // ── Output ────────────────────────────────────────────────────────────
        if (writer.shouldPrint(eq_U.step)) {
            writer.printProgress(eq_U.step, eq_U.time);
            printStats("phi   ", phi);
            printStats("U     ", U);
            printStats("a_cc  ", a_cc);
            printStats("dphi  ", dphi);
        }
        if (writer.shouldWrite(eq_U.step)) {
            writer.writeFields(phi, eq_U.step, eq_U.time);
            writer.writeFields(U,   eq_U.step, eq_U.time);
        }
        if (pfhub && eq_U.step % pfhubEvery == 0)
            pfhubSample(eq_U.time);
    }

    std::cout << "Done.\n";
    return 0;
}

int main(int argc, char* argv[]) {
    return PhiX::runGuarded(phixMain, argc, argv);
}
