/***********************************************************************\
 *
 *  aniso_dendrite — Karma-Rappel dendrite on the framework's anisotropy
 *  operator, for the anisotropy-strength showcase.
 *
 *  WHY A NEW SOLVER.  applications/solvers/dendrite_growth/2D predates
 *  operators/Anisotropy.h and hand-builds the anisotropic flux out of six
 *  FaceFields and a faceGrad / interp / facePW / divFace chain.  The
 *  framework operator expresses the same variational term as a single fused
 *  kernel, so this solver is both the shorter statement of the model and the
 *  one that exercises the module the paper describes.
 *
 *  Model (Karma & Rappel, Phys. Rev. E 57, 4323 (1998)):
 *
 *      tau(theta) dphi/dt = div J(phi) + (1-phi^2)(phi - lambda U (1-phi^2))
 *      dU/dt              = D lap(U) + (1/2) dphi/dt
 *
 *      a(theta) = 1 + eps cos(m (theta - theta0)),  theta = atan2(phi_y, phi_x)
 *      J        = W0^2 [ a^2 grad phi + a a' (-phi_y, phi_x) ]     <- anisoDiv
 *      tau      = tau0 a(theta)^2                                  <- anisoFactor
 *      lambda   = D tau0 / (0.6267 W0^2)                (thin-interface limit)
 *
 *  TWO DETAILS THAT MATTER.
 *
 *  1. anisoDiv reads DIAGONAL neighbours, and face-patch boundary conditions
 *     do not fill corner ghosts.  A quarter dendrite is seeded AT a domain
 *     corner, which is exactly the case the header warns about, so boundary
 *     conditions are applied through BCBatch, whose second pass fills the
 *     corners.  Using the per-BC path here would corrupt the seed.
 *
 *  2. Beyond the convexity limit eps > 1/(m^2-1) the surface stiffness turns
 *     negative in cones around the energy maxima: orientations go missing and
 *     the evolution is ill-posed.  AnisoParams::regularize switches on the
 *     Eggleston continuation.  For m = 4 the limit is 1/15 ~ 0.0667, so the
 *     sweep can be run on both sides of it, with and without the fix.
 *
 *  Usage:
 *      ./aniso_dendrite settings/settings.jsonc
 *
 *  Config adds to the BM3 blocks:  constants.regularize (0/1).
 *
\***********************************************************************/

#include "mesh/Mesh.h"
#include "core/RunGuard.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "field/ReducePW.h"
#include "boundary/BCFactory.h"
#include "boundary/BCBatch.h"
#include "equation/Equation.h"
#include "operators/Anisotropy.h"
#include "operators/Laplacian.h"
#include "diagnostics/Interface.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"
#include "IO/PFHubWriter.h"
#include "perf/Perf.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

static int phixMain(int argc, char* argv[])
{
    using namespace PhiX;
    IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);

    // === 1. Mesh =============================================================
    const int    nx = cfg["mesh"]["nx"];
    const double dx = cfg["mesh"]["dx"];
    const int    ny = cfg["mesh"]["ny"];
    const double dy = cfg["mesh"]["dy"];
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    nx, dx, cfg["mesh"]["x0"],
                                    ny, dy, cfg["mesh"]["y0"]);

    // === 2. Time parameters ==================================================
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];

    // === 3. Fields and initialisation ========================================
    ScalarField phi (mesh, "phi",  1);
    ScalarField U   (mesh, "U",    1);
    ScalarField a_cc(mesh, "a_cc", 1);   // a(theta) at cell centres, for tau
    ScalarField dphi(mesh, "dphi", 1);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);
    IO::initField(phi, start_step);
    IO::initField(U,   start_step);
    a_cc.fill(1.0);
    dphi.fill(0.0);
    for (ScalarField* f : {&phi, &U, &a_cc, &dphi}) {
        f->allocDevice(); f->uploadAllToDevice();
    }

    // === 4. Boundary conditions ==============================================
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;
    // Corner ghosts are mandatory for anisoDiv — see the header note.
    BCBatch batchPhi, batchU;
    batchPhi.build(phi, bcs);
    batchU.build(U, bcs);

    // === 5. Equations ========================================================
    const double D       = cfg["constants"]["D"];
    const double tau_0   = cfg["constants"]["tau_0"];
    const double W_0     = cfg["constants"]["W_0"];
    const double eps_m   = cfg["constants"]["epsilon_m"];
    const int    m_order = static_cast<int>(double(cfg["constants"]["m"]));
    const double theta_0 = cfg["constants"]["theta_0"];
    const bool   regular = cfg["constants"].value("regularize", 0) != 0;
    const double lambda  = D * tau_0 / (0.6267 * W_0 * W_0);

    AnisoParams ap;
    ap.W0 = W_0; ap.eps = eps_m; ap.m = m_order;
    ap.theta0 = theta_0; ap.regularize = regular;
    ap.validate();

    const double convexLimit = 1.0 / (double(m_order) * m_order - 1.0);
    AnisoReg reg = anisoComputeRegularization(eps_m, m_order);
    std::cout << "aniso_dendrite: eps=" << eps_m << "  m=" << m_order
              << "  convexity limit 1/(m^2-1)=" << convexLimit
              << (eps_m > convexLimit ? "  [NON-CONVEX]" : "  [convex]")
              << "  regularize=" << (regular ? "on" : "off")
              << "  theta_m=" << reg.thetaM << "\n"
              << "  lambda=" << lambda << "  dt=" << dt
              << "  steps=" << nSteps << "\n";

    // dphi/dt = [ div J(phi) + N(phi,U) ] / (tau0 a^2)
    auto N_bulk = pw(phi, U, PHIX_FN (double p, double u) {
        return (1.0 - p * p) * (p - lambda * u * (1.0 - p * p));
    });
    auto inv_tau = pw(a_cc, PHIX_FN (double a) {
        return 1.0 / (tau_0 * a * a);
    });

    Equation eq_dphi(dphi, "dphi_dt");
    eq_dphi.setRHS(inv_tau * (anisoDiv(phi, ap) + N_bulk));

    Equation eq_phi(phi, "phi_update");
    eq_phi.setRHS(1.0 * dphi);

    Equation eq_U(U, "U_diffusion");
    eq_U.setRHS(lap(U, D) + 0.5 * dphi);

    // === 6. Diagnostics ======================================================
    const double dV = dx * dy;
    auto sample = [&](double t) {
        const double solidFrac = reduce::fieldSumPW(phi, PHIX_FN (Real p) {
            return Real(0.5) * (Real(1) + p);
        }) * dV;
        double tipX = 0.0;
        try { tipX = interfacePosition(phi, 0, 0, 0, 0.0, true); }
        catch (const std::exception&) { tipX = -1.0; }   // no crossing yet
        return std::pair<double, double>{solidFrac, tipX};
    };

    std::unique_ptr<IO::PFHubWriter> csv;
    int csvEvery = 0;
    if (cfg.has("pfhub")) {
        csvEvery = cfg["pfhub"]["energy_interval"];
        csv = std::make_unique<IO::PFHubWriter>(
            std::string(cfg["pfhub"]["csv"]),
            std::vector<std::string>{"time", "solid_fraction", "tip_x"});
    }
    IO::OutputWriter writer(cfg["output"]);

    // === 7. Time loop ========================================================
    // anisoFactorOnGPU sits between the ghost refresh and the RHS, which is
    // why this is an explicit loop rather than a multi-step Solver chain.
    double time = start_step * dt;
    if (start_step == 0) {
        writer.writeFields(phi, 0, 0.0);
        if (csv) { auto [sf, tx] = sample(0.0); csv->addRow({0.0, sf, tx}); }
    }
    writer.resetTimer();
    perf::WallTimer wall;

    for (int s = start_step + 1; s <= nSteps; ++s) {
        batchPhi.applyOnGPU(phi);            // includes the corner pass
        anisoFactorOnGPU(phi, a_cc, ap);     // a(theta) for tau(theta)
        eq_dphi.advanceSteady(bcs, &phi);    // dphi := RHS
        batchU.applyOnGPU(U);
        eq_U.advanceTransient({}, dt, &U);   // U += dt (D lap U + dphi/2)
        eq_phi.advanceTransient({}, dt, &phi);
        time = s * dt;

        if (writer.shouldPrint(s)) writer.printProgress(s, time);
        if (writer.shouldWrite(s)) writer.writeFields(phi, s, time);
        if (csv && s % csvEvery == 0) {
            auto [sf, tx] = sample(time);
            csv->addRow({time, sf, tx});
        }
    }
    cudaDeviceSynchronize();
    const double sec = wall.seconds();
    std::cout << "done: " << sec << " s, "
              << sec * 1e3 / std::max(nSteps - start_step, 1) << " ms/step\n";
    return 0;
}

int main(int argc, char* argv[]) {
    return PhiX::runGuarded(phixMain, argc, argv);
}
