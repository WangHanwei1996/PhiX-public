/***********************************************************************\
 *
 *  Cahn-Hilliard + FFT Elasticity Solver -- Misfitting Precipitate (2D)
 *  (PFHub Benchmark 4 model, homogeneous-modulus variant)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *      F = Int[ w*P(eta) + kappa/2 |grad eta|^2 + f_el ] dV
 *      P(eta)   : 10th-order double well, wells at eta = 0, 1
 *                 (PFHub BM4 coefficients a0..a10, hard-coded below)
 *      eps*(eta) = eps_T * h(eta) * delta_ij   (dilatational misfit)
 *      h(eta)   = eta^3(6eta^2 - 15eta + 10)
 *      f_el     = 1/2 (eps - eps*) : C : (eps - eps*)
 *
 *      mu       = w*P'(eta) - eps_T*h'(eta)*(sigma11 + sigma22)
 *                 - kappa*lap(eta)
 *      deta/dt  = M * lap(mu)
 *
 *  Mechanical equilibrium div(sigma) = 0 is re-solved EVERY step with
 *  the spectral solver (mechanics/ElasticityFFT.h): homogeneous cubic
 *  modulus, fully periodic cell, <sigma> = 0 mean convention.
 *  sigma11 + sigma22 = (C11 + C12)(e11 + e22 - 2 eps_T h(eta)).
 *
 *  BCs must be Periodic on both axes (FFT consistency).
 *
 *  Optional cfg["pfhub"] section writes a CSV:
 *      time, free_energy, elastic_energy, grad_energy, precip_area, a10
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "core/RunGuard.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "field/ReducePW.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "mechanics/ElasticityFFT.h"
#include "diagnostics/Interface.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"
#include "IO/PFHubWriter.h"

#include <cmath>
#include <cstdlib>
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
    const double x0 = cfg["mesh"]["x0"];
    const int    ny = cfg["mesh"]["ny"];
    const double dy = cfg["mesh"]["dy"];
    const double y0 = cfg["mesh"]["y0"];

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    nx, dx, x0, ny, dy, y0);
    mesh.print();

    // === 2. Time parameters ==================================================
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];

    // === 3. Constants ========================================================
    const double w_well = cfg["constants"]["w"];        // aJ/nm^3
    const double kappa  = cfg["constants"]["kappa"];    // aJ/nm
    const double M      = cfg["constants"]["M"];
    const double eps_T  = cfg["constants"]["eps_T"];    // misfit strain
    ElasticParams2D C;
    C.C11 = cfg["constants"]["C11"];                    // aJ/nm^3 = GPa
    C.C12 = cfg["constants"]["C12"];
    C.C44 = cfg["constants"]["C44"];

    // PFHub BM4 10th-order double well: P(eta) = sum a_i eta^i
    // (wells at eta = 0 and eta = 1; multiply by w for the energy density).
    // Plain local array so device lambdas capture a COPY by value.
    const double A[11] = {
        0.0, 0.0, 8.072789087, -81.24549382, 408.0297321,
        -1244.129167, 2444.046270, -3120.635139, 2506.663551,
        -1151.003178, 230.2006355
    };

    // === 4. Fields ===========================================================
    ScalarField eta(mesh, "eta", 1), mu(mesh, "mu", 1);
    ScalarField eStar(mesh, "eStar", 1);
    ScalarField e11(mesh, "e11", 1), e22(mesh, "e22", 1),
                e12(mesh, "e12", 1), eEl(mesh, "eEl", 1);

    eta.fill(0); mu.fill(0); eStar.fill(0);
    e11.fill(0); e22.fill(0); e12.fill(0); eEl.fill(0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from, "eta");
    IO::initField(eta, start_step);

    for (ScalarField* f : {&eta, &mu, &eStar, &e11, &e22, &e12, &eEl}) {
        f->allocDevice();
        f->uploadAllToDevice();
    }

    // === 5. Boundary conditions (must be periodic — FFT mechanics) ===========
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;
    std::vector<BoundaryCondition*> noBCs;

    // === 6. Elasticity solver ================================================
    ElasticityFFT2D elast(mesh, C);   // zeroMeanStress = true (free cell)

    // === 7. Equations ========================================================
    // eStar = eps_T * h(eta)
    Equation eqEStar(eStar, "eStar");
    eqEStar.setRHS(pw(eta, PHIX_FN (double e) {
        return eps_T * e * e * e * (6.0 * e * e - 15.0 * e + 10.0);
    }));

    // mu = w*P'(eta) - eps_T*h'(eta)*(C11+C12)*(e11 + e22 - 2*eStar)
    //      - kappa*lap(eta)
    const double Cfac = C.C11 + C.C12;
    Equation eqMu(mu, "CH_mu");
    eqMu.setRHS(
        pw(eta, e11, e22, PHIX_FN (double e, double s11, double s22) {
            double dP = 0.0, epow = 1.0;                 // P'(eta) by Horner-ish
            for (int i = 1; i <= 10; ++i) {
                dP += i * A[i] * epow;
                epow *= e;
            }
            const double h  = e * e * e * (6.0 * e * e - 15.0 * e + 10.0);
            const double hp = 30.0 * e * e * (1.0 - e) * (1.0 - e);
            const double sigTrace = Cfac * (s11 + s22 - 2.0 * eps_T * h);
            return w_well * dP - eps_T * hp * sigTrace;
        })
        - kappa * lap(eta)
    );

    // deta/dt = M * lap(mu)
    Equation eqEta(eta, "CH_eta");
    eqEta.setRHS(M * lap(mu));

    // === 8. Output ===========================================================
    eqEta.step = start_step;
    eqEta.time = start_step * dt;

    IO::OutputWriter writer(cfg["output"]);

    // Optional PFHub CSV: time, F, F_el, F_grad, precip_area, a10
    std::unique_ptr<IO::PFHubWriter> pfhub;
    int pfhubEvery = 0;
    const double dV = dx * dy;
    const double cx = x0 + 0.5 * nx * dx;   // precipitate centre (domain centre)
    auto pfhubSample = [&](double time) {
        // eEl is refreshed by the in-loop elastic solve; bulk + grad here
        const double Fbulk = reduce::fieldSumPW(eta, PHIX_FN (Real e) {
            Real P = Real(0), epow = Real(1);
            for (int i = 0; i <= 10; ++i) { P += Real(A[i]) * epow; epow *= e; }
            return Real(w_well) * P;
        }) * dV;
        for (auto* bc : bcs) bc->applyOnGPU(eta);
        const double Fgrad = 0.5 * kappa * reduce::fieldGradSq(eta) * dV;
        const double Fel   = reduce::fieldSum(eEl) * dV;
        const double area  = reduce::fieldSumPW(eta, PHIX_FN (Real e) {
            return e * e * e * (Real(6) * e * e - Real(15) * e + Real(10));
        }) * dV;
        const double a10 = interfacePosition(eta, 0, ny / 2, 0, 0.5, true) - cx;
        pfhub->addRow({time, Fbulk + Fgrad + Fel, Fel, Fgrad, area, a10});
    };
    if (cfg.has("pfhub")) {
        pfhubEvery = cfg["pfhub"]["energy_interval"];
        const std::string csvPath = cfg["pfhub"]["csv"];
        pfhub = std::make_unique<IO::PFHubWriter>(
            csvPath, std::vector<std::string>{
                "time", "free_energy", "elastic_energy", "grad_energy",
                "precip_area", "a10"});
    }

    if (start_step == 0) {
        writer.writeFields(eta, 0, eqEta.time);
        std::cout << "Starting CH + FFT-elasticity simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming from step " << start_step
                  << " (t=" << start_step * dt << ")\n";
    }

    writer.resetTimer();

    // === 9. Time loop ========================================================
    for (int s = start_step; s < nSteps; ++s) {
        eqEStar.advanceSteady(noBCs);                  // eps* = eps_T h(eta^n)
        elast.solve(eStar, &e11, &e22, &e12, &eEl);    // div(sigma) = 0
        eqMu.advanceSteady(bcs, &eta);                 // mu(eta^n, eps^n)
        eqEta.advanceTransient(bcs, dt, &mu);          // eta^{n+1}

        if (pfhub && (eqEta.step % pfhubEvery == 0 || s == start_step))
            pfhubSample(eqEta.time);
        if (writer.shouldPrint(eqEta.step))
            writer.printProgress(eqEta.step, eqEta.time);
        if (writer.shouldWrite(eqEta.step)) {
            writer.writeFields(eta, eqEta.step, eqEta.time);
            writer.writeFields(eEl, eqEta.step, eqEta.time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}

int main(int argc, char* argv[]) {
    return PhiX::runGuarded(phixMain, argc, argv);
}
