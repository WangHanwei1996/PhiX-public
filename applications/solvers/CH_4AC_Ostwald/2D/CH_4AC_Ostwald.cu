/***********************************************************************\
 *
 *  Cahn-Hilliard + 4x Allen-Cahn Solver -- Ostwald Ripening (2D)
 *  (PFHub Benchmark 2 model: two-parabola free energy, p = 4 grains)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *      f_chem = f_a(c)[1-h] + f_b(c)h + w*g
 *      f_a = rho^2(c-ca)^2 ,  f_b = rho^2(cb-c)^2
 *      h   = sum_i eta_i^3(6eta_i^2 - 15eta_i + 10)
 *      g   = sum_i eta_i^2(1-eta_i)^2 + alpha*sum_i sum_{j!=i} eta_i^2 eta_j^2
 *
 *      mu        = df/dc - kappa_c*lap(c) ;   dc/dt = M*lap(mu)
 *      deta_i/dt = -L[ (f_b - f_a)*30eta_i^2(1-eta_i)^2
 *                      + 2w*eta_i(1-eta_i)(1-2eta_i)
 *                      + 2w*alpha*eta_i*(S - eta_i^2)
 *                      - kappa_eta*lap(eta_i) ] ,   S = sum_j eta_j^2
 *
 *  S is rebuilt (STEADY) before the eta updates each step, so the four
 *  Allen-Cahn equations see each other at the SAME time level.
 *
 *  Optional cfg["pfhub"] section writes free_energy.csv (time, F, total_c).
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "core/RunGuard.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "field/ReducePW.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
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
                                    nx, dx, x0,
                                    ny, dy, y0);
    mesh.print();

    // === 2. Time parameters ==================================================
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];

    // === 3. Fields & initialization ==========================================
    constexpr int P = 4;
    ScalarField c (mesh, "c",  /*ghost=*/1);
    ScalarField mu(mesh, "mu", /*ghost=*/1);
    ScalarField S (mesh, "S",  /*ghost=*/1);
    ScalarField eta1(mesh, "eta1", 1), eta2(mesh, "eta2", 1),
                eta3(mesh, "eta3", 1), eta4(mesh, "eta4", 1);
    ScalarField* eta[P] = {&eta1, &eta2, &eta3, &eta4};

    c.fill(0); mu.fill(0); S.fill(0);
    for (auto* e : eta) e->fill(0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    IO::initField(c, start_step);
    for (auto* e : eta) IO::initField(*e, start_step);
    // mu, S are STEADY auxiliaries — always rebuilt, never read from disk.

    for (ScalarField* f : {&c, &mu, &S, eta[0], eta[1], eta[2], eta[3]}) {
        f->allocDevice();
        f->uploadAllToDevice();
    }

    // === 4. Boundary conditions ==============================================
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;
    std::vector<BoundaryCondition*> noBCs;   // pointwise steps need no ghosts

    // === 5. Equations ========================================================
    const double rho       = cfg["constants"]["rho"];
    const double ca        = cfg["constants"]["ca"];
    const double cb        = cfg["constants"]["cb"];
    const double kappa_c   = cfg["constants"]["kappa_c"];
    const double kappa_eta = cfg["constants"]["kappa_eta"];
    const double M         = cfg["constants"]["M"];
    const double L         = cfg["constants"]["L"];
    const double w         = cfg["constants"]["w"];
    const double alpha     = cfg["constants"]["alpha"];

    // mu = 2rho^2(c-ca) + 2rho^2(ca-cb)*h(eta) - kappa_c*lap(c)
    // (h is additive over grains: one pw term per eta_i)
    auto hTerm = [&](ScalarField& e) {
        return pw(e, PHIX_FN (double v) {
            const double h = v * v * v * (6.0 * v * v - 15.0 * v + 10.0);
            return 2.0 * rho * rho * (ca - cb) * h;
        });
    };
    Equation eqMu(mu, "CH_mu");
    eqMu.setRHS(
        pw(c, PHIX_FN (double v) { return 2.0 * rho * rho * (v - ca); })
        + hTerm(eta1) + hTerm(eta2) + hTerm(eta3) + hTerm(eta4)
        - kappa_c * lap(c)
    );

    // S = sum_j eta_j^2   (frozen at time level n for the cross term)
    auto sqTerm = [](ScalarField& e) {
        return pw(e, PHIX_FN (double v) { return v * v; });
    };
    Equation eqS(S, "S");
    eqS.setRHS(sqTerm(eta1) + sqTerm(eta2) + sqTerm(eta3) + sqTerm(eta4));

    // dc/dt = M * lap(mu)
    Equation eqC(c, "CH_c");
    eqC.setRHS(M * lap(mu));

    // deta_i/dt = -L[(f_b - f_a)h'(eta_i) + w*dg/deta_i] + L*kappa_eta*lap(eta_i)
    std::vector<std::unique_ptr<Equation>> eqEta;
    for (int i = 0; i < P; ++i) {
        eqEta.emplace_back(std::make_unique<Equation>(*eta[i],
                           "AC_eta" + std::to_string(i + 1)));
        eqEta.back()->setRHS(
            pw(*eta[i], c, PHIX_FN (double e, double cv) {
                const double dfba = rho * rho * ((cb - cv) * (cb - cv)
                                               - (cv - ca) * (cv - ca));
                const double hp   = 30.0 * e * e * (1.0 - e) * (1.0 - e);
                const double dw   = 2.0 * w * e * (1.0 - e) * (1.0 - 2.0 * e);
                return -L * (dfba * hp + dw);
            })
            + pw(*eta[i], S, PHIX_FN (double e, double s) {
                return -L * 2.0 * w * alpha * e * (s - e * e);
            })
            + L * kappa_eta * lap(*eta[i])
        );
    }

    // === 6. Output ===========================================================
    eqC.step = start_step;
    eqC.time = start_step * dt;

    IO::OutputWriter writer(cfg["output"]);

    auto writeAll = [&](int step, double time) {
        writer.writeFields(c, step, time);
        for (auto* e : eta) writer.writeFields(*e, step, time);
    };

    // Optional PFHub free-energy CSV:
    //   F = Sum[ f_a + (f_b - f_a)h + w*g + kc/2|grad c|^2
    //            + ke/2 sum|grad eta_i|^2 ] dV
    // cross-grain term via S:  alpha*sum_{i!=j} = alpha*(S^2 - sum eta_i^4)
    std::unique_ptr<IO::PFHubWriter> pfhub;
    int pfhubEvery = 0;
    const double dV = dx * dy;
    auto freeEnergy = [&]() {
        for (auto* bc : bcs) {
            bc->applyOnGPU(c);
            for (auto* e : eta) bc->applyOnGPU(*e);
        }
        eqS.advanceSteady(noBCs);           // refresh S = sum eta_j^2
        double F = reduce::fieldSumPW(c, PHIX_FN (Real v) {
            return Real(rho) * Real(rho) * (v - Real(ca)) * (v - Real(ca));
        });
        for (auto* e : eta) {
            F += reduce::fieldSumPW(c, *e, PHIX_FN (Real cv, Real ev) {
                const Real dfba = Real(rho) * Real(rho)
                    * ((Real(cb) - cv) * (Real(cb) - cv)
                       - (cv - Real(ca)) * (cv - Real(ca)));
                const Real h  = ev * ev * ev
                    * (Real(6) * ev * ev - Real(15) * ev + Real(10));
                const Real g1 = ev * ev * (Real(1) - ev) * (Real(1) - ev)
                              - Real(alpha) * ev * ev * ev * ev;
                return dfba * h + Real(w) * g1;
            });
            F += 0.5 * kappa_eta * reduce::fieldGradSq(*e);
        }
        F += reduce::fieldSumPW(S, PHIX_FN (Real s) {
            return Real(w) * Real(alpha) * s * s;
        });
        F += 0.5 * kappa_c * reduce::fieldGradSq(c);
        return F * dV;
    };
    if (cfg.has("pfhub")) {
        pfhubEvery = cfg["pfhub"]["energy_interval"];
        const std::string csvPath = cfg["pfhub"]["csv"];
        pfhub = std::make_unique<IO::PFHubWriter>(
            csvPath,
            std::vector<std::string>{"time", "free_energy", "total_c"});
        if (start_step == 0)
            pfhub->addRow({0.0, freeEnergy(), reduce::fieldSum(c) * dV});
    }

    if (start_step == 0) {
        writeAll(0, eqC.time);
        std::cout << "Starting CH+4AC (Ostwald) simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming CH+4AC from step " << start_step
                  << " (t=" << start_step * dt << ")\n";
    }

    writer.resetTimer();

    // === 7. Time loop ========================================================
    for (int s = start_step; s < nSteps; ++s) {
        eqMu.advanceSteady(bcs, &c);          // mu from c^n, eta^n
        eqS.advanceSteady(noBCs);             // S  from eta^n (pointwise)
        eqC.advanceTransient(bcs, dt, &mu);   // c^{n+1}
        for (int i = 0; i < P; ++i)           // eta_i^{n+1} (cross via frozen S)
            eqEta[i]->advanceTransient(bcs, dt, eta[i]);

        if (writer.shouldPrint(eqC.step))
            writer.printProgress(eqC.step, eqC.time);
        if (writer.shouldWrite(eqC.step))
            writeAll(eqC.step, eqC.time);
        if (pfhub && eqC.step % pfhubEvery == 0)
            pfhub->addRow({eqC.time, freeEnergy(),
                           reduce::fieldSum(c) * dV});
    }

    std::cout << "Done.\n";
    return 0;
}

int main(int argc, char* argv[]) {
    return PhiX::runGuarded(phixMain, argc, argv);
}
