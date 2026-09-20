/***********************************************************************\
 *
 *  Cahn-Hilliard + Poisson Electrostatics Solver (2D)
 *  (PFHub Benchmark 6 model)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *      F = Int[ rho(c-ca)^2(cb-c)^2 + kappa/2 |grad c|^2 + k c Phi/2 ] dV
 *
 *      mu      = 2rho(c-ca)(cb-c)(ca+cb-2c) - kappa*lap(c) + k*Phi
 *      dc/dt   = M * lap(mu)                      [explicit Euler]
 *      lap(Phi) = -k c / eps                      [CG, every step]
 *
 *  Poisson BCs (PFHub BM6a): Phi = 0 at x-min, Phi = sin(y/7) at x-max
 *  (non-uniform Dirichlet), Neumann at y-min/max.  The CG operator must
 *  stay LINEAR, so the solve uses HOMOGENEOUS BCs (Fixed 0 / NoFlux) and
 *  the inhomogeneous Dirichlet data is folded into the RHS analytically:
 *  the ghost lift contributes  2*sin(y_j/7)/dx^2  in the right-most
 *  column only (time-independent, precomputed once).  Phi is
 *  warm-started from the previous step (CG converges in a few
 *  iterations).  Phi enters the CH step only POINTWISE (k*Phi), so no
 *  Phi ghost refresh is needed there; mu carries NoFlux ghosts (zero
 *  boundary flux => solute conservation).
 *
 *  Optional cfg["pfhub"] section writes free_energy.csv and, at the
 *  final step, c/Phi cross-sections along x = x_cut and y = y_cut.
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "core/RunGuard.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "field/ReducePW.h"
#include "boundary/BCFactory.h"
#include "boundary/FixedBC.h"
#include "boundary/NoFluxBC.h"
#include "equation/Equation.h"
#include "solver/LinearSolver.h"
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
    const double rho   = cfg["constants"]["rho"];
    const double ca    = cfg["constants"]["ca"];
    const double cb    = cfg["constants"]["cb"];
    const double kappa = cfg["constants"]["kappa"];
    const double M     = cfg["constants"]["M"];
    const double kchg  = cfg["constants"]["k"];      // charge coefficient
    const double eps   = cfg["constants"]["epsilon"];

    // === 4. Fields ===========================================================
    ScalarField c  (mesh, "c",   1), mu (mesh, "mu",  1);
    ScalarField Phi(mesh, "Phi", 1), rhs(mesh, "rhs", 1);
    ScalarField bbc(mesh, "bbc", 1);   // Dirichlet ghost lift (constant in t)

    c.fill(0); mu.fill(0); Phi.fill(0); rhs.fill(0); bbc.fill(0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);
    IO::initField(c, start_step);

    // Right-column lift.  PhiX FixedBC is a CONSTANT ghost fill
    // (ghost = value, Dirichlet enforced at the ghost-cell centre, not the
    // wall midpoint) — the matching lift is therefore sin(y_j/7)/dx^2, NOT
    // the midpoint form 2g/dx^2.  Left boundary value is 0 (no lift).
    for (int j = 0; j < ny; ++j) {
        const double g = std::sin(mesh.coord(1, j) / 7.0);
        bbc.curr[static_cast<std::size_t>(bbc.index(nx - 1, j))]
            = g / (dx * dx);
    }

    for (ScalarField* f : {&c, &mu, &Phi, &rhs, &bbc}) {
        f->allocDevice();
        f->uploadAllToDevice();
    }

    // === 5. Boundary conditions ==============================================
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);   // for c, mu
    auto& bcs   = bcSet.ptrs;
    std::vector<BoundaryCondition*> noBCs;

    // Poisson (homogeneous versions; inhomogeneity lives in bbc)
    FixedBC  bcPxl(mesh.facePatch(Axis::X, Side::LOW),  0.0);
    FixedBC  bcPxh(mesh.facePatch(Axis::X, Side::HIGH), 0.0);
    NoFluxBC bcPyl(mesh.facePatch(Axis::Y, Side::LOW));
    NoFluxBC bcPyh(mesh.facePatch(Axis::Y, Side::HIGH));
    std::vector<BoundaryCondition*> bcsPhi = {&bcPxl, &bcPxh, &bcPyl, &bcPyh};

    PoissonSolver poisson(mesh, 1, 1.0, bcsPhi);
    poisson.projectNullspace = false;    // Dirichlet sides fix the level

    // === 6. Equations ========================================================
    // rhs = k*c/eps + bbc   (PoissonSolver solves -lap(Phi) = rhs)
    Equation eqRhs(rhs, "P_rhs");
    eqRhs.setRHS(pw(c, bbc, PHIX_FN (double cv, double bv) {
        return kchg * cv / eps + bv;
    }));

    // mu = f'(c) + k*Phi - kappa*lap(c)
    Equation eqMu(mu, "CH_mu");
    eqMu.setRHS(
        pw(c, Phi, PHIX_FN (double cv, double pv) {
            return 2.0 * rho * (cv - ca) * (cv - cb)
                       * (2.0 * cv - ca - cb)
                 + kchg * pv;
        })
        - kappa * lap(c)
    );

    // dc/dt = M * lap(mu)
    Equation eqC(c, "CH_c");
    eqC.setRHS(M * lap(mu));

    // === 7. Output ===========================================================
    eqC.step = start_step;
    eqC.time = start_step * dt;

    IO::OutputWriter writer(cfg["output"]);

    std::unique_ptr<IO::PFHubWriter> pfhub;
    int pfhubEvery = 0;
    const double dV = dx * dy;
    auto freeEnergy = [&]() {
        for (auto* bc : bcs) bc->applyOnGPU(c);
        double F = reduce::fieldSumPW(c, Phi, PHIX_FN (Real cv, Real pv) {
            const Real da = cv - Real(ca), db = Real(cb) - cv;
            return Real(rho) * da * da * db * db
                 + Real(0.5) * Real(kchg) * cv * pv;
        });
        F += 0.5 * kappa * reduce::fieldGradSq(c);
        return F * dV;
    };
    if (cfg.has("pfhub")) {
        pfhubEvery = cfg["pfhub"]["energy_interval"];
        const std::string csvPath = cfg["pfhub"]["csv"];
        pfhub = std::make_unique<IO::PFHubWriter>(
            csvPath, std::vector<std::string>{"time", "free_energy",
                                              "total_c"});
    }

    if (start_step == 0) {
        writer.writeFields(c, 0, eqC.time);
        std::cout << "Starting CH+Poisson simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming from step " << start_step << "\n";
    }
    writer.resetTimer();

    // === 8. Time loop ========================================================
    int cgIters = 0;
    for (int s = start_step; s < nSteps; ++s) {
        eqRhs.advanceSteady(noBCs);                    // rhs(c^n)
        auto r = poisson.solve(Phi, rhs, 1e-8, 2000);  // warm-started
        cgIters = r.iterations;
        eqMu.advanceSteady(bcs, &c);                   // mu(c^n, Phi^n)
        eqC.advanceTransient(bcs, dt, &mu);            // c^{n+1}

        if (pfhub && (eqC.step % pfhubEvery == 0 || s == start_step))
            pfhub->addRow({eqC.time, freeEnergy(),
                           reduce::fieldSum(c) * dV});
        if (writer.shouldPrint(eqC.step)) {
            writer.printProgress(eqC.step, eqC.time);
            std::cout << "    CG iters (last) = " << cgIters << "\n";
        }
        if (writer.shouldWrite(eqC.step)) {
            writer.writeFields(c,   eqC.step, eqC.time);
            writer.writeFields(Phi, eqC.step, eqC.time);
        }
    }

    // === 9. Final cross-sections (PFHub BM6 deliverable) =====================
    if (cfg.has("pfhub")) {
        const double xCut = cfg["pfhub"]["x_cut"];
        const double yCut = cfg["pfhub"]["y_cut"];
        const std::string xCsv = cfg["pfhub"]["x_csv"];
        const std::string yCsv = cfg["pfhub"]["y_csv"];
        c.downloadCurrFromDevice();
        Phi.downloadCurrFromDevice();
        auto V = [&](ScalarField& f, int i, int j) {
            return static_cast<double>(f.curr[static_cast<std::size_t>(
                f.index(i, j))]);
        };
        const int iC = std::min(std::max(
            static_cast<int>(std::floor((xCut - x0) / dx)), 0), nx - 1);
        const int jC = std::min(std::max(
            static_cast<int>(std::floor((yCut - y0) / dy)), 0), ny - 1);
        {
            IO::PFHubWriter wx(xCsv, {"y", "concentration", "potential"});
            for (int j = 0; j < ny; ++j)
                wx.addRow({mesh.coord(1, j), V(c, iC, j), V(Phi, iC, j)});
        }
        {
            IO::PFHubWriter wy(yCsv, {"x", "concentration", "potential"});
            for (int i = 0; i < nx; ++i)
                wy.addRow({mesh.coord(0, i), V(c, i, jC), V(Phi, i, jC)});
        }
        std::cout << "  wrote " << xCsv << ", " << yCsv << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}

int main(int argc, char* argv[]) {
    return PhiX::runGuarded(phixMain, argc, argv);
}
