/***********************************************************************\
 *
 *  Cahn-Hilliard Solver — Double-Well Free Energy (2D)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *  Solves the Cahn-Hilliard equation with a double-well bulk free-energy
 *  density for spinodal decomposition:
 *
 *      dc/dt = M ∇²μ
 *      μ     = f'(c) − κ ∇²c
 *      f'(c) = 2ρ (c − ca)(c − cb)(2c − ca − cb)
 *
 *  Architecture (v2.6+)
 *  ---------------------
 *  · μ equation uses FusedTerm — fpw + flap compile into a single GPU
 *    kernel, eliminating the intermediate global-memory write.
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "core/RunGuard.h"
#include "core/Error.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "field/ReducePW.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "equation/FusedTerm.h"
#include "solver/Solver.h"
#include "IO/ConfigFile.h"
#include "scheme/Schemes.h"
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

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    nx, dx, x0,
                                    ny, dy, y0);
    mesh.print();

    // === 2. Time parameters ==================================================
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];

    // === 3. Fields & initialization ==========================================
    ScalarField c (mesh, "c",  /*ghost=*/1);
    ScalarField mu(mesh, "mu", /*ghost=*/1);
    c.fill(0);
    mu.fill(0);

    const std::string start_from = cfg["initialize"]["start_from"];
    const int         start_step = IO::resolveStartStep(start_from);

    // Cold start: named initializer from config (default = random noise
    // around c = 0.5); set "c_init": "" to read settings/initial_field/c.field
    // instead.  Warm restart reads output/c_<step>.field regardless.
    const std::string c_init = cfg["initialize"].value("c_init", "random:0.45:0.55");
    IO::initField(c, start_step, c_init);
    // mu needs no initial data: the STEADY step recomputes it from c before
    // its first use (on cold AND warm starts alike).

    c.allocDevice();
    c.uploadAllToDevice();
    mu.allocDevice();
    mu.uploadAllToDevice();

    // === 4. Boundary conditions ==============================================
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // === 5. Equations ========================================================
    const double rho   = cfg["constants"]["rho"];
    const double ca    = cfg["constants"]["ca"];
    const double cb    = cfg["constants"]["cb"];
    const double kappa = cfg["constants"]["kappa"];
    const double M     = cfg["constants"]["M"];

    using namespace PhiX::Fused;

    // μ = f'(c) − κ ∇²c
    // FusedTerm fuses fpw + flap into a single GPU kernel: no intermediate
    // scratch write for the pointwise term before the Laplacian is computed.
    Equation eqMu(mu, "CH_mu");
    eqMu.setRHS(fuse(
        fpw(c, PHIX_FN (double cv) {
            return 2.0 * rho * (cv - ca) * (cv - cb)
                       * (2.0 * cv - ca - cb);
        }) + flap(c) * (-kappa),
        c));

    // dc/dt = M ∇²μ
    Equation eqC(c, "CH_c");
    eqC.setRHS(M * lap(mu, sch));

    // === 6. Solver (multi-step: STEADY μ, then TRANSIENT c) =================
    //  Step 1 (STEADY)    — algebraic: μ = f'(c) − κ∇²c
    //  Step 2 (TRANSIENT) — time-integrated: dc/dt = M∇²μ
    // Fused kernels are EULER-only (a fused RHS never sees RK4 stage buffers).
    if (sch.ddt() != TimeScheme::EULER)
        throw ConfigError("schemes", "this solver evaluates its fused RHS with EULER only",
                          "set \"ddt\": { \"default\": \"EULER\" } in settings/schemes.jsonc");
    Solver solver({
        {&c,  bcs, &eqMu, EquationType::STEADY},
        {&mu, bcs, &eqC,  EquationType::TRANSIENT}
    }, dt, sch.ddt());
    sch.warnUnused();   // per-term keys nobody looked up = misspelled field names

    solver.step = start_step;
    solver.time = start_step * dt;

    // === 7. Output & time loop ===============================================
    IO::OutputWriter writer(cfg["output"]);

    // Optional PFHub-style free-energy CSV (cfg["pfhub"] section):
    //   F(t) = Σ [ ρ(c−ca)²(cb−c)² + κ/2 |∇c|² ] dV   (+ total solute)
    std::unique_ptr<IO::PFHubWriter> pfhub;
    int pfhubEvery = 0;
    const double dV = dx * dy;
    auto freeEnergy = [&]() {
        for (auto* bc : bcs) bc->applyOnGPU(c);   // fresh ghosts for ∇c
        const double fChem = reduce::fieldSumPW(c, PHIX_FN (Real v) {
            const Real da = v - Real(ca), db = Real(cb) - v;
            return Real(rho) * da * da * db * db;
        });
        return (fChem + 0.5 * kappa * reduce::fieldGradSq(c)) * dV;
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
        writer.writeFields(c, 0, solver.time);
        std::cout << "Starting Cahn-Hilliard simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming Cahn-Hilliard simulation from step " << start_step
                  << " (t=" << start_step * dt << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt
                  << "\n";
    }

    writer.resetTimer();

    solver.run(nSteps - start_step, 1, [&](const Solver& s) {
        if (writer.shouldPrint(s.step))
            writer.printProgress(s.step, s.time);
        if (writer.shouldWrite(s.step))
            writer.writeFields(c, s.step, s.time);
        if (pfhub && s.step % pfhubEvery == 0)
            pfhub->addRow({s.time, freeEnergy(),
                           reduce::fieldSum(c) * dV});
    });

    std::cout << "Done.\n";
    return 0;
}

int main(int argc, char* argv[]) {
    return PhiX::runGuarded(phixMain, argc, argv);
}
