// ---------------------------------------------------------------------------
// tutorials/quickstart/quickstart.cu — PhiX "first solver": 2D Allen-Cahn
//
//     ∂φ/∂t = M ∇²φ + M (φ − φ³)      (gradient flow of  F = ∫ ¼(φ²−1)² + ½|∇φ|²)
//
// Every number comes from settings/settings.jsonc, every numerical scheme
// from settings/schemes.jsonc.  README.md in this
// directory explains each line below and each config key.
// ---------------------------------------------------------------------------

#include "core/RunGuard.h"
#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "solver/Solver.h"
#include "IO/ConfigFile.h"
#include "scheme/Schemes.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

static int phixMain(int argc, char* argv[])
{
    using namespace PhiX;

    // Config file: argv[1], or settings/settings.jsonc.  Every cfg[...] is
    // checked — a missing key throws ConfigError naming the full path.
    IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);
    // Numerical schemes from settings/schemes.jsonc (missing file → the
    // defaults below, i.e. what this solver used before v3.8.8).
    Schemes sch = Schemes::load(cfg, {{"ddt", "RK4"}});

    // 1. Mesh — a parameter container (dims, spacing, origin), no big arrays
    const int    nx = cfg["mesh"]["nx"], ny = cfg["mesh"]["ny"];
    const double dx = cfg["mesh"]["dx"], dy = cfg["mesh"]["dy"];
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, dx, 0.0, ny, dy, 0.0);

    // 2. Field — one ghost layer; initialised on the CPU, then uploaded
    ScalarField phi(mesh, "phi", /*ghost=*/1);
    const std::string init = cfg["initialize"]["phi_init"];   // e.g. "random:-0.05:0.05"
    IO::initField(phi, /*startStep=*/0, init);
    phi.allocDevice();
    phi.uploadAllToDevice();

    // 3. Boundary conditions from config (Periodic / NoFlux / Fixed)
    auto bcs = buildBCs(mesh, cfg["boundary_conditions"]);

    // 4. Equation — the right-hand side in near-mathematical notation
    const double M = cfg["constants"]["M"];
    Equation eq(phi, "AllenCahn");
    eq.setRHS( M * lap(phi, sch)
             + M * pw(phi, PHIX_FN (double p) { return p - p * p * p; }) );

    // 5. Solver (time scheme from schemes.jsonc, RK4 here; BCs applied
    //    automatically) + output writer
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];
    Solver solver(eq, bcs.ptrs, dt, sch.ddt());
    sch.warnUnused();   // per-term keys nobody looked up = misspelled field names
    IO::OutputWriter writer(cfg["output"]);       // cadences + formats from "output"

    writer.writeFields(phi, 0, 0.0);
    solver.run(nSteps, /*callbackEvery=*/1, [&](const Solver& s) {
        if (writer.shouldPrint(s.step)) writer.printProgress(s.step, s.time);
        if (writer.shouldWrite(s.step)) writer.writeFields(phi, s.step, s.time);
    });
    return 0;
}

int main(int argc, char* argv[]) { return PhiX::runGuarded(phixMain, argc, argv); }
