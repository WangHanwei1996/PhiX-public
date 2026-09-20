#include "core/RunGuard.h"
#include "core/CudaCheck.h"
#include "numerics/Symbolic.h"
#include "boundary/BCFactory.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"
#include <cmath>
#include <iostream>

static int phixMain(int argc, char **argv) {
    using namespace PhiX;
    namespace num = PhiX::numerics;
    namespace sym = num::symbolic;

    // === 1. Settings and mesh ================================================
    const auto cfg = IO::ConfigFile::fromArgs(argc, argv);
    const auto schemes = Schemes::load(cfg);
    const auto m = cfg["mesh"];
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, m["nx"], m["dx"], 0,
                                   m["ny"], m["dy"], 0);
    const int ghost = m.value("ghost", 1);
    const double dt = cfg["initialize"]["dt"];
    const int steps = cfg["initialize"]["nSteps"];
    const int start = IO::resolveStartStep(
        cfg["initialize"].value("start_from", std::string("initial_field")));
    const double diffusion = cfg["constants"]["diffusion"];
    const double reaction = cfg["constants"]["reaction"];

    // === 2. State and boundary conditions ====================================
    ScalarField c(mesh, "c", ghost);
    if (start)
        IO::initField(c, start);
    else {
        const double lx = mesh.n[0]*mesh.d[0], ly = mesh.n[1]*mesh.d[1];
        c.initialize([=](double x, double y, double) {
            return .2 + .1*std::cos(2*M_PI*x/lx)*std::cos(2*M_PI*y/ly);
        });
    }
    auto bcs = buildBCs(mesh, cfg["boundary_conditions"]);

    // === 3. Original equation ================================================
    // d_t c = div(D(c) grad(c)) + r(c-c^3). D is an inline formula, not a field.
    auto C = sym::field(c);
    auto D = sym::named("D(c)", diffusion*(1 + C*C));
    num::System system(schemes);
    system.bc(c, bcs.ptrs);
    system.add(sym::ddt(c) == sym::div(D*sym::grad(C)) + reaction*(C - C*C*C));
    system.requiredOperators().print(std::cout);

    // === 4. Compile and inspect the execution plan ===========================
    const auto execution = cfg.data().value("execution", nlohmann::json::object());
    const std::string backend = execution.value("backend", std::string("cuda"));
    const std::string fusion = execution.value("fusion", std::string("auto"));
    if (backend != "cpu" && backend != "cuda")
        throw std::invalid_argument("backend must be cpu or cuda");
    if (fusion != "off" && fusion != "auto" && fusion != "required")
        throw std::invalid_argument("fusion must be off, auto or required");
    num::ExecutionOptions options;
    options.backend = backend == "cpu" ? num::Backend::CPU : num::Backend::CUDA;
    options.fusion = fusion == "off" ? num::Fusion::Off :
                     fusion == "required" ? num::Fusion::Required : num::Fusion::Auto;
    if (options.backend == num::Backend::CUDA) {
        c.allocDevice();
        c.uploadAllToDevice();
    }
    if (execution.value("report_expansion", false))
        system.reportEquations(std::cout);
    system.compile(options);
    system.restoreClock(start*dt, start);
    system.report(std::cout);
    schemes.warnUnused();

    // === 5. Time loop and output (run from a PhiX-run case directory) =========
    IO::OutputWriter writer(cfg["output"]);
    auto write = [&] {
        if (options.backend == num::Backend::CPU)
            writer.writeHostFields(c, system.step(), system.time());
        else
            writer.writeFields(c, system.step(), system.time());
    };
    if (!start) write();
    for (int step = start; step < steps; ++step) {
        system.advance(dt);
        if (writer.shouldPrint(system.step()))
            writer.printProgress(system.step(), system.time());
        if (writer.shouldWrite(system.step())) write();
    }
    return 0;
}
int main(int argc, char **argv) {
    return PhiX::runGuarded(phixMain, argc, argv);
}
