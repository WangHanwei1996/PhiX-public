#include "core/RunGuard.h"
#include "numerics/Face.h"
#include "boundary/BCFactory.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"
#include <iostream>

static int phixMain(int argc, char **argv) {
    using namespace PhiX;
    namespace num = PhiX::numerics;
    const auto cfg = IO::ConfigFile::fromArgs(argc, argv);
    const auto sch = Schemes::load(cfg);
    const auto m = cfg["mesh"];
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, m["nx"], m["dx"], 0, m["ny"], m["dy"], 0);
    const int ghost = m.value("ghost", 1);
    ScalarField c(mesh, "c", ghost), mu(mesh, "mu", ghost), mobility(mesh, "mobility", ghost);
    const double dt = cfg["initialize"]["dt"];
    const int steps = cfg["initialize"]["nSteps"];
    const int start =
        IO::resolveStartStep(cfg["initialize"].value("start_from", std::string("initial_field")));
    if (start)
        IO::initField(c, start);
    else
        c.initialize([](double x, double y, double) {
            return .2 + .02 * cos(6.283185307179586 * x) * cos(6.283185307179586 * y);
        });
    auto bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    const auto execution = cfg.data().value("execution", nlohmann::json::object());
    const std::string backend = execution.value("backend", std::string("cuda")),
                      fusion = execution.value("fusion", std::string("auto"));
    if (backend != "cuda" && backend != "cpu")
        throw std::invalid_argument("backend must be cpu or cuda");
    if (fusion != "off" && fusion != "auto" && fusion != "required")
        throw std::invalid_argument("fusion must be off, auto or required");
    num::ExecutionOptions options;
    options.backend = backend == "cpu" ? num::Backend::CPU : num::Backend::CUDA;
    options.fusion = fusion == "off"        ? num::Fusion::Off
                     : fusion == "required" ? num::Fusion::Required
                                            : num::Fusion::Auto;
    if (options.backend == num::Backend::CUDA)
        for (auto *f : {&c, &mu, &mobility}) {
            f->allocDevice();
            f->uploadAllToDevice();
        }
    num::Spatial ops(sch);
    num::Faces faces(sch);
    num::System system(sch);
    for (auto *f : {&c, &mu, &mobility})
        system.bc(*f, bcSet.ptrs);
    system.define(mu, ops.pw(c, num::pure(PHIX_FN(Real x) { return x * x * x - x; })) -
                          .002 * ops.lap(c));
    system.define(mobility,
                  ops.pw(c, num::pure(PHIX_FN(Real x) { return Real(.05) * (1 + x * x); })));
    const auto flux = faces.flux("diffusion", [&](auto face) {
        return faces.interp(mobility, face) * faces.snGrad(mu, face);
    });
    system.add(num::ddt(c) == faces.div(flux));
    system.compile(options);
    system.restoreClock(start * dt, start);
    system.report(std::cout);
    sch.warnUnused();
    IO::OutputWriter writer(cfg["output"]);
    auto write = [&] {
        system.refresh(mu);
        for (auto *f : {&c, &mu}) {
            if (options.backend == num::Backend::CPU)
                writer.writeHostFields(*f, system.step(), system.time());
            else
                writer.writeFields(*f, system.step(), system.time());
        }
    };
    if (!start)
        write();
    for (int i = start; i < steps; ++i) {
        system.advance(dt);
        if (writer.shouldPrint(system.step()))
            writer.printProgress(system.step(), system.time());
        if (writer.shouldWrite(system.step()))
            write();
    }
    return 0;
}
int main(int argc, char **argv) {
    return PhiX::runGuarded(phixMain, argc, argv);
}
