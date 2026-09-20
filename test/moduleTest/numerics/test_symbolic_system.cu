#include "numerics/Symbolic.h"
#include "boundary/PeriodicBC.h"
#include "core/CudaCheck.h"
#include <cmath>
#include <iostream>
#include <sstream>

using namespace PhiX;
namespace num = PhiX::numerics;
namespace sym = num::symbolic;
namespace {
constexpr double tolerance = sizeof(Real) == sizeof(float) ? 3e-5 : 3e-12;
void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}
void close(double actual, double expected, const std::string &message) {
    require(std::isfinite(actual) && std::abs(actual - expected) < tolerance, message);
}
template <class F> void rejects(F operation, const std::string &message) {
    bool caught = false;
    try { operation(); } catch (const std::exception &) { caught = true; }
    require(caught, message);
}
void upload(ScalarField &f) {
    f.allocDevice();
    f.uploadAllToDevice();
}
// A reusable model function returns a formula, without allocating a grid field.
sym::Expr mobility(sym::Expr c) { return sym::named("mobility", 1 + c*c); }

void coupledDiffusion() {
    constexpr int n = 12;
    constexpr double h = .3, dt = .005;
    auto mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, n, h, 0, n, h, 0);
    PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW)), by(mesh.facePatch(Axis::Y, Side::LOW));
    // A second configuration checks that fusion keeps separately selected stencils.
    for (bool wider : {false, true})
        for (auto backend : {num::Backend::CPU, num::Backend::CUDA})
            for (auto fusion : {num::Fusion::Off, num::Fusion::Auto, num::Fusion::Required}) {
                ScalarField c(mesh, "c", 2), v(mesh, "v", 2);
                c.fill(123); v.fill(456); // stale ghosts must be replaced by System
                std::vector<double> referenceC(n*n), referenceV(n*n);
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) {
                        const int k = i + n*j;
                        c.curr[c.index(i,j)] = Real(.2 + .1*std::sin(2*M_PI*i/n)*std::cos(2*M_PI*j/n));
                        v.curr[v.index(i,j)] = Real(.1*std::cos(2*M_PI*(i+j)/n));
                        referenceC[k] = c.curr[c.index(i,j)];
                        referenceV[k] = v.curr[v.index(i,j)];
                    }
                auto schemes = Schemes::fromJson({
                    {"schema", 2}, {"policy", "strict"},
                    {"ddt", {{"ddt(c)", "EULER"}, {"ddt(v)", "EULER"}}},
                    {"gradient", {{"grad(c,x)", wider ? "CD4" : "CD2"},
                                  {"grad(c,y)", wider ? "CD4" : "CD2"}}},
                    {"laplacian", {{"lap(c)", wider ? "Iso9" : "CD2"}, {"lap(v)", "CD2"}}}});
                num::System system(schemes);
                auto C = sym::field(c), V = sym::field(v);
                system.add(sym::ddt(c) == sym::div(mobility(C)*sym::grad(C)) + V - C*C*C);
                system.add(2*sym::ddt(v) == .2*sym::lap(V) + C - V);
                system.bc(c, {&bx, &by}); system.bc(v, {&bx, &by});
                auto inventory = system.requiredOperators();
                require(inventory.entries.size() == 6, "automatic equation inventory");
                require(inventory.schemeTemplate()["ddt"].contains("ddt(v)"), "weighted ddt inventory");
                std::ostringstream equations;
                system.reportEquations(equations);
                require(equations.str().find("mobility") != std::string::npos, "original formula missing");
                // Registration and inspection precede device allocation and compile.
                if (backend == num::Backend::CUDA) { upload(c); upload(v); }
                if (backend == num::Backend::CUDA && fusion == num::Fusion::Auto)
                    system.compile(); // CUDA + Auto + SameLevel are the public defaults
                else
                    system.compile({backend, fusion, num::Coupling::SameLevel});
                std::ostringstream execution;
                system.report(execution);
                require(execution.str().find(fusion == num::Fusion::Off ? "CUDA kernels=4" :
                                            "fused scalar DAG; CUDA kernels=1") != std::string::npos,
                        "System did not select expected fusion plan");
                require(system.persistentBytes() == c.storedBytes() + v.storedBytes(),
                        "unexpected intermediate persistent fields");
                for (int step = 0; step < 5; ++step) {
                    auto nextC = referenceC, nextV = referenceV;
                    auto at = [&](const std::vector<double> &a, int i, int j) {
                        return a[(i+n)%n + n*((j+n)%n)];
                    };
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            const int k = i + n*j;
                            double c0 = referenceC[k], v0 = referenceV[k];
                            double gx = (at(referenceC,i+1,j)-at(referenceC,i-1,j))/(2*h);
                            double gy = (at(referenceC,i,j+1)-at(referenceC,i,j-1))/(2*h);
                            double lc = (at(referenceC,i+1,j)+at(referenceC,i-1,j)+
                                         at(referenceC,i,j+1)+at(referenceC,i,j-1)-4*c0)/(h*h);
                            if (wider) {
                                gx = (-at(referenceC,i+2,j)+8*at(referenceC,i+1,j)-
                                      8*at(referenceC,i-1,j)+at(referenceC,i-2,j))/(12*h);
                                gy = (-at(referenceC,i,j+2)+8*at(referenceC,i,j+1)-
                                      8*at(referenceC,i,j-1)+at(referenceC,i,j-2))/(12*h);
                                lc = (4*(at(referenceC,i+1,j)+at(referenceC,i-1,j)+
                                         at(referenceC,i,j+1)+at(referenceC,i,j-1))+
                                      at(referenceC,i+1,j+1)+at(referenceC,i+1,j-1)+
                                      at(referenceC,i-1,j+1)+at(referenceC,i-1,j-1)-20*c0)/(6*h*h);
                            }
                            double lv = (at(referenceV,i+1,j)+at(referenceV,i-1,j)+
                                         at(referenceV,i,j+1)+at(referenceV,i,j-1)-4*v0)/(h*h);
                            nextC[k] += dt*((1+c0*c0)*lc + 2*c0*(gx*gx+gy*gy) + v0 - c0*c0*c0);
                            nextV[k] += dt*(.2*lv + c0-v0)/2;
                        }
                    referenceC = std::move(nextC); referenceV = std::move(nextV);
                    system.advance(dt);
                }
                if (backend == num::Backend::CUDA) {
                    c.downloadCurrFromDevice(); v.downloadCurrFromDevice();
                }
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) {
                        close(c.curr[c.index(i,j)], referenceC[i+n*j], "nonlinear diffusion stencil reference");
                        close(v.curr[v.index(i,j)], referenceV[i+n*j], "same-level coupled reference");
                    }
            }
}
void definitionsAndReplacement() {
    auto mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, 6, .3, 0, 5, .3, 0);
    PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW)), by(mesh.facePatch(Axis::Y, Side::LOW));
    for (auto backend : {num::Backend::CPU, num::Backend::CUDA}) {
        ScalarField c(mesh, "c", 1), auxiliary(mesh, "auxiliary", 1), coordinate(mesh, "coordinate", 1);
        c.fill(.2); auxiliary.fill(0); coordinate.fill(0);
        if (backend == num::Backend::CUDA)
            for (auto f : {&c, &auxiliary, &coordinate}) upload(*f);
        num::System system(Schemes{});
        auto C = sym::field(c);
        system.bc(c, {&bx, &by});
        system.define(auxiliary, sym::lap(C) + C*C);
        system.define(coordinate, sym::coordinate(0)); // no input field: use output layout
        system.add(sym::ddt(c) == sym::lap(C) + sym::field(auxiliary));
        auto inventory = system.requiredOperators();
        require(inventory.entries.size() == 2, "merged symbolic inventory duplicated a key");
        for (const auto &e : inventory.entries)
            if (e.key == "lap(c)") require(e.sources.size() == 2, "merged inventory lost origins");
        system.compile({backend});
        system.refresh(coordinate);
        if (backend == num::Backend::CUDA) coordinate.downloadCurrFromDevice();
        close(coordinate.curr[coordinate.index(2,1)], mesh.coord(0,2), "output layout binding");
        system.advance(.1);
        if (backend == num::Backend::CUDA) c.downloadCurrFromDevice();
        close(c.curr[c.index(2,1)], .204, "automatic stored definition");
        // Same reads and halo, changed stencil family. Inventory follows successful replacement.
        system.replace(auxiliary, sym::dxx(C) + 2*C*C);
        auto replaced = system.requiredOperators().schemeTemplate();
        require(replaced["secondDerivative"].contains("dxx(c)"), "replacement metadata stale");
        system.refresh(auxiliary);
        if (backend == num::Backend::CUDA) auxiliary.downloadCurrFromDevice();
        close(auxiliary.curr[auxiliary.index(2,1)], 2*.204*.204, "replacement cache invalidation");
        rejects([&] { system.replace(auxiliary, sym::Expr(7)); }, "replacement changed dependencies");
        require(system.requiredOperators().schemeTemplate() == replaced, "failed replacement changed inventory");
        system.replace(c, sym::lap(C) + 2*sym::field(auxiliary));
        require(system.requiredOperators().schemeTemplate()["ddt"].contains("ddt(c)"), "replacement lost ddt");
        system.replace(auxiliary, (sym::lap(C)+3*C*C).expand().discretize(Schemes{}));
        require(!system.requiredOperators().schemeTemplate().contains("secondDerivative"),
                "explicit discrete replacement retained old symbolic metadata");
    }
}
void invalidEquations() {
    auto mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, 6, .3, 0, 5, .3, 0);
    ScalarField c(mesh, "c", 1);
    PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW)), by(mesh.facePatch(Axis::Y, Side::LOW));
    auto C = sym::field(c);
    auto config = (sym::ddt(c) == sym::lap(C)).expand().requiredOperators().schemeTemplate();
    config["laplacian"].erase("lap(c)");
    num::System missing(Schemes::fromJson(config));
    missing.add(sym::ddt(c) == sym::lap(C)); // no scheme binding during registration
    missing.bc(c, {&bx, &by});
    require(missing.requiredOperators().entries.size() == 2, "cannot inspect unbound equation");
    rejects([&] { missing.compile({num::Backend::CPU}); }, "missing scheme accepted by compile");
    num::System noHalo(Schemes{});
    noHalo.add(sym::ddt(c) == sym::lap(C));
    rejects([&] { noHalo.compile({num::Backend::CPU}); }, "missing halo accepted by automatic path");
    num::System higher(Schemes{});
    rejects([&] { higher.add(sym::ddt(c) == sym::lap(sym::lap(C))); }, "unsupported spatial order accepted");
    rejects([&] { higher.add(sym::Equation{nullptr, C}); }, "null state accepted");
    require(higher.requiredOperators().entries.empty(), "failed registration mutated system");
    num::System timeScheme(Schemes::builtin({{"ddt", "RK4"}}));
    timeScheme.add(sym::ddt(c) == C);
    rejects([&] { timeScheme.compile({num::Backend::CPU}); }, "unsupported time scheme accepted");
}
} // namespace
int main() {
    try {
        coupledDiffusion(); definitionsAndReplacement(); invalidEquations();
        std::cout << "generic symbolic System: independent coupled stencils, CPU/CUDA, fusion, "
                     "inventory, replacement and rejection checks passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
