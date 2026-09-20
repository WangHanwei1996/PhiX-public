// Catalogue integration: numerical evaluation through every registered factory,
// selection provenance, and case/fixed-backend conflicts. No duplicate name list.
#include "scheme/SchemeCatalog.h"
#include "scheme/Schemes.h"
#include "equation/Equation.h"
#include "equation/FusedTerm.h"
#include "operators/Laplacian.h"
#include "operators/Gradient.h"
#include "operators/GradSq.h"
#include "operators/Advection.h"
#include "operators/Anisotropy.h"
#include "solver/Solver.h"
#include "core/Error.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <unistd.h>

using namespace PhiX;
namespace fs = std::filesystem;
static int failures = 0;
static void expect(bool ok, const std::string& message) {
    std::cout << (ok ? "PASS " : "FAIL ") << message << '\n';
    if (!ok) ++failures;
}
template<class E, class Fn>
static void throws(Fn fn, const std::string& message) {
    try { fn(); expect(false, message); }
    catch (const E&) { expect(true, message); }
    catch (const std::exception& e) { expect(false, message + ": " + e.what()); }
}
struct TempCase {
    fs::path dir = fs::temp_directory_path() / ("phix_catalog_" + std::to_string(::getpid()));
    TempCase() { fs::create_directories(dir); }
    ~TempCase() { std::error_code ec; fs::remove_all(dir, ec); }
};

int main() {
    TempCase tmp;
    const auto file = tmp.dir / "schemes.jsonc";
    std::ofstream(file) << R"JSON({
        "laplacian": {"default":"CD4", "lap(f)":"Iso9", "lap(typo)":"CD6"},
        "gradient": {"grad(f)":"CD2", "grad(a)":"CD4"}
    })JSON";
    const auto s = Schemes::fromFile(file.string(), {{"gradient","CD6"}});
    auto chosen = s.select("laplacian", "lap(f)");
    expect(chosen.name == "Iso9" && chosen.source == Schemes::Source::Entry
           && chosen.matchedKey == "lap(f)" && chosen.path == file.string(), "exact entry provenance");
    chosen = s.select("laplacian", "lap(other)");
    expect(chosen.name == "CD4" && chosen.source == Schemes::Source::SectionDefault
           && chosen.matchedKey == "default", "section default provenance");
    chosen = s.select("gradient", "grad(other)");
    expect(chosen.name == "CD6" && chosen.source == Schemes::Source::SolverDefault
           && chosen.path.empty(), "solver default provenance");
    chosen = s.select("advection", "adv(f)");
    expect(chosen.name == "UW1" && chosen.source == Schemes::Source::Builtin, "built-in provenance");
    chosen = Schemes::fromFile(file.string()).select("laplacian", "lap(f)");
    expect(chosen.describe().find(file.string()+":laplacian/lap(f)") != std::string::npos,
           "selection survives destruction of its Schemes object");
    std::ostringstream warnings;
    auto* previous = std::cerr.rdbuf(warnings.rdbuf());
    s.warnUnused(); std::cerr.rdbuf(previous);
    expect(warnings.str().find("lap(typo)") != std::string::npos
           && warnings.str().find("lap(f)") == std::string::npos,
           "provenance lookup preserves unused-entry tracking");
    try {
        s.requireFixed("laplacian", "lap(f)", "CD2", "test-fixed");
        expect(false, "fixed mismatch rejected");
    } catch (const ValidationError& e) {
        const std::string message = e.what();
        expect(message.find("lap(f)") != std::string::npos
               && message.find("Iso9") != std::string::npos
               && message.find("CD2") != std::string::npos
               && message.find(file.string()) != std::string::npos,
               "fixed mismatch explains source and both schemes");
    }
    throws<ValidationError>([&] { s.requireFixed("laplacian","lap(f)","UW1","test"); },
                            "fixed scheme must belong to its operator");
    throws<ValidationError>([&] { s.select("unknown", ""); }, "unknown family rejected");
    const auto defaults = Schemes::builtin();
    for (const auto& family : scheme::families()) {
        const auto& entries = scheme::catalog(family.id);
        expect(!entries.empty() && defaults.resolve(family.section, "") == entries.front().name,
               std::string(family.section) + " default comes from executable table");
        std::set<std::string> unique;
        for (const auto& d : entries) {
            expect(unique.insert(d.name).second && d.meaning[0] && d.effective2D[0],
                   std::string(family.section) + "/" + d.name + " has unique identity and semantics");
            // Every advertised name is accepted by configuration, including enum selectors.
            const auto config = Schemes::builtin({{family.section, d.name}});
            expect(config.resolve(family.section, "") == d.name, "catalogue/config agreement");
            if (family.id == scheme::Family::Ddt) (void)config.ddt();
            if (family.id == scheme::Family::AnisoDiv) (void)config.anisoDiv();
        }
    }
    expect(scheme::find(scheme::Family::Laplacian,"CD4")->primaryHalo == 2,
           "CD4 radius is available before field allocation");
    expect(scheme::find(scheme::Family::Laplacian,"Iso9")->grid2D == scheme::Grid2D::SquareRequired
           && scheme::find(scheme::Family::Gradient,"Iso9")->grid2D == scheme::Grid2D::SquareForIsotropy,
           "same name retains operator-specific grid conditions");
    expect(std::string(scheme::find(scheme::Family::Laplacian,"Iso27")->effective2D) == "CD2",
           "legacy Iso27 2D fallback is explicit");
    expect(!scheme::find(scheme::Family::Gradient,"UW1"), "names do not leak between families");

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, 12,.25,0.,9,.25,0.);
    ScalarField f(mesh,"f",3), a(mesh,"a",3), rhs(mesh,"rhs",3);
    VectorField u(mesh,"u",2,3);
    for (int j=-3; j<12; ++j) for (int i=-3; i<15; ++i) {
        const double x=mesh.coord(0,i), y=mesh.coord(1,j);
        const auto c=f.index(i,j);
        f.curr[c]=x*x+y*y; a.curr[c]=2*x-3*y+1;
        u[0].curr[c]=x<1.5 ? 1.2 : -1.2; u[1].curr[c]=-.3+.2*y;
    }
    for (auto* field : {&f,&a,&rhs,&u[0],&u[1]}) {
        field->allocDevice(); field->uploadAllToDevice();
    }
    auto check = [&](Term term, auto exact, const std::string& label) {
        Equation eq(f,"catalogue"); eq.setRHS(term);
        for (bool gpu : {false,true}) {
            if (gpu) { eq.computeRHS(rhs); rhs.downloadCurrFromDevice(); }
            else eq.computeRHSCPU(rhs);
            double error=0;
            for(int j=0;j<9;++j) for(int i=0;i<12;++i)
                error=std::max(error,std::fabs(double(rhs.curr[rhs.index(i,j)])-exact(i,j)));
#ifdef PHIX_REAL_FLOAT
            constexpr double tolerance=2e-3;
#else
            constexpr double tolerance=2e-10;
#endif
            expect(error<tolerance,label+(gpu?" GPU":" CPU")+" polynomial error="+std::to_string(error));
        }
    };
    for (const auto& d : scheme::catalog(scheme::Family::Laplacian))
        check(lap(f,std::string(d.name)), [](int,int){return 4.;}, std::string("lap ")+d.name);
    for (const auto& d : scheme::catalog(scheme::Family::Gradient))
        for(int axis=0;axis<2;++axis)
            check(grad(f,axis,std::string(d.name)),[&](int i,int j){return 2*mesh.coord(axis,axis?j:i);},
                  std::string("grad ")+d.name+" axis="+std::to_string(axis));
    for (const auto& d : scheme::catalog(scheme::Family::GradSq))
        check(gradSq(f,std::string(d.name)),[&](int i,int j){
            const double x=mesh.coord(0,i),y=mesh.coord(1,j); return 4*(x*x+y*y);
        },std::string("gradSq ")+d.name);
    for (const auto& d : scheme::catalog(scheme::Family::Advection))
        check(adv(u,a,std::string(d.name)),[&](int i,int j){
            auto c=a.index(i,j); return double(2*u[0].curr[c]-3*u[1].curr[c]);
        },std::string("adv ")+d.name);
    for (const auto& d : scheme::catalog(scheme::Family::AnisoDiv)) {
        AnisoParams params; params.eps=0; params.scheme=anisoSchemeFromString(d.name);
        check(anisoDiv(f,params),[](int,int){return 4.;},std::string("anisoDiv zero-epsilon ")+d.name);
    }
    throws<std::invalid_argument>([&]{lap(f,"unknown");},"direct operator retains exception type");
    throws<std::invalid_argument>([&]{anisoSchemeFromString("");},"empty anisotropy name still rejected");
    check(lap(f,""),[](int,int){return 4.;},"empty lap name retains CD2 default");
    // Fixed fused overloads must really execute CD2 when compatible.
    auto fused = Fused::flap(f, defaults);
    Equation eq(f,"fixed"); eq.setRHS(Fused::fuse(fused,f));
    eq.computeRHS(rhs); rhs.downloadCurrFromDevice();
    expect(std::fabs(rhs.curr[rhs.index(4,4)]-4.)<1e-6,"case-aware fused lap executes");
    throws<ValidationError>([&]{Fused::flap(f,s);},"fused lap rejects requested Iso9");
    throws<ValidationError>([&]{Fused::fgrad_dot(f,a,s);},"fused grad_dot checks both input selections");
    (void)Fused::fgrad_dot(f,a,defaults);
    std::cout << "failures=" << failures << '\n';
    return failures ? 1 : 0;
}
