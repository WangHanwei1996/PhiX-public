// ---------------------------------------------------------------------------
// module_schemes — Schemes: case-level scheme selection (settings/schemes.jsonc)
//
// 1. builtin(): hard defaults (CD2 / UW1 / EULER); solver defaults override.
// 2. fromFile(): per-term entry → section default → solver default → builtin.
// 3. load(cfg): finds schemes.jsonc next to the settings file; missing file
//    falls back to solver defaults without throwing.
// 4. Validation: unknown section / unknown name / non-string / non-object
//    → ConfigError; bad solver defaults → ValidationError; missing explicit
//    file → IOError; resolve() on an unknown section → ValidationError.
// 5. warnUnused(): an entry never looked up is reported, a used one is not.
// 6. Operator overloads lap/grad/gradSq/adv(…, sch, …) are BITWISE the
//    explicit-scheme calls (GPU).
// 7. Every name Schemes admits is accepted by its operator's string dispatch
//    (table-drift guard), and lap Iso9 on dx != dy throws ValidationError.
// ---------------------------------------------------------------------------

#include "scheme/Schemes.h"
#include "IO/ConfigFile.h"
#include "core/Error.h"
#include "equation/Equation.h"
#include "operators/Laplacian.h"
#include "operators/Gradient.h"
#include "operators/GradSq.h"
#include "operators/Advection.h"
#include "operators/Anisotropy.h"
#include "solver/Solver.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace PhiX;
namespace fs = std::filesystem;

static int g_pass = 0, g_fail = 0;
static void expect(bool ok, const std::string& what) {
    std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << what << "\n";
    (ok ? g_pass : g_fail)++;
}
template<class E, class Fn>
static void expectThrow(Fn fn, const std::string& what) {
    try { fn(); expect(false, what + " (no throw)"); }
    catch (const E&) { expect(true, what); }
    catch (const std::exception& e) { expect(false, what + " (wrong type: " + e.what() + ")"); }
}

static void writeFile(const fs::path& p, const std::string& text) {
    std::ofstream(p) << text;
}

// Capture a std::ostream for the duration of fn().
template<class Fn>
static std::string capture(std::ostream& os, Fn fn) {
    std::ostringstream buf;
    auto* old = os.rdbuf(buf.rdbuf());
    try { fn(); } catch (...) { os.rdbuf(old); throw; }
    os.rdbuf(old);
    return buf.str();
}

static const char* kSchemesFile = R"JSON({
    // schemes.jsonc — JSON with comments, like settings.jsonc
    "ddt"       : { "default": "RK4" },
    "laplacian" : { "default": "CD4", "lap(psi)": "Iso9", "lap(Typo)": "CD2" },
    "gradSq"    : { "gradSq(psi)": "Iso9" },      // no "default" here on purpose
    "anisoDiv"  : { "default": "Iso9" }
})JSON";

// Physical-cell RHS of `term` applied to f, GPU path.
static std::vector<double> rhsGPU(ScalarField& f, const Term& term) {
    Equation eq(f, "schemes_test");
    eq.setRHS(term);
    ScalarField rhs(f.mesh, "rhs", f.ghost);
    rhs.allocDevice();
    eq.computeRHS(rhs);
    rhs.downloadCurrFromDevice();
    std::vector<double> out;
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i)
        out.push_back(rhs.curr[static_cast<std::size_t>(rhs.index(i, j))]);
    return out;
}

int main() {
    const fs::path dir = fs::temp_directory_path()
                       / ("phix_schemes_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const fs::path settings = dir / "settings.jsonc";
    const fs::path schemes  = dir / "schemes.jsonc";
    writeFile(settings, "{ \"mesh\": { \"nx\": 4 } }\n");

    // ---- 1. builtin ------------------------------------------------------
    std::cout << "[1] builtin defaults\n";
    {
        Schemes s = Schemes::builtin();
        expect(s.lap() == "CD2" && s.lap("psi") == "CD2", "lap → CD2");
        expect(s.grad() == "CD2" && s.gradSq() == "CD2", "grad/gradSq → CD2");
        expect(s.adv() == "UW1", "adv → UW1");
        expect(s.anisoDiv() == AnisoScheme::CD2, "anisoDiv → CD2");
        expect(s.ddt() == TimeScheme::EULER, "ddt → EULER");
        expect(!s.hasFile() && s.path().empty(), "no file");
        Schemes d = Schemes::builtin({{"laplacian", "Iso9"}, {"ddt", "RK4"}});
        expect(d.lap("psi") == "Iso9" && d.grad() == "CD2", "solver default laplacian=Iso9, gradient untouched");
        expect(d.ddt() == TimeScheme::RK4, "solver default ddt=RK4");
        expect(d.summary().find("laplacian=Iso9") != std::string::npos, "summary lists solver default");
    }

    // ---- 2. fromFile resolution order ------------------------------------
    std::cout << "[2] fromFile resolution\n";
    writeFile(schemes, kSchemesFile);
    {
        Schemes s = Schemes::fromFile(schemes.string(), {{"gradSq", "Iso9"}, {"gradient", "CD4"}});
        expect(s.hasFile() && s.path() == schemes.string(), "hasFile/path");
        expect(s.lap("psi") == "Iso9", "per-term entry wins: lap(psi)=Iso9");
        expect(s.lap("U") == "CD4" && s.lap() == "CD4", "section default: lap(U)=CD4");
        expect(s.gradSq("psi") == "Iso9", "gradSq(psi) entry");
        expect(s.gradSq("U") == "Iso9", "no section default → solver default (gradSq=Iso9)");
        expect(s.grad("phi") == "CD4", "section absent → solver default (gradient=CD4)");
        expect(s.adv() == "UW1", "section absent, no solver default → builtin UW1");
        expect(s.anisoDiv("phi") == AnisoScheme::Iso9, "anisoDiv default Iso9");
        expect(s.ddt() == TimeScheme::RK4, "ddt RK4");
        expect(s.resolve("laplacian", "lap(psi)") == "Iso9", "resolve() generic form");
    }

    // ---- 3. load(cfg): sibling discovery + missing-file fallback ---------
    std::cout << "[3] load(cfg)\n";
    {
        IO::ConfigFile cfg(settings.string());
        expect(cfg.path() == settings.string(), "ConfigFile::path()");
        std::string out;
        Schemes s = Schemes::builtin();
        out = capture(std::cout, [&] { s = Schemes::load(cfg); });
        expect(s.hasFile() && s.lap("psi") == "Iso9", "found sibling schemes.jsonc");
        expect(out.find("[PhiX] schemes: ") == 0 && out.find(schemes.string()) != std::string::npos,
              "notice names the file");
        fs::remove(schemes);
        out = capture(std::cout, [&] { s = Schemes::load(cfg, {{"laplacian", "Iso9"}}); });
        expect(!s.hasFile() && s.lap("psi") == "Iso9" && s.ddt() == TimeScheme::EULER,
              "missing file → solver defaults, no throw");
        expect(out.find("not found") != std::string::npos && out.find("laplacian=Iso9") != std::string::npos,
              "notice says 'not found' and lists defaults");
        writeFile(schemes, kSchemesFile);
    }

    // ---- 4. validation ---------------------------------------------------
    std::cout << "[4] validation\n";
    {
        const fs::path bad = dir / "bad.jsonc";
        writeFile(bad, "{ \"laplacians\": { \"default\": \"CD2\" } }");
        expectThrow<ConfigError>([&] { Schemes::fromFile(bad.string()); }, "unknown section → ConfigError");
        writeFile(bad, "{ \"laplacian\": { \"default\": \"Iso8\" } }");
        expectThrow<ConfigError>([&] { Schemes::fromFile(bad.string()); }, "unknown scheme name → ConfigError");
        writeFile(bad, "{ \"laplacian\": { \"default\": 2 } }");
        expectThrow<ConfigError>([&] { Schemes::fromFile(bad.string()); }, "non-string entry → ConfigError");
        writeFile(bad, "{ \"laplacian\": \"CD2\" }");
        expectThrow<ConfigError>([&] { Schemes::fromFile(bad.string()); }, "non-object section → ConfigError");
        writeFile(bad, "{ \"advection\": { \"default\": \"Iso9\" } }");
        expectThrow<ConfigError>([&] { Schemes::fromFile(bad.string()); }, "name from another section → ConfigError");
        expectThrow<ValidationError>([&] { Schemes::builtin({{"laplacian", "UW1"}}); }, "bad solver default → ValidationError");
        expectThrow<ValidationError>([&] { Schemes::builtin({{"lapl", "CD2"}}); }, "solver default unknown section → ValidationError");
        expectThrow<IOError>([&] { Schemes::fromFile((dir / "nope.jsonc").string()); }, "explicit missing file → IOError");
        expectThrow<ValidationError>([&] { Schemes::builtin().resolve("nope", ""); }, "resolve unknown section → ValidationError");
    }

    // ---- 5. warnUnused ---------------------------------------------------
    std::cout << "[5] warnUnused\n";
    {
        Schemes s = Schemes::fromFile(schemes.string());
        s.lap("psi");
        s.gradSq("psi");
        const std::string err = capture(std::cerr, [&] { s.warnUnused(); });
        expect(err.find("lap(Typo)") != std::string::npos, "unused entry lap(Typo) reported");
        expect(err.find("lap(psi)") == std::string::npos && err.find("gradSq(psi)") == std::string::npos,
              "used entries not reported");
        expect(err.find("[PhiX] WARNING") != std::string::npos, "goes through the WARNING channel");
        Schemes clean = Schemes::builtin({{"laplacian", "Iso9"}});
        expect(capture(std::cerr, [&] { clean.warnUnused(); }).empty(), "builtin: nothing to report");
    }

    // ---- 6. operator overloads are bitwise the explicit calls ------------
    std::cout << "[6] operator overloads (GPU, bitwise)\n";
    {
        const int N = 32;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, 0.25, 0.0, N, 0.25, 0.0);
        ScalarField psi(mesh, "psi", 3);
        VectorField u(mesh, "u", 2, 3);
        for (int j = -3; j < N + 3; ++j)
        for (int i = -3; i < N + 3; ++i) {
            const double x = mesh.coord(0, i), y = mesh.coord(1, j);
            psi.curr [static_cast<std::size_t>(psi.index(i, j))]  = std::sin(1.3 * x) + 0.7 * std::cos(2.1 * y) + 0.1 * x * y;
            u[0].curr[static_cast<std::size_t>(u[0].index(i, j))] = (x < 3.0) ? 1.5 : -2.0;
            u[1].curr[static_cast<std::size_t>(u[1].index(i, j))] = -0.8 + 0.05 * y;
        }
        psi.allocDevice();  psi.uploadAllToDevice();
        u[0].allocDevice(); u[0].uploadAllToDevice();
        u[1].allocDevice(); u[1].uploadAllToDevice();

        const fs::path ops = dir / "ops.jsonc";
        writeFile(ops, R"JSON({
            "laplacian": { "default": "CD2",  "lap(psi)": "Iso9" },
            "gradient" : { "default": "CD4" },
            "gradSq"   : { "default": "Iso9" },
            "advection": { "default": "UW1",  "adv(psi)": "WENO5" }
        })JSON");
        Schemes s = Schemes::fromFile(ops.string());
        expect(rhsGPU(psi, lap(psi, s, 0.7))     == rhsGPU(psi, lap<scheme::Iso9>(psi, 0.7)),   "lap(psi, sch)     == lap<Iso9>");
        expect(rhsGPU(psi, grad(psi, 1, s, 2.0)) == rhsGPU(psi, grad<scheme::CD4>(psi, 1, 2.0)), "grad(psi, 1, sch) == grad<CD4>");
        expect(rhsGPU(psi, gradSq(psi, s))       == rhsGPU(psi, gradSq<scheme::Iso9>(psi)),      "gradSq(psi, sch)  == gradSq<Iso9>");
        expect(rhsGPU(psi, adv(u, psi, s, -1.0)) == rhsGPU(psi, adv(u, psi, "WENO5", -1.0)),   "adv(u, psi, sch)  == adv WENO5");
        expect(rhsGPU(psi, lap(psi, s))          != rhsGPU(psi, lap(psi)),                        "…and differs from the CD2 default (test is live)");
        ScalarField other(mesh, "other", 3);
        other.allocDevice(); other.uploadAllToDevice();
        expect(s.lap(other.name) == "CD2" && s.adv(other.name) == "UW1", "unlisted field gets section defaults");
    }

    // ---- 7. table-drift guard + Iso9 square-mesh guard -------------------
    std::cout << "[7] every admitted name reaches its operator\n";
    {
        const int N = 8;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, 0.5, 0.0, N, 0.5, 0.0);
        ScalarField f(mesh, "f", 3);
        VectorField u(mesh, "u", 2, 3);
        bool ok = true;
        try {
            for (const char* n : {"CD2", "CD4", "CD6", "Iso9", "Iso27"}) lap(f, std::string(n));
            for (const char* n : {"CD2", "CD4", "CD6", "Iso9"})          grad(f, 0, std::string(n));
            for (const char* n : {"CD2", "Iso9"})                        gradSq(f, std::string(n));
            for (const char* n : {"UW1", "UW2", "WENO5"})                adv(u, f, std::string(n));
            for (const char* n : {"CD2", "Iso9"})                        anisoSchemeFromString(n);
        } catch (const std::exception& e) { ok = false; std::cout << "    " << e.what() << "\n"; }
        expect(ok, "lap/grad/gradSq/adv/anisoDiv accept every name Schemes admits");

        Mesh rect = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, 0.5, 0.0, N, 0.25, 0.0);
        ScalarField g(rect, "g", 1);
        expectThrow<ValidationError>([&] { lap(g, "Iso9"); }, "lap Iso9 on dx != dy → ValidationError");
        Schemes s = Schemes::builtin({{"laplacian", "Iso9"}});
        expectThrow<ValidationError>([&] { lap(g, s); }, "…also through the Schemes overload");
    }

    fs::remove_all(dir);
    std::cout << "\nmodule_schemes: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
