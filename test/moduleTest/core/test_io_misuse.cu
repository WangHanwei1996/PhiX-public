// ---------------------------------------------------------------------------
// module_io_misuse — IO-layer error feedback: ConfigView checked nested
// access, fromArgs without exit(), OutputWriter interval guards,
// restart/init validation, BCFactory axis gating, stale-host-write warning.
// ---------------------------------------------------------------------------

#include "core/Error.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"
#include "boundary/BCFactory.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace PhiX;
using PhiX::IO::ConfigFile;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s\n", msg); } \
    } while (0)

template <class E, class Fn>
static bool throwsE(Fn fn) {
    try { fn(); }
    catch (const E&) { return true; }
    catch (...)      { return false; }
    return false;
}

template <class Fn>
static bool noThrow(Fn fn) {
    try { fn(); }
    catch (...) { return false; }
    return true;
}

template <typename Fn>
static std::string captureCerr(Fn fn) {
    std::ostringstream oss;
    std::streambuf* old = std::cerr.rdbuf(oss.rdbuf());
    fn();
    std::cerr.rdbuf(old);
    return oss.str();
}

static const char* kCfgPath = "io_misuse_tmp.jsonc";

static void writeTestConfig() {
    std::ofstream f(kCfgPath);
    f << "{\n"
         "  // JSONC comment must be stripped\n"
         "  \"mesh\": { \"nx\": 16, \"dx\": 0.5, \"label\": \"grid\" },\n"
         "  \"output\": { \"print_interval\": 10, \"write_interval\": 20,\n"
         "                \"format\": \"BINARY\" }\n"
         "}\n";
}

static void test1_config_view() {
    printf("[1] ConfigView checked nested access\n");
    ConfigFile cfg(kCfgPath);

    const int    nx = cfg["mesh"]["nx"];
    const double dx = cfg["mesh"]["dx"];
    const std::string label = cfg["mesh"]["label"];
    CHECK(nx == 16 && dx == 0.5 && label == "grid",
          "valid nested reads convert to int/double/string");

    CHECK(throwsE<ConfigError>([&] { double v = cfg["mesh"]["nxx"]; (void)v; }),
          "missing NESTED key throws ConfigError (was nlohmann UB)");
    CHECK(throwsE<ConfigError>([&] { cfg["missing_section"]; }),
          "missing top-level key throws ConfigError");

    bool hintListsKeys = false;
    try { cfg["mesh"]["nxx"]; }
    catch (const ConfigError& e) {
        hintListsKeys = e.hint().find("nx") != std::string::npos
                     && e.where().find("[\"mesh\"][\"nxx\"]") != std::string::npos;
    }
    CHECK(hintListsKeys, "error names the full path and lists existing keys");

    CHECK(throwsE<ConfigError>([&] { double v = cfg["mesh"]["label"]; (void)v; }),
          "type mismatch (string -> double) throws ConfigError");

    CHECK(cfg["mesh"].contains("nx") && !cfg["mesh"].contains("nq"),
          "contains() forwards");
    CHECK(cfg["mesh"].value("nq", 7) == 7 && cfg["mesh"].value("nx", 7) == 16,
          "value(key, default) forwards with checking");

    CHECK(noThrow([&] { PhiX::IO::OutputWriter w(cfg["output"]); }),
          "cfg[section] still passes to APIs taking const json&");
}

static void test2_from_args_throws() {
    printf("[2] fromArgs throws instead of exit(1)\n");
    char prog[] = "testapp";
    char path[] = "definitely_missing_config.jsonc";
    char* argv[] = { prog, path };

    bool hasUsage = false;
    try { ConfigFile::fromArgs(2, argv); }
    catch (const PhixError& e) {
        hasUsage = std::string(e.what()).find("usage") != std::string::npos
                || !e.hint().empty();
    }
    CHECK(hasUsage, "missing file throws PhixError (process survives, usage in hint)");
}

static void test3_output_writer_intervals() {
    printf("[3] OutputWriter interval guards\n");
    nlohmann::json bad = { {"print_interval", 0}, {"write_interval", 20},
                           {"format", "BINARY"} };
    CHECK(throwsE<ConfigError>([&] { PhiX::IO::OutputWriter w(bad); }),
          "print_interval = 0 throws ConfigError (was SIGFPE at step 0)");

    nlohmann::json neg = { {"print_interval", 10}, {"write_interval", -5},
                           {"format", "BINARY"} };
    CHECK(throwsE<ConfigError>([&] { PhiX::IO::OutputWriter w(neg); }),
          "negative write_interval throws ConfigError");
}

static void test4_restart_and_init() {
    printf("[4] restart / named-init validation\n");
    CHECK(PhiX::IO::resolveStartStep("initial_field", "phi") == 0,
          "initial_field resolves to step 0");
    CHECK(throwsE<ConfigError>([&] { PhiX::IO::resolveStartStep("abc", "phi"); }),
          "non-numeric start_from throws ConfigError (was raw stoi throw)");

    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);
    ScalarField f(m, "f", 1);
    CHECK(noThrow([&] { PhiX::IO::initField(f, 0, "linear:x:0.0:1.0"); }),
          "linear:x named init works");
    CHECK(throwsE<ConfigError>([&] { PhiX::IO::initField(f, 0, "linear:q:0:1"); }),
          "unknown linear axis throws ConfigError (was silently z)");
    CHECK(throwsE<ConfigError>([&] { PhiX::IO::initField(f, 0, "linear:z:0:1"); }),
          "inactive axis z on a 2D mesh throws ConfigError");
}

static void test5_bcfactory_axis_gating() {
    printf("[5] BCFactory both-or-neither axis keys\n");
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);

    nlohmann::json full = { {"x_min", "NoFlux"}, {"x_max", "NoFlux"},
                            {"y_min", "NoFlux"}, {"y_max", "NoFlux"} };
    CHECK(noThrow([&] { buildBCs(m, full); }), "full x+y config accepted");

    nlohmann::json oneSided = { {"x_min", "NoFlux"}, {"x_max", "NoFlux"},
                                {"y_min", "NoFlux"} };
    CHECK(throwsE<ConfigError>([&] { buildBCs(m, oneSided); }),
          "y_min without y_max throws ConfigError (was silent axis skip)");

    nlohmann::json xOnly = { {"x_min", "NoFlux"}, {"x_max", "NoFlux"} };
    std::string out = captureCerr([&] { buildBCs(m, xOnly); });
    CHECK(out.find("WARNING") != std::string::npos
              && out.find("y_min") != std::string::npos,
          "2D mesh with no Y BCs warns once (hand-written kernels stay legal)");
}

static void test6_stale_host_write() {
    printf("[6] stale host copy warning on write()\n");
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN, 8, 0.1, 0.0, 8, 0.1, 0.0);

    ScalarField stale(m, "stale_f", 1);
    stale.fill(1.0);
    stale.allocDevice();
    stale.uploadAllToDevice();
    stale.trackPrev = false;
    stale.advanceTimeLevelGPU();   // marks the device copy as advanced
    std::string out = captureCerr([&] { stale.write("io_misuse_stale.field"); });
    CHECK(out.find("downloadCurrFromDevice") != std::string::npos,
          "write() after GPU advance without download warns");

    ScalarField clean(m, "clean_f", 1);
    clean.fill(1.0);
    clean.allocDevice();
    clean.uploadAllToDevice();
    clean.advanceTimeLevelGPU();
    clean.downloadCurrFromDevice();
    out = captureCerr([&] { clean.write("io_misuse_clean.field"); });
    CHECK(out.empty(), "write() after download stays silent");
}

int main() {
    printf("=== module_io_misuse: IO-layer error feedback ===\n");
    writeTestConfig();
    test1_config_view();
    test2_from_args_throws();
    test3_output_writer_intervals();
    test4_restart_and_init();
    test5_bcfactory_axis_gating();
    test6_stale_host_write();
    std::remove(kCfgPath);
    std::remove("io_misuse_stale.field");
    std::remove("io_misuse_clean.field");
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
