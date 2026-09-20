// ---------------------------------------------------------------------------
// module_error — PhixError hierarchy, what() formatting, runGuarded,
// warn/warnOnce.  Host-only.
// ---------------------------------------------------------------------------

#include "core/Error.h"
#include "core/RunGuard.h"
#include "core/Version.h"

#include <cstdio>
#include <sstream>
#include <string>

using namespace PhiX;

static int pass_count = 0, fail_count = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { ++pass_count; printf("  PASS: %s\n", msg); } \
        else      { ++fail_count; printf("  FAIL: %s\n", msg); } \
    } while (0)

static bool contains(const std::string& s, const std::string& sub) {
    return s.find(sub) != std::string::npos;
}

// Capture everything written to std::cerr while fn() runs.
template <typename Fn>
static std::string captureCerr(Fn fn) {
    std::ostringstream oss;
    std::streambuf* old = std::cerr.rdbuf(oss.rdbuf());
    fn();
    std::cerr.rdbuf(old);
    return oss.str();
}

// Capture everything written to std::cout while fn() runs (runGuarded
// prints the build stamp there).
template <typename Fn>
static std::string captureCout(Fn fn) {
    std::ostringstream oss;
    std::streambuf* old = std::cout.rdbuf(oss.rdbuf());
    fn();
    std::cout.rdbuf(old);
    return oss.str();
}

static void test1_what_format() {
    printf("[1] what() formatting\n");
    ValidationError e("Equation::setRHS",
                      "term requires ghost >= 2 but field 'phi' has ghost = 1",
                      "construct the field with ghost=2");
    const std::string w = e.what();
    CHECK(contains(w, "PhixError[Validation] Equation::setRHS: term requires"),
          "what() = PhixError[category] where: message");
    CHECK(contains(w, "\n  hint: construct the field with ghost=2"),
          "hint appears on its own indented line");
    CHECK(e.category() == "Validation" && e.where() == "Equation::setRHS"
              && contains(e.message(), "ghost >= 2")
              && e.hint() == "construct the field with ghost=2",
          "structured accessors return the four parts");

    ConfigError noHint("ConfigFile", "key not found");
    CHECK(!contains(noHint.what(), "hint:"), "empty hint emits no hint line");
}

static void test2_subclasses() {
    printf("[2] subclass categories\n");
    CHECK(ConfigError("w", "m").category() == "Config",     "ConfigError -> Config");
    CHECK(DeviceError("w", "m").category() == "Device",     "DeviceError -> Device");
    CHECK(ValidationError("w", "m").category() == "Validation",
          "ValidationError -> Validation");
    CHECK(NumericsError("w", "m").category() == "Numerics", "NumericsError -> Numerics");
    CHECK(IOError("w", "m").category() == "IO",             "IOError -> IO");
}

static void test3_catchable_as_base() {
    printf("[3] catch compatibility\n");
    bool asPhix = false, asRuntime = false, asStd = false;
    try { throw NumericsError("Solver::advance", "NaN"); }
    catch (const PhixError& e) { asPhix = (e.category() == "Numerics"); }
    try { throw ConfigError("cfg", "missing"); }
    catch (const std::runtime_error&) { asRuntime = true; }
    try { throw IOError("FieldIO", "open failed"); }
    catch (const std::exception&) { asStd = true; }
    CHECK(asPhix,    "subclass caught as PhixError with category intact");
    CHECK(asRuntime, "PhixError caught as std::runtime_error");
    CHECK(asStd,     "PhixError caught as std::exception");
}

static void test4_run_guarded() {
    printf("[4] runGuarded\n");
    int rcOk = -1;
    std::string stamp;
    std::string out = captureCerr([&] {
        stamp = captureCout([&] {
            rcOk = runGuarded([](int a, int b) { return a - b; }, 7, 7);
        });
    });
    CHECK(rcOk == 0 && out.empty(),
          "success passes return value through, no cerr output");
    CHECK(contains(stamp, versionString()) && contains(stamp, "git "),
          "startup prints the build stamp (version + git SHA) to stdout");

    int rcPhix = -1;
    out = captureCerr([&] {
        captureCout([&] {
            rcPhix = runGuarded([]() -> int {
                throw ValidationError("Mesh::makeUniform2D", "nx must be > 0",
                                      "check the config");
            });
        });
    });
    CHECK(rcPhix == 1, "PhixError -> exit code 1");
    CHECK(contains(out, "[PhiX] FATAL PhixError[Validation] Mesh::makeUniform2D")
              && contains(out, "hint: check the config"),
          "PhixError printed structured with hint");

    int rcStd = -1;
    out = captureCerr([&] {
        captureCout([&] {
            rcStd = runGuarded([]() -> int { throw std::logic_error("plain"); });
        });
    });
    CHECK(rcStd == 1 && contains(out, "FATAL unhandled std::exception: plain"),
          "std::exception -> exit code 1 with message");

    int rcUnknown = -1;
    out = captureCerr([&] {
        captureCout([&] {
            rcUnknown = runGuarded([]() -> int { throw 42; });
        });
    });
    CHECK(rcUnknown == 1 && contains(out, "FATAL unknown exception"),
          "unknown exception -> exit code 1");
}

static void test5_warn_once() {
    printf("[5] warn / warnOnce\n");
    std::string out = captureCerr([] { warn("plain warning"); });
    CHECK(contains(out, "[PhiX] WARNING: plain warning"), "warn prefixes [PhiX] WARNING");

    out = captureCerr([] {
        warnOnce("k1", "first");
        warnOnce("k1", "first");
        warnOnce("k1", "first");
        warnOnce("k2", "second");
    });
    std::size_t n = 0;
    for (std::size_t p = out.find("WARNING"); p != std::string::npos;
         p = out.find("WARNING", p + 1))
        ++n;
    CHECK(n == 2 && contains(out, "first") && contains(out, "second"),
          "warnOnce dedupes per key (2 warnings for 4 calls, 2 keys)");
}

int main() {
    printf("=== module_error: PhixError hierarchy + runGuarded ===\n");
    test1_what_format();
    test2_subclasses();
    test3_catchable_as_base();
    test4_run_guarded();
    test5_warn_once();
    printf("Results: %d passed, %d failed\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
