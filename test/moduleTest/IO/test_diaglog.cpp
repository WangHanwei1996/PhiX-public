// ---------------------------------------------------------------------------
// module_io_diaglog — scalar time-series CSV logger (IO/DiagnosticLog.h).
//
// 1. Header is ON DISK immediately after construction (the 0-byte-file
//    trap: a header buffered until the first data row makes a healthy run
//    look dead).
// 2. Rows are flushed per call and parse back to the written values.
// 3. Value-count mismatch vs the declared columns throws.
// 4. due(): cadence gating; every = 0 disables.
// 5. append = true continues an existing file without duplicating the
//    header; on a missing/empty file it writes the header normally.
// ---------------------------------------------------------------------------

#include "IO/DiagnosticLog.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace PhiX;
namespace fs = std::filesystem;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static std::string slurp(const std::string& path) {
    std::ifstream in(path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main() {
try {
    const std::string dir = "diaglog_test_out";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string path = dir + "/run_log.csv";

    // === 1. header flushed at construction ================================
    {
        IO::DiagnosticLog log(path, {"step", "time_s", "front_W"});
        require(slurp(path) == "step,time_s,front_W\n",
                "header on disk immediately after construction");

        // === 2. rows flushed per call ======================================
        log.row({100.0, 0.5, 12.25});
        require(slurp(path).find("100,0.5,12.25") != std::string::npos,
                "row visible on disk right after row()");
        log.row({200.0, 1.0, 13.5});
        {
            std::istringstream is(slurp(path));
            std::string line;
            int n = 0;
            while (std::getline(is, line)) ++n;
            require(n == 3, "header + 2 rows");
        }

        // === 3. column-count mismatch throws ===============================
        bool threw = false;
        try { log.row({1.0, 2.0}); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "wrong value count throws");

        // === 4. cadence ====================================================
        require(log.due(0) && log.due(1) && log.due(7),
                "every = 1 default: every step due");
    }
    {
        IO::DiagnosticLog log(dir + "/cad.csv", {"a"}, 50);
        require(log.due(0) && log.due(100) && !log.due(75),
                "every = 50 gates correctly");
        IO::DiagnosticLog off(dir + "/off.csv", {"a"}, 0);
        require(!off.due(0) && !off.due(100), "every = 0 disables");
    }

    // === 5. append mode =====================================================
    {
        IO::DiagnosticLog log(path, {"step", "time_s", "front_W"}, 1, true);
        log.row({300.0, 1.5, 14.75});
    }
    {
        const std::string blob = slurp(path);
        std::istringstream is(blob);
        std::string line;
        int n = 0, headers = 0;
        while (std::getline(is, line)) {
            ++n;
            if (line == "step,time_s,front_W") ++headers;
        }
        require(headers == 1, "append: no duplicate header");
        require(n == 4 && blob.find("300,1.5,14.75") != std::string::npos,
                "append: earlier rows kept, new row added");
    }
    {
        IO::DiagnosticLog log(dir + "/fresh.csv", {"x"}, 1, true);
        require(slurp(dir + "/fresh.csv") == "x\n",
                "append on missing file still writes the header");
    }

    fs::remove_all(dir);
    std::printf("module_io_diaglog: all passed\n");
    return 0;
} catch (const std::exception& ex) {
    std::printf("FAIL: %s\n", ex.what());
    return 1;
}
}
