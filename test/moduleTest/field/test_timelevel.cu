// ---------------------------------------------------------------------------
// module_timelevel — opt-in prev-time-level tracking (v2.18.0)
//
// Default (trackPrev == false): allocDevice() must not allocate d_prev,
// advanceTimeLevelGPU/CPU are no-ops, prev accessors throw, and
// uploadAll/downloadAll skip the prev buffer.
//
// Opt-in (trackPrev == true): the pre-v2.18.0 semantics hold exactly —
// after a rotation prev equals curr on both host and device.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, 16, 0.1);

    // ---- default: prev untracked -------------------------------------------
    {
        ScalarField f(mesh, "f", 1);
        f.initialize([](double x, double, double) { return 1.0 + x; });
        f.fillPrev(-7.0);
        f.allocDevice();
        require(f.d_prev == nullptr, "d_prev allocated despite trackPrev=false");

        f.uploadAllToDevice();          // must not touch (missing) prev
        f.advanceTimeLevelGPU();        // must be a no-op, not a throw
        f.advanceTimeLevelCPU();
        require(f.prev[static_cast<std::size_t>(f.index(0))] == -7.0,
                "CPU rotation ran despite trackPrev=false");

        f.downloadAllFromDevice();      // must skip prev

        bool threw = false;
        try { f.uploadPrevToDevice(); } catch (const std::runtime_error&) { threw = true; }
        require(threw, "uploadPrevToDevice did not throw without trackPrev");

        threw = false;
        try { f.downloadPrevFromDevice(); } catch (const std::runtime_error&) { threw = true; }
        require(threw, "downloadPrevFromDevice did not throw without trackPrev");
    }

    // ---- opt-in: pre-v2.18.0 semantics --------------------------------------
    {
        ScalarField f(mesh, "g", 1);
        f.trackPrev = true;
        f.initialize([](double x, double, double) { return 2.0 * x; });
        f.fillPrev(0.0);
        f.allocDevice();
        require(f.d_prev != nullptr, "d_prev missing despite trackPrev=true");

        f.uploadAllToDevice();
        f.advanceTimeLevelGPU();        // d_prev <- d_curr
        f.fillPrev(-1.0);               // poison host prev, then fetch device
        f.downloadPrevFromDevice();
        for (int i = 0; i < mesh.n[0]; ++i) {
            const std::size_t idx = static_cast<std::size_t>(f.index(i));
            require(std::fabs(f.prev[idx] - f.curr[idx]) < 1e-15,
                    "GPU rotation did not copy curr to prev");
        }

        f.curr[0] = 123.0;
        f.advanceTimeLevelCPU();
        require(f.prev[0] == 123.0, "CPU rotation did not copy curr to prev");
    }

    return 0;
}
