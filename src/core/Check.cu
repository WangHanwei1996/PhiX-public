#include "core/Check.h"
#include "field/Reduce.h"

#include <cmath>
#include <string>

namespace PhiX {
namespace check {

namespace {

[[noreturn]] void throwNonFinite(const ScalarField& f, int step, double time) {
    throw NumericsError("check::checkFieldHealth",
        "NaN/Inf detected in field '" + f.name + "' at step "
        + std::to_string(step) + " (t=" + std::to_string(time) + ")",
        "the run has diverged — reduce dt, verify BCs and the initial "
        "condition, or enable adaptive dt; the sentinel cadence is "
        "configurable (0 disables)");
}

[[noreturn]] void throwBlowUp(const ScalarField& f, double maxAbs,
                              double maxAbsLimit, int step, double time) {
    throw NumericsError("check::checkFieldHealth",
        "field '" + f.name + "' exceeds maxAbsLimit "
        + std::to_string(maxAbsLimit) + " (|max| = " + std::to_string(maxAbs)
        + ") at step " + std::to_string(step)
        + " (t=" + std::to_string(time) + ")",
        "the solution is blowing up — reduce dt or check the model "
        "parameters; raise maxAbsLimit if large values are expected");
}

// Host scan over physical cells: returns (hasNonFinite, maxAbs).
void hostScan(const ScalarField& f, bool& nonFinite, double& maxAbs) {
    nonFinite = false;
    maxAbs    = 0.0;
    const int nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const double v =
            f.curr[static_cast<std::size_t>(f.index(i, j, k))];
        if (!std::isfinite(v)) { nonFinite = true; return; }
        maxAbs = std::max(maxAbs, std::fabs(v));
    }
}

} // namespace

void checkFinite(const ScalarField& f, const char* who) {
    checkOnDevice(f, who);
    if (reduce::fieldHasNonFinite(f))
        throw NumericsError(who,
            "field '" + f.name + "' contains NaN or Inf",
            "reduce dt, verify BCs and the initial condition; inspect with "
            "reduce::fieldMax/fieldMin to locate the blow-up");
}

void checkFieldHealth(const ScalarField& f, int step, double time,
                      double maxAbsLimit) {
    if (f.deviceAllocated()) {
        if (reduce::fieldHasNonFinite(f))
            throwNonFinite(f, step, time);
        if (maxAbsLimit > 0.0) {
            const double m = reduce::fieldMaxAbs(f);
            if (m > maxAbsLimit)
                throwBlowUp(f, m, maxAbsLimit, step, time);
        }
    } else {
        checkFieldHealthCPU(f, step, time, maxAbsLimit);
    }
}

void checkFieldHealthCPU(const ScalarField& f, int step, double time,
                         double maxAbsLimit) {
    bool nonFinite;
    double maxAbs;
    hostScan(f, nonFinite, maxAbs);
    if (nonFinite)
        throwNonFinite(f, step, time);
    if (maxAbsLimit > 0.0 && maxAbs > maxAbsLimit)
        throwBlowUp(f, maxAbs, maxAbsLimit, step, time);
}

} // namespace check
} // namespace PhiX
