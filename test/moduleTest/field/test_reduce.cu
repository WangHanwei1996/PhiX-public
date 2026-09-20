// ---------------------------------------------------------------------------
// module_reduce — device-side reductions over physical cells (Reduce.h)
//
// Checks against CPU references on 1D/2D/3D fields whose ghost cells are
// poisoned with huge values: any halo leakage flips max/min/sum immediately.
// Also verifies that fieldHasNonFinite ignores ghost NaN but catches physical
// NaN and Inf.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/Reduce.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static void requireClose(double a, double b, double relTol,
                         const std::string& msg) {
    const double scale = std::max({std::fabs(a), std::fabs(b), 1e-300});
    if (std::fabs(a - b) > relTol * scale)
        throw std::runtime_error(msg + "  (" + std::to_string(a)
                                 + " vs " + std::to_string(b) + ")");
}

// Fill ALL stored cells (incl. ghost) with `poison`, then set physical cells
// from fn(i,j,k) and upload.  Returns CPU references {max,min,sum,sumsq}.
template<typename Fn>
static void setupField(ScalarField& f, double poison, Fn fn,
                       double& refMax, double& refMin,
                       double& refSum, double& refSumSq) {
    f.fillCurr(poison);
    refMax = -std::numeric_limits<double>::infinity();
    refMin =  std::numeric_limits<double>::infinity();
    refSum = 0.0;
    refSumSq = 0.0;
    for (int k = 0; k < f.mesh.n[2]; ++k)
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i) {
        const double v = fn(i, j, k);
        f.curr[static_cast<std::size_t>(f.index(i, j, k))] = v;
        refMax = std::max(refMax, v);
        refMin = std::min(refMin, v);
        refSum += v;
        refSumSq += v * v;
    }
    if (!f.deviceAllocated()) f.allocDevice();
    f.uploadCurrToDevice();
}

static void checkField(ScalarField& f, const std::string& tag) {
    double refMax, refMin, refSum, refSumSq;
    setupField(f, 1e300,
               [](int i, int j, int k) {
                   return std::sin(0.37 * i + 0.61 * j + 0.23 * k)
                        + 0.5 * std::cos(1.3 * j - 0.7 * i);
               },
               refMax, refMin, refSum, refSumSq);

    // max/min/maxAbs must match the CPU reference exactly (same doubles,
    // order-independent operations).
    require(reduce::fieldMax(f) == refMax, tag + ": fieldMax mismatch");
    require(reduce::fieldMin(f) == refMin, tag + ": fieldMin mismatch");
    require(reduce::fieldMaxAbs(f) == std::max(std::fabs(refMax),
                                               std::fabs(refMin)),
            tag + ": fieldMaxAbs mismatch");

    // Sums differ only by association order.
    requireClose(reduce::fieldSum(f),   refSum,   1e-12, tag + ": fieldSum");
    requireClose(reduce::fieldSumSq(f), refSumSq, 1e-12, tag + ": fieldSumSq");
    requireClose(reduce::fieldL2(f), std::sqrt(refSumSq), 1e-12,
                 tag + ": fieldL2");

    // Finite everywhere physically (ghost poison is finite).
    require(!reduce::fieldHasNonFinite(f), tag + ": false NonFinite positive");

    // NaN in a ghost cell must be ignored...
    f.curr[static_cast<std::size_t>(f.index(-1, 0, 0))] =
        std::numeric_limits<double>::quiet_NaN();
    f.uploadCurrToDevice();
    require(!reduce::fieldHasNonFinite(f),
            tag + ": ghost NaN leaked into NonFinite");

    // ...but NaN / Inf in a physical cell must be caught.
    const int mid = f.index(f.mesh.n[0] / 2, f.mesh.n[1] / 2, f.mesh.n[2] / 2);
    const double saved = f.curr[static_cast<std::size_t>(mid)];
    f.curr[static_cast<std::size_t>(mid)] =
        std::numeric_limits<double>::quiet_NaN();
    f.uploadCurrToDevice();
    require(reduce::fieldHasNonFinite(f), tag + ": physical NaN missed");

    f.curr[static_cast<std::size_t>(mid)] =
        std::numeric_limits<double>::infinity();
    f.uploadCurrToDevice();
    require(reduce::fieldHasNonFinite(f), tag + ": physical Inf missed");

    f.curr[static_cast<std::size_t>(mid)] = saved;
}

int main() {
    // Odd, non-power-of-two sizes to exercise the index arithmetic.
    Mesh m1 = Mesh::makeUniform1D(CoordSys::CARTESIAN, 17, 0.1);
    Mesh m2 = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                  37, 0.1, 0.0, 23, 0.2, -1.0);
    Mesh m3 = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                  9, 0.1, 0.0, 7, 0.2, 0.0, 5, 0.3, 0.0);

    ScalarField f1(m1, "f1", 2);
    ScalarField f2(m2, "f2", 2);
    ScalarField f3(m3, "f3", 1);

    checkField(f1, "1D");
    checkField(f2, "2D");
    checkField(f3, "3D");

    // Calling on a field without device allocation must throw.
    ScalarField g(m1, "g", 1);
    bool threw = false;
    try { reduce::fieldMax(g); } catch (const std::runtime_error&) { threw = true; }
    require(threw, "missing-device-allocation did not throw");

    reduce::freeScratch();
    return 0;
}
