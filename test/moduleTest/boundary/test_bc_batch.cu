// ---------------------------------------------------------------------------
// module_bc_batch — batched ghost refresh (boundary/BCBatch.h)
//
// 1. Bitwise equivalence: a mixed BC set (periodic X + no-flux Y-low +
//    fixed Y-high, ghost = 2) applied via BCBatch must produce exactly the
//    same stored array as the sequential per-BC applyOnGPU calls.
// 2. Fallback: an unknown BoundaryCondition subclass goes to the fallback
//    list and the combined result still matches the sequential reference.
// 3. Pointer-swap safety: rebinding f.d_curr (the RK4 trick) between
//    build() and applyOnGPU() must affect the swapped buffer.
// ---------------------------------------------------------------------------

#include "boundary/BCBatch.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"
#include "field/ScalarField.h"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// A BC subclass BCBatch does not know — must land on the fallback path.
// (Fills the X-low ghost layers with a marker value.)
class MarkerBC : public BoundaryCondition {
public:
    explicit MarkerBC(const Patch& p) : BoundaryCondition(p) {}
    void applyOnCPU(ScalarField& f) const override {
        for (int j = 0; j < f.mesh.n[1]; ++j)
            for (int g = 1; g <= f.ghost; ++g)
                f.curr[static_cast<std::size_t>(f.index(-g, j))] = 42.5;
    }
    void applyOnGPU(ScalarField& f) const override {
        f.downloadCurrFromDevice();
        applyOnCPU(f);
        f.uploadCurrToDevice();
    }
};

static void fillField(ScalarField& f) {
    const int g = f.ghost;
    for (int j = -g; j < f.mesh.n[1] + g; ++j)
    for (int i = -g; i < f.mesh.n[0] + g; ++i)
        f.curr[static_cast<std::size_t>(f.index(i, j))] =
            std::sin(0.7 * i) + std::cos(1.3 * j) + 0.01 * i * j;
    f.uploadAllToDevice();
}

// Bitwise comparison over all stored cells EXCEPT the corner-ghost regions
// (cells outside the physical range in >= 2 axes): since v2.24.0 the batch
// fills corners (second kernel pass) while the sequential per-BC path never
// did — corners are checked separately for self-consistency.
static double maxDiffNonCorner(ScalarField& a, ScalarField& b) {
    a.downloadCurrFromDevice();
    b.downloadCurrFromDevice();
    const int g = a.ghost, nx = a.mesh.n[0], ny = a.mesh.n[1];
    double m = 0.0;
    for (int j = -g; j < ny + g; ++j)
    for (int i = -g; i < nx + g; ++i) {
        const bool gi = (i < 0 || i >= nx), gj = (j < 0 || j >= ny);
        if (gi && gj) continue;   // corner region
        const std::size_t idx = static_cast<std::size_t>(a.index(i, j));
        m = std::max(m, std::fabs(
            static_cast<double>(a.curr[idx])
            - static_cast<double>(b.curr[idx])));
    }
    return m;
}

int main() {
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    23, 0.1, 0.0, 17, 0.2, 0.0);

    PeriodicBC bcX(mesh.facePatch(Axis::X, Side::LOW));
    NoFluxBC   bcYlo(mesh.facePatch(Axis::Y, Side::LOW));
    FixedBC    bcYhi(mesh.facePatch(Axis::Y, Side::HIGH), 3.75);

    // ---- 1. equivalence vs sequential application ---------------------------
    {
        ScalarField fSeq(mesh, "seq", 2), fBat(mesh, "bat", 2);
        fSeq.allocDevice(); fBat.allocDevice();
        fillField(fSeq); fillField(fBat);

        bcX.applyOnGPU(fSeq);
        bcYlo.applyOnGPU(fSeq);
        bcYhi.applyOnGPU(fSeq);

        BCBatch batch;
        batch.build(fBat, {&bcX, &bcYlo, &bcYhi});
        require(batch.batchedCount() == 3 && batch.fallbackCount() == 0,
                "built-in BCs not fully batched");
        batch.applyOnGPU(fBat);

        require(maxDiffNonCorner(fSeq, fBat) == 0.0,
                "batched ghost refresh differs from sequential");

        // Corner self-consistency: the X-periodic corner pass must make
        // every corner-ghost cell the periodic image (±N in x) of the
        // pass-0-filled y-ghost band.
        fBat.downloadCurrFromDevice();
        const int nxp = mesh.n[0], nyp = mesh.n[1];
        for (int j : {-2, -1, nyp, nyp + 1})
        for (int i : {-2, -1, nxp, nxp + 1}) {
            const int ii = (i < 0) ? i + nxp : i - nxp;   // periodic image
            const double got = fBat.curr[static_cast<std::size_t>(
                fBat.index(i, j))];
            const double exp = fBat.curr[static_cast<std::size_t>(
                fBat.index(ii, j))];
            require(got == exp, "corner ghost is not the periodic image at ("
                                + std::to_string(i) + "," + std::to_string(j)
                                + ")");
        }
    }

    // ---- 2. unknown subclass goes to fallback -------------------------------
    {
        MarkerBC marker(mesh.facePatch(Axis::X, Side::LOW));

        ScalarField fSeq(mesh, "seq2", 2), fBat(mesh, "bat2", 2);
        fSeq.allocDevice(); fBat.allocDevice();
        fillField(fSeq); fillField(fBat);

        bcYlo.applyOnGPU(fSeq);
        bcYhi.applyOnGPU(fSeq);
        marker.applyOnGPU(fSeq);

        BCBatch batch;
        batch.build(fBat, {&bcYlo, &bcYhi, &marker});
        require(batch.batchedCount() == 2 && batch.fallbackCount() == 1,
                "unknown BC subclass not routed to fallback");
        batch.applyOnGPU(fBat);

        require(maxDiffNonCorner(fSeq, fBat) == 0.0,
                "batched + fallback result differs from sequential");
    }

    // ---- 3. pointer-swap safety (RK4 pattern) -------------------------------
    {
        ScalarField f(mesh, "f", 2), tmp(mesh, "tmp", 2);
        f.allocDevice(); tmp.allocDevice();
        fillField(f); fillField(tmp);

        BCBatch batch;
        batch.build(f, {&bcX, &bcYlo, &bcYhi});

        // reference: apply sequentially to tmp
        ScalarField ref(mesh, "ref", 2);
        ref.allocDevice(); fillField(ref);
        bcX.applyOnGPU(ref); bcYlo.applyOnGPU(ref); bcYhi.applyOnGPU(ref);

        // swap f's device pointer to tmp's buffer, apply batch through f
        Real* orig = f.d_curr;
        f.d_curr = tmp.d_curr;
        batch.applyOnGPU(f);
        f.d_curr = orig;

        require(maxDiffNonCorner(tmp, ref) == 0.0,
                "batch did not follow the swapped device pointer");
    }

    return 0;
}
