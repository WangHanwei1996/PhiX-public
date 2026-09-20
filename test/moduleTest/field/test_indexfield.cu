// ---------------------------------------------------------------------------
// module_indexfield — IndexField + adopt/gather primitives (v2.43.0, P-2)
//
// Acceptance from PHIX_IMPROVEMENTS.md: the PoolSectionGPU k_gid and
// k_orientOut kernels must be expressible with the official API with
// BIT-IDENTICAL results.  This test embeds both reference kernels (verbatim
// logic, explicit params instead of __constant__ P) and compares:
//
//   1. adoptFromNeighborGPU(gid, phi, pred, &theta, table)  vs  k_gid —
//      three front-advance rounds on a 2D multi-grain strip; gid (int32)
//      and theta (double) compared bitwise after every round.
//      Inputs are constructed race-free: scores decrease strictly away from
//      the assigned region and the predicate reaches exactly one column per
//      round, so in-launch adoption order cannot change the result.
//   2. gatherPWGPU(orient, gid, table, phi, fold)  vs  k_orientOut —
//      bitwise on the folded-angle output (liquid = -60 branch included).
//   3. gatherGPU vs CPU reference (assigned + fallback cells).
//   4. 3D (26-neighbour) adoption vs a host reference, one race-free round.
// ---------------------------------------------------------------------------

#include "field/IndexField.h"
#include "field/ScalarField.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

#define CUDA_REQUIRE(call)                                                   \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        require(_e == cudaSuccess,                                           \
                std::string("CUDA: ") + cudaGetErrorString(_e));             \
    } while (0)

// ============================================================================
// Reference kernels — verbatim PoolSectionGPU logic (PoolSectionGPU.cu
// k_gid / k_orientOut) with explicit params instead of __constant__ P.
// ============================================================================

__device__ inline int refIDX(int i, int j, int sx, int sy, int g) {
    return (i + g) + sx * ((j + g) + sy * g);
}

__global__ void ref_k_gid(const double* phi, int* gid, double* theta,
                          const double* thetaTab,
                          int nx, int ny, int sx, int sy, int g,
                          double arriveThresh)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int c = refIDX(i, j, sx, sy, g);
    if (gid[c] >= 0) return;
    if (phi[c] < arriveThresh) return;

    double best = -2.0; int bestId = -1;
    for (int dj = -1; dj <= 1; dj++)
    for (int di = -1; di <= 1; di++) {
        if (di == 0 && dj == 0) continue;
        const int ii = i + di, jj = j + dj;
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;
        const int n = refIDX(ii, jj, sx, sy, g);
        if (gid[n] >= 0 && phi[n] > best) { best = phi[n]; bestId = gid[n]; }
    }
    if (bestId >= 0) { gid[c] = bestId; theta[c] = thetaTab[bestId]; }
}

__global__ void ref_k_orientOut(const double* phi, const int* gid,
                                const double* thetaTab, double* out,
                                int nx, int ny, int sx, int sy, int g)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int c = refIDX(i, j, sx, sy, g);
    if (gid[c] < 0 || phi[c] < 0.0) { out[c] = -60.0; return; }
    double deg = thetaTab[gid[c]] * 180.0 / M_PI;
    deg = fmod(deg, 90.0);
    if (deg < -45.0) deg += 90.0;
    if (deg >= 45.0) deg -= 90.0;
    out[c] = deg;
}

// ============================================================================
// 2D bitwise comparison: primitive vs reference over 3 front-advance rounds
// ============================================================================

static void test2D() {
    const int nx = 64, ny = 48, nGrains = 7;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, 1.0, 0.0,
                                    ny, 1.0, 0.0);

    // Score field (plays the role of phi): strictly decreasing along +x with
    // a deterministic per-cell wiggle — argmax neighbours are unique and the
    // assigned region always outranks the unassigned one (race-free).
    // Scores near the adoption zone stay well above -2.0, the (incidental)
    // argmax sentinel of the reference k_gid, so both argmax semantics agree
    // exactly as they do for a physical phi in [-1, 1].
    const int i0 = 10;   // assigned region at round 0: i <= i0
    auto scoreAt = [i0](int i, int j, int round) {
        return 1.0 - 0.30 * (i - i0) + 0.30 * round
             + 0.011 * std::sin(3.7 * i + 1.3 * j)
             + 0.007 * std::cos(2.1 * j);
    };
    // Threshold reaches exactly one more column per round
    // (0.30 per column, +0.30 per round): column i0+1 scores ~0.7 at round 0.
    const double arriveThresh = 0.55;

    std::vector<double> tab(nGrains);
    for (int n = 0; n < nGrains; ++n) tab[n] = -2.4 + 0.83 * n;  // radians

    // -- build identical initial states for reference (A) and primitive (B) --
    ScalarField phi(mesh, "phi", 1), thetaA(mesh, "thA", 1),
                thetaB(mesh, "thB", 1);
    IndexField  gidB(mesh, "gidB", 1);
    std::vector<int> gidInit(gidB.storedSize, -1);

    phi.fillCurr(0.0);
    thetaA.fillCurr(0.0);
    thetaB.fillCurr(0.0);
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t c = static_cast<std::size_t>(phi.index(i, j));
        phi.curr[c] = scoreAt(i, j, 0);
        if (i <= i0) {
            const int gidv = (j / 7 + i / 4) % nGrains;
            gidInit[c]     = gidv;
            thetaA.curr[c] = tab[gidv];
            thetaB.curr[c] = tab[gidv];
        }
    }
    gidB.curr.assign(gidInit.begin(), gidInit.end());

    for (ScalarField* f : {&phi, &thetaA, &thetaB}) {
        f->allocDevice();
        f->uploadAllToDevice();
    }
    gidB.allocDevice();
    gidB.uploadToDevice();

    // reference gid: raw int buffer with the same stored layout
    int* d_gidA = nullptr;
    CUDA_REQUIRE(cudaMalloc(&d_gidA, gidB.storedBytes()));
    CUDA_REQUIRE(cudaMemcpy(d_gidA, gidInit.data(), gidB.storedBytes(),
                            cudaMemcpyHostToDevice));

    DeviceTable dTab(tab);

    const dim3 blk(32, 8);
    const dim3 grd((nx + blk.x - 1) / blk.x, (ny + blk.y - 1) / blk.y);

    int adoptedTotal = 0;
    for (int round = 0; round < 3; ++round) {
        // advance the "front": refresh score field for this round
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            phi.curr[static_cast<std::size_t>(phi.index(i, j))] =
                scoreAt(i, j, round);
        phi.uploadCurrToDevice();

        ref_k_gid<<<grd, blk>>>(phi.d_curr, d_gidA, thetaA.d_curr, dTab.data(),
                                nx, ny, phi.storedDims[0], phi.storedDims[1],
                                phi.ghost, arriveThresh);
        adoptFromNeighborGPU(gidB, phi,
            [arriveThresh] __host__ __device__ (double p)
                { return p >= arriveThresh; },
            &thetaB, dTab.data());
        CUDA_REQUIRE(cudaDeviceSynchronize());

        // bitwise comparison of gid + theta over the full stored arrays
        std::vector<int> gidA(gidB.storedSize);
        CUDA_REQUIRE(cudaMemcpy(gidA.data(), d_gidA, gidB.storedBytes(),
                                cudaMemcpyDeviceToHost));
        gidB.downloadFromDevice();
        thetaA.downloadCurrFromDevice();
        thetaB.downloadCurrFromDevice();

        int assigned = 0;
        for (std::size_t c = 0; c < gidB.storedSize; ++c) {
            require(gidA[c] == gidB.curr[c],
                    "2D round " + std::to_string(round) + ": gid mismatch");
            require(thetaA.curr[c] == thetaB.curr[c],
                    "2D round " + std::to_string(round) + ": theta mismatch");
            if (gidB.curr[c] >= 0) ++assigned;
        }
        // each round must adopt exactly one more column
        require(assigned == (i0 + 2 + round) * ny,
                "2D round " + std::to_string(round) + ": adoption did not "
                "advance exactly one column (" + std::to_string(assigned) + ")");
        adoptedTotal = assigned;
    }
    require(adoptedTotal == (i0 + 4) * ny, "2D: final adoption count");

    // ---- k_orientOut vs gatherPWGPU (phi has both signs downstream) ----
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
        phi.curr[static_cast<std::size_t>(phi.index(i, j))] =
            (i <= i0 + 2) ? 0.8 : -0.8;   // solid / liquid split
    phi.uploadCurrToDevice();

    ScalarField orientA(mesh, "orA", 1), orientB(mesh, "orB", 1);
    orientA.fillCurr(0.0); orientB.fillCurr(0.0);
    orientA.allocDevice(); orientA.uploadAllToDevice();
    orientB.allocDevice(); orientB.uploadAllToDevice();

    ref_k_orientOut<<<grd, blk>>>(phi.d_curr, d_gidA, dTab.data(),
                                  orientA.d_curr, nx, ny, phi.storedDims[0],
                                  phi.storedDims[1], phi.ghost);
    gatherPWGPU(orientB, gidB, dTab.data(), phi,
        [] __host__ __device__ (int id, double th, double p) {
            if (id < 0 || p < 0.0) return -60.0;
            double deg = fmod(th * 180.0 / M_PI, 90.0);
            if (deg < -45.0) deg += 90.0;
            if (deg >= 45.0) deg -= 90.0;
            return deg;
        });
    CUDA_REQUIRE(cudaDeviceSynchronize());

    orientA.downloadCurrFromDevice();
    orientB.downloadCurrFromDevice();
    bool sawSolid = false, sawLiquid = false;
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t c = static_cast<std::size_t>(phi.index(i, j));
        require(orientA.curr[c] == orientB.curr[c], "orient mismatch");
        if (orientA.curr[c] == -60.0) sawLiquid = true; else sawSolid = true;
    }
    require(sawSolid && sawLiquid, "orient test must exercise both branches");

    // ---- plain gatherGPU vs CPU ----
    ScalarField gathered(mesh, "gath", 1);
    gathered.allocDevice(); gathered.uploadAllToDevice();
    gatherGPU(gathered, gidB, dTab.data(), -9.5);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    gathered.downloadCurrFromDevice();
    bool sawFallback = false;
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t c = static_cast<std::size_t>(phi.index(i, j));
        const int id = gidB.curr[c];
        const double expect = (id >= 0) ? tab[id] : -9.5;
        require(gathered.curr[c] == expect, "gatherGPU mismatch");
        if (id < 0) sawFallback = true;
    }
    require(sawFallback, "gatherGPU test must exercise the fallback branch");

    cudaFree(d_gidA);
}

// ============================================================================
// 3D: one race-free adoption round vs a host reference (26-neighbourhood)
// ============================================================================

static void test3D() {
    const int nx = 12, ny = 9, nz = 7, nGrains = 4;
    Mesh mesh = Mesh::makeUniform3D(CoordSys::CARTESIAN, nx, 1.0, 0.0,
                                    ny, 1.0, 0.0, nz, 1.0, 0.0);

    auto scoreAt = [](int i, int j, int k) {
        return 1.0 - 0.40 * i + 0.013 * std::sin(2.9 * i + 1.7 * j + 0.9 * k);
    };
    const int    i0 = 4;
    const double arriveThresh = 1.0 - 0.40 * (i0 + 1) - 0.06;

    std::vector<double> tab(nGrains);
    for (int n = 0; n < nGrains; ++n) tab[n] = 0.31 * n - 0.5;

    ScalarField phi(mesh, "phi3", 1), theta(mesh, "th3", 1);
    IndexField  gid(mesh, "gid3", 1);
    phi.fillCurr(0.0); theta.fillCurr(0.0);
    gid.fill(-1);
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t c = static_cast<std::size_t>(phi.index(i, j, k));
        phi.curr[c] = scoreAt(i, j, k);
        if (i <= i0) {
            gid.curr[c]   = (j + 2 * k) % nGrains;
            theta.curr[c] = tab[gid.curr[c]];
        }
    }

    // host reference (same scan order dk→dj→di, strict '>')
    std::vector<int32_t> gidRef = gid.curr;
    std::vector<double>  thRef(theta.curr.begin(), theta.curr.end());
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t c = static_cast<std::size_t>(gid.index(i, j, k));
        if (gid.curr[c] >= 0) continue;          // decided on pre-round state
        if (phi.curr[c] < arriveThresh) continue;
        double best = -1e300; int bestId = -1;
        for (int dk = -1; dk <= 1; ++dk)
        for (int dj = -1; dj <= 1; ++dj)
        for (int di = -1; di <= 1; ++di) {
            if (di == 0 && dj == 0 && dk == 0) continue;
            const int ii = i + di, jj = j + dj, kk = k + dk;
            if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz)
                continue;
            const std::size_t n =
                static_cast<std::size_t>(gid.index(ii, jj, kk));
            if (gid.curr[n] >= 0 && phi.curr[n] > best) {
                best = phi.curr[n]; bestId = gid.curr[n];
            }
        }
        if (bestId >= 0) { gidRef[c] = bestId; thRef[c] = tab[bestId]; }
    }

    phi.allocDevice();   phi.uploadAllToDevice();
    theta.allocDevice(); theta.uploadAllToDevice();
    gid.allocDevice();   gid.uploadToDevice();
    DeviceTable dTab(tab);

    adoptFromNeighborGPU(gid, phi,
        [arriveThresh] __host__ __device__ (double p)
            { return p >= arriveThresh; },
        &theta, dTab.data());
    CUDA_REQUIRE(cudaDeviceSynchronize());

    gid.downloadFromDevice();
    theta.downloadCurrFromDevice();
    int adopted = 0;
    for (std::size_t c = 0; c < gid.storedSize; ++c) {
        require(gid.curr[c] == gidRef[c], "3D: gid mismatch");
        require(theta.curr[c] == thRef[c], "3D: theta mismatch");
        if (gid.curr[c] >= 0) ++adopted;
    }
    require(adopted == (i0 + 2) * ny * nz, "3D: adoption count");
}

// minSourceId: only grains with id >= minSourceId may propagate (the
// Pool-family "substrate grains are frozen decoration" rule).
static void testMinSourceId() {
    const int nx = 6, ny = 5;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, 1.0, 0.0,
                                    ny, 1.0, 0.0);
    ScalarField phi(mesh, "phi", 1), theta(mesh, "th", 1);
    phi.fillCurr(1.0);      // every unassigned cell "arrives"
    theta.fillCurr(0.0);

    auto build = [&](IndexField& gid) {
        gid.curr.assign(gid.storedSize, -1);
        for (int j = 0; j < ny; ++j) {
            gid.curr[gid.index(0, j)] = 0;                       // substrate
            phi.curr[phi.index(0, j)] = 5.0;
        }
        gid.curr[gid.index(2, 1)] = 3;                           // seed grain
        phi.curr[phi.index(2, 1)] = 10.0;
        gid.allocDevice();
        gid.uploadToDevice();
    };
    std::vector<double> tab = {0.1, 0.2, 0.3, 0.4};
    DeviceTable dTab(tab);
    phi.allocDevice();
    theta.allocDevice();

    IndexField gidAll(mesh, "gidAll", 1), gidSeed(mesh, "gidSeed", 1);
    build(gidAll);
    build(gidSeed);
    phi.uploadAllToDevice();
    theta.uploadAllToDevice();

    auto arrived = [] __host__ __device__ (double p) { return p >= 0.5; };
    adoptFromNeighborGPU(gidAll,  phi, arrived, &theta, dTab.data());       // default: all propagate
    adoptFromNeighborGPU(gidSeed, phi, arrived, &theta, dTab.data(), /*minSourceId=*/1);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    gidAll.downloadFromDevice();
    gidSeed.downloadFromDevice();

    // (1,4): only substrate (id 0) neighbours — adopted by default,
    //        frozen out when minSourceId = 1.
    require(gidAll.curr [gidAll.index (1, 4)] == 0,  "minSrc: default adopts substrate");
    require(gidSeed.curr[gidSeed.index(1, 4)] == -1, "minSrc: substrate blocked as source");
    // (1,1): sees the seed grain (score 10 beats substrate 5) — adopts 3 both ways.
    require(gidAll.curr [gidAll.index (1, 1)] == 3,  "minSrc: seed adopted (default)");
    require(gidSeed.curr[gidSeed.index(1, 1)] == 3,  "minSrc: seed adopted (filtered)");
    // Far cells with no assigned neighbours stay unassigned.
    require(gidSeed.curr[gidSeed.index(5, 2)] == -1, "minSrc: far cell untouched");
    std::printf("  PASS: adoptFromNeighborGPU minSourceId filter\n");
}

int main() {
    // storage layout must mirror ScalarField exactly
    {
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, 6, 1.0, 0.0,
                                        4, 1.0, 0.0);
        ScalarField f(mesh, "f", 1);
        IndexField  idx(mesh, "idx", 1);
        require(idx.storedSize == f.storedSize &&
                idx.planeOffset() == f.planeOffset() &&
                idx.storedBytes() == f.storedSize * sizeof(int32_t),
                "layout mismatch vs ScalarField");
        require(idx.index(2, 3) == f.index(2, 3), "index mismatch");
        require(idx.curr[0] == -1, "default init must be -1 (unassigned)");
    }

    test2D();
    test3D();
    testMinSourceId();

    std::printf("module_indexfield: ALL PASSED\n");
    return 0;
}
