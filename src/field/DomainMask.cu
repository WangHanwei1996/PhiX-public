#include "field/DomainMask.h"
#include "core/CudaCheck.h"
#include "field/ReducePW.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

// Zero faces (normal axis `a`) that are not flanked by two active cells.
// Faces adjacent to ghost cells see mask 0 there, so exterior faces are
// zeroed too — the mask boundary is uniformly no-flux.
__global__ void kernel_mask_faces(Real* F, const Real* mask,
                                  int a, int nfa, int nt0, int nt1,
                                  int fs0, int fs1,
                                  int ms0, int ms1, int mg, int fg)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nfa * nt0 * nt1) return;
    const int fi = idx % nfa;              // face index along normal axis
    const int t0 = (idx / nfa) % nt0;      // tangential physical indices
    const int t1 = idx / (nfa * nt0);

    // physical (i,j,k) of the two adjacent cells
    int lo[3], hi[3];
    int t = 0;
    for (int ax = 0; ax < 3; ++ax) {
        if (ax == a) { lo[ax] = fi - 1; hi[ax] = fi; }
        else         { lo[ax] = hi[ax] = (t++ == 0) ? t0 : t1; }
    }
    const int cLo = (lo[0] + mg) + ms0 * ((lo[1] + mg) + ms1 * (lo[2] + mg));
    const int cHi = (hi[0] + mg) + ms0 * ((hi[1] + mg) + ms1 * (hi[2] + mg));

    if (mask[cLo] < Real(0.5) || mask[cHi] < Real(0.5)) {
        int s[3];
        t = 0;
        for (int ax = 0; ax < 3; ++ax)
            s[ax] = (ax == a) ? fi : ((t++ == 0) ? t0 : t1) + fg;
        F[s[0] + fs0 * (s[1] + fs1 * s[2])] = Real(0);
    }
}

// Mirror closure: each INACTIVE physical cell bordering the active region
// takes the average of its active face-neighbours.  Active cells are only
// read, never written, so the in-place update is race-free.
__global__ void kernel_closure(Real* f, const Real* mask,
                               int nx, int ny, int nz,
                               int fs0, int fs1, int fgh,
                               int ms0, int ms1, int mg, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;
    const int i = idx % nx;
    const int j = (idx / nx) % ny;
    const int k = idx / (nx * ny);

    const int cm = (i + mg) + ms0 * ((j + mg) + ms1 * (k + mg));
    if (mask[cm] >= Real(0.5)) return;     // active cells untouched

    Real acc = Real(0);
    int  cnt = 0;
    const int p[3] = {i, j, k};
    for (int ax = 0; ax < dim; ++ax) {
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            int q[3] = {p[0], p[1], p[2]};
            q[ax] += sgn;
            const int nm = (q[0] + mg) + ms0 * ((q[1] + mg) + ms1 * (q[2] + mg));
            if (mask[nm] >= Real(0.5)) {
                const int nf = (q[0] + fgh)
                             + fs0 * ((q[1] + fgh) + fs1 * (q[2] + fgh));
                acc += f[nf];
                ++cnt;
            }
        }
    }
    if (cnt > 0) {
        const int cf = (i + fgh) + fs0 * ((j + fgh) + fs1 * (k + fgh));
        f[cf] = acc / Real(cnt);
    }
}

} // namespace

DomainMask::DomainMask(const Mesh& mesh,
                       const std::function<bool(double, double, double)>& inside,
                       int ghost)
    : mesh_(mesh), mask_(mesh, "domainMask", ghost)
{
    mask_.fill(0.0);
    for (int k = 0; k < mesh.n[2]; ++k)
    for (int j = 0; j < mesh.n[1]; ++j)
    for (int i = 0; i < mesh.n[0]; ++i) {
        if (inside(mesh.coord(0, i), mesh.coord(1, j), mesh.coord(2, k))) {
            mask_.curr[static_cast<std::size_t>(mask_.index(i, j, k))]
                = Real(1);
            ++nActive_;
        }
    }
    if (nActive_ == 0)
        throw std::invalid_argument("DomainMask: predicate selects no cells");
    mask_.allocDevice();
    mask_.uploadAllToDevice();
}

void DomainMask::maskFaces(FaceField& F) const {
    if (!F.d_data)
        throw std::runtime_error("DomainMask::maskFaces: face field not on device");
    const int a = F.normalAxis;
    int tAxes[2], t = 0;
    for (int ax = 0; ax < 3; ++ax)
        if (ax != a) tAxes[t++] = ax;
    const int nfa = mesh_.n[a] + 1;
    const int nt0 = mesh_.n[tAxes[0]];
    const int nt1 = mesh_.n[tAxes[1]];
    const int total = nfa * nt0 * nt1;
    kernel_mask_faces<<<(total + 255) / 256, 256>>>(
        F.d_data, mask_.d_curr, a, nfa, nt0, nt1,
        F.storedDims[0], F.storedDims[1],
        mask_.storedDims[0], mask_.storedDims[1], mask_.ghost, F.ghost);
    CUDA_CHECK(cudaGetLastError());
}

void DomainMask::applyClosure(ScalarField& f) const {
    if (!f.d_curr)
        throw std::runtime_error("DomainMask::applyClosure: field not on device");
    const int total = mesh_.n[0] * mesh_.n[1] * mesh_.n[2];
    kernel_closure<<<(total + 255) / 256, 256>>>(
        f.d_curr, mask_.d_curr,
        mesh_.n[0], mesh_.n[1], mesh_.n[2],
        f.storedDims[0], f.storedDims[1], f.ghost,
        mask_.storedDims[0], mask_.storedDims[1], mask_.ghost, mesh_.dim);
    CUDA_CHECK(cudaGetLastError());
}

double DomainMask::sum(const ScalarField& f) const {
    return reduce::fieldSumPW(f, mask_,
        [=] __host__ __device__ (Real v, Real m) { return v * m; });
}

} // namespace PhiX
