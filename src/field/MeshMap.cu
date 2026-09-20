#include "field/MeshMap.h"
#include "boundary/BoundaryCondition.h"
#include "core/Check.h"
#include "core/CudaCheck.h"

#include <cmath>
#include <string>

namespace PhiX {

namespace {

// Plain-value geometry pack for one field (mesh + storage).
struct MapGeom {
    int    n[3];
    double o[3], d[3];
    int    sx, sy, g;
};

__host__ __device__ inline int flatIdx(const MapGeom& m, int i, int j, int k) {
    return (i + m.g) + m.sx * ((j + m.g) + m.sy * (k + m.g));
}

MapGeom geomOf(const ScalarField& f) {
    MapGeom m;
    for (int ax = 0; ax < 3; ++ax) {
        m.n[ax] = f.mesh.n[ax];
        m.o[ax] = f.mesh.origin[ax];
        m.d[ax] = f.mesh.d[ax];
    }
    m.sx = f.storedDims[0];
    m.sy = f.storedDims[1];
    m.g  = f.ghost;
    return m;
}

// --- interpolation: bi/tri-linear sampling at destination cell centres ----
// Edge-clamped within the source's PHYSICAL cells (no ghost dependency).
__global__ void k_meshmap_interp(const Real* __restrict__ src,
                                 Real* __restrict__ dst,
                                 MapGeom s, MapGeom t)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = t.n[0] * t.n[1] * t.n[2];
    if (tid >= total) return;

    const int i = tid % t.n[0];
    const int j = (tid / t.n[0]) % t.n[1];
    const int k = tid / (t.n[0] * t.n[1]);
    const int idx[3] = {i, j, k};

    int    i0[3];
    double w[3];
    for (int ax = 0; ax < 3; ++ax) {
        if (s.n[ax] == 1) { i0[ax] = 0; w[ax] = 0.0; continue; }
        const double x = t.o[ax] + (idx[ax] + 0.5) * t.d[ax];
        double u = (x - s.o[ax]) / s.d[ax] - 0.5;
        int    c = static_cast<int>(floor(u));
        if (c < 0) c = 0;
        if (c > s.n[ax] - 2) c = s.n[ax] - 2;
        double f = u - c;
        if (f < 0.0) f = 0.0;
        if (f > 1.0) f = 1.0;
        i0[ax] = c;
        w[ax]  = f;
    }

    double acc = 0.0;
    for (int dk = 0; dk <= 1; ++dk)
    for (int dj = 0; dj <= 1; ++dj)
    for (int di = 0; di <= 1; ++di) {
        const double ww = (di ? w[0] : 1.0 - w[0])
                        * (dj ? w[1] : 1.0 - w[1])
                        * (dk ? w[2] : 1.0 - w[2]);
        if (ww == 0.0) continue;
        const int si = (s.n[0] == 1) ? 0 : i0[0] + di;
        const int sj = (s.n[1] == 1) ? 0 : i0[1] + dj;
        const int sk = (s.n[2] == 1) ? 0 : i0[2] + dk;
        acc += ww * static_cast<double>(src[flatIdx(s, si, sj, sk)]);
    }
    dst[flatIdx(t, i, j, k)] = static_cast<Real>(acc);
}

// --- conservative: exact cell-overlap volume weighting --------------------
// dst[c] = Σ_src overlap(c, src)·f_src / V_dst.  Because both uniform grids
// tile the identical domain, Σ f·dV is preserved to roundoff.
__global__ void k_meshmap_conserve(const Real* __restrict__ src,
                                   Real* __restrict__ dst,
                                   MapGeom s, MapGeom t)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = t.n[0] * t.n[1] * t.n[2];
    if (tid >= total) return;

    const int i = tid % t.n[0];
    const int j = (tid / t.n[0]) % t.n[1];
    const int k = tid / (t.n[0] * t.n[1]);
    const int idx[3] = {i, j, k};

    int    lo[3], hi[3];
    double x0[3], x1[3];
    double vDst = 1.0;
    for (int ax = 0; ax < 3; ++ax) {
        if (t.n[ax] == 1 || s.n[ax] == 1) {   // inactive axis
            lo[ax] = hi[ax] = 0;
            x0[ax] = x1[ax] = 0.0;
            continue;
        }
        x0[ax] = t.o[ax] + idx[ax] * t.d[ax];
        x1[ax] = x0[ax] + t.d[ax];
        vDst  *= t.d[ax];
        int a = static_cast<int>(floor((x0[ax] - s.o[ax]) / s.d[ax]));
        int b = static_cast<int>(ceil ((x1[ax] - s.o[ax]) / s.d[ax])) - 1;
        if (a < 0) a = 0;
        if (b > s.n[ax] - 1) b = s.n[ax] - 1;
        lo[ax] = a;
        hi[ax] = b;
    }

    double acc = 0.0;
    for (int sk = lo[2]; sk <= hi[2]; ++sk)
    for (int sj = lo[1]; sj <= hi[1]; ++sj)
    for (int si = lo[0]; si <= hi[0]; ++si) {
        const int sidx[3] = {si, sj, sk};
        double w = 1.0;
        for (int ax = 0; ax < 3; ++ax) {
            if (t.n[ax] == 1 || s.n[ax] == 1) continue;
            const double a0 = s.o[ax] + sidx[ax] * s.d[ax];
            const double a1 = a0 + s.d[ax];
            const double ov = fmin(x1[ax], a1) - fmax(x0[ax], a0);
            w *= (ov > 0.0) ? ov : 0.0;
        }
        if (w > 0.0)
            acc += w * static_cast<double>(src[flatIdx(s, si, sj, sk)]);
    }
    dst[flatIdx(t, i, j, k)] = static_cast<Real>(acc / vDst);
}

void launch(bool conservative, const ScalarField& src, ScalarField& dst,
            cudaStream_t stream)
{
    const MapGeom s = geomOf(src), t = geomOf(dst);
    const int total   = t.n[0] * t.n[1] * t.n[2];
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    if (conservative)
        k_meshmap_conserve<<<blocks, threads, 0, stream>>>(
            src.d_curr, dst.d_curr, s, t);
    else
        k_meshmap_interp<<<blocks, threads, 0, stream>>>(
            src.d_curr, dst.d_curr, s, t);
    PHIX_KERNEL_CHECK("MeshMap transfer");
}

} // namespace

// ---------------------------------------------------------------------------

MeshMap::MeshMap(const Mesh& a, const Mesh& b)
    : a_(a), b_(b)
{
    check::checkMeshValid(a, "MeshMap");
    check::checkMeshValid(b, "MeshMap");
    check::checkCartesian(a, "MeshMap");
    check::checkCartesian(b, "MeshMap");
    check::checkSameDomain(a, b, "MeshMap");

    aligned_ = true;
    for (int ax = 0; ax < a.dim; ++ax) {
        const double r = (a.d[ax] > b.d[ax]) ? a.d[ax] / b.d[ax]
                                             : b.d[ax] / a.d[ax];
        const double m = std::round(r);
        if (m < 1.0 || std::fabs(r - m) > 1e-9 * m) { aligned_ = false; break; }
    }
}

void MeshMap::checkPair(const ScalarField& src, const ScalarField& dst,
                        const char* who) const
{
    const bool ab = check::sameMeshGeometry(src.mesh, a_)
                 && check::sameMeshGeometry(dst.mesh, b_);
    const bool ba = check::sameMeshGeometry(src.mesh, b_)
                 && check::sameMeshGeometry(dst.mesh, a_);
    if (!(ab || ba))
        throw ValidationError(who,
            "fields '" + src.name + "' / '" + dst.name
            + "' do not live on this map's mesh pair",
            "construct the MeshMap from the two meshes these fields were "
            "built on");
    check::checkOnDevice(src, who);
    check::checkOnDevice(dst, who);
}

void MeshMap::interpolate(const ScalarField& src, ScalarField& dst,
                          cudaStream_t stream) const
{
    checkPair(src, dst, "MeshMap::interpolate");
    launch(false, src, dst, stream);
}

void MeshMap::conserve(const ScalarField& src, ScalarField& dst,
                       cudaStream_t stream) const
{
    checkPair(src, dst, "MeshMap::conserve");
    launch(true, src, dst, stream);
}

void MeshMap::interpolate(const ScalarField& src, ScalarField& dst,
                          const std::vector<BoundaryCondition*>& dstBCs,
                          cudaStream_t stream) const
{
    interpolate(src, dst, stream);
    for (auto* bc : dstBCs)
        if (bc) bc->applyOnGPU(dst);
}

void MeshMap::conserve(const ScalarField& src, ScalarField& dst,
                       const std::vector<BoundaryCondition*>& dstBCs,
                       cudaStream_t stream) const
{
    conserve(src, dst, stream);
    for (auto* bc : dstBCs)
        if (bc) bc->applyOnGPU(dst);
}

void MeshMap::interpolate(const VectorField& src, VectorField& dst,
                          cudaStream_t stream) const
{
    if (src.nComponents() != dst.nComponents())
        throw ValidationError("MeshMap::interpolate",
            "VectorField component counts differ ("
            + std::to_string(src.nComponents()) + " vs "
            + std::to_string(dst.nComponents()) + ")");
    for (int c = 0; c < src.nComponents(); ++c)
        interpolate(src[c], dst[c], stream);
}

void MeshMap::conserve(const VectorField& src, VectorField& dst,
                       cudaStream_t stream) const
{
    if (src.nComponents() != dst.nComponents())
        throw ValidationError("MeshMap::conserve",
            "VectorField component counts differ ("
            + std::to_string(src.nComponents()) + " vs "
            + std::to_string(dst.nComponents()) + ")");
    for (int c = 0; c < src.nComponents(); ++c)
        conserve(src[c], dst[c], stream);
}

} // namespace PhiX
