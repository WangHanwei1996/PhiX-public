#include "operators/GradSq.h"
#include "core/Check.h"
#include "core/CudaCheck.h"
#include "core/Error.h"
#include "scheme/Schemes.h"
#include "scheme/SchemeCatalog.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace PhiX {

namespace {

// The G bases live here rather than in scheme/ — they are squared-difference
// composites specific to this operator (Ji 2022 Appendix A), not reusable
// derivative primitives like Scheme::d1/d2/laplacian.
template<typename Scheme>
__host__ __device__ inline Real gradsq_eval(
        const Real* s, int c, int sx, int sy, int dim,
        Real inv_dx, Real inv_dy, Real inv_dz)
{
    // G_{1,0}: squared CD2 differences per axis
    Real gx  = (s[c + 1] - s[c - 1]) * Real(0.5) * inv_dx;
    Real g10 = gx * gx;
    if (dim >= 2) {
        Real gy = (s[c + sx] - s[c - sx]) * Real(0.5) * inv_dy;
        g10 += gy * gy;
    }
    if (dim >= 3) {
        const int sz = sx * sy;
        Real gz = (s[c + sz] - s[c - sz]) * Real(0.5) * inv_dz;
        g10 += gz * gz;
    }
    if constexpr (std::is_same<Scheme, scheme::Iso9>::value) {
        if (dim == 2) {
            // G_{0,1}: squared diagonal differences (assumes dx == dy)
            Real d1  = s[c + 1 + sx] - s[c - 1 - sx];
            Real d2  = s[c - 1 + sx] - s[c + 1 - sx];
            Real g01 = (d1 * d1 + d2 * d2) * inv_dx * inv_dx * Real(0.125);
            return Real(2.0 / 3.0) * g10 + Real(1.0 / 3.0) * g01;
        }
    }
    return g10;
}

template<typename Scheme>
__global__ void kernel_gradsq_accumulate(
        Real*       rhs,
        const Real* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int dim,
        double inv_dx,
        double inv_dy,
        double inv_dz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int is = i + ghost;
    int js = j + ghost;
    int ks = k + ghost;

    int c = is + sx * (js + sy * ks);
    Real val = gradsq_eval<Scheme>(src, c, sx, sy, dim,
                                   Real(inv_dx), Real(inv_dy), Real(inv_dz));
    rhs[c] += coeff * val;
}

template<typename Scheme>
Term makeGradSqTerm(const ScalarField& f, double coeff) {
    check::checkCartesian(f.mesh, "gradSq");
    if (std::is_same<Scheme, scheme::Iso9>::value && f.mesh.dim == 2
        && std::fabs(f.mesh.d[0] - f.mesh.d[1])
           > 1e-12 * std::fabs(f.mesh.d[0]))
        throw ValidationError("gradSq",
                              "Iso9 isotropic weights require dx == dy",
                              "use scheme CD2 on non-square meshes");
    Term t;
    t.type  = TermType::GRADIENT;
    t.field = &f;
    t.inputs = {&f};
    t.coeff = coeff;
    t.ghostRequired = Scheme::ghostRequired();
    describeInputs(t, t.ghostRequired, std::string(Scheme::name()) == "Iso9");
    t.info.schemes.push_back("gradSq/" + f.name + "=" + Scheme::name());

    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    g  = f.ghost;
    int    dim = f.mesh.dim;
    double inv_dx = 1.0 / f.mesh.d[0];
    double inv_dy = (dim >= 2) ? 1.0 / f.mesh.d[1] : 0.0;
    double inv_dz = (dim >= 3) ? 1.0 / f.mesh.d[2] : 0.0;

    const ScalarField* pf = &f;

    t.gpu_launcher = [pf, nx, ny, nz, sx, sy, g, dim, inv_dx, inv_dy, inv_dz]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        const Real* d_src = pf->d_curr;
        if (!d_src)
            throw std::runtime_error("gradSq GPU: source field not on device");
        int total = nx * ny * nz;
        kernel_gradsq_accumulate<Scheme><<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, d_src, c,
            nx, ny, nz, sx, sy,
            g, dim, inv_dx, inv_dy, inv_dz);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("gradSq GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, nz, sx, sy, g, dim, inv_dx, inv_dy, inv_dz]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* src = pf->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i + g, js = j + g, ks = k + g;
            int ctr = is + sx * (js + sy * ks);
            Real val = gradsq_eval<Scheme>(src, ctr, sx, sy, dim,
                                           Real(inv_dx), Real(inv_dy),
                                           Real(inv_dz));
            rhs[ctr] += c * val;
        }
    };

    return t;
}

} // namespace

template<typename Scheme>
Term gradSq(const ScalarField& f, double coeff) {
    return makeGradSqTerm<Scheme>(f, coeff);
}

const std::vector<scheme::detail::Entry<scheme::detail::ScalarFactory>>&
scheme::detail::gradSqFactories() {
    static const std::vector<Entry<ScalarFactory>> entries = {
        entry<scheme::CD2>(&gradSq<scheme::CD2>, Grid2D::Rectangular, false, "sum of squared central gradients"),
        entry<scheme::Iso9>(&gradSq<scheme::Iso9>, Grid2D::SquareRequired, true, "weighted squared-difference bases")
    };
    return entries;
}

Term gradSq(const ScalarField& f, double coeff) {
    return gradSq<scheme::CD2>(f, coeff);
}

Term gradSq(const ScalarField& f, const Schemes& sch, double coeff) {
    auto selected = sch.select("gradSq", "gradSq(" + f.name + ")");
    auto t = gradSq(f, selected.name, coeff);
    t.info.schemes = {selected.describe()};
    return t;
}

Term gradSq(const ScalarField& f, const std::string& schemeName, double coeff) {
    return scheme::detail::lookup(scheme::detail::gradSqFactories(), schemeName, "gradSq").factory(f, coeff);
}

} // namespace PhiX
