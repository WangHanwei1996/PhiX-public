#include "operators/Gradient.h"
#include "core/Check.h"
#include "core/CudaCheck.h"
#include "scheme/Isotropic.h"
#include "scheme/Schemes.h"
#include "scheme/SchemeCatalog.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

template<typename Scheme>
__global__ void kernel_grad_accumulate(
        Real*       rhs,
        const Real* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int dim, int axis,
        double inv_dx, double inv_dy, double inv_dz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int is = i + ghost;
    int js = j + ghost;
    int ks = k + ghost;
    int c  = is + sx * (js + sy * ks);

    rhs[c] += coeff * Scheme::gradient(src, c, axis, sx, sy, dim,
                                        inv_dx, inv_dy, inv_dz);
}

template<typename Scheme>
Term makeGradientTerm(const ScalarField& f, int axis, double coeff) {
    if (axis < 0 || axis >= f.mesh.dim)
        throw std::invalid_argument("grad: axis out of range for this mesh dimension");
    check::checkCartesian(f.mesh, "grad");

    Term t;
    t.type  = TermType::GRADIENT;
    t.field = &f;
    t.inputs = {&f};
    t.coeff = coeff;
    t.axis  = axis;
    t.ghostRequired = Scheme::ghostRequired();
    describeInputs(t, t.ghostRequired, std::string(Scheme::name()) == "Iso9");
    t.info.schemes.push_back("gradient/" + f.name + "=" + Scheme::name());

    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    g  = f.ghost;
    int    dim = f.mesh.dim;
    double inv_dx = 1.0 / f.mesh.d[0];
    double inv_dy = (dim >= 2) ? 1.0 / f.mesh.d[1] : 0.0;
    double inv_dz = (dim >= 3) ? 1.0 / f.mesh.d[2] : 0.0;

    const ScalarField* pf = &f;

    t.gpu_launcher = [pf, nx, ny, nz, sx, sy, g, dim, axis, inv_dx, inv_dy, inv_dz]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        const Real* d_src = pf->d_curr;
        if (!d_src)
            throw std::runtime_error("grad GPU: source field not on device");
        int total = nx * ny * nz;
        kernel_grad_accumulate<Scheme><<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, d_src, c, nx, ny, nz, sx, sy, g, dim, axis, inv_dx, inv_dy, inv_dz);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("grad GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, nz, sx, sy, g, dim, axis, inv_dx, inv_dy, inv_dz]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* src = pf->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i + g, js = j + g, ks = k + g;
            int ctr = is + sx * (js + sy * ks);
            rhs[ctr] += c * Scheme::gradient(src, ctr, axis, sx, sy, dim,
                                              inv_dx, inv_dy, inv_dz);
        }
    };

    return t;
}

} // namespace

template<typename Scheme>
Term grad(const ScalarField& f, int axis, double coeff) {
    return makeGradientTerm<Scheme>(f, axis, coeff);
}

const std::vector<scheme::detail::Entry<scheme::detail::GradientFactory>>&
scheme::detail::gradientFactories() {
    static const std::vector<Entry<GradientFactory>> entries = {
        entry<scheme::CD2>(&grad<scheme::CD2>, Grid2D::Rectangular, false, "gradient component"),
        entry<scheme::CD4>(&grad<scheme::CD4>, Grid2D::Rectangular, false, "gradient component"),
        entry<scheme::CD6>(&grad<scheme::CD6>, Grid2D::Rectangular, false, "gradient component"),
        entry<scheme::Iso9>(&grad<scheme::Iso9>, Grid2D::SquareForIsotropy, true, "transversely averaged gradient")
    };
    return entries;
}

Term grad(const ScalarField& f, int axis, double coeff) {
    return grad<scheme::CD2>(f, axis, coeff);
}

Term grad(const ScalarField& f, int axis, const Schemes& sch, double coeff) {
    auto selected = sch.gradient(f.name, axis);
    auto t = grad(f, axis, selected.name, coeff);
    t.info.schemes = {selected.describe()};
    return t;
}

Term grad(const ScalarField& f, int axis, const std::string& schemeName, double coeff) {
    return scheme::detail::lookup(scheme::detail::gradientFactories(), schemeName, "grad").factory(f, axis, coeff);
}

} // namespace PhiX
