#include "operators/Advection.h"
#include "scheme/Schemes.h"
#include "scheme/SchemeCatalog.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

// ---------------------------------------------------------------------------
// Upwind directional-derivative schemes (selected by velocity sign).
// Each provides ghostRequired() and deriv(s, c, stride, inv_d, positive).
// ---------------------------------------------------------------------------

// 1st-order donor cell
struct UW1 {
    static constexpr int ghostRequired() { return 1; }
    static constexpr int order() { return 1; }
    static constexpr const char* name() { return "UW1"; }
    __host__ __device__ static
    Real deriv(const Real* s, int c, int st, Real inv_d, bool pos) {
        return pos ? (s[c] - s[c - st]) * inv_d
                   : (s[c + st] - s[c]) * inv_d;
    }
};

// 2nd-order fully one-sided upwind
struct UW2 {
    static constexpr int ghostRequired() { return 2; }
    static constexpr int order() { return 2; }
    static constexpr const char* name() { return "UW2"; }
    __host__ __device__ static
    Real deriv(const Real* s, int c, int st, Real inv_d, bool pos) {
        return pos
            ? ( Real(3)*s[c] - Real(4)*s[c-st] + s[c-2*st]) * inv_d * Real(0.5)
            : (-Real(3)*s[c] + Real(4)*s[c+st] - s[c+2*st]) * inv_d * Real(0.5);
    }
};

// 5th-order HJ-WENO (Jiang-Shu weights; Osher-Fedkiw formulation).
// Smooth fields: 5th order; near kinks the smoothness indicators bias the
// stencil away from the discontinuity (essentially non-oscillatory).
struct WENO5 {
    static constexpr int ghostRequired() { return 3; }
    static constexpr int order() { return 5; }
    static constexpr const char* name() { return "WENO5"; }

    __host__ __device__ static
    Real combine(Real v1, Real v2, Real v3, Real v4, Real v5) {
        const Real c13 = Real(13.0 / 12.0), c14 = Real(0.25);
        const Real b1 = c13 * (v1 - Real(2)*v2 + v3) * (v1 - Real(2)*v2 + v3)
                      + c14 * (v1 - Real(4)*v2 + Real(3)*v3)
                            * (v1 - Real(4)*v2 + Real(3)*v3);
        const Real b2 = c13 * (v2 - Real(2)*v3 + v4) * (v2 - Real(2)*v3 + v4)
                      + c14 * (v2 - v4) * (v2 - v4);
        const Real b3 = c13 * (v3 - Real(2)*v4 + v5) * (v3 - Real(2)*v4 + v5)
                      + c14 * (Real(3)*v3 - Real(4)*v4 + v5)
                            * (Real(3)*v3 - Real(4)*v4 + v5);
        const Real eps = Real(1e-6);
        const Real a1 = Real(0.1) / ((eps + b1) * (eps + b1));
        const Real a2 = Real(0.6) / ((eps + b2) * (eps + b2));
        const Real a3 = Real(0.3) / ((eps + b3) * (eps + b3));
        const Real inv = Real(1) / (a1 + a2 + a3);
        const Real s1 =  v1 * Real(1.0/3.0) - v2 * Real(7.0/6.0) + v3 * Real(11.0/6.0);
        const Real s2 = -v2 * Real(1.0/6.0) + v3 * Real(5.0/6.0) + v4 * Real(1.0/3.0);
        const Real s3 =  v3 * Real(1.0/3.0) + v4 * Real(5.0/6.0) - v5 * Real(1.0/6.0);
        return (a1 * s1 + a2 * s2 + a3 * s3) * inv;
    }

    __host__ __device__ static
    Real deriv(const Real* s, int c, int st, Real inv_d, bool pos) {
        if (pos) {   // D⁻: backward differences v1..v5 at i−2..i+2
            const Real v1 = (s[c-2*st] - s[c-3*st]) * inv_d;
            const Real v2 = (s[c-st]   - s[c-2*st]) * inv_d;
            const Real v3 = (s[c]      - s[c-st])   * inv_d;
            const Real v4 = (s[c+st]   - s[c])      * inv_d;
            const Real v5 = (s[c+2*st] - s[c+st])   * inv_d;
            return combine(v1, v2, v3, v4, v5);
        } else {     // D⁺: mirrored
            const Real v1 = (s[c+3*st] - s[c+2*st]) * inv_d;
            const Real v2 = (s[c+2*st] - s[c+st])   * inv_d;
            const Real v3 = (s[c+st]   - s[c])      * inv_d;
            const Real v4 = (s[c]      - s[c-st])   * inv_d;
            const Real v5 = (s[c-st]   - s[c-2*st]) * inv_d;
            return combine(v1, v2, v3, v4, v5);
        }
    }
};

template<typename D>
__global__ void kernel_adv_accumulate(
        Real*       rhs,
        const Real* src,
        const Real* ux,
        const Real* uy,
        const Real* uz,
        Real          coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int dim,
        Real inv_dx, Real inv_dy, Real inv_dz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int c = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));

    Real val = ux[c] * D::deriv(src, c, 1, inv_dx, ux[c] > Real(0));
    if (dim >= 2) val += uy[c] * D::deriv(src, c, sx, inv_dy, uy[c] > Real(0));
    if (dim >= 3) val += uz[c] * D::deriv(src, c, sx * sy, inv_dz,
                                          uz[c] > Real(0));

    rhs[c] += coeff * val;
}

} // namespace

template<typename D>
static Term makeAdvTerm(const VectorField& u, const ScalarField& f,
                        double coeff) {
    const int dim = f.mesh.dim;
    if (u.nComponents() < dim)
        throw std::invalid_argument(
            "adv: velocity field '" + u.name + "' has "
            + std::to_string(u.nComponents()) + " components but mesh is "
            + std::to_string(dim) + "D");
    for (int c = 0; c < dim; ++c)
        if (u[c].storedSize != f.storedSize)
            throw std::invalid_argument(
                "adv: velocity component '" + u[c].name
                + "' layout differs from field '" + f.name + "'");

    Term t;
    t.type  = TermType::COMPOSITE;
    t.field = &f;
    t.inputs.push_back(&f);
    for (int c = 0; c < f.mesh.dim; ++c) t.inputs.push_back(&u[c]);
    t.coeff = coeff;
    t.ghostRequired = D::ghostRequired();
    describeInputs(t);
    t.info.read(&f, t.ghostRequired);
    t.info.schemes.push_back("advection/" + f.name + "=" + D::name());

    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    g  = f.ghost;
    const Real inv_dx = static_cast<Real>(1.0 / f.mesh.d[0]);
    const Real inv_dy = (dim >= 2) ? static_cast<Real>(1.0 / f.mesh.d[1]) : Real(0);
    const Real inv_dz = (dim >= 3) ? static_cast<Real>(1.0 / f.mesh.d[2]) : Real(0);

    const ScalarField* pf  = &f;
    const ScalarField* pux = &u[0];
    const ScalarField* puy = (dim >= 2) ? &u[1] : &u[0];
    const ScalarField* puz = (dim >= 3) ? &u[2] : &u[0];

    t.gpu_launcher = [pf, pux, puy, puz, nx, ny, nz, sx, sy, g, dim,
                      inv_dx, inv_dy, inv_dz]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        if (!pf->d_curr || !pux->d_curr)
            throw std::runtime_error("adv GPU: source/velocity not on device");
        int total = nx * ny * nz;
        kernel_adv_accumulate<D><<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, pf->d_curr, pux->d_curr, puy->d_curr, puz->d_curr,
            static_cast<Real>(c),
            nx, ny, nz, sx, sy, g, dim, inv_dx, inv_dy, inv_dz);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("adv GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, pux, puy, puz, nx, ny, nz, sx, sy, g, dim,
                    inv_dx, inv_dy, inv_dz]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* src = pf->curr.data();
        const Real* vx  = pux->curr.data();
        const Real* vy  = puy->curr.data();
        const Real* vz  = puz->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int ctr = (i + g) + sx * ((j + g) + sy * (k + g));
            Real val = vx[ctr] * D::deriv(src, ctr, 1, inv_dx, vx[ctr] > Real(0));
            if (dim >= 2) val += vy[ctr] * D::deriv(src, ctr, sx, inv_dy,
                                                    vy[ctr] > Real(0));
            if (dim >= 3) val += vz[ctr] * D::deriv(src, ctr, sx * sy, inv_dz,
                                                    vz[ctr] > Real(0));
            rhs[ctr] += static_cast<Real>(c) * val;
        }
    };

    return t;
}

const std::vector<scheme::detail::Entry<scheme::detail::AdvectionFactory>>&
scheme::detail::advectionFactories() {
    static const std::vector<Entry<AdvectionFactory>> entries = {
        entry<UW1>(&makeAdvTerm<UW1>, Grid2D::Rectangular, false, "u dot grad(f), donor cell"),
        entry<UW2>(&makeAdvTerm<UW2>, Grid2D::Rectangular, false, "u dot grad(f), one-sided derivative"),
        entry<WENO5>(&makeAdvTerm<WENO5>, Grid2D::Rectangular, false, "u dot grad(f), nonlinear HJ-WENO derivative")
    };
    return entries;
}

Term adv(const VectorField& u, const ScalarField& f, double coeff) {
    return makeAdvTerm<UW1>(u, f, coeff);
}

Term adv(const VectorField& u, const ScalarField& f,
         const Schemes& sch, double coeff) {
    auto selected = sch.advection(u.name, f.name);
    auto t = adv(u, f, selected.name, coeff);
    t.info.schemes = {selected.describe()};
    return t;
}

Term adv(const VectorField& u, const ScalarField& f,
         const std::string& schemeName, double coeff) {
    return scheme::detail::lookup(scheme::detail::advectionFactories(), schemeName, "adv").factory(u, f, coeff);
}

} // namespace PhiX
