#pragma once
#include "core/Check.h"

// ---------------------------------------------------------------------------
// DivFaceAssembled.h — fused face-flux divergence operator (P-4).
//
//   rhs[c] += coeff * Σ_axis ( F_axis(hi face) − F_axis(lo face) ) / d[axis]
//
// where the flux F on every face is assembled ON THE FLY by a user functor
// from register-resident face quantities — no FaceField materialisation, no
// intermediate global-memory traffic.  This collapses the four-step chain
//
//   faceGrad → interp → facePW → divFace     (4+ kernels, 4+ field sweeps)
//
// into a single kernel, accepting the redundant arithmetic of evaluating
// each face twice (once per adjacent cell): stencil code is memory-bound,
// arithmetic is nearly free.  This is the hot pattern of solidification /
// phase-field solvers (anisotropic ∇·(W(n̂)²∇φ), degenerate-mobility
// ∇·(M(φ)∇μ), anti-trapping currents...).
//
// Face quantities handed to the functor (at each face of normal `axis`):
//   • full face gradient of f:  (gx, gy, gz) — normal component by exact
//     two-point difference, tangential components by the 4-point corner
//     average (the staggered stencil of dendrite_growth / PoolSectionGPU);
//     inactive dimensions are 0;
//   • fF — linear face interpolation of f  (0.5·(f_lo + f_hi));
//   • a1F, a2F — same interpolation of the optional aux cell fields
//     (per-cell θ tables, φ in mobility prefactors, ∂tφ, ...).
//
// Functor signatures (PHIX_FN, must compile host+device):
//   flux(int axis, Real gx, Real gy, Real gz, Real fF)                → Real
//   flux(int axis, Real gx, Real gy, Real gz, Real fF, Real a1F)      → Real
//   flux(int axis, Real gx, Real gy, Real gz, Real fF, Real a1F, Real a2F)
//
// Requirements:
//   • all fields on the same mesh with ghost >= 1;
//   • ghost cells of f and every aux field must be filled (BCs applied)
//     INCLUDING edge/corner ghosts — the tangential gradient at boundary
//     faces reads corner-adjacent ghosts.  Applying the field's BCs on all
//     axes (the normal PhiX pattern) satisfies this.
//
// Returns a COMPOSITE Term (GPU kernel + CPU fallback) for Equation::setRHS,
// composable with other Terms:  eq.setRHS(divFaceAssembled(...) + pw(...)).
//
// Example — per-cell-θ anisotropic flux of the EFKP φ equation:
//
//   eq.setRHS(divFaceAssembled(phi, theta,
//       PHIX_FN (int ax, double gx, double gy, double, double, double th) {
//           double psi = atan2(gy, gx) - th;
//           double a   = 1.0 + eps4 * cos(4.0 * psi);
//           double s4  = 4.0 * eps4 * sin(4.0 * psi);
//           return (ax == 0) ? W0sq * a * (a * gx + s4 * gy)
//                            : W0sq * a * (a * gy - s4 * gx);
//       }) + pw(phi, U, bulkFn));
// ---------------------------------------------------------------------------

#include "core/CudaCheck.h"
#include "field/ScalarField.h"
#include "equation/Term.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace PhiX {

namespace dfa_detail {

// Face gradient vector of f at the face between stored cells cA (low) and
// cB (high) along `ax`.  g[t] for tangential active axes uses the 4-point
// corner average; inactive dims stay 0.
__host__ __device__ inline void faceGradVec(
        const Real* f, int cA, int cB, int ax,
        const int str[3], const Real invd[3], int dim, Real g[3])
{
    g[0] = 0.0; g[1] = 0.0; g[2] = 0.0;
    g[ax] = (f[cB] - f[cA]) * invd[ax];
    for (int t = 0; t < dim; ++t) {
        if (t == ax) continue;
        g[t] = Real(0.25) * invd[t]
             * (f[cB + str[t]] + f[cA + str[t]]
                - f[cB - str[t]] - f[cA - str[t]]);
    }
}

// Divergence of the assembled flux at stored cell c — evaluates the functor
// on the lo and hi faces of every active axis (each face is computed twice
// grid-wide; register arithmetic, no extra memory traffic).
// FluxCall adapts the aux arity: fc(ax, g, fF, cA, cB).
template<typename FluxCall>
__host__ __device__ inline Real divAssembled(
        const Real* f, int c,
        const int str[3], const Real invd[3], int dim,
        FluxCall fc)
{
    Real div = 0.0;
    for (int ax = 0; ax < dim; ++ax) {
        const int cA = c - str[ax];
        const int cB = c + str[ax];
        Real g[3];

        faceGradVec(f, cA, c, ax, str, invd, dim, g);
        const Real Flo = fc(ax, g, Real(0.5) * (f[cA] + f[c]), cA, c);

        faceGradVec(f, c, cB, ax, str, invd, dim, g);
        const Real Fhi = fc(ax, g, Real(0.5) * (f[c] + f[cB]), c, cB);

        div += (Fhi - Flo) * invd[ax];
    }
    return div;
}

// Aux-arity adapters (usable from host and device paths alike).
template<typename Fn>
struct Call0 {
    Fn fn;
    __host__ __device__ Real operator()(int ax, const Real g[3], Real fF,
                                        int, int) const {
        return fn(ax, g[0], g[1], g[2], fF);
    }
};

template<typename Fn>
struct Call1 {
    Fn fn;
    const Real* a1;
    __host__ __device__ Real operator()(int ax, const Real g[3], Real fF,
                                        int cA, int cB) const {
        return fn(ax, g[0], g[1], g[2], fF,
                  Real(0.5) * (a1[cA] + a1[cB]));
    }
};

template<typename Fn>
struct Call2 {
    Fn fn;
    const Real* a1;
    const Real* a2;
    __host__ __device__ Real operator()(int ax, const Real g[3], Real fF,
                                        int cA, int cB) const {
        return fn(ax, g[0], g[1], g[2], fF,
                  Real(0.5) * (a1[cA] + a1[cB]),
                  Real(0.5) * (a2[cA] + a2[cB]));
    }
};

template<typename FluxCall>
__global__ void kernel_div_face_assembled(
        Real* rhs, const Real* f, FluxCall fc, double coeff,
        int nx, int ny, int nz,
        int sx, int sy, int g, int dim,
        Real invdx, Real invdy, Real invdz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);
    int c = (i + g) + sx * ((j + g) + sy * (k + g));

    const int  str[3]  = {1, sx, sx * sy};
    const Real invd[3] = {invdx, invdy, invdz};
    rhs[c] += coeff * divAssembled(f, c, str, invd, dim, fc);
}

inline void checkSameLayout(const ScalarField& f, const ScalarField& a,
                            const char* who) {
    if (!check::sameMeshGeometry(f.mesh, a.mesh) || f.ghost != a.ghost)
        throw std::invalid_argument(std::string(who) +
            ": all fields must share mesh dimensions and ghost width");
}

// Build the COMPOSITE Term around a prepared FluxCall.  aux pointers are
// ScalarFields whose curr/d_curr are re-read at launch time via MakeCall.
template <typename MakeCallGPU, typename MakeCallCPU>
Term makeDfaTerm(const ScalarField &f, double coeff, MakeCallGPU makeGpu, MakeCallCPU makeCpu,
                 std::vector<const ScalarField *> aux = {}) {
    if (f.ghost < 1)
        throw std::invalid_argument("divFaceAssembled: f needs ghost >= 1");

    Term t;
    t.type  = TermType::COMPOSITE;
    t.field = &f;
    t.coeff = coeff;
    t.inputs = std::move(aux);
    t.inputs.push_back(&f);
    t.ghostRequired = 1;
    describeInputs(t, 1);
    t.info.read(&f, 1, true);

    const ScalarField* pf = &f;
    const int nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    const int sx = f.storedDims[0], sy = f.storedDims[1];
    const int g  = f.ghost;
    const int dim = f.mesh.dim;
    const Real invdx = Real(1) / f.mesh.d[0];
    const Real invdy = (dim >= 2) ? Real(1) / f.mesh.d[1] : Real(0);
    const Real invdz = (dim >= 3) ? Real(1) / f.mesh.d[2] : Real(0);

    t.gpu_launcher = [=](Real* d_rhs, double c, ScratchPool& pool) {
        if (!pf->d_curr)
            throw std::runtime_error(
                "divFaceAssembled GPU: field not on device");
        auto fc = makeGpu();
        const int total = nx * ny * nz;
        kernel_div_face_assembled<decltype(fc)>
            <<<(total + 255) / 256, 256, 0, pool.stream>>>(
                d_rhs, pf->d_curr, fc, c,
                nx, ny, nz, sx, sy, g, dim, invdx, invdy, invdz);
        PHIX_KERNEL_CHECK("divFaceAssembled");
    };

    t.cpu_kernel = [=](Real* rhs, double c, ScratchPool&) {
        auto fc = makeCpu();
        const Real* fd = pf->curr.data();
        const int  str[3]  = {1, sx, sx * sy};
        const Real invd[3] = {invdx, invdy, invdz};
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const int idx = (i + g) + sx * ((j + g) + sy * (k + g));
            rhs[idx] += c * divAssembled(fd, idx, str, invd, dim, fc);
        }
    };

    return t;
}

} // namespace dfa_detail

// ---------------------------------------------------------------------------
// divFaceAssembled — public factories (0 / 1 / 2 aux fields)
// ---------------------------------------------------------------------------

template<typename Fn>
Term divFaceAssembled(const ScalarField& f, Fn flux, double coeff = 1.0)
{
    const ScalarField* pf = &f;
    return dfa_detail::makeDfaTerm(f, coeff,
        [flux, pf] { return dfa_detail::Call0<Fn>{flux}; },
        [flux, pf] { return dfa_detail::Call0<Fn>{flux}; });
}

template<typename Fn>
Term divFaceAssembled(const ScalarField& f, const ScalarField& a1,
                      Fn flux, double coeff = 1.0)
{
    dfa_detail::checkSameLayout(f, a1, "divFaceAssembled(a1)");
    const ScalarField* p1 = &a1;
    return dfa_detail::makeDfaTerm(
        f, coeff,
        [flux, p1] {
            if (!p1->d_curr)
                throw std::runtime_error(
                    "divFaceAssembled GPU: aux field not on device");
            return dfa_detail::Call1<Fn>{flux, p1->d_curr};
        },
        [flux, p1] { return dfa_detail::Call1<Fn>{flux, p1->curr.data()}; }, {&a1});
}

template<typename Fn>
Term divFaceAssembled(const ScalarField& f, const ScalarField& a1,
                      const ScalarField& a2, Fn flux, double coeff = 1.0)
{
    dfa_detail::checkSameLayout(f, a1, "divFaceAssembled(a1)");
    dfa_detail::checkSameLayout(f, a2, "divFaceAssembled(a2)");
    const ScalarField* p1 = &a1;
    const ScalarField* p2 = &a2;
    return dfa_detail::makeDfaTerm(
        f, coeff,
        [flux, p1, p2] {
            if (!p1->d_curr || !p2->d_curr)
                throw std::runtime_error(
                    "divFaceAssembled GPU: aux field not on device");
            return dfa_detail::Call2<Fn>{flux, p1->d_curr, p2->d_curr};
        },
        [flux, p1, p2] { return dfa_detail::Call2<Fn>{flux, p1->curr.data(), p2->curr.data()}; },
        {&a1, &a2});
}

} // namespace PhiX
