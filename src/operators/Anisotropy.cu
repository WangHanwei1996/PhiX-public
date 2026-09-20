#include "scheme/SchemeCatalog.h"
#include "operators/Anisotropy.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace PhiX {

namespace {
AnisoScheme selectAnisoCD2()  { return AnisoScheme::CD2; }
AnisoScheme selectAnisoIso9() { return AnisoScheme::Iso9; }
}

const std::vector<scheme::detail::Entry<scheme::detail::AnisoFactory>>&
scheme::detail::anisoFactories() {
    static const std::vector<Entry<AnisoFactory>> entries = {
        {{"CD2", 2, 1, true, Grid2D::Rectangular, "CD2", "anisotropic face-flux divergence"}, &selectAnisoCD2},
        {{"Iso9", 2, 1, true, Grid2D::SquareForIsotropy, "Iso9", "compact isotropic-error face-flux divergence"}, &selectAnisoIso9}
    };
    return entries;
}

AnisoScheme anisoSchemeFromString(const std::string& name) {
    return scheme::detail::lookup(scheme::detail::anisoFactories(), name,
                                   "anisoSchemeFromString", false).factory();
}

void AnisoParams::validate() const {
    if (interfacePhi && (!std::isfinite(interfaceCutoff) || interfaceCutoff <= 0 || interfaceCutoff >= 1))
        throw std::invalid_argument("AnisoParams: interfaceCutoff must be finite and in (0,1)");
    if (W0 <= 0.0)
        throw std::invalid_argument("AnisoParams: W0 must be > 0");
    if (eps < 0.0)
        throw std::invalid_argument("AnisoParams: eps must be >= 0");
    if (m < 1)
        throw std::invalid_argument("AnisoParams: fold symmetry m must be >= 1");
    // |ε| < 1/(m²−1) is the convexity (no missing-orientation) limit;
    // beyond it set regularize = true (Eggleston continuation) — the flag
    // is a no-op below the limit.
    if (eps >= 1.0)
        throw std::invalid_argument("AnisoParams: eps must be < 1");
}

// ---------------------------------------------------------------------------
// Eggleston matching: solve tan(t)·γ(t) − ε·m·sin(m·t) = 0 for the smallest
// positive root (γ = 1 + ε·cos(m·t)).  Below the convexity limit the only
// root is t = 0 → θ_m = 0 (regularisation never triggers).
// ---------------------------------------------------------------------------
AnisoReg anisoComputeRegularization(double eps, int m) {
    if (eps * (static_cast<double>(m) * m - 1.0) <= 1.0)
        return {0.0, 1.0 + eps};             // sub-critical: no-op

    auto f = [&](double t) {
        return std::tan(t) * (1.0 + eps * std::cos(m * t))
             - eps * m * std::sin(m * t);
    };
    double lo = 1e-8, hi = M_PI / m;         // f(lo) < 0, f(hi) > 0
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (f(mid) < 0.0) lo = mid; else hi = mid;
    }
    const double thetaM = 0.5 * (lo + hi);
    const double A = (1.0 + eps * std::cos(m * thetaM)) / std::cos(thetaM);
    return {thetaM, A};
}

namespace {

// ---------------------------------------------------------------------------
// Face flux along the normal direction `nrm` given the face-local gradient
// (pn = normal component, pt = tangential component).  Kobayashi convention
// (matches the dendrite solver's facePW functors):
//   J_n = W0²·a·(a·pn + s·pt),   s = ε·m·sin(m(θ−θ0)) = −a'(θ)
// with θ = atan2(φ_y, φ_x) built from (pn, pt) in the correct (x, y) order.
// ---------------------------------------------------------------------------
// cos(m(θ−θ0)) and sin(m(θ−θ0)) WITHOUT transcendentals: rotate the
// gradient by −θ0 (host-precomputed ct0/st0), then take the m-th complex
// power of the unit direction via multiply-add recurrence.  FP64 atan2 +
// sincos run at 1/64 throughput on consumer GPUs and dominated this kernel
// — the algebraic path benchmarked >2× faster end to end.
__host__ __device__ inline
void cosSinM(Real px, Real py, Real ct0, Real st0, int m,
             Real& cosm, Real& sinm)
{
    const Real cx = ct0 * px + st0 * py;    // rotation by −θ0
    const Real cy = ct0 * py - st0 * px;
    const Real p2 = cx * cx + cy * cy;
    if (p2 <= Real(1e-150)) {               // no interface direction (margin
        cosm = Real(0);                     // against px² underflow)
        sinm = Real(0);
        return;
    }
    // Normalise FIRST: powers of the unit direction stay O(1), whereas
    // p2^m under/overflows for extreme-magnitude gradients (0·inf → NaN).
#ifdef __CUDA_ARCH__
    const Real inv = rsqrt(p2);
#else
    const Real inv = Real(1) / std::sqrt(p2);
#endif
    const Real ux = cx * inv, uy = cy * inv;
    Real zr = ux, zi = uy;                  // (ux + i·uy)^m, |z| == 1
    for (int k = 1; k < m; ++k) {
        const Real t = zr * ux - zi * uy;
        zi = zr * uy + zi * ux;
        zr = t;
    }
    cosm = zr;
    sinm = zi;
}

// (a, s = −a') at the face orientation, with the optional Eggleston
// continuation inside the missing-orientation cones (cos(mθ') > cos(mθ_m)).
__host__ __device__ inline
void anisoAS(Real cosm, Real sinm, Real eps, int m,
             Real cosMthm, Real regA, Real& a, Real& s)
{
    if (cosm > cosMthm) {                     // inside a cone (reg enabled)
        const Real delta = atan2(sinm, cosm) / Real(m);
        a = regA * cos(delta);                // γ̃ = A·cosδ (γ̃+γ̃'' ≡ 0)
        s = regA * sin(delta);                // s = −γ̃' = A·sinδ
    } else {
        a = Real(1) + eps * cosm;
        s = eps * Real(m) * sinm;
    }
}

// ---------------------------------------------------------------------------
// Face-normal gradient, scheme-selectable (S = 0: CD2, S = 1: Iso9).
//
// CD2 is the compact two-point difference — the legacy path, bit-identical.
// Iso9 transverse-averages the compact differences with weights
// (1/12, 5/6, 1/12) = [1 + (Δt²/12)δ²_t](f_R − f_L)/Δn.  Assembling the
// face divergence of these fluxes at ε = 0 gives EXACTLY the 9-point
// Patra–Karttunen Laplacian (4·face + corner − 20·center)/(6dx²): the
// fourfold O(Δx²) truncation term of the 5-point form (−(kΔx)²cos4θ/48)
// vanishes identically.  One value per face → telescoping-conservative;
// checkerboard symbol −16/3 (a two-pass NODAL Iso9 composite has symbol 0
// there — zero damping, blows up; that variant is deliberately NOT offered).
// `no` = normal stride, `to` = transverse stride; ghost = 1 suffices.
// ---------------------------------------------------------------------------
template<int S> __host__ __device__ inline
Real pnFaceHigh(const Real* f, int c, int no, int to, Real inv) {
    const Real d0 = f[c + no] - f[c];
    if constexpr (S == 0) {
        return d0 * inv;
    } else {
        const Real dp = f[c + no + to] - f[c + to];
        const Real dm = f[c + no - to] - f[c - to];
        return (Real(5.0 / 6.0) * d0 + Real(1.0 / 12.0) * (dp + dm)) * inv;
    }
}
template<int S> __host__ __device__ inline
Real pnFaceLow(const Real* f, int c, int no, int to, Real inv) {
    const Real d0 = f[c] - f[c - no];
    if constexpr (S == 0) {
        return d0 * inv;
    } else {
        const Real dp = f[c + to] - f[c - no + to];
        const Real dm = f[c - to] - f[c - no - to];
        return (Real(5.0 / 6.0) * d0 + Real(1.0 / 12.0) * (dp + dm)) * inv;
    }
}

// TK2015-style localization selects the isotropic RHS in bulk cells.
// Keep the mask outside the divergence: applying it to faces adds a different
// discrete interface term. The early branch also skips orientation work.
__host__ __device__ inline
bool inBulk(const Real* phase, int c, Real cutoff) {
    return phase && fabs(Real(1) - phase[c]*phase[c]) < cutoff;
}
template<int S> __host__ __device__ inline
Real isotropicDivCell(const Real* f, int c, int sx, Real idx, Real idy, Real W0sq) {
    return W0sq * ((pnFaceHigh<S>(f,c,1,sx,idx)-pnFaceLow<S>(f,c,1,sx,idx))*idx
                 +(pnFaceHigh<S>(f,c,sx,1,idy)-pnFaceLow<S>(f,c,sx,1,idy))*idy);
}

// Cell-centre gradient for the a(θ) factor kernels (S = 0: CD2, S = 1: the
// 9-point isotropic-error nodal derivative [4(f₊−f₋) + diag pairs]/(12Δ)).
// Pointwise output only — the nodal Iso9 caveat above does not apply.
template<int S> __host__ __device__ inline
Real gradCentered(const Real* f, int c, int no, int to, Real inv2) {
    if constexpr (S == 0) {
        return (f[c + no] - f[c - no]) * inv2;
    } else {
        return (Real(4) * (f[c + no] - f[c - no])
                + (f[c + no + to] - f[c - no + to])
                + (f[c + no - to] - f[c - no - to])) * (inv2 / Real(6));
    }
}

__host__ __device__ inline
Real fluxN(Real pn, Real pt, bool nIsX,
           Real W0sq, Real eps, int m, Real ct0, Real st0,
           Real cosMthm, Real regA)
{
    const Real px = nIsX ? pn : pt;
    const Real py = nIsX ? pt : pn;
    Real cosm, sinm;
    cosSinM(px, py, ct0, st0, m, cosm, sinm);
    Real a, s;
    anisoAS(cosm, sinm, eps, m, cosMthm, regA, a, s);
    // x-face: J = W0² a (a·px + s·py);  y-face: J = W0² a (a·py − s·px)
    return nIsX ? W0sq * a * (a * pn + s * pt)
                : W0sq * a * (a * pn - s * pt);
}

// ---------------------------------------------------------------------------
// Fused divergence: rhs[c] += coeff·[ (Jx_e − Jx_w)/dx + (Jy_n − Jy_s)/dy ]
// Every face flux is built from the face-normal difference and the averaged
// tangential central differences — identical inputs from both adjacent
// cells → conservative.
// ---------------------------------------------------------------------------
template<int S>
__global__ void kernel_aniso_div(
        Real* rhs, const Real* f,
        Real coeff,
        int nx, int ny,
        int sx, int sy, int g,
        Real inv_dx, Real inv_dy,
        Real W0sq, Real eps, int m, Real ct0, Real st0,
        Real cosMthm, Real regA, const Real* phase, Real cutoff)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;
    const int i = tid % nx;
    const int j = tid / nx;
    // NOTE: 2D fields are ghost-padded in z as well — the k=0 slice sits at
    // offset sy*g, exactly like cell_idx(i, j, 0) in FaceOps.
    const int c = (i + g) + sx * ((j + g) + sy * g);
    if (inBulk(phase,c,cutoff)) {
        rhs[c] += coeff * isotropicDivCell<S>(f,c,sx,inv_dx,inv_dy,W0sq);
        return;
    }

    const Real q = Real(0.25);

    // west x-face (i−½): normal grad + averaged tangential φ_y
    Real pn = pnFaceLow<S>(f, c, 1, sx, inv_dx);
    Real pt = q * inv_dy * (f[c - 1 + sx] - f[c - 1 - sx]
                            + f[c + sx] - f[c - sx]);
    const Real jw = fluxN(pn, pt, true, W0sq, eps, m, ct0, st0, cosMthm, regA);

    // east x-face (i+½)
    pn = pnFaceHigh<S>(f, c, 1, sx, inv_dx);
    pt = q * inv_dy * (f[c + sx] - f[c - sx]
                       + f[c + 1 + sx] - f[c + 1 - sx]);
    const Real je = fluxN(pn, pt, true, W0sq, eps, m, ct0, st0, cosMthm, regA);

    // south y-face (j−½): normal grad + averaged tangential φ_x
    pn = pnFaceLow<S>(f, c, sx, 1, inv_dy);
    pt = q * inv_dx * (f[c - sx + 1] - f[c - sx - 1]
                       + f[c + 1] - f[c - 1]);
    const Real js = fluxN(pn, pt, false, W0sq, eps, m, ct0, st0, cosMthm, regA);

    // north y-face (j+½)
    pn = pnFaceHigh<S>(f, c, sx, 1, inv_dy);
    pt = q * inv_dx * (f[c + 1] - f[c - 1]
                       + f[c + sx + 1] - f[c + sx - 1]);
    const Real jn = fluxN(pn, pt, false, W0sq, eps, m, ct0, st0, cosMthm, regA);

    rhs[c] += coeff * ((je - jw) * inv_dx + (jn - js) * inv_dy);
}

// CPU mirror of one cell (shared with the Term's cpu_kernel)
template<int S>
inline Real anisoDivCell(const Real* f, int c, int sx,
                         Real inv_dx, Real inv_dy,
                         Real W0sq, Real eps, int m, Real ct0, Real st0,
                         Real cosMthm, Real regA, const Real* phase, Real cutoff)
{
    if (inBulk(phase,c,cutoff)) return isotropicDivCell<S>(f,c,sx,inv_dx,inv_dy,W0sq);
    const Real q = Real(0.25);
    Real pn = pnFaceLow<S>(f, c, 1, sx, inv_dx);
    Real pt = q * inv_dy * (f[c - 1 + sx] - f[c - 1 - sx]
                            + f[c + sx] - f[c - sx]);
    const Real jw = fluxN(pn, pt, true, W0sq, eps, m, ct0, st0, cosMthm, regA);
    pn = pnFaceHigh<S>(f, c, 1, sx, inv_dx);
    pt = q * inv_dy * (f[c + sx] - f[c - sx]
                       + f[c + 1 + sx] - f[c + 1 - sx]);
    const Real je = fluxN(pn, pt, true, W0sq, eps, m, ct0, st0, cosMthm, regA);
    pn = pnFaceLow<S>(f, c, sx, 1, inv_dy);
    pt = q * inv_dx * (f[c - sx + 1] - f[c - sx - 1]
                       + f[c + 1] - f[c - 1]);
    const Real js = fluxN(pn, pt, false, W0sq, eps, m, ct0, st0, cosMthm, regA);
    pn = pnFaceHigh<S>(f, c, sx, 1, inv_dy);
    pt = q * inv_dx * (f[c + 1] - f[c - 1]
                       + f[c + sx + 1] - f[c + sx - 1]);
    const Real jn = fluxN(pn, pt, false, W0sq, eps, m, ct0, st0, cosMthm, regA);
    return (je - jw) * inv_dx + (jn - js) * inv_dy;
}

template<int S>
__global__ void kernel_aniso_factor(
        Real* out, const Real* f,
        int nx, int ny, int sx, int sy, int g,
        Real inv_2dx, Real inv_2dy,
        Real eps, int m, Real ct0, Real st0,
        Real cosMthm, Real regA, const Real* phase, Real cutoff)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;
    const int i = tid % nx;
    const int j = tid / nx;
    const int c = (i + g) + sx * ((j + g) + sy * g);
    if (inBulk(phase,c,cutoff)) { out[c] = Real(1); return; }
    const Real px = gradCentered<S>(f, c, 1, sx, inv_2dx);
    const Real py = gradCentered<S>(f, c, sx, 1, inv_2dy);
    Real cosm, sinm;
    cosSinM(px, py, ct0, st0, m, cosm, sinm);
    Real a, sdum;
    anisoAS(cosm, sinm, eps, m, cosMthm, regA, a, sdum);
    out[c] = a;
}

// ---------------------------------------------------------------------------
// Per-cell-orientation variants (v3.7.0).  A face between two cells takes
// the θ0 of the MORE-SOLID cell (larger φ): the interface's crystal
// structure belongs to the grain growing through that face.  With a uniform
// theta0Field this is the scalar-θ0 operator evaluated with device cos/sin
// of the same angle.
// ---------------------------------------------------------------------------
__device__ __host__ inline Real fluxNTh(Real pn, Real pt, bool nIsX,
        Real th, Real W0sq, Real eps, int m, Real cosMthm, Real regA)
{
    return fluxN(pn, pt, nIsX, W0sq, eps, m,
                 static_cast<Real>(cos(th)), static_cast<Real>(sin(th)),
                 cosMthm, regA);
}

template<int S>
__global__ void kernel_aniso_div_thfield(
        Real* rhs, const Real* f, const Real* th,
        Real coeff,
        int nx, int ny,
        int sx, int sy, int g,
        Real inv_dx, Real inv_dy,
        Real W0sq, Real eps, int m,
        Real cosMthm, Real regA, const Real* phase, Real cutoff)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;
    const int i = tid % nx;
    const int j = tid / nx;
    const int c = (i + g) + sx * ((j + g) + sy * g);
    if (inBulk(phase,c,cutoff)) {
        rhs[c] += coeff * isotropicDivCell<S>(f,c,sx,inv_dx,inv_dy,W0sq);
        return;
    }

    const Real q = Real(0.25);
    // Solid-side orientation per face; equal values choose west/south on both sides.
    const Real thW = (f[c - 1]  >= f[c]) ? th[c - 1]  : th[c];
    const Real thE = (f[c + 1]  >  f[c]) ? th[c + 1]  : th[c];
    const Real thS = (f[c - sx] >= f[c]) ? th[c - sx] : th[c];
    const Real thN = (f[c + sx] >  f[c]) ? th[c + sx] : th[c];

    Real pn = pnFaceLow<S>(f, c, 1, sx, inv_dx);
    Real pt = q * inv_dy * (f[c - 1 + sx] - f[c - 1 - sx]
                            + f[c + sx] - f[c - sx]);
    const Real jw = fluxNTh(pn, pt, true,  thW, W0sq, eps, m, cosMthm, regA);

    pn = pnFaceHigh<S>(f, c, 1, sx, inv_dx);
    pt = q * inv_dy * (f[c + sx] - f[c - sx]
                       + f[c + 1 + sx] - f[c + 1 - sx]);
    const Real je = fluxNTh(pn, pt, true,  thE, W0sq, eps, m, cosMthm, regA);

    pn = pnFaceLow<S>(f, c, sx, 1, inv_dy);
    pt = q * inv_dx * (f[c - sx + 1] - f[c - sx - 1]
                       + f[c + 1] - f[c - 1]);
    const Real js = fluxNTh(pn, pt, false, thS, W0sq, eps, m, cosMthm, regA);

    pn = pnFaceHigh<S>(f, c, sx, 1, inv_dy);
    pt = q * inv_dx * (f[c + 1] - f[c - 1]
                       + f[c + sx + 1] - f[c + sx - 1]);
    const Real jn = fluxNTh(pn, pt, false, thN, W0sq, eps, m, cosMthm, regA);

    rhs[c] += coeff * ((je - jw) * inv_dx + (jn - js) * inv_dy);
}

// CPU mirror of one cell
template<int S>
inline Real anisoDivCellTh(const Real* f, const Real* th, int c, int sx,
                           Real inv_dx, Real inv_dy,
                           Real W0sq, Real eps, int m,
                           Real cosMthm, Real regA, const Real* phase, Real cutoff)
{
    if (inBulk(phase,c,cutoff)) return isotropicDivCell<S>(f,c,sx,inv_dx,inv_dy,W0sq);
    const Real q = Real(0.25);
    const Real thW = (f[c - 1]  >= f[c]) ? th[c - 1]  : th[c];
    const Real thE = (f[c + 1]  >  f[c]) ? th[c + 1]  : th[c];
    const Real thS = (f[c - sx] >= f[c]) ? th[c - sx] : th[c];
    const Real thN = (f[c + sx] >  f[c]) ? th[c + sx] : th[c];
    Real pn = pnFaceLow<S>(f, c, 1, sx, inv_dx);
    Real pt = q * inv_dy * (f[c - 1 + sx] - f[c - 1 - sx]
                            + f[c + sx] - f[c - sx]);
    const Real jw = fluxNTh(pn, pt, true,  thW, W0sq, eps, m, cosMthm, regA);
    pn = pnFaceHigh<S>(f, c, 1, sx, inv_dx);
    pt = q * inv_dy * (f[c + sx] - f[c - sx]
                       + f[c + 1 + sx] - f[c + 1 - sx]);
    const Real je = fluxNTh(pn, pt, true,  thE, W0sq, eps, m, cosMthm, regA);
    pn = pnFaceLow<S>(f, c, sx, 1, inv_dy);
    pt = q * inv_dx * (f[c - sx + 1] - f[c - sx - 1]
                       + f[c + 1] - f[c - 1]);
    const Real js = fluxNTh(pn, pt, false, thS, W0sq, eps, m, cosMthm, regA);
    pn = pnFaceHigh<S>(f, c, sx, 1, inv_dy);
    pt = q * inv_dx * (f[c + 1] - f[c - 1]
                       + f[c + sx + 1] - f[c + sx - 1]);
    const Real jn = fluxNTh(pn, pt, false, thN, W0sq, eps, m, cosMthm, regA);
    return (je - jw) * inv_dx + (jn - js) * inv_dy;
}

template<int S>
__global__ void kernel_aniso_factor_thfield(
        Real* out, const Real* f, const Real* th,
        int nx, int ny, int sx, int sy, int g,
        Real inv_2dx, Real inv_2dy,
        Real eps, int m,
        Real cosMthm, Real regA, const Real* phase, Real cutoff)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;
    const int i = tid % nx;
    const int j = tid / nx;
    const int c = (i + g) + sx * ((j + g) + sy * g);
    if (inBulk(phase,c,cutoff)) { out[c] = Real(1); return; }
    const Real px = gradCentered<S>(f, c, 1, sx, inv_2dx);
    const Real py = gradCentered<S>(f, c, sx, 1, inv_2dy);
    Real cosm, sinm;
    cosSinM(px, py, static_cast<Real>(cos(th[c])),
            static_cast<Real>(sin(th[c])), m, cosm, sinm);
    Real a, sdum;
    anisoAS(cosm, sinm, eps, m, cosMthm, regA, a, sdum);
    out[c] = a;
}

void checkInterface(const ScalarField& input, const AnisoParams& p, bool gpu) {
    if (!p.interfacePhi) return;
    const auto& phase = *p.interfacePhi;
    if (phase.ghost != input.ghost || phase.mesh.dim != input.mesh.dim ||
        phase.storedSize != input.storedSize)
        throw std::invalid_argument("anisotropy interfacePhi: layout mismatch");
    for (int ax=0; ax<3; ++ax)
        if (phase.mesh.n[ax] != input.mesh.n[ax] || phase.mesh.d[ax] != input.mesh.d[ax])
            throw std::invalid_argument("anisotropy interfacePhi: mesh mismatch");
    if (gpu && !phase.d_curr)
        throw std::runtime_error("anisotropy interfacePhi: field not on device");
}

void checkField(const ScalarField& phi, const char* fn) {
    if (phi.mesh.dim != 2)
        throw std::invalid_argument(
            std::string(fn) + ": 2D meshes only (m-fold in-plane anisotropy)");
    if (phi.ghost < 1)
        throw std::invalid_argument(std::string(fn) + ": ghost >= 1 required");
}

} // namespace

Term anisoDiv(const ScalarField& phi, const AnisoParams& p, double coeff) {
    p.validate();
    checkInterface(phi, p, false);
    checkField(phi, "anisoDiv");

    Term t;
    t.type  = TermType::COMPOSITE;
    t.field = &phi;
    t.inputs = {&phi};
    t.coeff = coeff;
    t.ghostRequired = 1;
    describeInputs(t, 1, true);
    if (p.interfacePhi) {
        t.inputs.push_back(p.interfacePhi);
        t.info.read(p.interfacePhi);
    }

    const int  nx = phi.mesh.n[0], ny = phi.mesh.n[1];
    const int  sx = phi.storedDims[0], sy = phi.storedDims[1];
    const int  g  = phi.ghost;
    const Real inv_dx = static_cast<Real>(1.0 / phi.mesh.d[0]);
    const Real inv_dy = static_cast<Real>(1.0 / phi.mesh.d[1]);
    const Real W0sq   = static_cast<Real>(p.W0 * p.W0);
    const Real eps    = static_cast<Real>(p.eps);
    const Real ct0    = static_cast<Real>(std::cos(p.theta0));
    const Real st0    = static_cast<Real>(std::sin(p.theta0));
    const int  m      = p.m;
    // regularisation constants: cos(m·θ_m) as the in-cone trigger; with
    // regularize off (or sub-critical) the trigger is set unreachable (>1)
    const AnisoReg reg = p.regularize
        ? anisoComputeRegularization(p.eps, p.m) : AnisoReg{0.0, 0.0};
    const Real cosMthm = (p.regularize && reg.thetaM > 0.0)
        ? static_cast<Real>(std::cos(m * reg.thetaM)) : Real(2);
    const Real regA    = static_cast<Real>(reg.A);
    const bool iso9    = (p.scheme == AnisoScheme::Iso9);

    const ScalarField* pf = &phi;
    const auto* phase = p.interfacePhi;
    const Real cutoff = static_cast<Real>(p.interfaceCutoff);

    t.gpu_launcher = [pf, nx, ny, sx, sy, g, inv_dx, inv_dy, W0sq, eps, m, ct0, st0,
                      cosMthm, regA, iso9, phase, cutoff]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        if (phase && !phase->d_curr)
            throw std::runtime_error("anisotropy interfacePhi: field not on device");
        if (!pf->d_curr)
            throw std::runtime_error("anisoDiv GPU: field not on device");
        const int total = nx * ny;
        if (iso9)
            kernel_aniso_div<1><<<(total + 255) / 256, 256, 0, pool.stream>>>(
                d_rhs, pf->d_curr, static_cast<Real>(c),
                nx, ny, sx, sy, g, inv_dx, inv_dy, W0sq, eps, m, ct0, st0, cosMthm, regA, phase ? phase->d_curr : nullptr, cutoff);
        else
            kernel_aniso_div<0><<<(total + 255) / 256, 256, 0, pool.stream>>>(
                d_rhs, pf->d_curr, static_cast<Real>(c),
                nx, ny, sx, sy, g, inv_dx, inv_dy, W0sq, eps, m, ct0, st0, cosMthm, regA, phase ? phase->d_curr : nullptr, cutoff);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("anisoDiv kernel error: ")
                + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, sx, sy, g, inv_dx, inv_dy, W0sq, eps, m, ct0, st0,
                    cosMthm, regA, iso9, phase, cutoff]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* f = pf->curr.data();
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const int ctr = (i + g) + sx * ((j + g) + sy * g);
            rhs[ctr] += static_cast<Real>(c)
                      * (iso9
                         ? anisoDivCell<1>(f, ctr, sx, inv_dx, inv_dy,
                                           W0sq, eps, m, ct0, st0, cosMthm, regA, phase ? phase->curr.data() : nullptr, cutoff)
                         : anisoDivCell<0>(f, ctr, sx, inv_dx, inv_dy,
                                           W0sq, eps, m, ct0, st0, cosMthm, regA, phase ? phase->curr.data() : nullptr, cutoff));
        }
    };

    return t;
}

void anisoFactorOnGPU(const ScalarField& phi, ScalarField& aOut,
                      const AnisoParams& p) {
    p.validate();
    checkInterface(phi, p, true);
    checkField(phi, "anisoFactorOnGPU");
    if (aOut.storedSize != phi.storedSize)
        throw std::invalid_argument("anisoFactorOnGPU: layout mismatch");
    if (!phi.d_curr || !aOut.d_curr)
        throw std::runtime_error("anisoFactorOnGPU: fields not on device");

    const int total = phi.mesh.n[0] * phi.mesh.n[1];
    const Real cosMthmF = [&]{
        if (!p.regularize) return Real(2);
        const AnisoReg r = anisoComputeRegularization(p.eps, p.m);
        return (r.thetaM > 0.0)
            ? static_cast<Real>(std::cos(p.m * r.thetaM)) : Real(2); }();
    const Real regAF = static_cast<Real>(p.regularize
        ? anisoComputeRegularization(p.eps, p.m).A : 0.0);
    auto* kern = (p.scheme == AnisoScheme::Iso9)
        ? kernel_aniso_factor<1> : kernel_aniso_factor<0>;
    kern<<<(total + 255) / 256, 256>>>(
        aOut.d_curr, phi.d_curr,
        phi.mesh.n[0], phi.mesh.n[1], phi.storedDims[0], phi.storedDims[1],
        phi.ghost,
        static_cast<Real>(0.5 / phi.mesh.d[0]),
        static_cast<Real>(0.5 / phi.mesh.d[1]),
        static_cast<Real>(p.eps), p.m,
        static_cast<Real>(std::cos(p.theta0)),
        static_cast<Real>(std::sin(p.theta0)),
        cosMthmF, regAF, p.interfacePhi ? p.interfacePhi->d_curr : nullptr,
        static_cast<Real>(p.interfaceCutoff));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("anisoFactor kernel error: ")
                                 + cudaGetErrorString(err));
}

void anisoFactorOnCPU(const ScalarField& phi, ScalarField& aOut,
                      const AnisoParams& p) {
    p.validate();
    checkInterface(phi, p, false);
    checkField(phi, "anisoFactorOnCPU");
    if (aOut.storedSize != phi.storedSize)
        throw std::invalid_argument("anisoFactorOnCPU: layout mismatch");

    const Real* f = phi.curr.data();
    const int nx = phi.mesh.n[0], ny = phi.mesh.n[1];
    const int sx = phi.storedDims[0], sy = phi.storedDims[1];
    const int g = phi.ghost;
    const Real i2dx = static_cast<Real>(0.5 / phi.mesh.d[0]);
    const Real i2dy = static_cast<Real>(0.5 / phi.mesh.d[1]);
    const AnisoReg regC = p.regularize
        ? anisoComputeRegularization(p.eps, p.m) : AnisoReg{0.0, 0.0};
    const Real cosMthmC = (p.regularize && regC.thetaM > 0.0)
        ? static_cast<Real>(std::cos(p.m * regC.thetaM)) : Real(2);
    const Real regAC = static_cast<Real>(regC.A);
    const bool iso9 = (p.scheme == AnisoScheme::Iso9);
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const int c = (i + g) + sx * ((j + g) + sy * g);
        if (inBulk(p.interfacePhi ? p.interfacePhi->curr.data() : nullptr,
                   c, static_cast<Real>(p.interfaceCutoff))) {
            aOut.curr[c] = Real(1);
            continue;
        }
        const Real px = iso9 ? gradCentered<1>(f, c, 1, sx, i2dx)
                             : gradCentered<0>(f, c, 1, sx, i2dx);
        const Real py = iso9 ? gradCentered<1>(f, c, sx, 1, i2dy)
                             : gradCentered<0>(f, c, sx, 1, i2dy);
        Real cosm, sinm;
        cosSinM(px, py, static_cast<Real>(std::cos(p.theta0)),
                static_cast<Real>(std::sin(p.theta0)), p.m, cosm, sinm);
        Real a, sdum;
        anisoAS(cosm, sinm, static_cast<Real>(p.eps), p.m, cosMthmC, regAC,
                a, sdum);
        aOut.curr[static_cast<std::size_t>(c)] = a;
    }
}

// ===========================================================================
// Per-cell orientation variants (v3.7.0)
// ===========================================================================

Term anisoDiv(const ScalarField& phi, const ScalarField& theta0Field,
              const AnisoParams& p, double coeff) {
    p.validate();
    checkInterface(phi, p, false);
    checkField(phi, "anisoDiv(theta0Field)");
    if (theta0Field.storedSize != phi.storedSize
        || theta0Field.ghost != phi.ghost)
        throw std::invalid_argument(
            "anisoDiv(theta0Field): orientation field must share phi's "
            "mesh and ghost layout");

    Term t;
    t.type  = TermType::COMPOSITE;
    t.field = &phi;
    t.inputs = {&phi, &theta0Field};
    t.coeff = coeff;
    t.ghostRequired = 1;
    describeInputs(t, 1, true);
    if (p.interfacePhi) {
        t.inputs.push_back(p.interfacePhi);
        t.info.read(p.interfacePhi);
    }

    const int  nx = phi.mesh.n[0], ny = phi.mesh.n[1];
    const int  sx = phi.storedDims[0], sy = phi.storedDims[1];
    const int  g  = phi.ghost;
    const Real inv_dx = static_cast<Real>(1.0 / phi.mesh.d[0]);
    const Real inv_dy = static_cast<Real>(1.0 / phi.mesh.d[1]);
    const Real W0sq   = static_cast<Real>(p.W0 * p.W0);
    const Real eps    = static_cast<Real>(p.eps);
    const int  m      = p.m;
    const AnisoReg reg = p.regularize
        ? anisoComputeRegularization(p.eps, p.m) : AnisoReg{0.0, 0.0};
    const Real cosMthm = (p.regularize && reg.thetaM > 0.0)
        ? static_cast<Real>(std::cos(m * reg.thetaM)) : Real(2);
    const Real regA    = static_cast<Real>(reg.A);
    const bool iso9    = (p.scheme == AnisoScheme::Iso9);

    const ScalarField* pf  = &phi;
    const auto* phase = p.interfacePhi;
    const Real cutoff = static_cast<Real>(p.interfaceCutoff);
    const ScalarField* pth = &theta0Field;

    t.gpu_launcher = [pf, pth, nx, ny, sx, sy, g, inv_dx, inv_dy,
                      W0sq, eps, m, cosMthm, regA, iso9, phase, cutoff]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        if (phase && !phase->d_curr)
            throw std::runtime_error("anisotropy interfacePhi: field not on device");
        if (!pf->d_curr || !pth->d_curr)
            throw std::runtime_error(
                "anisoDiv(theta0Field) GPU: a field not on device");
        const int total = nx * ny;
        if (iso9)
            kernel_aniso_div_thfield<1><<<(total + 255) / 256, 256, 0,
                                          pool.stream>>>(
                d_rhs, pf->d_curr, pth->d_curr, static_cast<Real>(c),
                nx, ny, sx, sy, g, inv_dx, inv_dy, W0sq, eps, m, cosMthm, regA, phase ? phase->d_curr : nullptr, cutoff);
        else
            kernel_aniso_div_thfield<0><<<(total + 255) / 256, 256, 0,
                                          pool.stream>>>(
                d_rhs, pf->d_curr, pth->d_curr, static_cast<Real>(c),
                nx, ny, sx, sy, g, inv_dx, inv_dy, W0sq, eps, m, cosMthm, regA, phase ? phase->d_curr : nullptr, cutoff);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("anisoDiv(theta0Field) kernel error: ")
                + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, pth, nx, ny, sx, sy, g, inv_dx, inv_dy,
                    W0sq, eps, m, cosMthm, regA, iso9, phase, cutoff]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* f  = pf->curr.data();
        const Real* th = pth->curr.data();
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const int ctr = (i + g) + sx * ((j + g) + sy * g);
            rhs[ctr] += static_cast<Real>(c)
                      * (iso9
                         ? anisoDivCellTh<1>(f, th, ctr, sx, inv_dx, inv_dy,
                                             W0sq, eps, m, cosMthm, regA, phase ? phase->curr.data() : nullptr, cutoff)
                         : anisoDivCellTh<0>(f, th, ctr, sx, inv_dx, inv_dy,
                                             W0sq, eps, m, cosMthm, regA, phase ? phase->curr.data() : nullptr, cutoff));
        }
    };

    return t;
}

void anisoFactorOnGPU(const ScalarField& phi, const ScalarField& theta0Field,
                      ScalarField& aOut, const AnisoParams& p) {
    p.validate();
    checkInterface(phi, p, true);
    checkField(phi, "anisoFactorOnGPU(theta0Field)");
    if (aOut.storedSize != phi.storedSize
        || theta0Field.storedSize != phi.storedSize)
        throw std::invalid_argument(
            "anisoFactorOnGPU(theta0Field): layout mismatch");
    if (!phi.d_curr || !aOut.d_curr || !theta0Field.d_curr)
        throw std::runtime_error(
            "anisoFactorOnGPU(theta0Field): fields not on device");

    const AnisoReg reg = p.regularize
        ? anisoComputeRegularization(p.eps, p.m) : AnisoReg{0.0, 0.0};
    const int total = phi.mesh.n[0] * phi.mesh.n[1];
    auto* kernTh = (p.scheme == AnisoScheme::Iso9)
        ? kernel_aniso_factor_thfield<1> : kernel_aniso_factor_thfield<0>;
    kernTh<<<(total + 255) / 256, 256>>>(
        aOut.d_curr, phi.d_curr, theta0Field.d_curr,
        phi.mesh.n[0], phi.mesh.n[1], phi.storedDims[0], phi.storedDims[1],
        phi.ghost,
        static_cast<Real>(0.5 / phi.mesh.d[0]),
        static_cast<Real>(0.5 / phi.mesh.d[1]),
        static_cast<Real>(p.eps), p.m,
        (p.regularize && reg.thetaM > 0.0)
            ? static_cast<Real>(std::cos(p.m * reg.thetaM)) : Real(2),
        static_cast<Real>(p.regularize ? reg.A : 0.0),
        p.interfacePhi ? p.interfacePhi->d_curr : nullptr, static_cast<Real>(p.interfaceCutoff));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("anisoFactor(theta0Field) kernel error: ")
            + cudaGetErrorString(err));
}

void anisoFactorOnCPU(const ScalarField& phi, const ScalarField& theta0Field,
                      ScalarField& aOut, const AnisoParams& p) {
    p.validate();
    checkInterface(phi, p, false);
    checkField(phi, "anisoFactorOnCPU(theta0Field)");
    if (aOut.storedSize != phi.storedSize
        || theta0Field.storedSize != phi.storedSize)
        throw std::invalid_argument(
            "anisoFactorOnCPU(theta0Field): layout mismatch");

    const AnisoReg reg = p.regularize
        ? anisoComputeRegularization(p.eps, p.m) : AnisoReg{0.0, 0.0};
    const Real cosMthm = (p.regularize && reg.thetaM > 0.0)
        ? static_cast<Real>(std::cos(p.m * reg.thetaM)) : Real(2);
    const Real regA = static_cast<Real>(p.regularize ? reg.A : 0.0);
    const int  nx = phi.mesh.n[0], ny = phi.mesh.n[1];
    const int  sx = phi.storedDims[0], sy = phi.storedDims[1];
    const int  g  = phi.ghost;
    const Real i2dx = static_cast<Real>(0.5 / phi.mesh.d[0]);
    const Real i2dy = static_cast<Real>(0.5 / phi.mesh.d[1]);
    const Real* f  = phi.curr.data();
    const Real* th = theta0Field.curr.data();
    const bool iso9 = (p.scheme == AnisoScheme::Iso9);
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const int c = (i + g) + sx * ((j + g) + sy * g);
        if (inBulk(p.interfacePhi ? p.interfacePhi->curr.data() : nullptr,
                   c, static_cast<Real>(p.interfaceCutoff))) {
            aOut.curr[c] = Real(1);
            continue;
        }
        const Real px = iso9 ? gradCentered<1>(f, c, 1, sx, i2dx)
                             : gradCentered<0>(f, c, 1, sx, i2dx);
        const Real py = iso9 ? gradCentered<1>(f, c, sx, 1, i2dy)
                             : gradCentered<0>(f, c, sx, 1, i2dy);
        Real cosm, sinm;
        cosSinM(px, py, static_cast<Real>(std::cos(th[c])),
                static_cast<Real>(std::sin(th[c])), p.m, cosm, sinm);
        Real a, sdum;
        anisoAS(cosm, sinm, static_cast<Real>(p.eps), p.m, cosMthm, regA,
                a, sdum);
        aOut.curr[c] = a;
    }
}

// ===========================================================================
// 3D cubic anisotropy (Karma–Rappel), axis-aligned crystal axes.
// ===========================================================================

void Aniso3DParams::setEulerZXZ(double phi1, double Phi, double phi2) {
    const double c1 = std::cos(phi1), s1 = std::sin(phi1);
    const double cF = std::cos(Phi),  sF = std::sin(Phi);
    const double c2 = std::cos(phi2), s2 = std::sin(phi2);
    // Bunge g-matrix (lab → crystal): g = Rz(φ2)·Rx(Φ)·Rz(φ1)
    R[0] =  c1 * c2 - s1 * s2 * cF;
    R[1] =  s1 * c2 + c1 * s2 * cF;
    R[2] =  s2 * sF;
    R[3] = -c1 * s2 - s1 * c2 * cF;
    R[4] = -s1 * s2 + c1 * c2 * cF;
    R[5] =  c2 * sF;
    R[6] =  s1 * sF;
    R[7] = -c1 * sF;
    R[8] =  cF;
}

void Aniso3DParams::validate() const {
    if (W0 <= 0.0)
        throw std::invalid_argument("Aniso3DParams: W0 must be > 0");
    if (eps < 0.0 || eps >= 0.3)
        throw std::invalid_argument(
            "Aniso3DParams: eps must be in [0, 0.3) — the 2D Eggleston "
            "regularisation does not yet extend to 3D");
    // R must be orthonormal (RᵀR = I)
    double devI = 0.0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            double dot = 0.0;
            for (int k = 0; k < 3; ++k)
                dot += R[3 * k + a] * R[3 * k + b];
            devI = std::max(devI, std::fabs(dot - (a == b ? 1.0 : 0.0)));
        }
    if (devI > 1e-8)
        throw std::invalid_argument(
            "Aniso3DParams: R is not orthonormal (use setEulerZXZ or a "
            "proper rotation matrix)");
}

namespace {

// Rotation carrier (passed to kernels by value): r maps lab → crystal.
struct Rot9 { Real r[9]; };

// Face flux along the normal slot `nIdx` (0=x,1=y,2=z) given the face-local
// LAB gradient (px, py, pz).  Evaluated in the crystal frame and rotated
// back:  J_lab = W0²·a·[a·p_lab + Rᵀ·v_c],  v_c,i = 16ε·p_c,i·(n_c,i² − S).
__host__ __device__ inline
Real flux3(Real px, Real py, Real pz, int nIdx, Real W0sq, Real eps,
           const Rot9& q)
{
    const Real pn = (nIdx == 0) ? px : (nIdx == 1) ? py : pz;
    const Real p2 = px * px + py * py + pz * pz;   // rotation-invariant
    if (p2 <= Real(1e-150)) return W0sq * pn;      // margin vs p² underflow

    const Real pcx = q.r[0] * px + q.r[1] * py + q.r[2] * pz;
    const Real pcy = q.r[3] * px + q.r[4] * py + q.r[5] * pz;
    const Real pcz = q.r[6] * px + q.r[7] * py + q.r[8] * pz;

    // Normalised direction cosines squared FIRST — S = Σ(n_i²)² stays O(1)
    // (a raw Σp⁴/|p|⁴ under/overflows for extreme gradients → 0·inf NaN).
    const Real invp2 = Real(1) / p2;
    const Real nx2 = pcx * pcx * invp2;
    const Real ny2 = pcy * pcy * invp2;
    const Real nz2 = pcz * pcz * invp2;
    const Real S = nx2 * nx2 + ny2 * ny2 + nz2 * nz2;
    const Real a = Real(1) - Real(3) * eps + Real(4) * eps * S;

    const Real vcx = Real(16) * eps * pcx * (nx2 - S);
    const Real vcy = Real(16) * eps * pcy * (ny2 - S);
    const Real vcz = Real(16) * eps * pcz * (nz2 - S);
    // (Rᵀ v_c) component nIdx = Σ_k R[k][nIdx]·v_k
    const Real vn = q.r[0 + nIdx] * vcx + q.r[3 + nIdx] * vcy
                  + q.r[6 + nIdx] * vcz;

    return W0sq * a * (a * pn + vn);
}

// Face-local gradient at the face between cells L and R along `ax`:
// normal = 2-point difference; each tangential = averaged central diffs of
// the two adjacent cells (identical inputs from both sides → conservative).
__host__ __device__ inline
Real aniso3dDivCell(const Real* f, int c, int sx, int sz,
                    Real inv_dx, Real inv_dy, Real inv_dz,
                    Real W0sq, Real eps, const Rot9& rot)
{
    const Real q = Real(0.25);
    const int st[3] = {1, sx, sz};
    const Real invd[3] = {inv_dx, inv_dy, inv_dz};

    Real div = Real(0);
    for (int ax = 0; ax < 3; ++ax) {
        const int  sn  = st[ax];
        const int  t1  = (ax == 0) ? 1 : 0;          // first tangential axis
        const int  t2  = (ax == 2) ? 1 : 2;          // second tangential axis
        const int  s1  = st[t1], s2 = st[t2];
        const Real id1 = invd[t1], id2 = invd[t2];

        Real p[2][3];                                // [west|east][x,y,z]
        for (int side = 0; side < 2; ++side) {
            const int R = c + side * sn;             // right cell of the face
            const int L = R - sn;                    // left cell
            const Real pn  = (f[R] - f[L]) * invd[ax];
            const Real pt1 = q * id1 * (f[L + s1] - f[L - s1]
                                        + f[R + s1] - f[R - s1]);
            const Real pt2 = q * id2 * (f[L + s2] - f[L - s2]
                                        + f[R + s2] - f[R - s2]);
            p[side][ax] = pn;
            p[side][t1] = pt1;
            p[side][t2] = pt2;
        }
        const Real jw = flux3(p[0][0], p[0][1], p[0][2], ax, W0sq, eps, rot);
        const Real je = flux3(p[1][0], p[1][1], p[1][2], ax, W0sq, eps, rot);
        div += (je - jw) * invd[ax];
    }
    return div;
}

__global__ void kernel_aniso3d_div(
        Real* rhs, const Real* f,
        Real coeff,
        int nx, int ny, int nz,
        int sx, int sy, int g,
        Real inv_dx, Real inv_dy, Real inv_dz,
        Real W0sq, Real eps, Rot9 q)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;
    const int i = tid % nx;
    const int j = (tid / nx) % ny;
    const int k = tid / (nx * ny);
    const int c = (i + g) + sx * ((j + g) + sy * (k + g));
    rhs[c] += coeff * aniso3dDivCell(f, c, sx, sx * sy,
                                     inv_dx, inv_dy, inv_dz, W0sq, eps, q);
}

__global__ void kernel_aniso3d_factor(
        Real* out, const Real* f,
        int nx, int ny, int nz,
        int sx, int sy, int g,
        Real i2dx, Real i2dy, Real i2dz, Real eps, Rot9 q)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;
    const int i = tid % nx;
    const int j = (tid / nx) % ny;
    const int k = tid / (nx * ny);
    const int c = (i + g) + sx * ((j + g) + sy * (k + g));
    const int sz = sx * sy;
    const Real px = (f[c + 1] - f[c - 1]) * i2dx;
    const Real py = (f[c + sx] - f[c - sx]) * i2dy;
    const Real pz = (f[c + sz] - f[c - sz]) * i2dz;
    out[c] = aniso::factor3D(q.r[0]*px + q.r[1]*py + q.r[2]*pz,
                             q.r[3]*px + q.r[4]*py + q.r[5]*pz,
                             q.r[6]*px + q.r[7]*py + q.r[8]*pz, eps);
}

void checkField3D(const ScalarField& phi, const char* fn) {
    if (phi.mesh.dim != 3)
        throw std::invalid_argument(
            std::string(fn) + ": 3D meshes only (use anisoDiv for 2D)");
    if (phi.ghost < 1)
        throw std::invalid_argument(std::string(fn) + ": ghost >= 1 required");
}

} // namespace

Term anisoDiv3D(const ScalarField& phi, const Aniso3DParams& p, double coeff) {
    p.validate();
    checkField3D(phi, "anisoDiv3D");

    Term t;
    t.type  = TermType::COMPOSITE;
    t.field = &phi;
    t.inputs = {&phi};
    t.coeff = coeff;
    t.ghostRequired = 1;

    const int  nx = phi.mesh.n[0], ny = phi.mesh.n[1], nz = phi.mesh.n[2];
    const int  sx = phi.storedDims[0], sy = phi.storedDims[1];
    const int  g  = phi.ghost;
    const Real idx = static_cast<Real>(1.0 / phi.mesh.d[0]);
    const Real idy = static_cast<Real>(1.0 / phi.mesh.d[1]);
    const Real idz = static_cast<Real>(1.0 / phi.mesh.d[2]);
    const Real W0sq = static_cast<Real>(p.W0 * p.W0);
    const Real eps  = static_cast<Real>(p.eps);
    Rot9 q;
    for (int r9 = 0; r9 < 9; ++r9) q.r[r9] = static_cast<Real>(p.R[r9]);

    const ScalarField* pf = &phi;

    t.gpu_launcher = [pf, nx, ny, nz, sx, sy, g, idx, idy, idz, W0sq, eps, q]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        if (!pf->d_curr)
            throw std::runtime_error("anisoDiv3D GPU: field not on device");
        const int total = nx * ny * nz;
        kernel_aniso3d_div<<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, pf->d_curr, static_cast<Real>(c),
            nx, ny, nz, sx, sy, g, idx, idy, idz, W0sq, eps, q);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("anisoDiv3D kernel error: ")
                + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, nz, sx, sy, g, idx, idy, idz, W0sq, eps, q]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* f = pf->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const int ctr = (i + g) + sx * ((j + g) + sy * (k + g));
            rhs[ctr] += static_cast<Real>(c)
                      * aniso3dDivCell(f, ctr, sx, sx * sy,
                                       idx, idy, idz, W0sq, eps, q);
        }
    };

    return t;
}

void anisoFactor3DOnGPU(const ScalarField& phi, ScalarField& aOut,
                        const Aniso3DParams& p) {
    p.validate();
    checkField3D(phi, "anisoFactor3DOnGPU");
    if (aOut.storedSize != phi.storedSize)
        throw std::invalid_argument("anisoFactor3DOnGPU: layout mismatch");
    if (!phi.d_curr || !aOut.d_curr)
        throw std::runtime_error("anisoFactor3DOnGPU: fields not on device");

    const int total = phi.mesh.n[0] * phi.mesh.n[1] * phi.mesh.n[2];
    kernel_aniso3d_factor<<<(total + 255) / 256, 256>>>(
        aOut.d_curr, phi.d_curr,
        phi.mesh.n[0], phi.mesh.n[1], phi.mesh.n[2],
        phi.storedDims[0], phi.storedDims[1], phi.ghost,
        static_cast<Real>(0.5 / phi.mesh.d[0]),
        static_cast<Real>(0.5 / phi.mesh.d[1]),
        static_cast<Real>(0.5 / phi.mesh.d[2]),
        static_cast<Real>(p.eps),
        [&]{ Rot9 qq;
             for (int r9 = 0; r9 < 9; ++r9)
                 qq.r[r9] = static_cast<Real>(p.R[r9]);
             return qq; }());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("anisoFactor3D kernel error: ")
                                 + cudaGetErrorString(err));
}

void anisoFactor3DOnCPU(const ScalarField& phi, ScalarField& aOut,
                        const Aniso3DParams& p) {
    p.validate();
    checkField3D(phi, "anisoFactor3DOnCPU");
    if (aOut.storedSize != phi.storedSize)
        throw std::invalid_argument("anisoFactor3DOnCPU: layout mismatch");

    const Real* f = phi.curr.data();
    const int nx = phi.mesh.n[0], ny = phi.mesh.n[1], nz = phi.mesh.n[2];
    const int sx = phi.storedDims[0], sy = phi.storedDims[1];
    const int g = phi.ghost, sz = sx * sy;
    const Real i2dx = static_cast<Real>(0.5 / phi.mesh.d[0]);
    const Real i2dy = static_cast<Real>(0.5 / phi.mesh.d[1]);
    const Real i2dz = static_cast<Real>(0.5 / phi.mesh.d[2]);
    Rot9 q;
    for (int r9 = 0; r9 < 9; ++r9) q.r[r9] = static_cast<Real>(p.R[r9]);
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const int c = (i + g) + sx * ((j + g) + sy * (k + g));
        const Real px = (f[c + 1] - f[c - 1]) * i2dx;
        const Real py = (f[c + sx] - f[c - sx]) * i2dy;
        const Real pz = (f[c + sz] - f[c - sz]) * i2dz;
        aOut.curr[static_cast<std::size_t>(c)] =
            aniso::factor3D(q.r[0]*px + q.r[1]*py + q.r[2]*pz,
                            q.r[3]*px + q.r[4]*py + q.r[5]*pz,
                            q.r[6]*px + q.r[7]*py + q.r[8]*pz,
                            static_cast<Real>(p.eps));
    }
}

} // namespace PhiX
