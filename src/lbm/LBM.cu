#include "lbm/LBM.h"
#include "core/CudaCheck.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace PhiX {

void LBMParams::validate() const {
    if (tau <= 0.5)
        throw std::invalid_argument("LBMParams: tau must be > 0.5 "
                                    "(nu = (tau-1/2)/3 must be positive)");
}

namespace {

// D2Q9 lattice: index, velocity (cx, cy), weight, opposite index
//   6 2 5
//   3 0 1
//   7 4 8
__device__ const int CX[9]  = { 0, 1, 0,-1, 0, 1,-1,-1, 1};
__device__ const int CY[9]  = { 0, 0, 1, 0,-1, 1, 1,-1,-1};
__device__ const int OPP[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6};

__host__ __device__ inline Real wq(int i) {
    return (i == 0) ? Real(4.0 / 9.0)
         : (i < 5)  ? Real(1.0 / 9.0)
                    : Real(1.0 / 36.0);
}

__host__ __device__ inline
Real feq(int i, Real rho, Real ux, Real uy, int cx, int cy) {
    const Real cu = Real(3) * (cx * ux + cy * uy);           // c·u / c_s²
    const Real u2 = Real(1.5) * (ux * ux + uy * uy);         // u² / (2c_s²)
    return wq(i) * rho * (Real(1) + cu + Real(0.5) * cu * cu - u2);
}

// ---------------------------------------------------------------------------
// Equilibrium initialisation
// ---------------------------------------------------------------------------
__global__ void kernel_lbm_init(Real* f, int n,
                                Real rho0, Real ux0, Real uy0)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    for (int i = 0; i < 9; ++i)
        f[i * n + idx] = feq(i, rho0, ux0, uy0, CX[i], CY[i]);
}

// ---------------------------------------------------------------------------
// BGK collision + Guo forcing (in place)
// ---------------------------------------------------------------------------
__global__ void kernel_lbm_collide(Real* f, int nx, int ny,
                                   Real invTau, Real fx, Real fy)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = nx * ny;
    if (idx >= n) return;

    Real fi[9], rho = Real(0), mx = Real(0), my = Real(0);
    for (int i = 0; i < 9; ++i) {
        fi[i] = f[i * n + idx];
        rho += fi[i];
        mx  += fi[i] * CX[i];
        my  += fi[i] * CY[i];
    }
    const Real invRho = Real(1) / rho;
    const Real ux = (mx + Real(0.5) * fx) * invRho;   // Guo velocity shift
    const Real uy = (my + Real(0.5) * fy) * invRho;

    const Real fw = Real(1) - Real(0.5) * invTau;     // (1 − 1/(2τ))
    for (int i = 0; i < 9; ++i) {
        const Real cu = Real(3) * (CX[i] * ux + CY[i] * uy);
        // Guo source: w_i [3(c−u) + 9(c·u)c] · F   (c_s² = 1/3)
        const Real Si = wq(i) * (Real(3) * ((CX[i] - ux) * fx
                                            + (CY[i] - uy) * fy)
                                 + Real(3) * cu * (CX[i] * fx + CY[i] * fy));
        f[i * n + idx] = fi[i]
            - invTau * (fi[i] - feq(i, rho, ux, uy, CX[i], CY[i]))
            + fw * Si;
    }
}

// ---------------------------------------------------------------------------
// Pull streaming with per-side periodic / halfway bounce-back.
// f_new[i](x) = f_post[i](x − c_i); pulling across a WALL side instead
// takes the opposite direction from the cell itself (halfway BB).
// ---------------------------------------------------------------------------
__global__ void kernel_lbm_stream(Real* fnew, const Real* f,
                                  const unsigned char* solid,
                                  int nx, int ny,
                                  int wxlo, int wxhi, int wylo, int wyhi)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = nx * ny;
    if (idx >= n) return;
    const int x = idx % nx;
    const int y = idx / nx;

    if (solid && solid[idx]) {              // solid cells stay frozen
        for (int i = 0; i < 9; ++i) fnew[i * n + idx] = f[i * n + idx];
        return;
    }

    // Non-periodic sides (walls/inlet/outflow) stream as bounce-back here;
    // inlet/outflow layers are overwritten by their BC kernels afterwards.
    for (int i = 0; i < 9; ++i) {
        int sx = x - CX[i];
        int sy = y - CY[i];
        bool bb = false;
        if (sx < 0)        { if (wxlo) bb = true; else sx += nx; }
        else if (sx >= nx) { if (wxhi) bb = true; else sx -= nx; }
        if (sy < 0)        { if (wylo) bb = true; else sy += ny; }
        else if (sy >= ny) { if (wyhi) bb = true; else sy -= ny; }

        if (!bb && solid && solid[sx + nx * sy]) bb = true;   // obstacle BB

        fnew[i * n + idx] = bb ? f[OPP[i] * n + idx]
                               : f[i * n + (sx + nx * sy)];
    }
}

// ---------------------------------------------------------------------------
// Zou-He velocity inlet on one straight side (2D, all four sides).
// prof: (uNormal, uTangential) pairs per boundary cell; uNormal > 0 = into
// the domain on either side.
// ---------------------------------------------------------------------------
__global__ void kernel_lbm_inlet(Real* f, const Real* prof,
                                 int nx, int ny, int axis, int side)
{
    const int nT = (axis == 0) ? ny : nx;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nT) return;
    const int n = nx * ny;

    const int x = (axis == 0) ? (side == 0 ? 0 : nx - 1) : t;
    const int y = (axis == 0) ? t : (side == 0 ? 0 : ny - 1);
    const int c = x + nx * y;

    const Real uN = prof[2 * t];        // into the domain
    const Real uT = prof[2 * t + 1];

    Real F[9];
    for (int i = 0; i < 9; ++i) F[i] = f[i * n + c];

    if (axis == 0 && side == 0) {           // x-low: ux=+uN, uy=uT
        const Real rho = (F[0]+F[2]+F[4] + Real(2)*(F[3]+F[6]+F[7]))
                       / (Real(1) - uN);
        F[1] = F[3] + Real(2.0/3.0)*rho*uN;
        F[5] = F[7] - Real(0.5)*(F[2]-F[4]) + Real(1.0/6.0)*rho*uN
             + Real(0.5)*rho*uT;
        F[8] = F[6] + Real(0.5)*(F[2]-F[4]) + Real(1.0/6.0)*rho*uN
             - Real(0.5)*rho*uT;
    } else if (axis == 0 && side == 1) {    // x-high: ux=−uN, uy=uT
        const Real rho = (F[0]+F[2]+F[4] + Real(2)*(F[1]+F[5]+F[8]))
                       / (Real(1) - uN);
        F[3] = F[1] - Real(2.0/3.0)*rho*uN;
        F[7] = F[5] + Real(0.5)*(F[2]-F[4]) - Real(1.0/6.0)*rho*uN
             - Real(0.5)*rho*uT;
        F[6] = F[8] - Real(0.5)*(F[2]-F[4]) - Real(1.0/6.0)*rho*uN
             + Real(0.5)*rho*uT;
    } else if (axis == 1 && side == 0) {    // y-low: uy=+uN, ux=uT
        const Real rho = (F[0]+F[1]+F[3] + Real(2)*(F[4]+F[7]+F[8]))
                       / (Real(1) - uN);
        F[2] = F[4] + Real(2.0/3.0)*rho*uN;
        F[5] = F[7] - Real(0.5)*(F[1]-F[3]) + Real(1.0/6.0)*rho*uN
             + Real(0.5)*rho*uT;
        F[6] = F[8] + Real(0.5)*(F[1]-F[3]) + Real(1.0/6.0)*rho*uN
             - Real(0.5)*rho*uT;
    } else {                                // y-high: uy=−uN, ux=uT
        const Real rho = (F[0]+F[1]+F[3] + Real(2)*(F[2]+F[5]+F[6]))
                       / (Real(1) - uN);
        F[4] = F[2] - Real(2.0/3.0)*rho*uN;
        F[7] = F[5] - Real(0.5)*(F[1]-F[3]) - Real(1.0/6.0)*rho*uN
             - Real(0.5)*rho*uT;
        F[8] = F[6] + Real(0.5)*(F[1]-F[3]) - Real(1.0/6.0)*rho*uN
             + Real(0.5)*rho*uT;
    }
    // knowns are written back unchanged, unknowns carry the Zou-He values
    for (int i = 0; i < 9; ++i) f[i * n + c] = F[i];
}

// ---------------------------------------------------------------------------
// Zou-He pressure outlet: density anchored at rho0, normal velocity floats,
// tangential velocity taken as zero.  (A pure zero-gradient copy does not
// fix the pressure level — paired with a velocity inlet the total mass
// integrates without bound.)
// ---------------------------------------------------------------------------
__global__ void kernel_lbm_outflow(Real* f, int nx, int ny,
                                   int axis, int side, Real rho0)
{
    const int nT = (axis == 0) ? ny : nx;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nT) return;
    const int n = nx * ny;

    const int x = (axis == 0) ? (side == 0 ? 0 : nx - 1) : t;
    const int y = (axis == 0) ? t : (side == 0 ? 0 : ny - 1);
    const int c = x + nx * y;

    Real F[9];
    for (int i = 0; i < 9; ++i) F[i] = f[i * n + c];

    if (axis == 0 && side == 1) {           // x-high outlet
        const Real ux = Real(-1)
            + (F[0]+F[2]+F[4] + Real(2)*(F[1]+F[5]+F[8])) / rho0;
        F[3] = F[1] - Real(2.0/3.0)*rho0*ux;
        F[7] = F[5] + Real(0.5)*(F[2]-F[4]) - Real(1.0/6.0)*rho0*ux;
        F[6] = F[8] - Real(0.5)*(F[2]-F[4]) - Real(1.0/6.0)*rho0*ux;
    } else if (axis == 0 && side == 0) {    // x-low outlet
        const Real ux = Real(1)
            - (F[0]+F[2]+F[4] + Real(2)*(F[3]+F[6]+F[7])) / rho0;
        F[1] = F[3] + Real(2.0/3.0)*rho0*ux;
        F[5] = F[7] - Real(0.5)*(F[2]-F[4]) + Real(1.0/6.0)*rho0*ux;
        F[8] = F[6] + Real(0.5)*(F[2]-F[4]) + Real(1.0/6.0)*rho0*ux;
    } else if (axis == 1 && side == 1) {    // y-high outlet
        const Real uy = Real(-1)
            + (F[0]+F[1]+F[3] + Real(2)*(F[2]+F[5]+F[6])) / rho0;
        F[4] = F[2] - Real(2.0/3.0)*rho0*uy;
        F[7] = F[5] - Real(0.5)*(F[1]-F[3]) - Real(1.0/6.0)*rho0*uy;
        F[8] = F[6] + Real(0.5)*(F[1]-F[3]) - Real(1.0/6.0)*rho0*uy;
    } else {                                // y-low outlet
        const Real uy = Real(1)
            - (F[0]+F[1]+F[3] + Real(2)*(F[4]+F[7]+F[8])) / rho0;
        F[2] = F[4] + Real(2.0/3.0)*rho0*uy;
        F[5] = F[7] - Real(0.5)*(F[1]-F[3]) + Real(1.0/6.0)*rho0*uy;
        F[6] = F[8] + Real(0.5)*(F[1]-F[3]) + Real(1.0/6.0)*rho0*uy;
    }
    for (int i = 0; i < 9; ++i) f[i * n + c] = F[i];
}

// ---------------------------------------------------------------------------
// Macroscopic export into ghost-padded ScalarFields (physical cells)
// ---------------------------------------------------------------------------
__global__ void kernel_lbm_macro(const Real* f, const unsigned char* solid,
                                 int nx, int ny,
                                 Real fx, Real fy,
                                 Real* rho_out, Real* ux_out, Real* uy_out,
                                 int sx, int sy, int g)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = nx * ny;
    if (idx >= n) return;
    const int x = idx % nx;
    const int y = idx / nx;
    const int c = (x + g) + sx * ((y + g) + sy * g);

    if (solid && solid[idx]) {
        if (rho_out) rho_out[c] = Real(1);
        if (ux_out)  ux_out[c]  = Real(0);
        if (uy_out)  uy_out[c]  = Real(0);
        return;
    }

    Real rho = Real(0), mx = Real(0), my = Real(0);
    for (int i = 0; i < 9; ++i) {
        const Real v = f[i * n + idx];
        rho += v;
        mx  += v * CX[i];
        my  += v * CY[i];
    }
    const Real invRho = Real(1) / rho;
    if (rho_out) rho_out[c] = rho;
    if (ux_out)  ux_out[c]  = (mx + Real(0.5) * fx) * invRho;
    if (uy_out)  uy_out[c]  = (my + Real(0.5) * fy) * invRho;
}

} // namespace

// ===========================================================================
// LBM2D
// ===========================================================================

LBM2D::LBM2D(const Mesh& mesh, const LBMParams& p)
    : mesh_(mesh)
    , tau_(p.tau)
    , fx_(static_cast<Real>(p.fx))
    , fy_(static_cast<Real>(p.fy))
    , nx_(mesh.n[0])
    , ny_(mesh.n[1])
{
    p.validate();
    if (mesh.dim != 2)
        throw std::invalid_argument("LBM2D: 2D meshes only (D2Q9)");
    const std::size_t bytes =
        9u * static_cast<std::size_t>(nx_) * ny_ * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&d_f_, bytes));
    CUDA_CHECK(cudaMalloc(&d_ftmp_, bytes));
}

LBM2D::~LBM2D() {
    if (d_f_)    cudaFree(d_f_);      // best-effort; no throw in dtor
    if (d_ftmp_) cudaFree(d_ftmp_);
    if (d_mask_) cudaFree(d_mask_);
    for (auto& row : d_inlet_)
        for (auto* p : row)
            if (p) cudaFree(p);
}

void LBM2D::setWall(Axis axis, Side side) {
    if (axis == Axis::Z)
        throw std::invalid_argument("LBM2D::setWall: 2D lattice has no Z");
    sideType_[static_cast<int>(axis)][side == Side::LOW ? 0 : 1] = 1;
}

void LBM2D::setVelocityInlet(Axis axis, Side side,
                             const std::vector<double>& uNormal,
                             const std::vector<double>& uTangential) {
    if (axis == Axis::Z)
        throw std::invalid_argument("LBM2D::setVelocityInlet: no Z in 2D");
    const int a = static_cast<int>(axis);
    const int sIdx = (side == Side::LOW) ? 0 : 1;
    const int nT = (a == 0) ? ny_ : nx_;
    if (static_cast<int>(uNormal.size()) != nT)
        throw std::invalid_argument(
            "LBM2D::setVelocityInlet: profile length "
            + std::to_string(uNormal.size()) + " != tangential extent "
            + std::to_string(nT));
    if (!uTangential.empty()
        && static_cast<int>(uTangential.size()) != nT)
        throw std::invalid_argument(
            "LBM2D::setVelocityInlet: uTangential length mismatch");

    std::vector<Real> prof(2 * static_cast<std::size_t>(nT));
    for (int t = 0; t < nT; ++t) {
        prof[2 * t]     = static_cast<Real>(uNormal[static_cast<std::size_t>(t)]);
        prof[2 * t + 1] = uTangential.empty()
            ? Real(0)
            : static_cast<Real>(uTangential[static_cast<std::size_t>(t)]);
    }
    if (!d_inlet_[a][sIdx])
        CUDA_CHECK(cudaMalloc(&d_inlet_[a][sIdx], 2 * nT * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_inlet_[a][sIdx], prof.data(),
                          2 * nT * sizeof(Real), cudaMemcpyHostToDevice));
    sideType_[a][sIdx] = 2;
}

void LBM2D::setOutflow(Axis axis, Side side, double rho0) {
    if (axis == Axis::Z)
        throw std::invalid_argument("LBM2D::setOutflow: no Z in 2D");
    const int a = static_cast<int>(axis);
    const int sIdx = (side == Side::LOW) ? 0 : 1;
    sideType_[a][sIdx] = 3;
    outletRho_[a][sIdx] = rho0;
}

void LBM2D::setObstacleMask(const ScalarField& mask) {
    if (mask.mesh.n[0] != nx_ || mask.mesh.n[1] != ny_)
        throw std::invalid_argument(
            "LBM2D::setObstacleMask: mask mesh differs from lattice");
    const int n = nx_ * ny_;
    std::vector<unsigned char> h(static_cast<std::size_t>(n));
    const int g = mask.ghost, sx = mask.storedDims[0], sy = mask.storedDims[1];
    for (int y = 0; y < ny_; ++y)
    for (int x = 0; x < nx_; ++x)
        h[static_cast<std::size_t>(x + nx_ * y)] =
            (mask.curr[static_cast<std::size_t>(
                 (x + g) + sx * ((y + g) + sy * g))] >= Real(0.5)) ? 1 : 0;
    if (!d_mask_)
        CUDA_CHECK(cudaMalloc(&d_mask_, n));
    CUDA_CHECK(cudaMemcpy(d_mask_, h.data(), n, cudaMemcpyHostToDevice));
}

void LBM2D::applyBoundaryKernels_() {
    for (int a = 0; a < 2; ++a)
        for (int sIdx = 0; sIdx < 2; ++sIdx) {
            const int nT = (a == 0) ? ny_ : nx_;
            if (sideType_[a][sIdx] == 2)
                kernel_lbm_inlet<<<(nT + 255) / 256, 256>>>(
                    d_f_, d_inlet_[a][sIdx], nx_, ny_, a, sIdx);
            else if (sideType_[a][sIdx] == 3)
                kernel_lbm_outflow<<<(nT + 255) / 256, 256>>>(
                    d_f_, nx_, ny_, a, sIdx,
                    static_cast<Real>(outletRho_[a][sIdx]));
        }
    CUDA_CHECK(cudaGetLastError());
}

void LBM2D::initialize(double rho0, double ux0, double uy0) {
    const int n = nx_ * ny_;
    kernel_lbm_init<<<(n + 255) / 256, 256>>>(
        d_f_, n, static_cast<Real>(rho0),
        static_cast<Real>(ux0), static_cast<Real>(uy0));
    CUDA_CHECK(cudaGetLastError());
    step_ = 0;
}

void LBM2D::step() {
    const int n = nx_ * ny_;
    kernel_lbm_collide<<<(n + 255) / 256, 256>>>(
        d_f_, nx_, ny_, static_cast<Real>(1.0 / tau_), fx_, fy_);
    CUDA_CHECK(cudaGetLastError());
    kernel_lbm_stream<<<(n + 255) / 256, 256>>>(
        d_ftmp_, d_f_, d_mask_, nx_, ny_,
        sideType_[0][0] != 0, sideType_[0][1] != 0,
        sideType_[1][0] != 0, sideType_[1][1] != 0);
    CUDA_CHECK(cudaGetLastError());
    Real* t = d_f_; d_f_ = d_ftmp_; d_ftmp_ = t;   // buffer swap, no copy
    applyBoundaryKernels_();
    ++step_;
}

void LBM2D::run(int nSteps) {
    for (int s = 0; s < nSteps; ++s) step();
}

void LBM2D::macroscopics(ScalarField* rho, ScalarField* ux, ScalarField* uy) {
    const ScalarField* ref = rho ? rho : (ux ? ux : uy);
    if (!ref) return;
    for (const ScalarField* fld : {static_cast<const ScalarField*>(rho),
                                   static_cast<const ScalarField*>(ux),
                                   static_cast<const ScalarField*>(uy)}) {
        if (!fld) continue;
        if (fld->mesh.n[0] != nx_ || fld->mesh.n[1] != ny_)
            throw std::invalid_argument(
                "LBM2D::macroscopics: field mesh differs from lattice");
        if (!fld->d_curr)
            throw std::runtime_error(
                "LBM2D::macroscopics: field not device-allocated");
    }
    const int n = nx_ * ny_;
    kernel_lbm_macro<<<(n + 255) / 256, 256>>>(
        d_f_, d_mask_, nx_, ny_, fx_, fy_,
        rho ? rho->d_curr : nullptr,
        ux  ? ux->d_curr  : nullptr,
        uy  ? uy->d_curr  : nullptr,
        ref->storedDims[0], ref->storedDims[1], ref->ghost);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace PhiX
