#include "solver/LinearSolver.h"
#include "solver/Preconditioner.h"
#include "core/CudaCheck.h"
#include "field/Reduce.h"
#include "scheme/CentralDifference.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace PhiX {

namespace {

// ---------------------------------------------------------------------------
// y = D·∇²x over physical cells (overwrite; CD2)
// ---------------------------------------------------------------------------
__global__ void kernel_lap_apply(
        Real* y, const Real* x,
        int nx, int ny, int nz,
        int sx, int sy, int g, int dim,
        Real D, Real shift, Real inv_dx2, Real inv_dy2, Real inv_dz2)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;
    const int i = tid % nx;
    const int j = (tid / nx) % ny;
    const int k = tid / (nx * ny);
    const int c = (i + g) + sx * ((j + g) + sy * (k + g));
    y[c] = D * scheme::CD2::laplacian(x, c, sx, sy, dim,
                                      inv_dx2, inv_dy2, inv_dz2)
         - shift * x[c];
}

// ---------------------------------------------------------------------------
// Elementwise CG helpers (whole stored array — scratch fields are
// zero-initialised so ghost slots stay finite)
// ---------------------------------------------------------------------------

// r = b − alpha·x + sigma·Lx   (initial residual for A = α·I − σ·L)
__global__ void kernel_residual(Real* r, const Real* b, const Real* x,
                                const Real* Lx, Real alpha, Real sigma,
                                int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - alpha * x[i] + sigma * Lx[i];
}

// Ap = alpha·p − sigma·Lp     (both read from device slots so a captured
//                              graph survives per-step coefficient changes)
__global__ void kernel_form_A(Real* Lp, const Real* p, const double* alpha,
                              const double* sigma, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        Lp[i] = static_cast<Real>(*alpha) * p[i]
              - static_cast<Real>(*sigma) * Lp[i];
}

// Write a host value into a device scalar slot (async, no pinned staging)
__global__ void kernel_set_scalar(double* slot, double v) { *slot = v; }

// x += alpha·p ;  r −= alpha·Ap        (alpha read from a device scalar)
__global__ void kernel_update_xr(Real* x, Real* r, const Real* p,
                                 const Real* Ap, const double* alpha, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const Real a = static_cast<Real>(*alpha);
        x[i] += a * p[i];
        r[i] -= a * Ap[i];
    }
}

// p = r + beta·p                        (beta read from a device scalar)
__global__ void kernel_update_p(Real* p, const Real* r, const double* beta,
                                int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i] + static_cast<Real>(*beta) * p[i];
}

// p = r  (first search direction)
__global__ void kernel_copy_p(Real* p, const Real* r, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i];
}

// Device scalar slots: s[0]=rho, s[1]=rhoNew, s[2]=pAp, s[3]=alpha, s[4]=beta
__global__ void kernel_cg_alpha(double* s)
{
    s[3] = s[0] / s[2];                 // alpha = rho / <p, Ap>
}

__global__ void kernel_cg_beta(double* s)
{
    s[4] = s[1] / s[0];                 // beta = rhoNew / rho
    s[0] = s[1];                        // rho <- rhoNew
}

ScalarField makeScratch(const Mesh& mesh, int ghost, const char* name) {
    ScalarField f(mesh, name, ghost);
    f.allocDevice();
    CUDA_CHECK(cudaMemset(f.d_curr, 0, f.storedSize * sizeof(Real)));
    return f;
}

inline int blocks(std::size_t n) { return static_cast<int>((n + 255) / 256); }

} // namespace

// ===========================================================================
// LaplacianOp
// ===========================================================================

LaplacianOp::LaplacianOp(double D, std::vector<BoundaryCondition*> bcs,
                         double shift)
    : D_(D), shift_(shift), bcs_(std::move(bcs))
{
    if (shift_ < 0.0)
        throw std::invalid_argument(
            "LaplacianOp: stabilisation shift must be >= 0 (SPD)");
}

void LaplacianOp::apply(ScalarField& x, ScalarField& y, cudaStream_t stream) {
    if (!x.d_curr || !y.d_curr)
        throw std::runtime_error("LaplacianOp::apply: fields not on device");
    if (!bcBatch_.built()) bcBatch_.build(x, bcs_);
    bcBatch_.applyOnGPU(x, stream);

    const Mesh& m = x.mesh;
    const int dim = m.dim;
    const Real inv_dx2 = static_cast<Real>(1.0 / (m.d[0] * m.d[0]));
    const Real inv_dy2 = (dim >= 2)
        ? static_cast<Real>(1.0 / (m.d[1] * m.d[1])) : Real(0);
    const Real inv_dz2 = (dim >= 3)
        ? static_cast<Real>(1.0 / (m.d[2] * m.d[2])) : Real(0);

    const int total = m.n[0] * m.n[1] * m.n[2];
    kernel_lap_apply<<<blocks(total), 256, 0, stream>>>(
        y.d_curr, x.d_curr, m.n[0], m.n[1], m.n[2],
        x.storedDims[0], x.storedDims[1], x.ghost, dim,
        static_cast<Real>(D_), static_cast<Real>(shift_),
        inv_dx2, inv_dy2, inv_dz2);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// VarCoeffLaplacianOp
// ===========================================================================

namespace {

// y = ∇·(a ∇x) over physical cells; x read with halo indexing (ghosts fresh),
// face arrays indexed by physical (i,j).
__global__ void kernel_varlap_apply(
        Real* y, const Real* x, const double* ax, const double* ay,
        int nx, int ny, int sx, int g, int planeOff,
        Real inv_dx2, Real inv_dy2)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;
    const int i = tid % nx;
    const int j = tid / nx;
    const int c = planeOff + (i + g) + sx * (j + g);
    const double xc = static_cast<double>(x[c]);
    const double xW = x[c - 1], xE = x[c + 1];
    const double xS = x[c - sx], xN = x[c + sx];
    const double axW = ax[i     + (nx + 1) * j];
    const double axE = ax[i + 1 + (nx + 1) * j];
    const double ayS = ay[i + nx * j];
    const double ayN = ay[i + nx * (j + 1)];
    y[c] = static_cast<Real>(
        (axE * (xE - xc) - axW * (xc - xW)) * static_cast<double>(inv_dx2)
      + (ayN * (xN - xc) - ayS * (xc - xS)) * static_cast<double>(inv_dy2));
}

} // namespace

VarCoeffLaplacianOp::VarCoeffLaplacianOp(const Mesh& mesh, const double* d_ax,
                                         const double* d_ay,
                                         std::vector<BoundaryCondition*> bcs)
    : ax_(d_ax), ay_(d_ay), bcs_(std::move(bcs))
{
    if (mesh.dim != 2)
        throw std::invalid_argument(
            "VarCoeffLaplacianOp: 2D only (dim == 2)");
    if (!d_ax || !d_ay)
        throw std::invalid_argument(
            "VarCoeffLaplacianOp: face-coefficient arrays must be non-null");
}

void VarCoeffLaplacianOp::apply(ScalarField& x, ScalarField& y,
                                cudaStream_t stream) {
    if (!x.d_curr || !y.d_curr)
        throw std::runtime_error(
            "VarCoeffLaplacianOp::apply: fields not on device");
    if (!bcBatch_.built()) bcBatch_.build(x, bcs_);
    bcBatch_.applyOnGPU(x, stream);

    const Mesh& m = x.mesh;
    const Real inv_dx2 = static_cast<Real>(1.0 / (m.d[0] * m.d[0]));
    const Real inv_dy2 = static_cast<Real>(1.0 / (m.d[1] * m.d[1]));
    const int total = m.n[0] * m.n[1];
    const int planeOff = x.storedDims[0] * x.storedDims[1] * x.ghost;
    kernel_varlap_apply<<<blocks(total), 256, 0, stream>>>(
        y.d_curr, x.d_curr, ax_, ay_, m.n[0], m.n[1],
        x.storedDims[0], x.ghost, planeOff, inv_dx2, inv_dy2);
    CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// BiharmonicOp
// ===========================================================================

BiharmonicOp::BiharmonicOp(double G,
                           std::vector<BoundaryCondition*> bcsX,
                           std::vector<BoundaryCondition*> bcsLap)
    : G_(G)
    , bcsX_(std::move(bcsX))
    , bcsLap_(std::move(bcsLap))
    , inner_(1.0, bcsX_)     // persistent: BC batches built once, reused
    , outer_(-G, bcsLap_)
{}

void BiharmonicOp::apply(ScalarField& x, ScalarField& y, cudaStream_t stream) {
    if (!lap_) {
        lap_ = std::make_unique<ScalarField>(x.mesh, "_bih_lap", x.ghost);
        lap_->allocDevice();
        CUDA_CHECK(cudaMemset(lap_->d_curr, 0,
                              lap_->storedSize * sizeof(Real)));
    }
    inner_.apply(x, *lap_, stream);     // lap = ∇²x
    outer_.apply(*lap_, y, stream);     // y = −G·∇²(lap)
}

// ===========================================================================
// ConjugateGradient
// ===========================================================================

ConjugateGradient::ConjugateGradient(const Mesh& mesh, int ghost)
    : r_(makeScratch(mesh, ghost, "_cg_r"))
    , p_(makeScratch(mesh, ghost, "_cg_p"))
    , Lp_(makeScratch(mesh, ghost, "_cg_Lp"))
{
    CUDA_CHECK(cudaMalloc(&d_s_, 8 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_s_, 0, 8 * sizeof(double)));
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

ConjugateGradient::~ConjugateGradient() {
    destroyGraph_();
    if (stream_) cudaStreamDestroy(stream_);
    if (d_s_) cudaFree(d_s_);   // best-effort; no throw in dtor
}

void ConjugateGradient::destroyGraph_() {
    if (graphExec_) {
        cudaGraphExecDestroy(graphExec_);
        graphExec_ = nullptr;
    }
    keyX_ = keyB_ = keyL_ = nullptr;
    keyBurst_ = 0;
}

// One (P)CG iteration, enqueued on `stream` with zero host round trips.
// M == nullptr: the plain-CG sequence, kernel-for-kernel as before (rho is
// r·r).  M != nullptr: classical PCG — rho is <r, z>, the search direction
// updates from z = M⁻¹r, and the TRUE residual r·r lands in slot 7 on the
// last iteration of each check burst (withRR) for the convergence readback.
void ConjugateGradient::enqueueIteration(LinearOperator& L, Preconditioner* M,
                                         ScalarField& x, cudaStream_t stream,
                                         int n, bool withRR)
{
    // Ap = p − σ·Lp   (in place in Lp_)
    L.apply(p_, Lp_, stream);
    kernel_form_A<<<blocks(n), 256, 0, stream>>>(Lp_.d_curr, p_.d_curr,
                                                 d_s_ + 6, d_s_ + 5, n);
    CUDA_CHECK(cudaGetLastError());

    reduce::fieldDotAsync(p_, Lp_, d_s_ + 2, stream);          // pAp
    kernel_cg_alpha<<<1, 1, 0, stream>>>(d_s_);                // α = ρ/pAp
    kernel_update_xr<<<blocks(n), 256, 0, stream>>>(
        x.d_curr, r_.d_curr, p_.d_curr, Lp_.d_curr, d_s_ + 3, n);
    CUDA_CHECK(cudaGetLastError());

    if (M) {
        M->apply(r_, *z_, d_s_ + 6, d_s_ + 5, stream);         // z = M⁻¹r
        reduce::fieldDotAsync(r_, *z_, d_s_ + 1, stream);      // rhoNew = <r,z>
        if (withRR)
            reduce::fieldDotAsync(r_, r_, d_s_ + 7, stream);   // true ‖r‖²
        kernel_cg_beta<<<1, 1, 0, stream>>>(d_s_);             // β, ρ←ρNew
        kernel_update_p<<<blocks(n), 256, 0, stream>>>(
            p_.d_curr, z_->d_curr, d_s_ + 4, n);               // p = z + βp
    } else {
        reduce::fieldDotAsync(r_, r_, d_s_ + 1, stream);       // rhoNew
        kernel_cg_beta<<<1, 1, 0, stream>>>(d_s_);             // β, ρ←ρNew
        kernel_update_p<<<blocks(n), 256, 0, stream>>>(
            p_.d_curr, r_.d_curr, d_s_ + 4, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

ConjugateGradient::Result ConjugateGradient::solveOperator(
        LinearOperator& L, double alpha, double sigma,
        ScalarField& x, const ScalarField& b,
        double relTol, int maxIter, bool throwOnFail, Preconditioner* M)
{
    if (!x.d_curr || !b.d_curr)
        throw std::runtime_error("ConjugateGradient::solve: x/b not on device");
    if (x.storedSize != r_.storedSize || b.storedSize != r_.storedSize)
        throw std::invalid_argument(
            "ConjugateGradient::solve: field layout differs from scratch");
    if (x.ghost < L.ghostRequired())
        throw std::invalid_argument(
            "ConjugateGradient::solve: operator needs ghost >= "
            + std::to_string(L.ghostRequired()));
    if (M) {
        if (r_.ghost < M->ghostRequired())
            throw std::invalid_argument(
                "ConjugateGradient::solve: preconditioner needs ghost >= "
                + std::to_string(M->ghostRequired()));
        if (!z_)
            z_ = std::make_unique<ScalarField>(
                makeScratch(r_.mesh, r_.ghost, "_cg_z"));
    }

    const int  n  = static_cast<int>(r_.storedSize);
    const Real sg = static_cast<Real>(sigma);

    const double bNorm = std::sqrt(reduce::fieldDot(b, b));
    if (bNorm == 0.0) {
        // A x = 0 with SPD A → x = 0
        CUDA_CHECK(cudaMemset(x.d_curr, 0, x.storedSize * sizeof(Real)));
        return {0, 0.0, true};
    }

    // σ and α into their device slots (graph kernels read via pointer)
    kernel_set_scalar<<<1, 1, 0, stream_>>>(d_s_ + 5, sigma);
    kernel_set_scalar<<<1, 1, 0, stream_>>>(d_s_ + 6, alpha);
    CUDA_CHECK(cudaGetLastError());

    // r = b − A x = b − x + σ·Lx ;  p = r
    L.apply(x, Lp_, stream_);
    kernel_residual<<<blocks(n), 256, 0, stream_>>>(
        r_.d_curr, b.d_curr, x.d_curr, Lp_.d_curr,
        static_cast<Real>(alpha), sg, n);
    CUDA_CHECK(cudaGetLastError());
    kernel_copy_p<<<blocks(n), 256, 0, stream_>>>(p_.d_curr, r_.d_curr, n);
    CUDA_CHECK(cudaGetLastError());

    // Initial rho — one synchronous readback (also catches x0 == solution).
    // NOTE: fieldDot runs on the default stream; order against stream_ by
    // draining it first (single cheap sync at solve start).
    // PCG: rho = <r, M⁻¹r> drives the recurrence, the TRUE ‖r‖² (slot 7)
    // drives convergence; p starts from z, not r.
    if (M) {
        M->apply(r_, *z_, d_s_ + 6, d_s_ + 5, stream_);
        kernel_copy_p<<<blocks(n), 256, 0, stream_>>>(p_.d_curr,
                                                      z_->d_curr, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    double rr  = reduce::fieldDot(r_, r_);
    double rho = M ? reduce::fieldDot(r_, *z_) : rr;
    CUDA_CHECK(cudaMemcpy(d_s_, &rho, sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_ + 7, &rr, sizeof(double),
                          cudaMemcpyHostToDevice));

    Result res;
    res.relResidual = std::sqrt(rr) / bNorm;
    if (res.relResidual <= relTol) {
        res.converged = true;
        return res;
    }

    const int cadence = (checkEvery > 0) ? checkEvery : 1;

    // --- graph path: capture one cadence-burst once, then replay ---------
    const bool graphOK = useGraph && L.streamSafe()
                       && (!M || M->streamSafe());
    if (graphOK && (graphExec_ == nullptr || keyX_ != x.d_curr
                    || keyB_ != b.d_curr || keyL_ != &L
                    || keyM_ != static_cast<const void*>(M)
                    || keyBurst_ != cadence)) {
        destroyGraph_();
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(stream_,
                                          cudaStreamCaptureModeThreadLocal));
        for (int k = 0; k < cadence; ++k)
            enqueueIteration(L, M, x, stream_, n,
                             M != nullptr && k == cadence - 1);
        CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph, nullptr,
                                        nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        keyX_ = x.d_curr;
        keyB_ = b.d_curr;
        keyL_ = &L;
        keyM_ = M;
        keyBurst_ = cadence;
    }

    int it = 0;
    while (it < maxIter) {
        const int burst = std::min(cadence, maxIter - it);
        if (graphOK && graphExec_ && burst == cadence) {
            CUDA_CHECK(cudaGraphLaunch(graphExec_, stream_));
        } else {
            for (int k = 0; k < burst; ++k)
                enqueueIteration(L, M, x, stream_, n,
                                 M != nullptr && k == burst - 1);
        }
        it += burst;

        // --- convergence checkpoint: single synchronous readback of the
        //     TRUE residual (slot 0 = rho = r·r for CG; slot 7 for PCG) ---
        CUDA_CHECK(cudaMemcpyAsync(&rr, M ? d_s_ + 7 : d_s_, sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (!std::isfinite(rr))
            throw std::runtime_error(
                "ConjugateGradient::solve: residual became non-finite — "
                "operator is likely not SPD (check BCs / sign of sigma·L)");
        res.iterations  = it;
        res.relResidual = std::sqrt(rr) / bNorm;
        if (res.relResidual <= relTol) {
            res.converged = true;
            return res;
        }
    }

    if (throwOnFail)
        throw std::runtime_error(
            "ConjugateGradient::solve: no convergence after "
            + std::to_string(maxIter) + " iterations (relResidual = "
            + std::to_string(res.relResidual) + ")");
    return res;
}

// ===========================================================================
// PoissonSolver
// ===========================================================================

namespace {

// out = in − mean   over physical cells (mean passed via device slot? host)
__global__ void kernel_sub_const(Real* dst, const Real* src, Real mean,
                                 int nx, int ny, int nz,
                                 int sx, int sy, int g)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;
    const int i = tid % nx;
    const int j = (tid / nx) % ny;
    const int k = tid / (nx * ny);
    const int c = (i + g) + sx * ((j + g) + sy * (k + g));
    dst[c] = src[c] - mean;
}

} // namespace

PoissonSolver::PoissonSolver(const Mesh& mesh, int ghost, double D,
                             std::vector<BoundaryCondition*> bcs)
    : L_(D, std::move(bcs))
    , cg_(mesh, ghost)
    , b_(makeScratch(mesh, ghost, "_poisson_b"))
    , cellCount_(static_cast<double>(mesh.n[0]) * mesh.n[1] * mesh.n[2])
{}

ConjugateGradient::Result PoissonSolver::solve(
        ScalarField& phi, const ScalarField& rhs,
        double relTol, int maxIter, bool throwOnFail)
{
    if (!phi.d_curr || !rhs.d_curr)
        throw std::runtime_error("PoissonSolver::solve: fields not on device");

    const Mesh& m = phi.mesh;
    const int total = m.n[0] * m.n[1] * m.n[2];

    Real meanB = Real(0);
    if (projectNullspace)
        meanB = static_cast<Real>(reduce::fieldSum(rhs) / cellCount_);
    kernel_sub_const<<<blocks(total), 256>>>(
        b_.d_curr, rhs.d_curr, meanB,
        m.n[0], m.n[1], m.n[2],
        phi.storedDims[0], phi.storedDims[1], phi.ghost);
    CUDA_CHECK(cudaGetLastError());

    // A = 0·I − 1·(D∇²) = −D∇²  (SPD on the mean-zero subspace)
    auto res = cg_.solveOperator(L_, 0.0, 1.0, phi, b_,
                                 relTol, maxIter, throwOnFail);

    if (projectNullspace) {
        const Real meanX = static_cast<Real>(
            reduce::fieldSum(phi) / cellCount_);
        kernel_sub_const<<<blocks(total), 256>>>(
            phi.d_curr, phi.d_curr, meanX,
            m.n[0], m.n[1], m.n[2],
            phi.storedDims[0], phi.storedDims[1], phi.ghost);
        CUDA_CHECK(cudaGetLastError());
    }
    return res;
}

} // namespace PhiX
