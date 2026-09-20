#include "equation/Equation.h"
#include "scheme/SchemeCatalog.h"
#include "scheme/Schemes.h"
#include "core/Check.h"
#include "core/CudaCheck.h"
#include "equation/EvalPlan.h"
#include "equation/Expr.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"
#include "equation/Term.h"
#include "equation/TermPW.inl"
#include "equation/FieldOps.inl"
#include "boundary/BoundaryCondition.h"
#include "boundary/BCBatch.h"
#include "boundary/Apply2D.h"
#include "operators/Gradient.h"
#include "scheme/Isotropic.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace PhiX {

// ===========================================================================
// GPU kernels for built-in differential operators
// ===========================================================================

// ---------------------------------------------------------------------------
// Laplacian accumulate:
//   rhs[i,j,k] += coeff * (d²f/dx² [+ d²f/dy²] [+ d²f/dz²])
//
// 2nd-order central FD:
//   d²f/dx² ≈ (f[i+1,j,k] - 2f[i,j,k] + f[i-1,j,k]) / dx²
//
// Ghost cells of `src` must be valid before calling (BCs applied by Solver).
// Only physical cells are written in `rhs`.
// ---------------------------------------------------------------------------
__global__ void kernel_lap_accumulate(
        Real*       rhs,
        const Real* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,          // storedDims[0], storedDims[1]
        int ghost, int dim,
        double inv_dx2, double inv_dy2, double inv_dz2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int is = i + ghost;
    int js = j + ghost;
    int ks = k + ghost;

    // Flat stored indices for stencil neighbours
    int c  = is       + sx * (js       + sy * ks);
    int xm = (is - 1) + sx * (js       + sy * ks);
    int xp = (is + 1) + sx * (js       + sy * ks);

    double val = (src[xp] - 2.0 * src[c] + src[xm]) * inv_dx2;

    if (dim >= 2) {
        int ym = is + sx * ((js - 1) + sy * ks);
        int yp = is + sx * ((js + 1) + sy * ks);
        val += (src[yp] - 2.0 * src[c] + src[ym]) * inv_dy2;
    }

    if (dim >= 3) {
        int zm = is + sx * (js + sy * (ks - 1));
        int zp = is + sx * (js + sy * (ks + 1));
        val += (src[zp] - 2.0 * src[c] + src[zm]) * inv_dz2;
    }

    rhs[c] += coeff * val;
}

// ---------------------------------------------------------------------------
// Gradient accumulate (one component):
//   rhs[i,j,k] += coeff * df/dx_axis
//
// 2nd-order central FD:
//   df/dx ≈ (f[i+1] - f[i-1]) / (2*dx)
// ---------------------------------------------------------------------------
__global__ void kernel_grad_accumulate(
        Real*       rhs,
        const Real* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int axis,
        double inv_2d)   // 1 / (2 * d[axis])
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int is = i + ghost;
    int js = j + ghost;
    int ks = k + ghost;

    int c   = is + sx * (js + sy * ks);
    int fwd, bwd;

    if (axis == 0) {
        fwd = (is + 1) + sx * (js + sy * ks);
        bwd = (is - 1) + sx * (js + sy * ks);
    } else if (axis == 1) {
        fwd = is + sx * ((js + 1) + sy * ks);
        bwd = is + sx * ((js - 1) + sy * ks);
    } else {
        fwd = is + sx * (js + sy * (ks + 1));
        bwd = is + sx * (js + sy * (ks - 1));
    }

    rhs[c] += coeff * (src[fwd] - src[bwd]) * inv_2d;
}

// Iso9 gradient kernel — used by iso_grad(Term/RHSExpr, ...) overload.
__global__ void kernel_iso9_grad_accumulate(
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

    rhs[c] += coeff * scheme::Iso9::gradient(src, c, axis, sx, sy, dim,
                                              inv_dx, inv_dy, inv_dz);
}

// ===========================================================================
// Scalar lap/grad base factories have moved to operators/* modules.
// This file keeps expression-based overloads (lap(expr,...), grad(expr,...)),
// isotropic operators, and Equation runtime logic.
// ===========================================================================

// ---------------------------------------------------------------------------
// iso_grad — 9-point isotropic gradient
// Delegates to grad<scheme::Iso9> (implemented in operators/Gradient.cu).
// ---------------------------------------------------------------------------
Term iso_grad(const ScalarField& f, int axis, double coeff) {
    if (axis < 0 || axis >= f.mesh.dim)
        throw std::invalid_argument("iso_grad: axis out of range for this mesh dimension");
    if (f.mesh.dim != 2 || axis > 1)
        warnOnce("iso_grad-fallback",
                 "iso_grad: the 9-point isotropic stencil is 2D-only (axis 0/1); "
                 "falling back to plain CD2 grad for field '" + f.name + "'");
    return grad<scheme::Iso9>(f, axis, coeff);
}

// ===========================================================================
// kernel_grad_dot_accumulate  --  rhs[idx] += coeff * (∇f · ∇g)
//
// Computes the pointwise dot product of the gradients of two scalar fields
// using 2nd-order central FD on all active axes:
//   result = Σ_a  (df/dx_a) * (dg/dx_a)
// Both fields must share the same mesh, storedDims, and ghost width.
// ===========================================================================
__global__ void kernel_grad_dot_accumulate(
        Real*       rhs,
        const Real* src_f,
        const Real* src_g,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int dim,
        double inv_2dx, double inv_2dy, double inv_2dz)
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

    double val = 0.0;

    // x-axis
    {
        int fwd = (is+1) + sx*(js + sy*ks);
        int bwd = (is-1) + sx*(js + sy*ks);
        val += (src_f[fwd] - src_f[bwd]) * inv_2dx
             * (src_g[fwd] - src_g[bwd]) * inv_2dx;
    }
    if (dim >= 2) {
        int fwd = is + sx*((js+1) + sy*ks);
        int bwd = is + sx*((js-1) + sy*ks);
        val += (src_f[fwd] - src_f[bwd]) * inv_2dy
             * (src_g[fwd] - src_g[bwd]) * inv_2dy;
    }
    if (dim >= 3) {
        int fwd = is + sx*(js + sy*(ks+1));
        int bwd = is + sx*(js + sy*(ks-1));
        val += (src_f[fwd] - src_f[bwd]) * inv_2dz
             * (src_g[fwd] - src_g[bwd]) * inv_2dz;
    }

    rhs[c] += coeff * val;
}

// grad_dot(f, g, coeff) — ∇f · ∇g, sum over all active axes.
Term grad_dot(const ScalarField& f, const ScalarField& g, double coeff) {
    check::checkCartesian(f.mesh, "grad_dot");
    check::checkSameMesh(f, g, "grad_dot");
    check::checkGhost(f, 1, "grad_dot");

    Term t;
    t.type  = TermType::COMPOSITE;
    t.field = &f;
    t.inputs = {&f, &g};
    describeInputs(t, 1);
    t.coeff = coeff;
    t.ghostRequired = 1;   // CD2 neighbours — lets validateTermGhosts catch ghost-0 fields

    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    gh = f.ghost;
    int    dim = f.mesh.dim;
    double inv_2dx = 0.5 / f.mesh.d[0];
    double inv_2dy = (dim >= 2) ? 0.5 / f.mesh.d[1] : 0.0;
    double inv_2dz = (dim >= 3) ? 0.5 / f.mesh.d[2] : 0.0;

    const ScalarField* pf = &f;
    const ScalarField* pg = &g;

    t.gpu_launcher = [pf, pg, nx, ny, nz, sx, sy, gh, dim, inv_2dx, inv_2dy, inv_2dz]
                     (Real* d_rhs, double c, ScratchPool& pool) {
        const Real* d_f = pf->d_curr;
        const Real* d_g = pg->d_curr;
        if (!d_f || !d_g)
            throw std::runtime_error(
                "grad_dot GPU: a field not on device");
        int total = nx * ny * nz;
        kernel_grad_dot_accumulate<<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, d_f, d_g, c, nx, ny, nz, sx, sy,
            gh, dim, inv_2dx, inv_2dy, inv_2dz);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("grad_dot GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, pg, nx, ny, nz, sx, sy, gh, dim, inv_2dx, inv_2dy, inv_2dz]
                   (Real* rhs, double c, ScratchPool&) {
        const Real* f_data = pf->curr.data();
        const Real* g_data = pg->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i+gh, js = j+gh, ks = k+gh;
            int ctr = is + sx*(js + sy*ks);
            double val = 0.0;
            {
                int fwd = (is+1) + sx*(js + sy*ks);
                int bwd = (is-1) + sx*(js + sy*ks);
                val += (f_data[fwd] - f_data[bwd]) * inv_2dx
                     * (g_data[fwd] - g_data[bwd]) * inv_2dx;
            }
            if (dim >= 2) {
                int fwd = is + sx*((js+1) + sy*ks);
                int bwd = is + sx*((js-1) + sy*ks);
                val += (f_data[fwd] - f_data[bwd]) * inv_2dy
                     * (g_data[fwd] - g_data[bwd]) * inv_2dy;
            }
            if (dim >= 3) {
                int fwd = is + sx*(js + sy*(ks+1));
                int bwd = is + sx*(js + sy*(ks-1));
                val += (f_data[fwd] - f_data[bwd]) * inv_2dz
                     * (g_data[fwd] - g_data[bwd]) * inv_2dz;
            }
            rhs[ctr] += c * val;
        }
    };

    return t;
}

// ===========================================================================
// Equation
// ===========================================================================

Equation::Equation(ScalarField& unknown_, const std::string& name_)
    : name(name_), unknown(unknown_) {}

// Explicit destructor: EvalPlan must be fully defined here (Equation.h only
// forward-declares it to break the circular include chain).
Equation::~Equation() = default;

// Every term's stencil width must fit inside its source field's halo —
// a CD4 term (ghost 2) on a ghost-1 field would silently read out of bounds.
static void validateTermGhosts(const RHSExpr& expr) {
    for (const auto &t : expr.terms) {
        if (t.info.complete) {
            for (const auto &r : t.info.reads)
                if (r.cell && r.cell->ghost < r.halo)
                    throw std::invalid_argument("Equation::setRHS: insufficient halo for input '" +
                                                r.cell->name + "'");
        } else if (t.field && t.field->ghost < t.ghostRequired)
            throw std::invalid_argument(
                "Equation::setRHS: insufficient halo for representative field '" + t.field->name +
                "'");
    }
}

void Equation::setRHS(const RHSExpr& expr) {
    validateTermGhosts(expr);
    rhs_expr_ = expr;
    requiredGhost_ = 0;
    for (const auto& t : rhs_expr_.terms)
        requiredGhost_ = std::max(requiredGhost_, t.ghostRequired);
    // A previously set ExprTree plan would otherwise shadow this RHS —
    // computeRHS executes eval_plan_ first when present.
    eval_plan_.reset();
}

void Equation::setRHS(const Term& t) {
    validateTermGhosts(RHSExpr(t));
    rhs_expr_ = RHSExpr(t);
    requiredGhost_ = t.ghostRequired;
    eval_plan_.reset();
}

void Equation::setRHS(const ExprTree& tree) {
    validateGhostRequirements(tree);
    eval_plan_ = std::make_unique<EvalPlan>(lowerExprTree(tree, bc_map_));
    requiredGhost_ = tree.ghostRequired();
    rhs_expr_ = eval_plan_->expression();
    validateTermGhosts(rhs_expr_);
}

void Equation::registerBC(const ScalarField& field,
                           std::vector<BoundaryCondition*> bcs) {
    bc_map_[&field] = std::move(bcs);
}

void Equation::computeRHS(ScalarField& rhs) const {
    if (!rhs.deviceAllocated())
        throw std::runtime_error("Equation::computeRHS: rhs device memory not allocated");

    if (rhs_expr_.terms.empty())
        throw std::runtime_error("Equation::computeRHS: RHS not set (call setRHS first)");

    // -----------------------------------------------------------------------
    // In-place safety (v3.7.0).  computeRHS zeroes its target before the
    // terms accumulate, so a term whose INPUT aliases the target would read
    // zeros instead of the current values -- a silent-wrong-answer trap
    // (found in the wild: an in-place clamp update pinned a solute field at
    // zero for an entire run).  If any term reads the target, evaluate into
    // a pool scratch buffer and copy back instead.  The detection relies on
    // Term::inputs (and Term::field as a fallback), which every ScalarField-
    // reading factory populates; the ExprTree/EvalPlan path (setRHS(ExprTree))
    // does not carry input lists yet and keeps the historical behaviour.
    // -----------------------------------------------------------------------
    bool aliased = false;
    for (const auto& term : rhs_expr_.terms) {
        if (term.field && (term.field == &rhs || term.field->d_curr == rhs.d_curr))
            aliased = true;
        for (const ScalarField* in : term.inputs)
            if (in && (in == &rhs || in->d_curr == rhs.d_curr))
                aliased = true;
    }

    scratch_pool_.reset();
    scratch_pool_.stream = stream_;

    Real* target = rhs.d_curr;
    if (aliased)
        target = scratch_pool_.acquireDevice(rhs.storedSize);

    // Zero the whole stored array asynchronously.
    CUDA_CHECK(cudaMemsetAsync(target, 0, rhs.storedSize * sizeof(Real), stream_));

    for (const auto& term : rhs_expr_.terms) {
        if (!term.gpu_launcher)
            throw std::runtime_error(
                "Equation::computeRHS: a Term has no GPU launcher. "
                "Did you build it with a non-CUDA path?");
        // Terms that index the rhs with a foreign layout (divFace uses the
        // face ghost) corrupt memory if the rhs ghost differs — reject.
        if (term.rhsGhost >= 0 && term.rhsGhost != rhs.ghost)
            throw ValidationError("Equation::computeRHS",
                "term indexes the rhs with ghost = "
                + std::to_string(term.rhsGhost) + " but rhs field '"
                + rhs.name + "' has ghost = " + std::to_string(rhs.ghost),
                "give the unknown/rhs field the same ghost width as the "
                "FaceFields entering divFace");
        term.gpu_launcher(target, term.coeff, scratch_pool_);
    }
    if (aliased)
        CUDA_CHECK(cudaMemcpyAsync(rhs.d_curr, target,
                                   rhs.storedSize * sizeof(Real),
                                   cudaMemcpyDeviceToDevice, stream_));
    // No DeviceSynchronize here — callers sync at step boundaries.
}

void Equation::computeRHSCPU(ScalarField& rhs) const {
    if (rhs_expr_.terms.empty())
        throw std::runtime_error("Equation::computeRHSCPU: RHS not set");

    // In-place safety, host mirror of the GPU path (v3.7.0): if a term reads
    // the target, evaluate into a host scratch and copy back.
    bool aliased = false;
    for (const auto& term : rhs_expr_.terms) {
        if (term.field == &rhs) aliased = true;
        for (const ScalarField* in : term.inputs)
            if (in == &rhs) aliased = true;
    }
    std::vector<Real> host_scratch;
    Real* target = rhs.curr.data();
    if (aliased) {
        host_scratch.assign(rhs.storedSize, Real(0));
        target = host_scratch.data();
    } else {
        std::fill(rhs.curr.begin(), rhs.curr.end(), 0.0);
    }

    scratch_pool_.reset();
    for (const auto& term : rhs_expr_.terms) {
        if (!term.cpu_kernel)
            throw std::runtime_error(
                "Equation::computeRHSCPU: a Term has no CPU kernel.");
        if (term.rhsGhost >= 0 && term.rhsGhost != rhs.ghost)
            throw ValidationError("Equation::computeRHSCPU",
                "term indexes the rhs with ghost = "
                + std::to_string(term.rhsGhost) + " but rhs field '"
                + rhs.name + "' has ghost = " + std::to_string(rhs.ghost),
                "give the unknown/rhs field the same ghost width as the "
                "FaceFields entering divFace");
        term.cpu_kernel(target, term.coeff, scratch_pool_);
    }
    if (aliased)
        std::copy(host_scratch.begin(), host_scratch.end(), rhs.curr.begin());
}

Term grad_dot(const ScalarField &f, const ScalarField &g, const Schemes &sch, double coeff) {
    check::checkSameMesh(f, g, "grad_dot");
    RHSExpr sum;
    for (int axis = 0; axis < f.mesh.dim; ++axis)
        sum += mul(grad(f, axis, sch), grad(g, axis, sch));
    Term out;
    out.type = TermType::COMPOSITE;
    out.field = &f;
    out.coeff = coeff;
    out.info.complete = true;
    inheritInputs(out, sum);
    out.gpu_launcher = [sum](Real *rhs, double c, ScratchPool &pool) {
        for (const auto &t : sum.terms)
            t.gpu_launcher(rhs, c * t.coeff, pool);
    };
    out.cpu_kernel = [sum](Real *rhs, double c, ScratchPool &pool) {
        for (const auto &t : sum.terms)
            t.cpu_kernel(rhs, c * t.coeff, pool);
    };
    return out;
}

namespace {
template <class E>
Term selectedIntermediate(E expression, const std::vector<BoundaryCondition *> &bcs,
                          const std::string &name, int axis, bool isLap, double coeff) {
    const auto *layout = detail::repField(expression);
    std::function<Term(const ScalarField &)> factory;
    if (isLap) {
        const auto create =
            scheme::detail::lookup(scheme::detail::laplacianFactories(), name, "lap(expr)", false)
                .factory;
        factory = [create](const ScalarField &f) { return create(f, 1); };
    } else {
        const auto create =
            scheme::detail::lookup(scheme::detail::gradientFactories(), name, "grad(expr)", false)
                .factory;
        factory = [create, axis](const ScalarField &f) { return create(f, axis, 1); };
    }
    auto prototype = factory(*layout);
    check::checkGhost(*layout, prototype.ghostRequired, "intermediate stencil");
    if (bcs.empty())
        throw std::invalid_argument("intermediate stencil requires explicit BCs");
    Term out;
    out.type = prototype.type;
    out.field = layout;
    out.coeff = coeff;
    out.rhsGhost = layout->ghost;
    out.info.complete = true;
    inheritInputs(out, expression);
    out.ghostRequired = std::max(out.ghostRequired, prototype.ghostRequired);
    out.info.schemes.push_back(std::string(isLap ? "lap" : "grad") + "(materialized)=" + name);
    out.gpu_launcher = [=](Real *rhs, double c, ScratchPool &pool) {
        Real *data = pool.acquireDevice(layout->storedSize);
        detail::materialiseGPU(expression, data, layout->storedSize, pool);
        auto field = ScalarField::makeShell(layout->mesh, layout->ghost, data);
        if (pool.stream)
            CUDA_CHECK(cudaStreamSynchronize(pool.stream));
        BCBatch batch;
        batch.build(field, bcs);
        batch.applyOnGPU(field);
        if (pool.stream) {
            cudaEvent_t done;
            CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(done));
            CUDA_CHECK(cudaStreamWaitEvent(pool.stream, done, 0));
            CUDA_CHECK(cudaEventDestroy(done));
        }
        auto term = factory(field);
        term.gpu_launcher(rhs, c, pool);
    };
    out.cpu_kernel = [=](Real *rhs, double c, ScratchPool &pool) {
        ScalarField field(layout->mesh, "intermediate", layout->ghost);
        detail::materialiseCPU(expression, field.curr.data(), field.storedSize, pool);
        applyBCsCPU2D(field, bcs);
        auto term = factory(field);
        term.cpu_kernel(rhs, c, pool);
    };
    return out;
}
} // namespace
Term lap(const Term &e, const std::vector<BoundaryCondition *> &b, const std::string &s, double c) {
    return selectedIntermediate(e, b, s, 0, true, c);
}
Term lap(const RHSExpr &e, const std::vector<BoundaryCondition *> &b, const std::string &s,
         double c) {
    return selectedIntermediate(e, b, s, 0, true, c);
}
Term grad(const Term &e, int a, const std::vector<BoundaryCondition *> &b, const std::string &s,
          double c) {
    return selectedIntermediate(e, b, s, a, false, c);
}
Term grad(const RHSExpr &e, int a, const std::vector<BoundaryCondition *> &b, const std::string &s,
          double c) {
    return selectedIntermediate(e, b, s, a, false, c);
}

// ===========================================================================
// Vector operator factories
// ===========================================================================

// lap(VectorField) — component-wise Laplacian
VectorRHSExpr lap(const VectorField& vf, double coeff) {
    VectorRHSExpr expr(vf.nComponents());
    for (int c = 0; c < vf.nComponents(); ++c)
        expr[c] = RHSExpr(lap(vf[c], coeff));
    return expr;
}

// grad(ScalarField) — returns mesh.dim-component gradient
VectorRHSExpr grad(const ScalarField& f, double coeff) {
    const int dim = f.mesh.dim;
    VectorRHSExpr expr(dim);
    for (int ax = 0; ax < dim; ++ax)
        expr[ax] = RHSExpr(grad(f, ax, coeff));
    return expr;
}

// div(VectorField) — scalar divergence
RHSExpr div(const VectorField& vf, double coeff) {
    if (vf.nComponents() < vf.mesh.dim)
        throw std::invalid_argument(
            "div: VectorField must have at least mesh.dim components");
    RHSExpr expr;
    for (int ax = 0; ax < vf.mesh.dim; ++ax)
        expr += grad(vf[ax], ax, coeff);
    return expr;
}

// curl(VectorField) — 3D only
VectorRHSExpr curl(const VectorField& vf, double coeff) {
    if (vf.mesh.dim != 3 || vf.nComponents() != 3)
        throw std::invalid_argument(
            "curl: VectorField must be 3-component on a 3D mesh");
    VectorRHSExpr expr(3);
    // curl[0] =  dv2/dy - dv1/dz
    expr[0] += grad(vf[2], 1,  coeff);
    expr[0] += grad(vf[1], 2, -coeff);
    // curl[1] =  dv0/dz - dv2/dx
    expr[1] += grad(vf[0], 2,  coeff);
    expr[1] += grad(vf[2], 0, -coeff);
    // curl[2] =  dv1/dx - dv0/dy
    expr[2] += grad(vf[1], 0,  coeff);
    expr[2] += grad(vf[0], 1, -coeff);
    return expr;
}

// ===========================================================================
// ScratchPool
// ===========================================================================

ScratchPool::~ScratchPool() {
    for (Real* p : dev_bufs_) {
        if (p) cudaFree(p);
    }
}

Real* ScratchPool::acquireDevice(std::size_t size) {
    if (next_dev_ < dev_bufs_.size()) {
        if (dev_sizes_[next_dev_] < size) {
            cudaFree(dev_bufs_[next_dev_]);
            CUDA_CHECK(cudaMalloc(&dev_bufs_[next_dev_], size * sizeof(Real)));
            dev_sizes_[next_dev_] = size;
        }
        return dev_bufs_[next_dev_++];
    }
    Real* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, size * sizeof(Real)));
    dev_bufs_.push_back(p);
    dev_sizes_.push_back(size);
    ++next_dev_;
    return p;
}

Real* ScratchPool::acquireHost(std::size_t size) {
    if (next_host_ < host_bufs_.size()) {
        if (host_bufs_[next_host_].size() < size)
            host_bufs_[next_host_].assign(size, 0.0);
        return host_bufs_[next_host_++].data();
    }
    host_bufs_.emplace_back(size, 0.0);
    ++next_host_;
    return host_bufs_.back().data();
}

// ===========================================================================
// kernel_mul_accumulate  --  rhs[idx] += coeff * s1[idx] * s2[idx]
// (physical cells only)
// ===========================================================================

__global__ void kernel_mul_accumulate(
        Real*       rhs,
        const Real* s1,
        const Real* s2,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * s1[idx] * s2[idx];
}

// ===========================================================================
// Composite expressions on the right-hand side
// ===========================================================================
//
// The two helpers below materialise a Term / RHSExpr into a scratch buffer
// (device or host).  They are used by:
//   * Term * Term / Term * RHSExpr / RHSExpr * RHSExpr  (Phase 2)
//   * lap(expr, bcs) / grad(expr, ax, bcs)              (Phase 3)
//
// For GPU evaluation the launchers accept a ScratchPool& and obtain device
// buffers via pool.acquireDevice(storedSize).  Buffers are zeroed before
// each child launcher is invoked so accumulation semantics match a fresh
// rhs.

namespace detail {

// Pick a representative source field from a Term / RHSExpr (for layout info
// and for rhs nullptr checks).  Throws if the expression has no field.
const ScalarField* repField(const Term& t) {
    if (!t.field)
        throw std::runtime_error(
            "Composite Term: no representative source field captured");
    return t.field;
}
const ScalarField* repField(const RHSExpr& e) {
    for (const auto& t : e.terms)
        if (t.field) return t.field;
    throw std::runtime_error(
        "Composite RHSExpr: no representative source field captured");
}

// Materialise an RHSExpr into d_buf on GPU.  d_buf must have been allocated
// to at least storedSize doubles; it is zeroed first.
void materialiseGPU(const RHSExpr& expr,
                    Real* d_buf, std::size_t storedSize,
                    ScratchPool& pool)
{
    CUDA_CHECK(cudaMemsetAsync(d_buf, 0, storedSize * sizeof(Real), pool.stream));
    for (const auto& t : expr.terms) {
        if (!t.gpu_launcher)
            throw std::runtime_error(
                "Composite GPU: a Term has no GPU launcher");
        t.gpu_launcher(d_buf, t.coeff, pool);
    }
}
void materialiseGPU(const Term& t,
                    Real* d_buf, std::size_t storedSize,
                    ScratchPool& pool)
{
    CUDA_CHECK(cudaMemsetAsync(d_buf, 0, storedSize * sizeof(Real), pool.stream));
    if (!t.gpu_launcher)
        throw std::runtime_error("Composite GPU: Term has no GPU launcher");
    t.gpu_launcher(d_buf, t.coeff, pool);
}

void materialiseCPU(const RHSExpr& expr,
                    Real* h_buf, std::size_t storedSize,
                    ScratchPool& pool)
{
    std::fill(h_buf, h_buf + storedSize, 0.0);
    for (const auto& t : expr.terms) {
        if (!t.cpu_kernel)
            throw std::runtime_error(
                "Composite CPU: a Term has no CPU kernel");
        t.cpu_kernel(h_buf, t.coeff, pool);
    }
}
void materialiseCPU(const Term& t,
                    Real* h_buf, std::size_t storedSize,
                    ScratchPool& pool)
{
    std::fill(h_buf, h_buf + storedSize, 0.0);
    if (!t.cpu_kernel)
        throw std::runtime_error("Composite CPU: Term has no CPU kernel");
    t.cpu_kernel(h_buf, t.coeff, pool);
}

// Aggregate the ScalarField inputs of a Term / RHSExpr for the in-place
// alias detection of computeRHS (v3.7.0).
inline void collectInputs(const Term& t,
                          std::vector<const ScalarField*>& out) {
    if (t.field) out.push_back(t.field);
    out.insert(out.end(), t.inputs.begin(), t.inputs.end());
}
inline void collectInputs(const RHSExpr& e,
                          std::vector<const ScalarField*>& out) {
    for (const Term& t : e.terms) collectInputs(t, out);
}

} // namespace detail

// ===========================================================================
// Term * Term / Term * ScalarField / RHSExpr * RHSExpr ...  -- Phase 2
// ===========================================================================
//
// Implemented in include/equation/FieldOps.inl as inline functions, but the
// underlying mul_accumulate launcher helper lives here so that the kernel
// symbol is emitted in this translation unit.

namespace detail {

// Launch mul_accumulate kernel.  Public to FieldOps.inl via detail::.
void mulAccumulateGPU(Real* d_rhs,
                      const Real* d_s1, const Real* d_s2,
                      double coeff,
                      int nx, int ny, int nz,
                      int sx, int sy, int g,
                      cudaStream_t stream)
{
    int total = nx * ny * nz;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    kernel_mul_accumulate<<<blocks, threads, 0, stream>>>(
        d_rhs, d_s1, d_s2, coeff, nx, ny, nz, sx, sy, g);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("mul_accumulate GPU error: ") + cudaGetErrorString(err));
}

void mulAccumulateCPU(Real* rhs,
                      const Real* s1, const Real* s2,
                      double coeff,
                      int nx, int ny, int nz,
                      int sx, int sy, int g)
{
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        int idx = (i+g) + sx*((j+g) + sy*(k+g));
        rhs[idx] += coeff * s1[idx] * s2[idx];
    }
}

} // namespace detail

// ===========================================================================
// lap(Term/RHSExpr, bcs) and grad(...) on composite expressions  -- Phase 3
// ===========================================================================

// Helper: build a Term that materialises `src_expr` into a scratch buffer,
// applies BCs, then runs `op` (lap or grad) on the scratch buffer.  Generic
// over the source-expression type (Term / RHSExpr) and the FD operator.

template<typename SrcExpr>
static Term makeStencilOnExprTerm(
        SrcExpr             src_expr,
        const ScalarField&  layout,                                // for mesh/ghost/storedSize
        std::vector<BoundaryCondition*> bcs,
        std::function<void(Real* /*rhs*/, const Real* /*src*/,
                           double /*coeff*/, ScratchPool& /*pool*/)> gpu_op,
        std::function<void(Real* /*rhs*/, const Real* /*src*/,
                           double /*coeff*/)> cpu_op,
        TermType  out_type,
        int       out_axis,
        double    coeff)
{
    check::checkCartesian(layout.mesh, "lap/grad(expr)");
    check::checkGhost(layout, 1, "lap/grad(expr)");   // composite path is CD2

    Term out;
    out.type  = out_type;
    out.axis  = out_axis;
    out.coeff = coeff;
    out.field = &layout;     // representative; ensures rhs sanity in nesting
    out.ghostRequired = 1;
    detail::collectInputs(src_expr, out.inputs);   // alias detection (v3.7.0)
    out.info.complete = true;
    inheritInputs(out, src_expr);

    // Capture by value for thread-safety of the std::function.
    const Mesh*  pmesh      = &layout.mesh;
    int          ghost      = layout.ghost;
    std::size_t  storedSize = layout.storedSize;

    out.gpu_launcher = [src_expr, pmesh, ghost, storedSize, bcs,
                        gpu_op]
                       (Real* d_rhs, double c, ScratchPool& pool) {
        // 1. Allocate / reuse scratch
        Real* d_scratch = pool.acquireDevice(storedSize);

        // 2. Evaluate src_expr into d_scratch (zeros first)
        detail::materialiseGPU(src_expr, d_scratch, storedSize, pool);

        // 3. Apply BCs on the shell view of d_scratch
        ScalarField shell = ScalarField::makeShell(*pmesh, ghost, d_scratch);
        if (pool.stream)
            CUDA_CHECK(cudaStreamSynchronize(pool.stream));
        for (auto* bc : bcs) bc->applyOnGPU(shell);
        if (pool.stream) {
            cudaEvent_t done;
            CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(done, nullptr));
            CUDA_CHECK(cudaStreamWaitEvent(pool.stream, done, 0));
            CUDA_CHECK(cudaEventDestroy(done));
        }

        // 4. Run the FD operator: rhs += c * op(scratch)
        gpu_op(d_rhs, d_scratch, c, pool);
    };

    out.cpu_kernel = [src_expr, pmesh, ghost, storedSize, bcs,
                      cpu_op]
                     (Real* rhs, double c, ScratchPool& pool) {
        Real* h_scratch = pool.acquireHost(storedSize);
        detail::materialiseCPU(src_expr, h_scratch, storedSize, pool);

        // CPU BCs require a CPU-resident ScalarField; build one and copy in.
        ScalarField tmp(*pmesh, "shell_cpu", ghost);
        std::copy(h_scratch, h_scratch + storedSize, tmp.curr.begin());
        for (auto* bc : bcs) bc->applyOnCPU(tmp);
        std::copy(tmp.curr.begin(), tmp.curr.end(), h_scratch);

        cpu_op(rhs, h_scratch, c);
    };

    return out;
}

// --- lap(Term, bcs) / lap(RHSExpr, bcs) -------------------------------------

template<typename SrcExpr>
static Term lapOnExpr(SrcExpr src_expr, const ScalarField& layout,
                      const std::vector<BoundaryCondition*>& bcs,
                      double coeff)
{
    int    nx = layout.mesh.n[0], ny = layout.mesh.n[1], nz = layout.mesh.n[2];
    int    sx = layout.storedDims[0], sy = layout.storedDims[1];
    int    g  = layout.ghost;
    int    dim = layout.mesh.dim;
    double inv_dx2 = 1.0 / (layout.mesh.d[0] * layout.mesh.d[0]);
    double inv_dy2 = (dim >= 2) ? 1.0 / (layout.mesh.d[1] * layout.mesh.d[1]) : 0.0;
    double inv_dz2 = (dim >= 3) ? 1.0 / (layout.mesh.d[2] * layout.mesh.d[2]) : 0.0;

    auto gpu_op = [nx, ny, nz, sx, sy, g, dim, inv_dx2, inv_dy2, inv_dz2]
                  (Real* d_rhs, const Real* d_src, double c, ScratchPool& pool) {
        int total = nx * ny * nz;
        kernel_lap_accumulate<<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, d_src, c, nx, ny, nz, sx, sy, g, dim,
            inv_dx2, inv_dy2, inv_dz2);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("lap(expr) GPU error: ") + cudaGetErrorString(err));
    };

    auto cpu_op = [nx, ny, nz, sx, sy, g, dim, inv_dx2, inv_dy2, inv_dz2]
                  (Real* rhs, const Real* src, double c) {
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i+g, js = j+g, ks = k+g;
            int ctr = is + sx*(js + sy*ks);
            double val =
                (src[(is+1)+sx*(js+sy*ks)] - 2.0*src[ctr] + src[(is-1)+sx*(js+sy*ks)]) * inv_dx2;
            if (dim >= 2)
                val += (src[is+sx*((js+1)+sy*ks)] - 2.0*src[ctr] + src[is+sx*((js-1)+sy*ks)]) * inv_dy2;
            if (dim >= 3)
                val += (src[is+sx*(js+sy*(ks+1))] - 2.0*src[ctr] + src[is+sx*(js+sy*(ks-1))]) * inv_dz2;
            rhs[ctr] += c * val;
        }
    };

    return makeStencilOnExprTerm(std::move(src_expr), layout, bcs,
                                 gpu_op, cpu_op,
                                 TermType::COMPOSITE, 0, coeff);
}

// TODO(vector): add VectorRHSExpr lap(VectorRHSExpr, bcs) overload.
Term lap(const Term& t, const std::vector<BoundaryCondition*>& bcs, double coeff) {
    return lapOnExpr<Term>(t, *detail::repField(t), bcs, coeff);
}
Term lap(const RHSExpr& e, const std::vector<BoundaryCondition*>& bcs, double coeff) {
    return lapOnExpr<RHSExpr>(e, *detail::repField(e), bcs, coeff);
}

// --- grad(Term, axis, bcs) / grad(RHSExpr, axis, bcs) -----------------------

template<typename SrcExpr>
static Term gradOnExpr(SrcExpr src_expr, const ScalarField& layout, int axis,
                       const std::vector<BoundaryCondition*>& bcs,
                       double coeff)
{
    if (axis < 0 || axis >= layout.mesh.dim)
        throw std::invalid_argument("grad(expr): axis out of range");

    int    nx = layout.mesh.n[0], ny = layout.mesh.n[1], nz = layout.mesh.n[2];
    int    sx = layout.storedDims[0], sy = layout.storedDims[1];
    int    g  = layout.ghost;
    double inv_2d = 0.5 / layout.mesh.d[axis];

    auto gpu_op = [nx, ny, nz, sx, sy, g, axis, inv_2d]
                  (Real* d_rhs, const Real* d_src, double c, ScratchPool& pool) {
        int total = nx * ny * nz;
        kernel_grad_accumulate<<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, d_src, c, nx, ny, nz, sx, sy, g, axis, inv_2d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("grad(expr) GPU error: ") + cudaGetErrorString(err));
    };

    auto cpu_op = [nx, ny, nz, sx, sy, g, axis, inv_2d]
                  (Real* rhs, const Real* src, double c) {
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i+g, js = j+g, ks = k+g;
            int ctr = is + sx*(js + sy*ks);
            int fwd, bwd;
            if (axis == 0) {
                fwd = (is+1) + sx*(js + sy*ks);
                bwd = (is-1) + sx*(js + sy*ks);
            } else if (axis == 1) {
                fwd = is + sx*((js+1) + sy*ks);
                bwd = is + sx*((js-1) + sy*ks);
            } else {
                fwd = is + sx*(js + sy*(ks+1));
                bwd = is + sx*(js + sy*(ks-1));
            }
            rhs[ctr] += c * (src[fwd] - src[bwd]) * inv_2d;
        }
    };

    return makeStencilOnExprTerm(std::move(src_expr), layout, bcs,
                                 gpu_op, cpu_op,
                                 TermType::COMPOSITE, axis, coeff);
}

Term grad(const Term& t, int axis,
          const std::vector<BoundaryCondition*>& bcs, double coeff) {
    return gradOnExpr<Term>(t, *detail::repField(t), axis, bcs, coeff);
}
Term grad(const RHSExpr& e, int axis,
          const std::vector<BoundaryCondition*>& bcs, double coeff) {
    return gradOnExpr<RHSExpr>(e, *detail::repField(e), axis, bcs, coeff);
}

// grad(expr, bcs)  ->  VectorRHSExpr (one component per mesh axis)
// TODO(vector): mirror this for grad(VectorRHSExpr, bcs) returning a
// rank-2 tensor expression once vector solvers need it.
VectorRHSExpr grad(const Term& t,
                   const std::vector<BoundaryCondition*>& bcs, double coeff) {
    const int dim = detail::repField(t)->mesh.dim;
    VectorRHSExpr expr(dim);
    for (int ax = 0; ax < dim; ++ax)
        expr[ax] = RHSExpr(grad(t, ax, bcs, coeff));
    return expr;
}
VectorRHSExpr grad(const RHSExpr& e,
                   const std::vector<BoundaryCondition*>& bcs, double coeff) {
    const int dim = detail::repField(e)->mesh.dim;
    VectorRHSExpr expr(dim);
    for (int ax = 0; ax < dim; ++ax)
        expr[ax] = RHSExpr(grad(e, ax, bcs, coeff));
    return expr;
}

// div(VectorRHSExpr, bcs) — divergence of expression-valued flux.
RHSExpr div(const VectorRHSExpr& v,
            const std::vector<BoundaryCondition*>& bcs, double coeff) {
    RHSExpr expr;
    const int n = v.nComponents();
    for (int ax = 0; ax < n; ++ax)
        expr += grad(v[ax], ax, bcs, coeff);
    return expr;
}

// --- iso_grad(Term, axis, bcs) / iso_grad(RHSExpr, axis, bcs) ---------------
//
// Like gradOnExpr but uses the 9-point isotropic stencil (kernel_iso_grad_accumulate).
// Falls back to gradOnExpr for non-2D meshes or axis >= 2.

template<typename SrcExpr>
static Term isoGradOnExpr(SrcExpr src_expr, const ScalarField& layout, int axis,
                           const std::vector<BoundaryCondition*>& bcs,
                           double coeff)
{
    if (axis < 0 || axis >= layout.mesh.dim)
        throw std::invalid_argument("iso_grad(expr): axis out of range");

    // Fallback to standard grad for 1D/3D or axis >= 2
    if (layout.mesh.dim != 2 || axis > 1) {
        warnOnce("iso_grad-expr-fallback",
                 "iso_grad(expr): the 9-point isotropic stencil is 2D-only "
                 "(axis 0/1); falling back to plain CD2 grad");
        return gradOnExpr(std::move(src_expr), layout, axis, bcs, coeff);
    }

    int    nx = layout.mesh.n[0], ny = layout.mesh.n[1], nz = layout.mesh.n[2];
    int    sx = layout.storedDims[0], sy = layout.storedDims[1];
    int    g  = layout.ghost;
    int    dim = layout.mesh.dim;
    double inv_dx = 1.0 / layout.mesh.d[0];
    double inv_dy = (dim >= 2) ? 1.0 / layout.mesh.d[1] : 0.0;
    double inv_dz = (dim >= 3) ? 1.0 / layout.mesh.d[2] : 0.0;

    auto gpu_op = [nx, ny, nz, sx, sy, g, dim, axis, inv_dx, inv_dy, inv_dz]
                  (Real* d_rhs, const Real* d_src, double c, ScratchPool& pool) {
        int total = nx * ny * nz;
        kernel_iso9_grad_accumulate<<<(total + 255) / 256, 256, 0, pool.stream>>>(
            d_rhs, d_src, c, nx, ny, nz, sx, sy, g, dim, axis, inv_dx, inv_dy, inv_dz);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("iso_grad(expr) GPU error: ") + cudaGetErrorString(err));
    };

    auto cpu_op = [nx, ny, nz, sx, sy, g, dim, axis, inv_dx, inv_dy, inv_dz]
                  (Real* rhs, const Real* src, double c) {
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i+g, js = j+g, ks = k+g;
            int ctr = is + sx*(js + sy*ks);
            rhs[ctr] += c * scheme::Iso9::gradient(src, ctr, axis, sx, sy, dim,
                                                    inv_dx, inv_dy, inv_dz);
        }
    };

    return makeStencilOnExprTerm(std::move(src_expr), layout, bcs,
                                 gpu_op, cpu_op,
                                 TermType::COMPOSITE, axis, coeff);
}

Term iso_grad(const Term& t, int axis,
              const std::vector<BoundaryCondition*>& bcs, double coeff) {
    return isoGradOnExpr<Term>(t, *detail::repField(t), axis, bcs, coeff);
}
Term iso_grad(const RHSExpr& e, int axis,
              const std::vector<BoundaryCondition*>& bcs, double coeff) {
    return isoGradOnExpr<RHSExpr>(e, *detail::repField(e), axis, bcs, coeff);
}

// ===========================================================================
// Equation::advanceSteady / advanceTransient
// ===========================================================================

// GPU kernel: dst[i] += coeff * src[i]  (scale-accumulate over full stored array)
__global__ void kernel_eq_axpy(Real* dst, const Real* src, double coeff, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] += coeff * src[tid];
}

// ---------------------------------------------------------------------------
// advanceSteady
//
//   1. Apply bcs to sourceField (nullptr → unknown).
//   2. Evaluate RHS directly into unknown (zero-then-fill via computeRHS).
//
// No time advancement, no advanceTimeLevelGPU call.
// ---------------------------------------------------------------------------
void Equation::advanceSteady(const std::vector<BoundaryCondition*>& bcs,
                              ScalarField* sourceField)
{
    ScalarField& src = sourceField ? *sourceField : unknown;
    for (auto* bc : bcs) bc->applyOnGPU(src);
    computeRHS(unknown);
}

// ---------------------------------------------------------------------------
// advanceTransient  (forward Euler)
//
//   1. Apply bcs to sourceField (nullptr → unknown).
//   2. Evaluate RHS into rhs_scratch_ (lazily allocated).
//   3. unknown.d_curr += dt * rhs_scratch_.d_curr  (axpy over storedSize).
//   4. unknown.advanceTimeLevelGPU()  (d_prev ← d_curr).
//   5. Increment step and time.
// ---------------------------------------------------------------------------
void Equation::advanceTransient(const std::vector<BoundaryCondition*>& bcs,
                                 double dt,
                                 ScalarField* sourceField)
{
    check::checkPositive(dt, "dt", "Equation::advanceTransient");
    check::checkFiniteScalar(dt, "dt", "Equation::advanceTransient");

    ScalarField& src = sourceField ? *sourceField : unknown;
    for (auto* bc : bcs) bc->applyOnGPU(src);

    // Lazily allocate rhs scratch field.
    if (!rhs_scratch_) {
        rhs_scratch_ = std::make_unique<ScalarField>(
            unknown.mesh, unknown.name + "_rhs", unknown.ghost);
        rhs_scratch_->allocDevice();
    }

    computeRHS(*rhs_scratch_);

    int n = static_cast<int>(unknown.storedSize);
    kernel_eq_axpy<<<(n + 255) / 256, 256, 0, stream_>>>(
        unknown.d_curr, rhs_scratch_->d_curr, dt, n);
    CUDA_CHECK(cudaGetLastError());
    // Sync before advanceTimeLevelGPU (d_prev ← d_curr) and before caller
    // might read results from CPU side.
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    unknown.advanceTimeLevelGPU();

    ++step;
    time += dt;
}

} // namespace PhiX
