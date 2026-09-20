#pragma once

// ---------------------------------------------------------------------------
// Multigrid.h — geometric multigrid preconditioner (2D, cell-centred).
//
// MultigridPreconditioner approximates z ≈ A⁻¹ r for the CG layer's system
//
//     A = alpha·I − sigma·∇·(a(x)∇)
//
// with ONE V(nPre,nPost)-cycle per apply — the MG-PCG combination: the
// V-cycle is an SPD operation for symmetric smoothing, so CG convergence
// theory holds, and the iteration count becomes O(1) in the stiffness
// σ·a/dx² where plain CG grows like its square root.
//
// Design (matches the layer's graph contract):
//   • alpha/sigma arrive as DEVICE pointers (the CG scalar slots) — every
//     kernel reads them through the pointer, so a captured burst graph stays
//     valid across per-solve coefficient changes.
//   • BCs are absorbed into FACE COEFFICIENTS + index arithmetic: no-flux =
//     zero boundary-face coefficient, periodic = wrapped neighbour index.
//     Coarse levels need no halo buffers and no BC objects.
//   • Coefficients live on faces per level: ax (nx+1)×ny, ay nx×(ny+1).
//     Level 0 is filled from a constant (setup) or from device face arrays
//     (setupFaces — the variable-coefficient case this module exists for);
//     coarser faces are arithmetic averages of the two overlapping fine
//     faces.  setup* is cheap and may be called every step.
//   • 2:1 full coarsening while both dims are even and > 4; an odd dimension
//     stops the hierarchy (2000 → 125 four levels down).  The coarsest level
//     is relaxed with `coarseSweeps` damped-Jacobi sweeps — exactness is not
//     required of a preconditioner, but a large forced-stop coarsest grid
//     under extreme σ degrades cycle quality (document per app).
//   • Smoother: damped Jacobi (omega, default 0.8), nPre/nPost sweeps;
//     restriction: 4-child full weighting; prolongation: bilinear.
//   • 2D only (dim == 2 enforced); 3D is a planned extension.
//
// See test/moduleTest/solver/test_mg.cu and develop/mg_precond/plan.md.
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"
#include "solver/Preconditioner.h"

#include <vector>

namespace PhiX {

class MultigridPreconditioner : public Preconditioner {
public:
    enum class BCKind { Periodic, NoFlux };

    MultigridPreconditioner(const Mesh& mesh, BCKind bcX, BCKind bcY,
                            int nPre = 2, int nPost = 2, double omega = 0.8,
                            int coarseSweeps = 50);
    ~MultigridPreconditioner();

    MultigridPreconditioner(const MultigridPreconditioner&)            = delete;
    MultigridPreconditioner& operator=(const MultigridPreconditioner&) = delete;

    // Constant coefficient a (the unit-Laplacian pattern: fold D into sigma
    // and pass a = 1).  Builds the whole face-coefficient hierarchy.
    void setup(double a = 1.0);

    // Variable coefficients from device face arrays of the FINE level:
    //   ax: (nx+1)·ny values, ax[i + (nx+1)·j] = a on face between cells
    //       (i−1,j) and (i,j);  ax[0] and ax[nx] are the boundary faces
    //       (equal under periodic wrap, zeroed internally under no-flux).
    //   ay: nx·(ny+1) values, analogous in y.
    // Coarser levels are rebuilt by face averaging.  Call per step when the
    // coefficients move (P3: M_c(phi)·chi faces).
    void setupFaces(const double* d_ax, const double* d_ay);

    void apply(ScalarField& r, ScalarField& z,
               const double* d_alpha, const double* d_sigma,
               cudaStream_t stream = nullptr) override;

    bool streamSafe() const override { return true; }
    int  ghostRequired() const override { return 0; }

    int levels() const { return static_cast<int>(lv_.size()); }

private:
    struct Level {
        int     nx = 0, ny = 0;
        double  inv_dx2 = 0.0, inv_dy2 = 0.0;
        double* ax = nullptr;   // (nx+1)·ny
        double* ay = nullptr;   // nx·(ny+1)
        Real*   z  = nullptr;   // nx·ny  (correction / solution)
        Real*   r  = nullptr;   // nx·ny  (restricted residual)
        Real*   t  = nullptr;   // nx·ny  (scratch: Az / residual)
    };

    void vcycle(int l, const double* d_alpha, const double* d_sigma,
                cudaStream_t stream);
    void smooth(Level& L, int sweeps, const double* d_alpha,
                const double* d_sigma, cudaStream_t stream);
    void buildCoarseFaces(cudaStream_t stream = nullptr);

    std::vector<Level> lv_;
    BCKind bcX_, bcY_;
    int    nPre_, nPost_, coarseSweeps_;
    double omega_;
    bool   facesReady_ = false;
};

} // namespace PhiX
