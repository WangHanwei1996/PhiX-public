#pragma once

// ---------------------------------------------------------------------------
// LBM.h — D2Q9 lattice-Boltzmann solver (BGK + Guo forcing), GPU-resident.
//
// Purpose: explicit velocity fields for future flow-coupled phase-field
// runs (e.g. melt convection during solidification).  The module owns the
// distribution functions entirely on the device and exports the
// macroscopic fields (ρ, u_x, u_y) into PhiX ScalarFields for coupling.
//
// Model:
//   • D2Q9, BGK single-relaxation-time collision, c_s² = 1/3;
//   • constant body force via Guo et al. (2002) forcing (2nd-order
//     consistent: u_macro = (Σ c_i f_i + F/2)/ρ, source term weighted by
//     (1 − 1/(2τ)));
//   • two-kernel update: in-place collision, then pull-streaming into a
//     second buffer (pointers swapped — no copies);
//   • boundaries per side: PERIODIC (default) or WALL (halfway
//     bounce-back, no-slip located half a cell outside the outermost row).
//
// Everything is in LATTICE UNITS (dx = dt = 1): kinematic viscosity
// ν = c_s²·(τ − ½).  The Mesh only supplies the lattice dimensions and the
// layout for the exported fields — physical scaling is the caller's
// mapping (choose dx_phys, dt_phys, convert ν and forces accordingly).
//
// Usage:
//     LBMParams p;  p.tau = 0.9;  p.fx = 1e-6;
//     LBM2D lbm(mesh, p);
//     lbm.setWall(Axis::Y, Side::LOW);
//     lbm.setWall(Axis::Y, Side::HIGH);
//     lbm.initialize(1.0, 0.0, 0.0);
//     lbm.run(20000);
//     lbm.macroscopics(rho, ux, uy);      // physical cells of the fields
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"

#include <vector>

namespace PhiX {

struct LBMParams {
    double tau = 1.0;    // BGK relaxation time (> 0.5); ν = (τ − ½)/3
    double fx  = 0.0;    // constant body force, lattice units
    double fy  = 0.0;

    void validate() const;   // throws std::invalid_argument
};

class LBM2D {
public:
    LBM2D(const Mesh& mesh, const LBMParams& p);
    ~LBM2D();

    LBM2D(const LBM2D&)            = delete;
    LBM2D& operator=(const LBM2D&) = delete;

    // Mark one side as a no-slip wall (halfway bounce-back); call once per
    // side before the first step.  Unmarked sides stream periodically.
    void setWall(Axis axis, Side side);

    // Zou-He velocity inlet on one side: prescribed (u_normal, u_tangential)
    // per boundary cell (profiles of length = tangential extent; uTan may be
    // empty = 0).  u_normal > 0 means INTO the domain on either side.
    void setVelocityInlet(Axis axis, Side side,
                          const std::vector<double>& uNormal,
                          const std::vector<double>& uTangential = {});

    // Pressure outlet (Zou-He) on one side: fixes the density level rho0
    // and lets the normal velocity float.  NOTE: a pure zero-gradient
    // outflow does NOT anchor the pressure — combined with a velocity
    // inlet the density integrates without bound; this outlet is the
    // stable pairing for channel flows.
    void setOutflow(Axis axis, Side side, double rho0 = 1.0);

    // Interior no-slip obstacles: cells with mask >= 0.5 are solid (halfway
    // bounce-back at their faces); solid cells report u = 0, rho = 1 in
    // macroscopics().  The mask field must share the constructor mesh.
    void setObstacleMask(const ScalarField& mask);

    // Equilibrium initialisation at uniform (rho0, u0).
    void initialize(double rho0, double ux0 = 0.0, double uy0 = 0.0);

    void step();                 // collide + stream (+ buffer swap)
    void run(int nSteps);

    // Export macroscopic fields into the PHYSICAL cells of PhiX fields
    // (fields must share the constructor mesh; ghosts untouched).
    // Any pointer may be null to skip that quantity.
    void macroscopics(ScalarField* rho, ScalarField* ux, ScalarField* uy);

    double latticeViscosity() const { return (tau_ - 0.5) / 3.0; }
    int    stepCount() const { return step_; }

private:
    const Mesh& mesh_;
    double      tau_;
    Real        fx_, fy_;
    int         nx_, ny_;
    int         step_ = 0;

    // sideType_[axis][side]: 0 periodic, 1 wall (BB), 2 velocity inlet,
    // 3 zero-gradient outflow
    int sideType_[2][2] = {{0, 0}, {0, 0}};

    Real* d_f_    = nullptr;   // 9 × nx·ny distributions (SoA)
    Real* d_ftmp_ = nullptr;

    unsigned char* d_mask_ = nullptr;         // 1 = solid, nullptr = none
    Real* d_inlet_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
                                              // (uN, uT) pairs per boundary cell
    double outletRho_[2][2] = {{1.0, 1.0}, {1.0, 1.0}};

    void applyBoundaryKernels_();             // inlet/outflow after streaming
};

} // namespace PhiX
