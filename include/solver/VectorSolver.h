#pragma once

#include "solver/Solver.h"           // TimeScheme enum
#include "equation/VectorEquation.h"
#include "boundary/BoundaryCondition.h"
#include "field/VectorField.h"

#include <functional>
#include <string>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// VectorSolver
//
// Drives explicit time advancement of one VectorEquation.
// Mirrors the scalar Solver API and time-stepping logic, but operates on
// all field components simultaneously — ensuring RK4 stage evaluations are
// consistent across components (important for coupled vector PDEs).
//
// Supported time schemes: EULER, RK4 (same as scalar Solver).
//
// Usage:
//   VectorSolver vs(veq, {&bc_x, &bc_y}, dt, TimeScheme::RK4);
//   vs.run(1000, [&](const VectorSolver& s) {
//       v.downloadCurrFromDevice();
//       v.write("output/v_" + std::to_string(s.step) + ".vfield");
//   });
// ---------------------------------------------------------------------------

class VectorSolver {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    VectorSolver(VectorEquation&                     equation,
                 std::vector<BoundaryCondition*>     bcs,
                 double                              dt,
                 TimeScheme                          scheme = TimeScheme::EULER);

    // Non-copyable
    VectorSolver(const VectorSolver&)            = delete;
    VectorSolver& operator=(const VectorSolver&) = delete;

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    double     dt;
    TimeScheme scheme;
    int        step = 0;
    double     time = 0.0;

    // -----------------------------------------------------------------------
    // Single time step (GPU path)
    // -----------------------------------------------------------------------
    void advance();

    // CPU fallback
    void advanceCPU();

    // -----------------------------------------------------------------------
    // Run multiple steps
    // -----------------------------------------------------------------------
    void run(int nSteps,
             int callbackEvery = 0,
             std::function<void(const VectorSolver&)> callback = nullptr);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    const VectorField& unknown() const { return equation_.unknown; }
    VectorField&       unknown()       { return equation_.unknown; }

    const VectorEquation&                       equation() const { return equation_; }
    const std::vector<BoundaryCondition*>&       bcs()     const { return bcs_; }

private:
    VectorEquation&                 equation_;
    std::vector<BoundaryCondition*> bcs_;
    bool                            use_rk4_;

    // Scratch VectorFields — same mesh / ghost / nComponents as unknown
    VectorField rhs_;
    VectorField k1_, k2_, k3_, k4_;
    VectorField phi_tmp_;

    void applyBCsGPU();
    void applyBCsCPU();
    void eulerUpdateGPU();
    void eulerUpdateCPU();
    void rk4AdvanceGPU();
    void rk4AdvanceCPU();
};

} // namespace PhiX
