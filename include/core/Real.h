#pragma once

// ---------------------------------------------------------------------------
// Real.h — the floating-point type of all field data and GPU kernels.
//
// Selected at configure time:
//
//   cmake .. -DPHIX_PRECISION=DOUBLE    (default — Real = double)
//   cmake .. -DPHIX_PRECISION=FLOAT     (Real = float)
//
// Scope of the switch:
//   • Field storage (ScalarField/VectorField/FaceField), boundary kernels,
//     stencil schemes, equation/solver kernels, noise, Gibbs projection.
//   • Host-side control scalars (dt, time, coefficients, mesh geometry) stay
//     double and are cast to Real at kernel-launch boundaries.
//   • Reductions (field/Reduce.h) always ACCUMULATE and return double, even
//     for float fields — sums/L2 keep full precision.
//   • On-disk formats (.field binary) remain double; IO converts on
//     read/write, so files are interchangeable between builds.
//   • FreeEnergyTable keeps double tables (lookup-bound; document in model
//     code if float tables are ever needed).
//
// In FLOAT builds the application solvers and the strict (1e-12-tolerance)
// test suites are not configured; test/floatSmoke covers the core paths with
// float-appropriate tolerances.
//
// Writing kernels/functors: use Real-typed parameters and Real(…) literals
// in hot device code — a stray double literal silently promotes the whole
// expression to FP64 (1/32–1/64 throughput on consumer GPUs).
// ---------------------------------------------------------------------------

namespace PhiX {

#ifdef PHIX_REAL_FLOAT
using Real = float;
#else
using Real = double;
#endif

} // namespace PhiX
