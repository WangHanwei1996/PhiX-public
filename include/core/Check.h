#pragma once

// ---------------------------------------------------------------------------
// Check.h — user-facing validation API (namespace PhiX::check).
//
// Small, composable checks that turn silent misuse into a structured
// ValidationError / NumericsError (core/Error.h).  The framework calls
// them at construction/launch entry points; solver authors are encouraged
// to call them directly in application code:
//
//   check::checkPositive(dt, "dt", "myApp");
//   check::checkOnDevice(phi, "myApp");
//   check::checkSameMesh(phi, c, "myApp");          // dims + spacing + origin
//   check::checkFinite(phi, "myApp");               // GPU NaN/Inf scan
//   check::checkFieldHealth(phi, step, t, 1e6);     // per-step sentinel
//
// All scalar/geometry checks are inline and header-only (host-safe: no
// CUDA includes).  checkFinite / checkFieldHealth run device reductions
// and are defined in src/core/Check.cu.
//
// Cost model: every inline check is O(1) at call time; none belong in
// per-cell loops except checkIndex, which exists precisely as an opt-in
// debug guard for hand-written index arithmetic.
// ---------------------------------------------------------------------------

#include "core/Error.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"

#include <cmath>
#include <string>

namespace PhiX {
namespace check {

// ---------------------------------------------------------------------------
// Scalar checks
// ---------------------------------------------------------------------------

inline void checkPositive(double v, const char* name, const char* who) {
    if (!(v > 0.0))
        throw ValidationError(who,
            std::string(name) + " must be > 0 (got " + std::to_string(v) + ")");
}

inline void checkNonNegative(double v, const char* name, const char* who) {
    if (!(v >= 0.0))
        throw ValidationError(who,
            std::string(name) + " must be >= 0 (got " + std::to_string(v) + ")");
}

inline void checkRange(double v, double lo, double hi,
                       const char* name, const char* who) {
    if (!(v >= lo && v <= hi))
        throw ValidationError(who,
            std::string(name) + " must be in [" + std::to_string(lo) + ", "
            + std::to_string(hi) + "] (got " + std::to_string(v) + ")");
}

inline void checkFiniteScalar(double v, const char* name, const char* who) {
    if (!std::isfinite(v))
        throw ValidationError(who,
            std::string(name) + " must be finite (got "
            + std::to_string(v) + ")");
}

// ---------------------------------------------------------------------------
// Mesh checks
// ---------------------------------------------------------------------------

// True if two meshes describe the same discretisation: dim, cell counts,
// spacings, and origins all equal.  (Equal n[] alone is NOT enough — two
// meshes with the same resolution but different dx describe different
// physical domains.)
inline bool sameMeshGeometry(const Mesh& a, const Mesh& b) {
    if (a.dim != b.dim) return false;
    for (int ax = 0; ax < 3; ++ax)
        if (a.n[ax] != b.n[ax] || a.d[ax] != b.d[ax]
            || a.origin[ax] != b.origin[ax])
            return false;
    return true;
}

inline void checkMeshValid(const Mesh& m, const char* who) {
    if (!m.isValid())
        throw ValidationError(who,
            "mesh is not valid (default-constructed or bad dims/spacing)",
            "build meshes with Mesh::makeUniform1D/2D/3D, which validate "
            "their parameters");
}

// All differential operators in PhiX implement Cartesian forms only; a
// CYLINDRICAL/SPHERICAL mesh would silently get the wrong equations.
// Pointwise terms (pw/fpw*) and hand-written kernels are coordinate-
// agnostic and remain usable on any CoordSys.
inline void checkCartesian(const Mesh& m, const char* who) {
    if (m.coordSys != CoordSys::CARTESIAN)
        throw ValidationError(who,
            "mesh has CoordSys::" + coordSysToString(m.coordSys)
            + " but all differential operators implement Cartesian forms only",
            "use CoordSys::CARTESIAN, or build curvilinear metric terms "
            "explicitly from pw() + mesh.coord() source terms / hand-written "
            "kernels");
}

// Same PHYSICAL domain: equal dim and, per active axis, equal origin and
// extent n·d within a relative tolerance.  (Resolutions may differ — this
// is the precondition of MeshMap, not of same-mesh algebra.)
inline void checkSameDomain(const Mesh& a, const Mesh& b, const char* who,
                            double relTol = 1e-12) {
    if (a.dim != b.dim)
        throw ValidationError(who,
            "meshes have different dimensionality ("
            + std::to_string(a.dim) + "D vs " + std::to_string(b.dim) + "D)");
    for (int ax = 0; ax < a.dim; ++ax) {
        const double La = a.n[ax] * a.d[ax], Lb = b.n[ax] * b.d[ax];
        const double scale = std::max(1.0, std::max(std::fabs(La), std::fabs(Lb)));
        if (std::fabs(a.origin[ax] - b.origin[ax]) > relTol * scale
            || std::fabs(La - Lb) > relTol * scale)
            throw ValidationError(who,
                "meshes do not cover the same physical domain on axis "
                + std::to_string(ax) + " (origin " + std::to_string(a.origin[ax])
                + " vs " + std::to_string(b.origin[ax]) + ", extent "
                + std::to_string(La) + " vs " + std::to_string(Lb) + ")",
                "MeshMap requires both meshes to span the identical domain; "
                "check n, d and origin");
    }
}

// Opt-in bounds check for hand-written index arithmetic (physical cells).
inline void checkIndex(const Mesh& m, int i, int j, int k, const char* who) {
    if (i < 0 || i >= m.n[0] || j < 0 || j >= m.n[1] || k < 0 || k >= m.n[2])
        throw ValidationError(who,
            "index (" + std::to_string(i) + ", " + std::to_string(j) + ", "
            + std::to_string(k) + ") outside physical cells ["
            + std::to_string(m.n[0]) + " x " + std::to_string(m.n[1]) + " x "
            + std::to_string(m.n[2]) + ")");
}

// ---------------------------------------------------------------------------
// Field checks
// ---------------------------------------------------------------------------

inline void checkOnDevice(const ScalarField& f, const char* who) {
    if (!f.deviceAllocated())
        throw ValidationError(who,
            "field '" + f.name + "' has no device buffer",
            "call allocDevice() + uploadAllToDevice() before GPU use");
}

inline void checkGhost(const ScalarField& f, int required, const char* who) {
    if (f.ghost < required)
        throw ValidationError(who,
            "requires ghost >= " + std::to_string(required) + " but field '"
            + f.name + "' has ghost = " + std::to_string(f.ghost),
            "construct the field with ScalarField(mesh, name, /*ghost=*/"
            + std::to_string(required) + ")");
}

inline void checkSameMesh(const ScalarField& a, const ScalarField& b,
                          const char* who) {
    if (!sameMeshGeometry(a.mesh, b.mesh) || a.ghost != b.ghost)
        throw ValidationError(who,
            "fields '" + a.name + "' and '" + b.name
            + "' do not share the same mesh geometry and ghost width "
            "(dims/spacing/origin/ghost must all match)",
            "build all fields entering one expression from the same Mesh "
            "object with the same ghost");
}

// ---------------------------------------------------------------------------
// GPU reductions (defined in src/core/Check.cu)
// ---------------------------------------------------------------------------

// Throws NumericsError if any physical cell of f holds NaN or ±Inf.
void checkFinite(const ScalarField& f, const char* who);

// Per-step health sentinel: NaN/Inf scan, plus (if maxAbsLimit > 0) a
// blow-up check |f|_max <= maxAbsLimit that often fires before the first
// NaN appears.  step/time label the error message.  Scans the DEVICE copy
// when the field is device-allocated, the host copy otherwise.
void checkFieldHealth(const ScalarField& f, int step, double time,
                      double maxAbsLimit = 0.0);

// Host-side variant: always scans the CPU `curr` array (the authoritative
// copy in advanceCPU workflows, where the device buffer may be stale).
void checkFieldHealthCPU(const ScalarField& f, int step, double time,
                         double maxAbsLimit = 0.0);

} // namespace check
} // namespace PhiX
