#pragma once

#include "IO/FieldFormat.h"
#include "mesh/Mesh.h"

#include <string>

namespace PhiX {

// Forward declarations — avoid pulling full field headers into every TU
class ScalarField;
class VectorField;

namespace IO {

// ---------------------------------------------------------------------------
// Scalar field IO
// ---------------------------------------------------------------------------

/// Write a ScalarField to disk in the specified format.
/// coordScale multiplies the VTK geometry only — VTS node coordinates and the
/// VTK_BIN / VTI Origin+Spacing (e.g. 1e9 → nm), so nm-scale grids are
/// visible in ParaView; DAT/BINARY are unaffected.
void writeField(const ScalarField& f,
                const std::string& path,
                FieldFormat fmt = FieldFormat::BINARY,
                double coordScale = 1.0);

/// Read a ScalarField from a binary .field file.
/// Physical cells are loaded into `curr`; `prev` is left zeroed.
/// Header nx/ny/nz are validated against the provided mesh.
ScalarField readScalarField(const Mesh& mesh,
                            const std::string& path,
                            int ghost = 1);

/// Read a binary .field file into an existing ScalarField (in-place).
/// Only `curr` is updated; `prev` is left unchanged.
/// Header dimensions are validated against the field's own mesh.
void readField(ScalarField& f, const std::string& path);

// ---------------------------------------------------------------------------
// Restart / initialization helpers
// ---------------------------------------------------------------------------

/// Resolve the starting step number from cfg["initialize"]["start_from"].
///
/// - "initial_field"  → returns 0
/// - "<integer>"      → scans output/ for files named `{refFieldName}_*.field`
///                      and returns the step number closest to the requested
///                      value.  Throws std::runtime_error if no files found.
///
/// @param startFrom    value of cfg["initialize"]["start_from"] (as string)
/// @param refFieldName name of the field used to enumerate available steps
///                     (default: "c")
int resolveStartStep(const std::string& startFrom,
                     const std::string& refFieldName = "c");

/// Read a field's initial data according to the resolved start step.
///
/// - startStep == 0  → reads  settings/initial_field/{f.name}.field
/// - startStep  > 0  → reads  output/{f.name}_{startStep}.field
///
/// Prints a status line to stdout.
void initField(ScalarField& f, int startStep);

/// Named initializer overload.  When startStep == 0 and no file is found
/// (or you want to bypass the file entirely), pass a non-empty `namedInit`
/// string to generate the initial condition analytically.
///
/// Supported formats:
///   "uniform:<value>"           → constant, e.g. "uniform:0.5"
///   "random:<lo>:<hi>"          → uniform random in [lo, hi]
///   "linear:x:<lo>:<hi>"        → linear ramp along x from lo (left) to hi (right)
///   "linear:y:<lo>:<hi>"        → linear ramp along y
///   "linear:z:<lo>:<hi>"        → linear ramp along z
///
/// If namedInit is empty and startStep == 0, falls back to file-based init.
void initField(ScalarField& f, int startStep, const std::string& namedInit);

// ---------------------------------------------------------------------------
// Vector field IO
// ---------------------------------------------------------------------------

/// Write a VectorField to disk in the specified format.
/// coordScale is honoured by the VTI/VTI_F64 path (Origin + Spacing); the
/// VTS path writes unscaled physical coordinates.
void writeField(const VectorField& vf,
                const std::string& path,
                FieldFormat fmt = FieldFormat::BINARY,
                double coordScale = 1.0);

/// Read a VectorField from a binary .vfield file.
/// Physical cells are loaded into each component's `curr`.
/// Header nx/ny/nz and nComponents are validated.
VectorField readVectorField(const Mesh& mesh,
                            const std::string& path,
                            int ghost = 1);

/// Read a binary .vfield file into an existing VectorField (in-place).
/// Only each component's `curr` is updated; `prev` is left unchanged.
/// Header dimensions and nComponents are validated against the field.
void readField(VectorField& vf, const std::string& path);

} // namespace IO
} // namespace PhiX
