#pragma once

#include "IO/FieldFormat.h"
#include "field/ScalarField.h"

#include <nlohmann/json.hpp>
#include <chrono>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace PhiX {
namespace IO {

// ---------------------------------------------------------------------------
// OutputWriter
//
// Manages field output and progress logging based on a JSON config block.
//
// Expected JSON layout:
//   {
//       "print_interval": 10000,
//       "write_interval": 100000,
//       "format"        : "BINARY+VTI",  // see the token list below
//       "vti_precision" : "float32",     // optional: "float32" | "float64"
//       "coord_scale"   : 1.0            // optional: scales VTK geometry
//   }
//
// Format tokens:
//   "BINARY"  .field  — full-precision restart format
//   "DAT"     .dat    — ASCII x y z value
//   "VTI"     .vti    — XML ImageData, appended raw float32, implicit
//                       geometry: ParaView's uniform-grid format of choice,
//                       ~41x smaller than VTS and read by memcpy
//   "VTK_BIN" .vtk    — legacy binary VTK, same size as VTI, but POINT_DATA
//                       (half-cell offset from the VTS/VTI convention)
//   "VTK"/"VTS" .vts  — ASCII StructuredGrid with explicit node coordinates;
//                       kept for compatibility, warns on use
//   "ALL"             — BINARY + DAT + VTS (historical meaning, unchanged)
//
// Several tokens may be joined with '+', e.g. "BINARY+VTI" to keep restart
// files alongside ParaView output, or "DAT+VTI" when a Python post-processing
// chain reads the .dat files.
//
// With "VTI" the writer also maintains output/<field>.pvd — a ParaView
// collection indexing every written .vti with its PHYSICAL time as the
// timestep, so the animation time axis is simulation time, not step count.
// On warm restarts the existing .pvd is reloaded and entries at/after the
// restart time are replaced.  "vti_precision": "float64" switches the payload
// to FieldFormat::VTI_F64 (same layout, double the bytes).
//
// Usage:
//   IO::OutputWriter writer(cfg["output"]);
//   writer.writeFields(c, step, simTime);       // write field files
//   writer.printProgress(step, simTime);         // print progress + elapsed
//   if (writer.shouldPrint(step)) { ... }
//   if (writer.shouldWrite(step)) { ... }
// ---------------------------------------------------------------------------
class OutputWriter {
public:
    explicit OutputWriter(const nlohmann::json& output_config);

    int printInterval;
    int writeInterval;

    /// Check whether this step should trigger a progress print.
    bool shouldPrint(int step) const { return step % printInterval == 0; }

    /// Check whether this step should trigger a file write.
    bool shouldWrite(int step) const { return step % writeInterval == 0; }

    /// Write field to output/ in all configured formats.
    /// Downloads from device, writes files, and prints a status line.
    void writeFields(ScalarField& f, int step, double simTime);

    /// Write current host values without an implicit device download.
    /// Use for CPU execution, including fields with separately allocated GPU storage.
    void writeHostFields(const ScalarField& f, int step, double simTime);

    /// Print a progress line with step, simulation time, and wall-clock elapsed.
    void printProgress(int step, double simTime);

    /// Reset the internal wall-clock timer (called automatically on construction).
    void resetTimer();

private:
    // Set from the (possibly '+'-joined) "format" token list; default false
    // so an unmentioned channel stays off.
    bool writeBinary_ = false;
    bool writeDat_    = false;
    bool writeVtk_    = false;
    bool writeVtkBin_ = false;
    bool writeVti_    = false;

    // VTI payload precision, from "vti_precision" (float32 default).
    FieldFormat vtiFormat_ = FieldFormat::VTI;

    // .pvd collection per field name (VTI only): (physical time, filename)
    struct PvdSeries {
        bool seeded = false;   // existing .pvd reloaded on first write
        std::vector<std::pair<double, std::string>> entries;
    };
    std::map<std::string, PvdSeries> pvd_;
    void updatePvd(const std::string& fieldName, const std::string& fileName,
                   double simTime);

    // Multiplies the VTK geometry — VTS node coordinates and VTK_BIN / VTI
    // Origin+Spacing (e.g. 1e9 → nm) — for ParaView visibility of nm-scale
    // grids; default 1.0 leaves physical (metre) coordinates.
    double coordScale_ = 1.0;

    using Clock = std::chrono::steady_clock;
    Clock::time_point t_start_;
};

} // namespace IO
} // namespace PhiX
