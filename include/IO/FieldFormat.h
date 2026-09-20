#pragma once

namespace PhiX {

// ---------------------------------------------------------------------------
// Output format selector for field IO
// ---------------------------------------------------------------------------
enum class FieldFormat {
    BINARY,   ///< Custom binary format (.field / .vfield)  — default; smallest
              ///< full-precision file; this is the restart format.
    DAT,      ///< ASCII text with x y z value columns — easy for gnuplot/matplotlib
    VTS,      ///< VTK XML StructuredGrid (.vts), ASCII float64 with EXPLICIT
              ///< node coordinates.  NOT recommended on uniform grids: ~41x
              ///< larger than VTI (≈163 B/cell vs 4 B/cell in 2D) and slow to
              ///< load — ParaView must parse every value as text and allocate
              ///< a full vtkPoints array.  Kept for compatibility; use VTI.
    VTK_BIN,  ///< Legacy VTK STRUCTURED_POINTS (.vtk), binary float32 big-endian
              ///< — ~40x smaller than ASCII VTS on large uniform grids.
              ///< NOTE: values are written as POINT_DATA on the nx*ny*nz
              ///< lattice, i.e. offset by half a cell relative to the
              ///< CellData-on-corner-lattice convention of VTS/VTI — the two
              ///< families do not overlay pixel-exactly.
    VTI,      ///< VTK XML ImageData (.vti), appended raw float32 — the mesh
              ///< is implicit (Origin + Spacing), ~41x smaller than ASCII
              ///< VTS (measured 858 MB → 21 MB at 2046x2556); ParaView's
              ///< preferred format for uniform grids.
              ///< Visualization output (float32) — use BINARY for restarts.
    VTI_F64   ///< As VTI but appended raw float64 — same geometry and layout,
              ///< twice the payload.  For post-processing that needs the
              ///< solver's full precision straight out of a .vti.
};

} // namespace PhiX
