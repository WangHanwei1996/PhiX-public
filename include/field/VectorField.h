#pragma once

#include "field/FieldLayout.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"

#include <string>
#include <vector>
#include <cstddef>

namespace PhiX {

// ---------------------------------------------------------------------------
// VectorField
//
// Vector-valued field stored as SoA (Structure of Arrays):
// N independent ScalarField objects, one per component.
//
// Component naming convention:
//   nComponents == 3  ->  name_x, name_y, name_z
//   otherwise        ->  name_0, name_1, ...
//
// Ghost layout, indexing, and GPU management are all delegated to the
// underlying ScalarField objects (see ScalarField.h for details).
//
// Output formats:
//   BINARY (.vfield) : small binary file; readable by VectorField::readFromFile
//   DAT    (.dat)    : ASCII, columns: x y z v0 v1 ...
//   VTI    (.vti)    : VTK XML ImageData, appended raw float32 vector
//                      CellData — ParaView's format of choice on uniform
//                      grids (VTI_F64 for float64)
//   VTS    (.vts)    : VTK XML StructuredGrid with vector CellData; ASCII,
//                      ~41x larger than VTI — kept for compatibility
// ---------------------------------------------------------------------------

class VectorField {
public:
    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------

    // ghost defaults to 1 to match ScalarField convention.
    explicit VectorField(const Mesh& mesh,
                         const std::string& name,
                         int nComponents,
                         int ghost = 1);

    explicit VectorField(const FieldLayout& layout,
                         const std::string& name,
                         int nComponents);

    // Non-copyable (each component ScalarField owns GPU memory)
    VectorField(const VectorField&)            = delete;
    VectorField& operator=(const VectorField&) = delete;

    // Movable
    VectorField(VectorField&& other) noexcept;
    VectorField& operator=(VectorField&& other) noexcept;

    ~VectorField() = default;

    // -----------------------------------------------------------------------
    // Component access
    // -----------------------------------------------------------------------
    ScalarField&       operator[](int c);
    const ScalarField& operator[](int c) const;

    int nComponents() const { return static_cast<int>(components_.size()); }

    // Convenience accessors for 3-component fields
    ScalarField& x() { return (*this)[0]; }
    ScalarField& y() { return (*this)[1]; }
    ScalarField& z() { return (*this)[2]; }
    const ScalarField& x() const { return (*this)[0]; }
    const ScalarField& y() const { return (*this)[1]; }
    const ScalarField& z() const { return (*this)[2]; }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------
    FieldLayout  layout;
    const Mesh&  mesh;    // shared mesh (from first component)
    int          ghost;   // ghost layers
    std::string  name;    // base name (components are name_x/y/z or name_0...)

    // -----------------------------------------------------------------------
    // Initialisation helpers (CPU) — delegates to all components
    // -----------------------------------------------------------------------
    void fill(double value);
    void fillCurr(double value);
    void fillPrev(double value);

    // Initialize each component's curr by evaluating fn(x, y, z) which returns
    // an array-like of N values (e.g. std::array<double,N> or a lambda returning
    // a value per component index).
    // Overload (a): fn(x,y,z) returns std::array<double,N> (or similar indexable).
    template<typename Fn>
    void initialize(Fn fn) {
        const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];
        const int nc = nComponents();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            double x = mesh.coord(0, i);
            double y = (mesh.dim >= 2) ? mesh.coord(1, j) : 0.0;
            double z = (mesh.dim >= 3) ? mesh.coord(2, k) : 0.0;
            auto vals = fn(x, y, z);
            int idx = layout.index(i, j, k);
            for (int c = 0; c < nc; ++c)
                (*this)[c].curr[static_cast<std::size_t>(idx)] = vals[c];
        }
    }

    // Overload (b): initialize a single component via fn(x,y,z)->double.
    template<typename Fn>
    void initializeComponent(int comp, Fn fn) {
        (*this)[comp].initialize(fn);
    }

    // -----------------------------------------------------------------------
    // Time-stepping — delegates to all components
    // -----------------------------------------------------------------------
    void advanceTimeLevelCPU();
    void advanceTimeLevelGPU();

    // -----------------------------------------------------------------------
    // GPU management — delegates to all components
    // -----------------------------------------------------------------------
    bool deviceAllocated() const;   // true if ALL components are allocated

    void allocDevice();
    void freeDevice();

    void uploadCurrToDevice()  const;
    void uploadPrevToDevice()  const;
    void uploadAllToDevice()   const;

    void downloadCurrFromDevice();
    void downloadPrevFromDevice();
    void downloadAllFromDevice();

    // -----------------------------------------------------------------------
    // IO
    //
    // BINARY (.vfield):
    //   Text header:
    //     "# PhiX VectorField\n"
    //     "name         <name>\n"
    //     "nComponents  <N>\n"
    //     "nx <nx>  ny <ny>  nz <nz>\n"
    //     "ghost        <ghost>\n"
    //     "---\n"
    //   Binary data:
    //     For each component c in [0, N):
    //       nx*ny*nz doubles, row-major (x fastest, z slowest)
    //
    // DAT (.dat):
    //   One line per cell:  x  y  z  v0  v1  ...
    //
    // VTI (.vti)  <-- recommended for ParaView on uniform grids:
    //   VTK XML ImageData, appended raw float32 vector CellData, interleaved
    //   v0 v1 v2 per cell and zero-padded to 3 components (VTK requires 3).
    //   Geometry is implicit (Origin + Spacing), scaled by coordScale.
    //   VTI_F64 is the same layout in float64.
    //
    // VTS (.vts):
    //   VTK XML StructuredGrid with vector CellData, ASCII float64.
    //   DataArray is interleaved: for each cell (x fastest) output v0 v1 ... vN-1.
    //   Kept for compatibility — ~41x larger than VTI on uniform grids and
    //   slow to load; note this path ignores coordScale.  Prefer VTI.
    // -----------------------------------------------------------------------
    void write(const std::string& path,
               FieldFormat fmt = FieldFormat::BINARY,
               double coordScale = 1.0) const;

    static VectorField readFromFile(const Mesh& mesh,
                                    const std::string& path,
                                    int ghost = 1);

    // Human-readable summary for each component
    void print() const;

private:
    std::vector<ScalarField> components_;

    static std::string componentName(const std::string& baseName,
                                     int c, int nComp);
};

} // namespace PhiX
