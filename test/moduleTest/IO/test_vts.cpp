// ---------------------------------------------------------------------------
// module_io_vts — VTK XML StructuredGrid writer (FieldFormat::VTS).
//
// VTS is the legacy ParaView path: ASCII float64 with an EXPLICIT node
// coordinate array.  On a uniform grid that costs ~163 B/cell against 4 B/cell
// for VTI, and ParaView must strtod every token and materialise a vtkPoints
// array.  The writer is therefore frozen, not improved — this test pins its
// current byte-level output so any accidental change is caught.  It records
// behaviour (including the VectorField path ignoring coordScale); it does not
// endorse it.  New work should use FieldFormat::VTI — see module_io_vti.
//
// 1. Structure: StructuredGrid, WholeExtent/Piece on the corner-node lattice,
//    ascii DataArrays.
// 2. Points: (nx+1)(ny+1)(nz+1) nodes — in 2D the z lattice has TWO layers,
//    which is what makes the geometry cost ~2x the cell count — each written
//    as scientific(12) x y z scaled by coordScale, 2D slab dz = dx.
// 3. CellData: nx*ny*nz values, scientific(12), ghost-free.
// 4. VectorField: 3 interleaved components (N<3 zero-padded) and coordScale
//    is NOT applied on this path.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/VectorField.h"
#include "IO/FieldIO.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PhiX;
namespace fs = std::filesystem;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// The exact text the writer produces for one double.
static std::string sci12(double v) {
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(12) << v;
    return ss.str();
}

// Lines of `blob` between the n-th "<DataArray" and the following "</DataArray>",
// with blank lines dropped.
static std::vector<std::string> dataArrayLines(const std::string& blob, int n) {
    std::size_t p = 0;
    for (int i = 0; i <= n; ++i) {
        p = blob.find("<DataArray", (i == 0) ? 0 : p + 1);
        require(p != std::string::npos, "DataArray not found");
    }
    const std::size_t open = blob.find('\n', p);
    const std::size_t close = blob.find("</DataArray>", open);
    require(close != std::string::npos, "unterminated DataArray");
    std::istringstream body(blob.substr(open + 1, close - open - 1));
    std::vector<std::string> out;
    std::string line;
    while (std::getline(body, line))
        if (line.find_first_not_of(" \t\r") != std::string::npos)
            out.push_back(line);
    return out;
}

int main() {
try {
    fs::create_directories("vts_test_out");

    const int NX = 7, NY = 5;
    const double DX = 0.25, DY = 0.5, OX = 1.0, OY = -2.0;
    const double scale = 2.0;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    NX, DX, OX, NY, DY, OY);

    // === 1-3. ScalarField =================================================
    ScalarField f(mesh, "phi", 1);
    f.fillCurr(1e200);                       // poison ghosts
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            f.curr[f.index(i, j, 0)] = 0.1 * i - 0.7 * j + 0.013 * i * j;

    const std::string path = "vts_test_out/phi.vts";
    IO::writeField(f, path, FieldFormat::VTS, scale);
    const std::string blob = slurp(path);

    require(blob.find("<VTKFile type=\"StructuredGrid\"") != std::string::npos,
            "VTKFile type is StructuredGrid");
    require(blob.find("WholeExtent=\"0 7 0 5 0 1\"") != std::string::npos,
            "WholeExtent covers the corner-node lattice");
    require(blob.find("<Piece Extent=\"0 7 0 5 0 1\">") != std::string::npos,
            "Piece Extent matches WholeExtent");
    require(blob.find("format=\"ascii\"") != std::string::npos,
            "VTS is ASCII (this is the cost being documented)");
    require(blob.find("format=\"appended\"") == std::string::npos,
            "VTS has no appended section");

    // Points: the 2D node lattice has two z layers -> 2x(nx+1)(ny+1) nodes
    {
        const auto pts = dataArrayLines(blob, 0);
        require(pts.size() ==
                static_cast<std::size_t>(NX + 1) * (NY + 1) * 2,
                "Points array holds (nx+1)(ny+1)(nz+1) nodes, nz+1 == 2 in 2D");
        std::size_t idx = 0;
        for (int k = 0; k <= 1; ++k)
            for (int j = 0; j <= NY; ++j)
                for (int i = 0; i <= NX; ++i) {
                    const std::string want =
                        "          " + sci12((OX + i * DX) * scale)
                      + " "          + sci12((OY + j * DY) * scale)
                      + " "          + sci12((0.0 + k * DX) * scale);
                    require(pts[idx++] == want,
                            "node line exact (coordScale, 2D slab dz = dx)");
                }
    }

    // CellData: one scientific(12) value per physical cell, ghosts excluded
    {
        const auto vals = dataArrayLines(blob, 1);
        require(vals.size() == static_cast<std::size_t>(NX) * NY,
                "CellData holds nx*ny values");
        std::size_t idx = 0;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                require(vals[idx++] ==
                            "          " + sci12(f.curr[f.index(i, j, 0)]),
                        "cell value exact, ghost-free");
    }

    // === 4. VectorField ====================================================
    {
        VectorField vf(mesh, "v", 2, 1);        // N = 2 -> padded to 3
        for (int c = 0; c < 2; ++c) {
            vf[c].fillCurr(1e200);
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i)
                    vf[c].curr[vf[c].index(i, j, 0)] =
                        (c + 1) * (0.3 * i - 0.11 * j);
        }
        const std::string vpath = "vts_test_out/v.vts";
        IO::writeField(vf, vpath, FieldFormat::VTS, scale);
        const std::string vblob = slurp(vpath);

        require(vblob.find("<CellData Vectors=\"v\">") != std::string::npos,
                "vector VTS uses Vectors CellData");
        require(vblob.find("NumberOfComponents=\"3\" format=\"ascii\"")
                != std::string::npos, "vector VTS pads to 3 components");

        // Recorded quirk: the VectorField VTS path ignores coordScale.
        const auto pts = dataArrayLines(vblob, 0);
        require(pts[0] == "          " + sci12(OX) + " " + sci12(OY)
                          + " " + sci12(0.0),
                "vector VTS writes UNSCALED coordinates (recorded quirk)");

        const auto vals = dataArrayLines(vblob, 1);
        require(vals.size() == static_cast<std::size_t>(NX) * NY,
                "vector CellData holds nx*ny lines");
        std::size_t idx = 0;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                const std::string want =
                    "          " + sci12(vf[0].curr[vf[0].index(i, j, 0)])
                  + " "          + sci12(vf[1].curr[vf[1].index(i, j, 0)])
                  + " 0.0";
                require(vals[idx++] == want,
                        "vector cell line interleaved and zero-padded");
            }
    }

    fs::remove_all("vts_test_out");
    std::printf("module_io_vts: all passed\n");
    return 0;
} catch (const std::exception& ex) {
    std::printf("FAIL: %s\n", ex.what());
    return 1;
}
}
