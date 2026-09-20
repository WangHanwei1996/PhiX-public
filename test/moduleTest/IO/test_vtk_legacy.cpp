// ---------------------------------------------------------------------------
// module_io_vtk — legacy binary VTK writer (FieldFormat::VTK_BIN, v2.41.0)
//
// 1. 2D field with mesh origin 0 and uniform dx: output must be BYTE-IDENTICAL
//    to the hand-written writeLegacyVTK() of develop/PoolSectionGPU (the
//    function this writer adopts) — reference implementation embedded below.
// 2. Header/payload parse check on 2D and 3D fields: DIMENSIONS / ORIGIN /
//    SPACING / POINT_DATA / SCALARS lines, then float32 big-endian payload
//    bit-compared against float(cell value); ghost cells are poisoned so any
//    halo leakage flips the payload immediately.
// 3. writeField auto-creates missing parent directories (no silent failure).
// 4. OutputWriter accepts the "VTK_BIN" token and writes <name>_<step>.vtk;
//    unknown tokens still throw.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PhiX;
namespace fs = std::filesystem;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// Reference: verbatim logic of develop/PoolSectionGPU writeLegacyVTK()
// (2D, origin 0, uniform dx) — the adopted writer must reproduce it exactly.
static void writeLegacyVTKRef(const std::string& path, const std::string& name,
                              const double* host, int nx, int ny,
                              int sx, int sy, int g, double dx)
{
    std::ofstream out(path, std::ios::binary);
    out << "# vtk DataFile Version 3.0\n" << name << "\nBINARY\n"
        << "DATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS " << nx << " " << ny << " 1\n"
        << "ORIGIN 0 0 0\n"
        << "SPACING " << dx << " " << dx << " 1\n"
        << "POINT_DATA " << (size_t)nx * ny << "\n"
        << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
    std::vector<uint32_t> row(nx);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            const float v = (float)host[(i + g) + (size_t)sx * ((j + g) + (size_t)sy * g)];
            uint32_t u; std::memcpy(&u, &v, 4);
            row[i] = ((u & 0x000000FFu) << 24) | ((u & 0x0000FF00u) << 8)
                   | ((u & 0x00FF0000u) >> 8)  | ((u & 0xFF000000u) >> 24);
        }
        out.write(reinterpret_cast<const char*>(row.data()), (size_t)nx * 4);
    }
}

static std::vector<char> slurp(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    require(bool(ifs), "cannot open " + path);
    return std::vector<char>(std::istreambuf_iterator<char>(ifs),
                             std::istreambuf_iterator<char>());
}

// Fill physical cells with a deterministic pattern, poison everything else.
static void fillPattern(ScalarField& f) {
    f.fillCurr(1e300);   // ghost poison
    for (int k = 0; k < f.mesh.n[2]; ++k)
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i)
        f.curr[static_cast<std::size_t>(f.index(i, j, k))] =
            0.1 * i - 2.5 * j + 0.03 * k + 0.7;
}

// Parse the text header of a legacy VTK file; returns the payload offset.
struct VtkHeader {
    std::string name;
    int nx = 0, ny = 0, nz = 0;
    double ox = 0, oy = 0, oz = 0, dx = 0, dy = 0, dz = 0;
    std::size_t nPoints = 0;
    std::size_t payloadOffset = 0;
};

static VtkHeader parseHeader(const std::vector<char>& bytes) {
    VtkHeader h;
    std::size_t pos = 0;
    auto nextLine = [&]() -> std::string {
        std::size_t e = pos;
        while (e < bytes.size() && bytes[e] != '\n') ++e;
        std::string line(bytes.begin() + pos, bytes.begin() + e);
        pos = e + 1;
        return line;
    };
    require(nextLine() == "# vtk DataFile Version 3.0", "bad magic line");
    h.name = nextLine();
    require(nextLine() == "BINARY", "expected BINARY");
    require(nextLine() == "DATASET STRUCTURED_POINTS", "expected STRUCTURED_POINTS");
    std::string tok;
    { std::istringstream ss(nextLine());
      ss >> tok >> h.nx >> h.ny >> h.nz;
      require(tok == "DIMENSIONS", "expected DIMENSIONS"); }
    { std::istringstream ss(nextLine());
      ss >> tok >> h.ox >> h.oy >> h.oz;
      require(tok == "ORIGIN", "expected ORIGIN"); }
    { std::istringstream ss(nextLine());
      ss >> tok >> h.dx >> h.dy >> h.dz;
      require(tok == "SPACING", "expected SPACING"); }
    { std::istringstream ss(nextLine());
      ss >> tok >> h.nPoints;
      require(tok == "POINT_DATA", "expected POINT_DATA"); }
    { std::istringstream ss(nextLine());
      std::string nm, ty; int nc = 0;
      ss >> tok >> nm >> ty >> nc;
      require(tok == "SCALARS" && ty == "float" && nc == 1, "bad SCALARS line");
      require(nm == h.name, "SCALARS name mismatch"); }
    require(nextLine() == "LOOKUP_TABLE default", "expected LOOKUP_TABLE");
    h.payloadOffset = pos;
    return h;
}

// Bit-exact payload check: byte-swapped float32 must equal float(cell value).
static void checkPayload(const std::vector<char>& bytes, const VtkHeader& h,
                         const ScalarField& f, const std::string& tag) {
    require(bytes.size() == h.payloadOffset + h.nPoints * 4,
            tag + ": payload size mismatch");
    std::size_t off = h.payloadOffset;
    for (int k = 0; k < f.mesh.n[2]; ++k)
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i) {
        uint32_t be;
        std::memcpy(&be, bytes.data() + off, 4);
        uint32_t u = ((be & 0x000000FFu) << 24) | ((be & 0x0000FF00u) << 8)
                   | ((be & 0x00FF0000u) >> 8)  | ((be & 0xFF000000u) >> 24);
        float v;
        std::memcpy(&v, &u, 4);
        const float expect =
            static_cast<float>(f.curr[static_cast<std::size_t>(f.index(i, j, k))]);
        uint32_t ue, uv;
        std::memcpy(&ue, &expect, 4);
        std::memcpy(&uv, &v, 4);
        require(ue == uv, tag + ": payload value mismatch at ("
                          + std::to_string(i) + "," + std::to_string(j) + ","
                          + std::to_string(k) + ")");
        off += 4;
    }
}

int main() {
    const std::string dir = "module_io_vtk_out";
    fs::remove_all(dir);
    fs::remove_all("output");

    // --- 1. byte-identity with the PoolSectionGPU reference (2D, origin 0) ---
    {
        const int nx = 7, ny = 5;
        const double dx = 1.0e-7;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, dx, 0.0,
                                        ny, dx, 0.0);
        ScalarField f(mesh, "phi", 1);
        fillPattern(f);

        IO::writeField(f, dir + "/lib.vtk", FieldFormat::VTK_BIN);
        writeLegacyVTKRef(dir + "/ref.vtk", "phi", f.curr.data(), nx, ny,
                          f.storedDims[0], f.storedDims[1], f.ghost, dx);

        require(slurp(dir + "/lib.vtk") == slurp(dir + "/ref.vtk"),
                "2D: library VTK_BIN output differs from PoolSectionGPU "
                "reference writeLegacyVTK");
    }

    // --- 2. header + payload on 2D (non-zero origin) and 3D fields ---
    {
        const int nx = 6, ny = 4;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, 0.5, 1.0,
                                        ny, 0.25, -2.0);
        ScalarField f(mesh, "U", 1);
        fillPattern(f);
        IO::writeField(f, dir + "/u2d.vtk", FieldFormat::VTK_BIN);

        const auto bytes = slurp(dir + "/u2d.vtk");
        const auto h = parseHeader(bytes);
        require(h.name == "U", "2D: name");
        require(h.nx == nx && h.ny == ny && h.nz == 1, "2D: DIMENSIONS");
        require(h.ox == 1.0 && h.oy == -2.0 && h.oz == 0.0, "2D: ORIGIN");
        require(h.dx == 0.5 && h.dy == 0.25 && h.dz == 1.0, "2D: SPACING");
        require(h.nPoints == std::size_t(nx) * ny, "2D: POINT_DATA");
        checkPayload(bytes, h, f, "2D");
    }
    {
        const int nx = 5, ny = 3, nz = 2;
        Mesh mesh = Mesh::makeUniform3D(CoordSys::CARTESIAN, nx, 0.1, 0.0,
                                        ny, 0.2, 0.0, nz, 0.4, 3.0);
        ScalarField f(mesh, "c3", 1);
        fillPattern(f);
        IO::writeField(f, dir + "/c3d.vtk", FieldFormat::VTK_BIN);

        const auto bytes = slurp(dir + "/c3d.vtk");
        const auto h = parseHeader(bytes);
        require(h.nx == nx && h.ny == ny && h.nz == nz, "3D: DIMENSIONS");
        require(h.oz == 3.0, "3D: ORIGIN z");
        require(h.dx == 0.1 && h.dy == 0.2 && h.dz == 0.4, "3D: SPACING");
        require(h.nPoints == std::size_t(nx) * ny * nz, "3D: POINT_DATA");
        checkPayload(bytes, h, f, "3D");
    }

    // --- 3. missing parent directories are created (all formats) ---
    {
        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, 4, 1.0);
        ScalarField f(mesh, "m", 1);
        fillPattern(f);
        IO::writeField(f, dir + "/deep/nested/m.vtk",  FieldFormat::VTK_BIN);
        IO::writeField(f, dir + "/deep/bin/m.field",   FieldFormat::BINARY);
        require(fs::exists(dir + "/deep/nested/m.vtk"), "mkdir: .vtk missing");
        require(fs::exists(dir + "/deep/bin/m.field"),  "mkdir: .field missing");
    }

    // --- 4. OutputWriter VTK_BIN token ---
    {
        const int nx = 6, ny = 4;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, 0.5, 0.0,
                                        ny, 0.5, 0.0);
        ScalarField f(mesh, "eta", 1);
        fillPattern(f);
        f.allocDevice();
        f.uploadAllToDevice();

        nlohmann::json cfg = { {"print_interval", 1},
                               {"write_interval", 1},
                               {"format", "VTK_BIN"} };
        IO::OutputWriter writer(cfg);
        writer.writeFields(f, 42, 0.0);
        require(fs::exists("output/eta_42.vtk"), "OutputWriter: .vtk missing");
        require(!fs::exists("output/eta_42.vts"), "OutputWriter: unexpected .vts");

        const auto bytes = slurp("output/eta_42.vtk");
        const auto h = parseHeader(bytes);
        checkPayload(bytes, h, f, "OutputWriter");

        bool threw = false;
        try {
            nlohmann::json bad = { {"print_interval", 1},
                                   {"write_interval", 1},
                                   {"format", "VTU"} };
            IO::OutputWriter w2(bad);
        } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "OutputWriter: unknown format must throw");
    }

    fs::remove_all(dir);
    fs::remove_all("output");
    std::printf("module_io_vtk: ALL PASSED\n");
    return 0;
}
