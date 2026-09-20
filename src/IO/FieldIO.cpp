#include "IO/FieldIO.h"
#include "core/Error.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace PhiX {
namespace IO {

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers
// ---------------------------------------------------------------------------
namespace {

// Number of physical cells (no ghost)
std::size_t physicalSize(const Mesh& mesh) {
    return static_cast<std::size_t>(mesh.n[0])
         * mesh.n[1]
         * mesh.n[2];
}

// Create the parent directory of `path` if it does not exist yet, so a
// missing output directory raises no silent ofstream failure downstream.
void ensureParentDir(const std::string& path) {
    namespace fs = std::filesystem;
    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
}

// Convert physical (i,j,k) into the stored flat index (ghost-padded)
int physIdx(const ScalarField& f, int i, int j, int k) {
    return (i + f.ghost)
         + f.storedDims[0] * ((j + f.ghost)
         + f.storedDims[1] *  (k + f.ghost));
}

// ---------------------------------------------------------------------------
// Common header parsing helpers
// ---------------------------------------------------------------------------

struct ScalarFileHeader {
    std::string fieldName;
    int nx = 0, ny = 0, nz = 0, ghost = 0;
};

struct VectorFileHeader {
    std::string fieldName;
    int nx = 0, ny = 0, nz = 0, ghost = 0, nComponents = 0;
};

ScalarFileHeader parseScalarHeader(std::ifstream& ifs, const std::string& path) {
    ScalarFileHeader h;
    bool headerDone = false;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "---") { headerDone = true; break; }

        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key) || key[0] == '#') continue;

        if      (key == "name")  { ss >> h.fieldName; }
        else if (key == "ghost") { ss >> h.ghost; }
        else {
            std::istringstream row(line);
            std::string tok;
            while (row >> tok) {
                if (tok == "nx")      row >> h.nx;
                else if (tok == "ny") row >> h.ny;
                else if (tok == "nz") row >> h.nz;
            }
        }
    }
    if (!headerDone)
        throw std::runtime_error("IO: missing '---' header terminator in " + path);
    return h;
}

VectorFileHeader parseVectorHeader(std::ifstream& ifs, const std::string& path) {
    VectorFileHeader h;
    bool headerDone = false;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "---") { headerDone = true; break; }

        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key) || key[0] == '#') continue;

        if      (key == "name")        { ss >> h.fieldName; }
        else if (key == "nComponents") { ss >> h.nComponents; }
        else if (key == "ghost")       { ss >> h.ghost; }
        else {
            std::istringstream row(line);
            std::string tok;
            while (row >> tok) {
                if (tok == "nx")      row >> h.nx;
                else if (tok == "ny") row >> h.ny;
                else if (tok == "nz") row >> h.nz;
            }
        }
    }
    if (!headerDone)
        throw std::runtime_error("IO: missing '---' header terminator in " + path);
    return h;
}

// ---------------------------------------------------------------------------
// DAT header parsing helpers
// ---------------------------------------------------------------------------

// ScalarField DAT header: lines starting with '#', stops (and seeks back)
// when a non-'#' line is encountered.
// Expected comment lines (order flexible):
//   # PhiX ScalarField - DAT
//   # name: <name>
//   # nx N  ny N  nz N
//   # x y z value
struct ScalarDatHeader {
    std::string fieldName;
    int nx = 0, ny = 0, nz = 0;
};

ScalarDatHeader parseScalarDatHeader(std::ifstream& ifs, const std::string& path) {
    ScalarDatHeader h;
    std::string line;
    while (true) {
        std::streampos pos = ifs.tellg();
        if (!std::getline(ifs, line)) break;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] != '#') {
            ifs.seekg(pos);   // put back the first data line
            break;
        }
        // Strip leading '#' and optional space
        std::string rest = line.substr(1);
        std::istringstream ss(rest);
        std::string key;
        if (!(ss >> key)) continue;
        if (key == "name:") {
            ss >> h.fieldName;
        } else {
            // scan for nx/ny/nz tokens anywhere on the line
            std::istringstream row(rest);
            std::string tok;
            while (row >> tok) {
                if      (tok == "nx") row >> h.nx;
                else if (tok == "ny") row >> h.ny;
                else if (tok == "nz") row >> h.nz;
            }
        }
    }
    return h;
}

// VectorField DAT header: same pattern.
// Expected comment lines:
//   # PhiX VectorField - DAT
//   # name: <name>  nComponents: N
//   # nx N  ny N  nz N
//   # x y z v0 v1 ...
struct VectorDatHeader {
    std::string fieldName;
    int nx = 0, ny = 0, nz = 0, nComponents = 0;
};

VectorDatHeader parseVectorDatHeader(std::ifstream& ifs, const std::string& path) {
    VectorDatHeader h;
    std::string line;
    while (true) {
        std::streampos pos = ifs.tellg();
        if (!std::getline(ifs, line)) break;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] != '#') {
            ifs.seekg(pos);
            break;
        }
        std::string rest = line.substr(1);
        std::istringstream row(rest);
        std::string tok;
        while (row >> tok) {
            if      (tok == "name:")        row >> h.fieldName;
            else if (tok == "nComponents:") row >> h.nComponents;
            else if (tok == "nx")           row >> h.nx;
            else if (tok == "ny")           row >> h.ny;
            else if (tok == "nz")           row >> h.nz;
        }
    }
    return h;
}

// Detect format from first line: returns true if DAT, false if BINARY.
// Seeks back to start of file afterwards.
bool isDatFormat(std::ifstream& ifs) {
    std::string line;
    std::getline(ifs, line);
    ifs.seekg(0);
    return line.find("- DAT") != std::string::npos;
}

// ===================================================================
// ScalarField write helpers
// ===================================================================

void writeScalarBinary(const ScalarField& f, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = f.mesh;

    // Text header
    ofs << "# PhiX ScalarField\n";
    ofs << "name    " << f.name << "\n";
    ofs << "nx " << mesh.n[0]
        << "  ny " << mesh.n[1]
        << "  nz " << mesh.n[2] << "\n";
    ofs << "ghost   " << f.ghost << "\n";
    ofs << "---\n";

    // Binary data: physical cells only, row-major (x fastest)
    const std::size_t phySize = physicalSize(mesh);
    std::vector<double> buf;
    buf.reserve(phySize);

    for (int k = 0; k < mesh.n[2]; ++k)
        for (int j = 0; j < mesh.n[1]; ++j)
            for (int i = 0; i < mesh.n[0]; ++i)
                buf.push_back(f.curr[physIdx(f, i, j, k)]);

    ofs.write(reinterpret_cast<const char*>(buf.data()),
              static_cast<std::streamsize>(phySize * sizeof(double)));
}

void writeScalarDat(const ScalarField& f, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = f.mesh;

    ofs << "# PhiX ScalarField - DAT\n";
    ofs << "# name: " << f.name << "\n";
    ofs << "# nx " << mesh.n[0]
        << "  ny " << mesh.n[1]
        << "  nz " << mesh.n[2] << "\n";
    ofs << "# x y z value\n";

    ofs << std::scientific << std::setprecision(12);
    const bool is2D = (mesh.n[2] == 1);
    for (int k = 0; k < mesh.n[2]; ++k)
        for (int j = 0; j < mesh.n[1]; ++j)
            for (int i = 0; i < mesh.n[0]; ++i) {
                double x = mesh.origin[0] + (i + 0.5) * mesh.d[0];
                double y = mesh.origin[1] + (j + 0.5) * mesh.d[1];
                double z = is2D ? mesh.origin[2] : mesh.origin[2] + (k + 0.5) * mesh.d[2];
                ofs << x << "  " << y << "  " << z << "  "
                    << f.curr[physIdx(f, i, j, k)] << "\n";
            }
}

void writeScalarVts(const ScalarField& f, const std::string& path,
                    double coordScale) {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = f.mesh;
    const int nx = mesh.n[0];
    const int ny = mesh.n[1];
    const int nz = mesh.n[2];

    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"StructuredGrid\" version=\"0.1\""
           " byte_order=\"LittleEndian\">\n";
    ofs << "  <StructuredGrid WholeExtent=\""
        << "0 " << nx << " 0 " << ny << " 0 " << nz << "\">\n";
    ofs << "    <Piece Extent=\""
        << "0 " << nx << " 0 " << ny << " 0 " << nz << "\">\n";

    // --- corner-node coordinates ---
    ofs << "      <Points>\n";
    ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\""
           " format=\"ascii\">\n";
    ofs << std::scientific << std::setprecision(12);
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                // For 2D the 3rd dimension is inactive (d[2] is a unit
                // placeholder); use dx as a thin proportional slab thickness so
                // the z extent doesn't dwarf the in-plane geometry in ParaView.
                const double dz = (mesh.dim >= 3) ? mesh.d[2] : mesh.d[0];
                double x = (mesh.origin[0] + i * mesh.d[0]) * coordScale;
                double y = (mesh.origin[1] + j * mesh.d[1]) * coordScale;
                double z = (mesh.origin[2] + k * dz)        * coordScale;
                ofs << "          " << x << " " << y << " " << z << "\n";
            }
    ofs << "        </DataArray>\n";
    ofs << "      </Points>\n";

    // --- cell-centred field values ---
    ofs << "      <CellData Scalars=\"" << f.name << "\">\n";
    ofs << "        <DataArray type=\"Float64\" Name=\"" << f.name
        << "\" format=\"ascii\">\n";
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                ofs << "          "
                    << f.curr[physIdx(f, i, j, k)] << "\n";
    ofs << "        </DataArray>\n";
    ofs << "      </CellData>\n";

    ofs << "    </Piece>\n";
    ofs << "  </StructuredGrid>\n";
    ofs << "</VTKFile>\n";
}

// Legacy VTK STRUCTURED_POINTS, binary float32 (big-endian per the legacy
// spec).  Cell values are written as POINT_DATA on the nx*ny*nz lattice —
// ~40x smaller than the ASCII VTS writer on large uniform grids (24M cells:
// 96 MB vs 3.9 GB).  ParaView/VisIt read it natively.
// Adopted from develop/PoolSectionGPU (PHIX_IMPROVEMENTS.md P-5).
void writeScalarVtkLegacy(const ScalarField& f, const std::string& path,
                          double coordScale) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = f.mesh;
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];
    const double dy = (mesh.dim >= 2) ? mesh.d[1] : mesh.d[0];
    const double dz = (mesh.dim >= 3) ? mesh.d[2] : 1.0;

    ofs << "# vtk DataFile Version 3.0\n" << f.name << "\nBINARY\n"
        << "DATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n"
        << "ORIGIN "  << mesh.origin[0] * coordScale
        << " "        << mesh.origin[1] * coordScale
        << " "        << mesh.origin[2] * coordScale << "\n"
        << "SPACING " << mesh.d[0] * coordScale
        << " "        << dy * coordScale
        << " "        << ((mesh.dim >= 3) ? dz * coordScale : dz) << "\n"
        << "POINT_DATA " << physicalSize(mesh) << "\n"
        << "SCALARS " << f.name << " float 1\nLOOKUP_TABLE default\n";

    std::vector<uint32_t> row(nx);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const float v = static_cast<float>(f.curr[physIdx(f, i, j, k)]);
                uint32_t u;
                std::memcpy(&u, &v, 4);
                // big-endian per the legacy VTK binary spec
                row[i] = ((u & 0x000000FFu) << 24) | ((u & 0x0000FF00u) << 8)
                       | ((u & 0x00FF0000u) >> 8)  | ((u & 0xFF000000u) >> 24);
            }
            ofs.write(reinterpret_cast<const char*>(row.data()),
                      static_cast<std::streamsize>(nx) * 4);
        }
}

// VTK XML ImageData (.vti), appended raw.  The uniform mesh is implicit
// (Origin + Spacing), so the file is essentially the payload — ~41x smaller
// than the ASCII VTS (measured 858 MB -> 21 MB at 2046x2556, i.e. ~163 B/cell
// vs 4 B/cell in 2D).  Values go as CellData on the corner-node lattice (same
// geometry convention as the VTS writer).  float32 is the visualization
// default; VTI_F64 keeps the solver's precision — BINARY stays the restart
// format either way.
//
// The XML header, shared by the scalar and vector writers.  Built in a local
// ostringstream so the global stream format state is never touched.
// dataKind is "Scalars" or "Vectors"; the NumberOfComponents attribute is
// emitted only for nComponents > 1, so the scalar header is unchanged.
std::string vtiHeader(const Mesh& mesh, double coordScale,
                      const char* dataKind, const std::string& name,
                      int nComponents, bool f64) {
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];
    const double dy = (mesh.dim >= 2) ? mesh.d[1] : mesh.d[0];
    // 2D: thin proportional slab, same convention as the VTS writer
    const double dz = (mesh.dim >= 3) ? mesh.d[2] : mesh.d[0];

    std::ostringstream head;
    head << std::scientific << std::setprecision(12);
    head << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"ImageData\" version=\"1.0\""
            " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
         << "  <ImageData WholeExtent=\"0 " << nx << " 0 " << ny
         << " 0 " << nz << "\" Origin=\""
         << mesh.origin[0] * coordScale << " "
         << mesh.origin[1] * coordScale << " "
         << mesh.origin[2] * coordScale << "\" Spacing=\""
         << mesh.d[0] * coordScale << " "
         << dy * coordScale << " "
         << dz * coordScale << "\">\n"
         << "    <Piece Extent=\"0 " << nx << " 0 " << ny
         << " 0 " << nz << "\">\n"
         << "      <CellData " << dataKind << "=\"" << name << "\">\n"
         << "        <DataArray type=\"" << (f64 ? "Float64" : "Float32")
         << "\" Name=\"" << name << "\"";
    if (nComponents > 1)
        head << " NumberOfComponents=\"" << nComponents << "\"";
    head << " format=\"appended\" offset=\"0\"/>\n"
         << "      </CellData>\n"
         << "    </Piece>\n"
         << "  </ImageData>\n"
         << "  <AppendedData encoding=\"raw\">\n_";
    return head.str();
}

void writeScalarVti(const ScalarField& f, const std::string& path,
                    double coordScale, bool f64) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = f.mesh;
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];

    ofs << vtiHeader(mesh, coordScale, "Scalars", f.name, 1, f64);

    const std::size_t wordSize = f64 ? sizeof(double) : sizeof(float);
    const std::uint64_t nBytes =
        static_cast<std::uint64_t>(physicalSize(mesh)) * wordSize;
    ofs.write(reinterpret_cast<const char*>(&nBytes), 8);

    const std::size_t rowWords = static_cast<std::size_t>(nx);
    std::vector<float>  row32(f64 ? 0 : rowWords);
    std::vector<double> row64(f64 ? rowWords : 0);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const double v = f.curr[physIdx(f, i, j, k)];
                if (f64) row64[static_cast<std::size_t>(i)] = v;
                else     row32[static_cast<std::size_t>(i)] =
                             static_cast<float>(v);
            }
            ofs.write(f64 ? reinterpret_cast<const char*>(row64.data())
                          : reinterpret_cast<const char*>(row32.data()),
                      static_cast<std::streamsize>(rowWords * wordSize));
        }

    ofs << "\n  </AppendedData>\n</VTKFile>\n";
}

// ===================================================================
// VectorField write helpers
// ===================================================================

void writeVectorBinary(const VectorField& vf, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = vf.mesh;
    const int N  = vf.nComponents();
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];

    ofs << "# PhiX VectorField\n";
    ofs << "name         " << vf.name << "\n";
    ofs << "nComponents  " << N    << "\n";
    ofs << "nx " << nx << "  ny " << ny << "  nz " << nz << "\n";
    ofs << "ghost        " << vf.ghost << "\n";
    ofs << "---\n";

    const std::size_t phySize = static_cast<std::size_t>(nx) * ny * nz;
    std::vector<double> buf(phySize);

    for (int c = 0; c < N; ++c) {
        const ScalarField& sf = vf[c];
        std::size_t idx = 0;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    buf[idx++] = sf.curr[physIdx(sf, i, j, k)];
        ofs.write(reinterpret_cast<const char*>(buf.data()),
                  static_cast<std::streamsize>(phySize * sizeof(double)));
    }
}

void writeVectorDat(const VectorField& vf, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = vf.mesh;
    const int N  = vf.nComponents();
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];

    ofs << "# PhiX VectorField - DAT\n";
    ofs << "# name: " << vf.name << "  nComponents: " << N << "\n";
    ofs << "# nx " << nx << "  ny " << ny << "  nz " << nz << "\n";
    ofs << "# x y z";
    for (int c = 0; c < N; ++c) ofs << " v" << c;
    ofs << "\n";

    ofs << std::scientific << std::setprecision(12);

    const bool is2Dv = (nz == 1);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                double x = mesh.origin[0] + (i + 0.5) * mesh.d[0];
                double y = mesh.origin[1] + (j + 0.5) * mesh.d[1];
                double z = is2Dv ? mesh.origin[2] : mesh.origin[2] + (k + 0.5) * mesh.d[2];
                ofs << x << "  " << y << "  " << z;
                for (int c = 0; c < N; ++c) {
                    const ScalarField& sf = vf[c];
                    ofs << "  " << sf.curr[physIdx(sf, i, j, k)];
                }
                ofs << "\n";
            }
}

void writeVectorVts(const VectorField& vf, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = vf.mesh;
    const int N  = vf.nComponents();
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];

    // VTK requires vectors to have exactly 3 components; pad with zeros if N<3.
    const int vtk_nc = (N <= 3) ? 3 : N;

    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"StructuredGrid\" version=\"0.1\""
           " byte_order=\"LittleEndian\">\n";
    ofs << "  <StructuredGrid WholeExtent=\""
        << "0 " << nx << " 0 " << ny << " 0 " << nz << "\">\n";
    ofs << "    <Piece Extent=\""
        << "0 " << nx << " 0 " << ny << " 0 " << nz << "\">\n";

    // Corner-node coordinates
    ofs << "      <Points>\n";
    ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\""
           " format=\"ascii\">\n";
    ofs << std::scientific << std::setprecision(12);
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i)
                ofs << "          "
                    << mesh.origin[0] + i * mesh.d[0] << " "
                    << mesh.origin[1] + j * mesh.d[1] << " "
                    << mesh.origin[2] + k * mesh.d[2] << "\n";
    ofs << "        </DataArray>\n";
    ofs << "      </Points>\n";

    // Vector cell data (interleaved: v0 v1 v2 per cell)
    ofs << "      <CellData Vectors=\"" << vf.name << "\">\n";
    ofs << "        <DataArray type=\"Float64\" Name=\"" << vf.name
        << "\" NumberOfComponents=\"" << vtk_nc << "\" format=\"ascii\">\n";

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                ofs << "         ";
                for (int c = 0; c < N; ++c) {
                    const ScalarField& sf = vf[c];
                    ofs << " " << sf.curr[physIdx(sf, i, j, k)];
                }
                // Pad to vtk_nc if needed
                for (int c = N; c < vtk_nc; ++c) ofs << " 0.0";
                ofs << "\n";
            }

    ofs << "        </DataArray>\n";
    ofs << "      </CellData>\n";
    ofs << "    </Piece>\n";
    ofs << "  </StructuredGrid>\n";
    ofs << "</VTKFile>\n";
}


// VectorField counterpart of writeScalarVti: one appended array of
// interleaved components (v0 v1 v2 per cell).  VTK wants exactly 3
// components for a vector, so N < 3 is zero-padded — the same convention as
// writeVectorVts.
void writeVectorVti(const VectorField& vf, const std::string& path,
                    double coordScale, bool f64) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("IO::writeField: cannot open file: " + path);

    const Mesh& mesh = vf.mesh;
    const int N  = vf.nComponents();
    const int nx = mesh.n[0], ny = mesh.n[1], nz = mesh.n[2];
    const int vtk_nc = (N <= 3) ? 3 : N;

    ofs << vtiHeader(mesh, coordScale, "Vectors", vf.name, vtk_nc, f64);

    const std::size_t wordSize = f64 ? sizeof(double) : sizeof(float);
    const std::uint64_t nBytes =
        static_cast<std::uint64_t>(physicalSize(mesh))
      * static_cast<std::uint64_t>(vtk_nc) * wordSize;
    ofs.write(reinterpret_cast<const char*>(&nBytes), 8);

    const std::size_t rowWords =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(vtk_nc);
    std::vector<float>  row32(f64 ? 0 : rowWords);
    std::vector<double> row64(f64 ? rowWords : 0);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i)
                for (int c = 0; c < vtk_nc; ++c) {
                    double v = 0.0;
                    if (c < N) {
                        const ScalarField& sf = vf[c];
                        v = sf.curr[physIdx(sf, i, j, k)];
                    }
                    const std::size_t w =
                        static_cast<std::size_t>(i)
                      * static_cast<std::size_t>(vtk_nc)
                      + static_cast<std::size_t>(c);
                    if (f64) row64[w] = v;
                    else     row32[w] = static_cast<float>(v);
                }
            ofs.write(f64 ? reinterpret_cast<const char*>(row64.data())
                          : reinterpret_cast<const char*>(row32.data()),
                      static_cast<std::streamsize>(rowWords * wordSize));
        }

    ofs << "\n  </AppendedData>\n</VTKFile>\n";
}

} // anonymous namespace

// ===========================================================================
// Public API — ScalarField
// ===========================================================================

void writeField(const ScalarField& f,
                const std::string& path,
                FieldFormat fmt,
                double coordScale) {
    ensureParentDir(path);
    switch (fmt) {
        case FieldFormat::BINARY:  writeScalarBinary(f, path); break;
        case FieldFormat::DAT:     writeScalarDat(f, path);    break;
        case FieldFormat::VTS:     writeScalarVts(f, path, coordScale); break;
        case FieldFormat::VTK_BIN: writeScalarVtkLegacy(f, path, coordScale); break;
        case FieldFormat::VTI:
            writeScalarVti(f, path, coordScale, /*f64=*/false); break;
        case FieldFormat::VTI_F64:
            writeScalarVti(f, path, coordScale, /*f64=*/true);  break;
    }
}

ScalarField readScalarField(const Mesh& mesh,
                             const std::string& path,
                             int ghost) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("IO::readScalarField: cannot open file: " + path);

    if (isDatFormat(ifs)) {
        // ---- DAT (text) format ----
        const auto h = parseScalarDatHeader(ifs, path);

        if (h.nx != mesh.n[0] || h.ny != mesh.n[1] || h.nz != mesh.n[2])
            throw std::runtime_error(
                "IO::readScalarField: mesh dimensions in file ("
                + std::to_string(h.nx) + "x" + std::to_string(h.ny) + "x" + std::to_string(h.nz)
                + ") do not match provided mesh ("
                + std::to_string(mesh.n[0]) + "x" + std::to_string(mesh.n[1]) + "x" + std::to_string(mesh.n[2])
                + ")");

        ScalarField f(mesh, h.fieldName.empty() ? "field" : h.fieldName, ghost);
        std::string line;
        for (int k = 0; k < mesh.n[2]; ++k)
        for (int j = 0; j < mesh.n[1]; ++j)
        for (int i = 0; i < mesh.n[0]; ++i) {
            if (!std::getline(ifs, line))
                throw std::runtime_error("IO::readScalarField: unexpected end of DAT data in " + path);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            double x, y, z, val;
            if (!(ss >> x >> y >> z >> val))
                throw std::runtime_error("IO::readScalarField: malformed DAT line in " + path);
            f.curr[physIdx(f, i, j, k)] = val;
        }
        return f;
    } else {
        // ---- BINARY format ----
        const auto h = parseScalarHeader(ifs, path);

        if (h.nx != mesh.n[0] || h.ny != mesh.n[1] || h.nz != mesh.n[2])
            throw std::runtime_error(
                "IO::readScalarField: mesh dimensions in file ("
                + std::to_string(h.nx) + "x" + std::to_string(h.ny) + "x" + std::to_string(h.nz)
                + ") do not match provided mesh ("
                + std::to_string(mesh.n[0]) + "x" + std::to_string(mesh.n[1]) + "x" + std::to_string(mesh.n[2])
                + ")");

        ScalarField f(mesh, h.fieldName, ghost);
        const std::size_t phySize = physicalSize(mesh);
        std::vector<double> buf(phySize);
        ifs.read(reinterpret_cast<char*>(buf.data()),
                 static_cast<std::streamsize>(phySize * sizeof(double)));
        if (!ifs)
            throw std::runtime_error("IO::readScalarField: unexpected end of binary data in " + path);

        std::size_t idx = 0;
        for (int k = 0; k < mesh.n[2]; ++k)
            for (int j = 0; j < mesh.n[1]; ++j)
                for (int i = 0; i < mesh.n[0]; ++i)
                    f.curr[physIdx(f, i, j, k)] = buf[idx++];
        return f;
    }
}

void readField(ScalarField& f, const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("IO::readField: cannot open file: " + path);

    if (isDatFormat(ifs)) {
        // ---- DAT (text) format ----
        const auto h = parseScalarDatHeader(ifs, path);

        if (h.nx != f.mesh.n[0] || h.ny != f.mesh.n[1] || h.nz != f.mesh.n[2]) {
            throw std::runtime_error(
                "IO::readField: file dimensions ("
                + std::to_string(h.nx) + "x" + std::to_string(h.ny) + "x" + std::to_string(h.nz)
                + ") do not match field \"" + f.name + "\" mesh ("
                + std::to_string(f.mesh.n[0]) + "x" + std::to_string(f.mesh.n[1]) + "x" + std::to_string(f.mesh.n[2])
                + ")");
        }

        std::string line;
        for (int k = 0; k < f.mesh.n[2]; ++k)
        for (int j = 0; j < f.mesh.n[1]; ++j)
        for (int i = 0; i < f.mesh.n[0]; ++i) {
            if (!std::getline(ifs, line))
                throw std::runtime_error("IO::readField: unexpected end of DAT data in " + path);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            double x, y, z, val;
            if (!(ss >> x >> y >> z >> val))
                throw std::runtime_error("IO::readField: malformed DAT line in " + path);
            f.curr[physIdx(f, i, j, k)] = val;
        }
    } else {
        // ---- BINARY format ----
        const auto h = parseScalarHeader(ifs, path);

        if (h.nx != f.mesh.n[0] || h.ny != f.mesh.n[1] || h.nz != f.mesh.n[2]) {
            throw std::runtime_error(
                "IO::readField: file dimensions ("
                + std::to_string(h.nx) + "x" + std::to_string(h.ny) + "x" + std::to_string(h.nz)
                + ") do not match field \"" + f.name + "\" mesh ("
                + std::to_string(f.mesh.n[0]) + "x" + std::to_string(f.mesh.n[1]) + "x" + std::to_string(f.mesh.n[2])
                + ")");
        }

        const std::size_t phySize = physicalSize(f.mesh);
        std::vector<double> buf(phySize);
        ifs.read(reinterpret_cast<char*>(buf.data()),
                 static_cast<std::streamsize>(phySize * sizeof(double)));
        if (!ifs)
            throw std::runtime_error("IO::readField: unexpected end of binary data in " + path);

        std::size_t idx = 0;
        for (int k = 0; k < f.mesh.n[2]; ++k)
            for (int j = 0; j < f.mesh.n[1]; ++j)
                for (int i = 0; i < f.mesh.n[0]; ++i)
                    f.curr[physIdx(f, i, j, k)] = buf[idx++];
    }
}

// ===========================================================================
// Public API — VectorField
// ===========================================================================

void writeField(const VectorField& vf,
                const std::string& path,
                FieldFormat fmt,
                double coordScale) {
    ensureParentDir(path);
    switch (fmt) {
        case FieldFormat::BINARY: writeVectorBinary(vf, path); break;
        case FieldFormat::DAT:    writeVectorDat(vf, path);    break;
        case FieldFormat::VTS:    writeVectorVts(vf, path);    break;
        case FieldFormat::VTK_BIN:
            throw std::invalid_argument(
                "IO::writeField(VectorField): VTK_BIN supports scalar fields "
                "only — write components individually");
        case FieldFormat::VTI:
            writeVectorVti(vf, path, coordScale, /*f64=*/false); break;
        case FieldFormat::VTI_F64:
            writeVectorVti(vf, path, coordScale, /*f64=*/true);  break;
    }
}

VectorField readVectorField(const Mesh& mesh,
                              const std::string& path,
                              int ghost) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("IO::readVectorField: cannot open file: " + path);

    if (isDatFormat(ifs)) {
        // ---- DAT (text) format ----
        const auto h = parseVectorDatHeader(ifs, path);

        if (h.nComponents <= 0)
            throw std::runtime_error("IO::readVectorField: invalid nComponents in " + path);
        if (h.nx != mesh.n[0] || h.ny != mesh.n[1] || h.nz != mesh.n[2])
            throw std::runtime_error("IO::readVectorField: mesh mismatch in " + path);

        const std::string fname = h.fieldName.empty() ? "field" : h.fieldName;
        VectorField vf(mesh, fname, h.nComponents, ghost);
        const int N = h.nComponents;
        std::string line;
        for (int k = 0; k < h.nz; ++k)
        for (int j = 0; j < h.ny; ++j)
        for (int i = 0; i < h.nx; ++i) {
            if (!std::getline(ifs, line))
                throw std::runtime_error("IO::readVectorField: unexpected end of DAT data in " + path);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            double x, y, z;
            if (!(ss >> x >> y >> z))
                throw std::runtime_error("IO::readVectorField: malformed DAT line in " + path);
            for (int c = 0; c < N; ++c) {
                double val;
                if (!(ss >> val))
                    throw std::runtime_error(
                        "IO::readVectorField: missing component " + std::to_string(c)
                        + " in DAT line in " + path);
                vf[c].curr[physIdx(vf[c], i, j, k)] = val;
            }
        }
        return vf;
    } else {
        // ---- BINARY format ----
        const auto h = parseVectorHeader(ifs, path);

        if (h.nComponents <= 0)
            throw std::runtime_error("IO::readVectorField: invalid nComponents in " + path);
        if (h.nx != mesh.n[0] || h.ny != mesh.n[1] || h.nz != mesh.n[2])
            throw std::runtime_error("IO::readVectorField: mesh mismatch in " + path);

        VectorField vf(mesh, h.fieldName, h.nComponents, ghost);

        const std::size_t phySize = static_cast<std::size_t>(h.nx) * h.ny * h.nz;
        std::vector<double> buf(phySize);

        for (int c = 0; c < h.nComponents; ++c) {
            ifs.read(reinterpret_cast<char*>(buf.data()),
                     static_cast<std::streamsize>(phySize * sizeof(double)));
            if (!ifs)
                throw std::runtime_error(
                    "IO::readVectorField: unexpected end of data for component "
                    + std::to_string(c) + " in " + path);

            ScalarField& sf = vf[c];
            std::size_t idx = 0;
            for (int k = 0; k < h.nz; ++k)
                for (int j = 0; j < h.ny; ++j)
                    for (int i = 0; i < h.nx; ++i)
                        sf.curr[physIdx(sf, i, j, k)] = buf[idx++];
        }
        return vf;
    }
}

void readField(VectorField& vf, const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("IO::readField: cannot open file: " + path);

    if (isDatFormat(ifs)) {
        // ---- DAT (text) format ----
        const auto h = parseVectorDatHeader(ifs, path);

        if (h.nComponents != vf.nComponents())
            throw std::runtime_error(
                "IO::readField: file nComponents (" + std::to_string(h.nComponents)
                + ") does not match field \"" + vf.name + "\" nComponents ("
                + std::to_string(vf.nComponents()) + ")");

        if (h.nx != vf.mesh.n[0] || h.ny != vf.mesh.n[1] || h.nz != vf.mesh.n[2])
            throw std::runtime_error(
                "IO::readField: mesh mismatch for field \"" + vf.name + "\" in " + path);

        const int N = vf.nComponents();
        std::string line;
        for (int k = 0; k < h.nz; ++k)
        for (int j = 0; j < h.ny; ++j)
        for (int i = 0; i < h.nx; ++i) {
            if (!std::getline(ifs, line))
                throw std::runtime_error("IO::readField: unexpected end of DAT data in " + path);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            double x, y, z;
            if (!(ss >> x >> y >> z))
                throw std::runtime_error("IO::readField: malformed DAT line in " + path);
            for (int c = 0; c < N; ++c) {
                double val;
                if (!(ss >> val))
                    throw std::runtime_error(
                        "IO::readField: missing component " + std::to_string(c)
                        + " in DAT line in " + path);
                vf[c].curr[physIdx(vf[c], i, j, k)] = val;
            }
        }
    } else {
        // ---- BINARY format ----
        const auto h = parseVectorHeader(ifs, path);

        if (h.nComponents != vf.nComponents())
            throw std::runtime_error(
                "IO::readField: file nComponents (" + std::to_string(h.nComponents)
                + ") does not match field \"" + vf.name + "\" nComponents ("
                + std::to_string(vf.nComponents()) + ")");

        if (h.nx != vf.mesh.n[0] || h.ny != vf.mesh.n[1] || h.nz != vf.mesh.n[2])
            throw std::runtime_error("IO::readField: mesh mismatch for field \"" + vf.name + "\" in " + path);

        const std::size_t phySize = static_cast<std::size_t>(h.nx) * h.ny * h.nz;
        std::vector<double> buf(phySize);

        for (int c = 0; c < h.nComponents; ++c) {
            ifs.read(reinterpret_cast<char*>(buf.data()),
                     static_cast<std::streamsize>(phySize * sizeof(double)));
            if (!ifs)
                throw std::runtime_error(
                    "IO::readField: unexpected end of data for component "
                    + std::to_string(c) + " in " + path);

            ScalarField& sf = vf[c];
            std::size_t idx = 0;
            for (int k = 0; k < h.nz; ++k)
                for (int j = 0; j < h.ny; ++j)
                    for (int i = 0; i < h.nx; ++i)
                        sf.curr[physIdx(sf, i, j, k)] = buf[idx++];
        }
    }
}

} // namespace IO
} // namespace PhiX

// ---------------------------------------------------------------------------
// Restart / initialization helpers
// ---------------------------------------------------------------------------
namespace PhiX {
namespace IO {

int resolveStartStep(const std::string& startFrom,
                     const std::string& refFieldName)
{
    if (startFrom == "initial_field")
        return 0;

    int target = 0;
    try {
        std::size_t used = 0;
        target = std::stoi(startFrom, &used);
        if (used != startFrom.size())
            throw std::invalid_argument("trailing characters");
    } catch (const std::exception&) {
        throw ConfigError("IO::resolveStartStep",
            "start_from must be \"initial_field\" or an integer step (got \""
            + startFrom + "\")",
            "e.g. \"start_from\": \"5000\" to restart from step 5000");
    }
    const std::string prefix = refFieldName + "_";

    std::vector<int> steps;
    namespace fs = std::filesystem;
    if (fs::is_directory("output")) {
        for (const auto& entry : fs::directory_iterator("output")) {
            const fs::path p = entry.path();
            if (p.extension() != ".field") continue;
            const std::string stem = p.stem().string();
            if (stem.rfind(prefix, 0) != 0) continue;
            try {
                steps.push_back(std::stoi(stem.substr(prefix.size())));
            } catch (...) {}
        }
    }

    if (steps.empty())
        throw std::runtime_error(
            "IO::resolveStartStep: start_from=\"" + startFrom +
            "\" but no " + prefix + "*.field files found in output/");

    const int resolved = *std::min_element(steps.begin(), steps.end(),
        [target](int a, int b) {
            return std::abs(a - target) < std::abs(b - target);
        });
    // Nearest-match restart is a convenience, but a silently shifted start
    // point corrupts time series — say it out loud.
    if (resolved != target)
        warn("IO::resolveStartStep: requested step " + std::to_string(target)
             + " not found in output/; restarting from nearest step "
             + std::to_string(resolved));
    return resolved;
}

void initField(ScalarField& f, int startStep)
{
    std::string path;
    if (startStep == 0) {
        path = "settings/initial_field/" + f.name + ".field";
        std::cout << "  cold start: loading " << path << "\n";
    } else {
        path = "output/" + f.name + "_" + std::to_string(startStep) + ".field";
        std::cout << "  warm start: loading " << path << "\n";
    }
    readField(f, path);
}

void initField(ScalarField& f, int startStep, const std::string& namedInit)
{
    // Warm start always reads from file regardless of namedInit.
    if (startStep > 0) {
        initField(f, startStep);
        return;
    }

    // Cold start: if namedInit is empty, try file-based init.
    if (namedInit.empty()) {
        initField(f, startStep);
        return;
    }

    // Parse namedInit string.
    // Split by ':'
    std::vector<std::string> tokens;
    {
        std::istringstream ss(namedInit);
        std::string tok;
        while (std::getline(ss, tok, ':'))
            tokens.push_back(tok);
    }

    const std::string& kind = tokens[0];

    if (kind == "uniform") {
        if (tokens.size() < 2)
            throw std::runtime_error(
                "IO::initField: \"uniform\" requires a value, e.g. \"uniform:0.5\"");
        double val = std::stod(tokens[1]);
        std::cout << "  named init: uniform(" << val << ") for field \"" << f.name << "\"\n";
        f.initialize([val](double, double, double) { return val; });

    } else if (kind == "random") {
        if (tokens.size() < 3)
            throw std::runtime_error(
                "IO::initField: \"random\" requires lo and hi, e.g. \"random:0.4:0.6\"");
        double lo = std::stod(tokens[1]);
        double hi = std::stod(tokens[2]);
        std::cout << "  named init: random[" << lo << ", " << hi
                  << "] for field \"" << f.name << "\"\n";
        // Simple LCG seeded from field name hash for reproducibility.
        std::size_t seed = std::hash<std::string>{}(f.name);
        f.initialize([lo, hi, &seed](double, double, double) mutable {
            // xorshift64
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            double t = static_cast<double>(seed) / static_cast<double>(
                std::numeric_limits<std::size_t>::max());
            return lo + t * (hi - lo);
        });

    } else if (kind == "linear") {
        if (tokens.size() < 4)
            throw std::runtime_error(
                "IO::initField: \"linear\" requires axis and lo/hi, "
                "e.g. \"linear:x:0.3:0.7\"");
        const std::string& axisStr = tokens[1];
        double lo = std::stod(tokens[2]);
        double hi = std::stod(tokens[3]);

        if (axisStr != "x" && axisStr != "y" && axisStr != "z")
            throw ConfigError("IO::initField",
                "\"linear\" axis must be x, y or z (got \"" + axisStr + "\")",
                "e.g. \"linear:x:0.3:0.7\"");
        int axis = (axisStr == "x") ? 0 : (axisStr == "y") ? 1 : 2;
        if (axis >= f.mesh.dim)
            throw ConfigError("IO::initField",
                "\"linear\" axis " + axisStr + " is inactive on a "
                + std::to_string(f.mesh.dim) + "D mesh");
        double origin = f.mesh.origin[axis];
        double length = f.mesh.n[axis] * f.mesh.d[axis];

        std::cout << "  named init: linear along " << axisStr
                  << " [" << lo << ", " << hi << "] for field \"" << f.name << "\"\n";

        f.initialize([lo, hi, axis, origin, length](double x, double y, double z) {
            double coord = (axis == 0) ? x : (axis == 1) ? y : z;
            double t = (length > 0.0) ? (coord - origin) / length : 0.0;
            return lo + t * (hi - lo);
        });

    } else {
        throw std::runtime_error(
            std::string("IO::initField: unknown named initializer \"") + kind
            + "\". Supported: uniform, random, linear");
    }
}

} // namespace IO
} // namespace PhiX
