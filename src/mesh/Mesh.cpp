#include "mesh/Mesh.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace PhiX {

// -----------------------------------------------------------------------
// String conversion helpers
// -----------------------------------------------------------------------

std::string coordSysToString(CoordSys cs) {
    switch (cs) {
        case CoordSys::CARTESIAN:   return "CARTESIAN";
        case CoordSys::CYLINDRICAL: return "CYLINDRICAL";
        case CoordSys::SPHERICAL:   return "SPHERICAL";
    }
    return "CARTESIAN";
}

CoordSys stringToCoordSys(const std::string& s) {
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    if (upper == "CYLINDRICAL") return CoordSys::CYLINDRICAL;
    if (upper == "SPHERICAL")   return CoordSys::SPHERICAL;
    if (upper == "CARTESIAN")   return CoordSys::CARTESIAN;
    throw std::invalid_argument("Unknown CoordSys: " + s);
}

// -----------------------------------------------------------------------
// Private init
// -----------------------------------------------------------------------

void Mesh::init(int dim_, CoordSys cs_,
                int nx, double dx, double x0,
                int ny, double dy, double y0,
                int nz, double dz, double z0) {
    dim      = dim_;
    coordSys = cs_;
    n[0] = nx; d[0] = dx; origin[0] = x0;
    n[1] = ny; d[1] = dy; origin[1] = y0;
    n[2] = nz; d[2] = dz; origin[2] = z0;
}

// -----------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------

Mesh::Mesh() {
    init(1, CoordSys::CARTESIAN,
         0, 0.0, 0.0,
         1, 1.0, 0.0,
         1, 1.0, 0.0);
}

Mesh::Mesh(int dim_, CoordSys cs_,
           int nx, double dx, double x0,
           int ny, double dy, double y0,
           int nz, double dz, double z0) {
    init(dim_, cs_, nx, dx, x0, ny, dy, y0, nz, dz, z0);
    validate();
    defaultPatches();
}

// -----------------------------------------------------------------------
// Static factory methods
// -----------------------------------------------------------------------

Mesh Mesh::makeUniform1D(CoordSys cs, int nx, double dx, double x0) {
    return Mesh(1, cs, nx, dx, x0, 1, 1.0, 0.0, 1, 1.0, 0.0);
}

Mesh Mesh::makeUniform2D(CoordSys cs,
                         int nx, double dx, double x0,
                         int ny, double dy, double y0) {
    return Mesh(2, cs, nx, dx, x0, ny, dy, y0, 1, 1.0, 0.0);
}

Mesh Mesh::makeUniform3D(CoordSys cs,
                         int nx, double dx, double x0,
                         int ny, double dy, double y0,
                         int nz, double dz, double z0) {
    return Mesh(3, cs, nx, dx, x0, ny, dy, y0, nz, dz, z0);
}

// -----------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------

void Mesh::validate() const {
    if (dim < 1 || dim > 3)
        throw std::invalid_argument("Mesh: dim must be 1, 2, or 3");

    for (int ax = 0; ax < dim; ++ax) {
        if (n[ax] <= 0)
            throw std::invalid_argument("Mesh: n[" + std::to_string(ax) + "] must be > 0");
        if (d[ax] <= 0.0)
            throw std::invalid_argument("Mesh: d[" + std::to_string(ax) + "] must be > 0");
    }

    // Axes beyond dim must be degenerate (n=1)
    for (int ax = dim; ax < 3; ++ax) {
        if (n[ax] != 1)
            throw std::invalid_argument(
                "Mesh: n[" + std::to_string(ax) + "] must be 1 for unused axes");
    }
}

bool Mesh::isValid() const noexcept {
    try {
        validate();
        return true;
    } catch (...) {
        return false;
    }
}

// -----------------------------------------------------------------------
// IO: write
// -----------------------------------------------------------------------

void Mesh::write(const std::string& path) const {
    validate();

    std::ofstream ofs(path);
    if (!ofs)
        throw std::runtime_error("Mesh::write: cannot open file: " + path);

    ofs << "# PhiX Mesh\n";
    ofs << "dim      " << dim << "\n";
    ofs << "coordSys " << coordSysToString(coordSys) << "\n";
    ofs << "nx " << n[0] << "  dx " << d[0] << "  x0 " << origin[0] << "\n";
    ofs << "ny " << n[1] << "  dy " << d[1] << "  y0 " << origin[1] << "\n";
    ofs << "nz " << n[2] << "  dz " << d[2] << "  z0 " << origin[2] << "\n";
}

// -----------------------------------------------------------------------
// IO: read
// -----------------------------------------------------------------------

Mesh Mesh::readFromFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs)
        throw std::runtime_error("Mesh::readFromFile: cannot open file: " + path);

    int dim_  = 1;
    CoordSys cs = CoordSys::CARTESIAN;
    int    n_[3]      = {1, 1, 1};
    double d_[3]      = {1.0, 1.0, 1.0};
    double origin_[3] = {0.0, 0.0, 0.0};

    // Direction labels used in the file: nx/dx/x0, ny/dy/y0, nz/dz/z0
    const char* nKey[3]   = {"nx", "ny", "nz"};
    const char* dKey[3]   = {"dx", "dy", "dz"};
    const char* o0Key[3]  = {"x0", "y0", "z0"};

    std::string line;
    while (std::getline(ifs, line)) {
        // Strip comments
        auto pos = line.find('#');
        if (pos != std::string::npos) line = line.substr(0, pos);

        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key)) continue;

        if (key == "dim") {
            ss >> dim_;
        } else if (key == "coordSys") {
            std::string val; ss >> val;
            cs = stringToCoordSys(val);
        } else {
            // Try to match nx/dx/x0 etc. (any order on the line is fine)
            // First token already read as `key`; push it back by re-scanning the line
            std::istringstream row(line);
            std::string tok;
            while (row >> tok) {
                for (int ax = 0; ax < 3; ++ax) {
                    if (tok == nKey[ax])  { row >> n_[ax];      break; }
                    if (tok == dKey[ax])  { row >> d_[ax];      break; }
                    if (tok == o0Key[ax]) { row >> origin_[ax]; break; }
                }
            }
        }
    }

    return Mesh(dim_, cs,
                n_[0], d_[0], origin_[0],
                n_[1], d_[1], origin_[1],
                n_[2], d_[2], origin_[2]);
}

// -----------------------------------------------------------------------
// print
// -----------------------------------------------------------------------

void Mesh::print() const {
    std::cout << "=== Mesh ===\n";
    std::cout << "  dim      : " << dim << "\n";
    std::cout << "  coordSys : " << coordSysToString(coordSys) << "\n";
    const char* axName[3] = {"x", "y", "z"};
    for (int ax = 0; ax < 3; ++ax) {
        std::cout << "  " << axName[ax] << ": n=" << n[ax]
                  << "  d=" << d[ax]
                  << "  origin=" << origin[ax];
        if (ax >= dim) std::cout << "  (inactive)";
        std::cout << "\n";
    }
    std::cout << "  totalSize: " << totalSize() << "\n";
    std::cout << "  patches  : " << patches_.size() << "\n";
    for (const auto& p : patches_) {
        std::cout << "    - " << p.name
                  << "  axis=" << (p.axis == Axis::X ? "X" : p.axis == Axis::Y ? "Y" : "Z")
                  << "  side=" << (p.side == Side::LOW ? "LOW" : "HIGH")
                  << "  cells=" << p.numCells()
                  << "\n";
    }
}

// =======================================================================
// Patch topology
// =======================================================================

IndexBox Mesh::fullFaceBox(Axis axis, Side side) const {
    const int a = static_cast<int>(axis);
    IndexBox box;
    for (int t = 0; t < 3; ++t) {
        box.lo[t] = 0;
        box.hi[t] = n[t];
    }
    if (side == Side::LOW) {
        box.lo[a] = 0;
        box.hi[a] = 1;
    } else {
        box.lo[a] = n[a] - 1;
        box.hi[a] = n[a];
    }
    return box;
}

void Mesh::validatePatchBox(Axis axis, Side side, const IndexBox& box) const {
    const int a = static_cast<int>(axis);
    if (a >= dim)
        throw std::invalid_argument("Mesh::addPatch: axis exceeds mesh dim");

    // Normal-axis: must be the single cell at the correct boundary.
    const int expected_lo = (side == Side::LOW) ? 0          : n[a] - 1;
    const int expected_hi = (side == Side::LOW) ? 1          : n[a];
    if (box.lo[a] != expected_lo || box.hi[a] != expected_hi)
        throw std::invalid_argument(
            "Mesh::addPatch: patch normal-axis extent must be the single "
            "boundary cell");

    // Tangential axes: must lie inside the mesh and have positive extent.
    for (int t = 0; t < 3; ++t) {
        if (t == a) continue;
        if (box.lo[t] < 0 || box.hi[t] > n[t] || box.lo[t] >= box.hi[t])
            throw std::invalid_argument(
                "Mesh::addPatch: tangential extent out of range");
    }
}

void Mesh::defaultPatches() {
    patches_.clear();
    for (int a = 0; a < dim; ++a) {
        Axis axis = static_cast<Axis>(a);
        for (Side side : {Side::LOW, Side::HIGH}) {
            Patch p;
            p.name   = defaultPatchName(axis, side);
            p.axis   = axis;
            p.side   = side;
            p.region = fullFaceBox(axis, side);
            p.kind   = PatchKind::PHYSICAL;
            patches_.push_back(p);
        }
    }
    validatePatches();
}

const Patch& Mesh::patch(const std::string& name) const {
    for (const auto& p : patches_) if (p.name == name) return p;
    throw std::out_of_range("Mesh::patch: no patch named \"" + name + "\"");
}

Patch& Mesh::patch(const std::string& name) {
    for (auto& p : patches_) if (p.name == name) return p;
    throw std::out_of_range("Mesh::patch: no patch named \"" + name + "\"");
}

bool Mesh::hasPatch(const std::string& name) const {
    for (const auto& p : patches_) if (p.name == name) return true;
    return false;
}

std::vector<const Patch*> Mesh::facePatches(Axis axis, Side side) const {
    std::vector<const Patch*> out;
    for (const auto& p : patches_)
        if (p.axis == axis && p.side == side) out.push_back(&p);
    return out;
}

const Patch& Mesh::facePatch(Axis axis, Side side) const {
    const Patch* found = nullptr;
    int count = 0;
    for (const auto& p : patches_) {
        if (p.axis == axis && p.side == side) {
            found = &p;
            ++count;
        }
    }
    if (count == 0)
        throw std::runtime_error("Mesh::facePatch: no patch on requested face");
    if (count > 1)
        throw std::runtime_error(
            "Mesh::facePatch: face has been split into multiple patches; "
            "use facePatches() instead");
    return *found;
}

void Mesh::addPatch(const Patch& p) {
    if (p.name.empty())
        throw std::invalid_argument("Mesh::addPatch: patch name is empty");
    if (hasPatch(p.name))
        throw std::invalid_argument(
            "Mesh::addPatch: duplicate patch name \"" + p.name + "\"");
    validatePatchBox(p.axis, p.side, p.region);
    patches_.push_back(p);
}

bool Mesh::removePatch(const std::string& name) {
    for (auto it = patches_.begin(); it != patches_.end(); ++it) {
        if (it->name == name) { patches_.erase(it); return true; }
    }
    return false;
}

void Mesh::removeFacePatches(Axis axis, Side side) {
    patches_.erase(
        std::remove_if(patches_.begin(), patches_.end(),
                       [&](const Patch& p) {
                           return p.axis == axis && p.side == side;
                       }),
        patches_.end());
}

void Mesh::validatePatches() const {
    // For each active (axis, side) face, verify that the patches form a
    // disjoint exact tiling of the full face.
    for (int a = 0; a < dim; ++a) {
        Axis axis = static_cast<Axis>(a);
        for (Side side : {Side::LOW, Side::HIGH}) {
            std::vector<const Patch*> fps = facePatches(axis, side);

            if (fps.empty())
                throw std::runtime_error(
                    std::string("Mesh::validatePatches: no patches on face ")
                    + defaultPatchName(axis, side));

            long long covered = 0;
            for (std::size_t i = 0; i < fps.size(); ++i) {
                covered += fps[i]->numCells();
                for (std::size_t j = i + 1; j < fps.size(); ++j) {
                    if (fps[i]->region.overlaps(fps[j]->region))
                        throw std::runtime_error(
                            std::string("Mesh::validatePatches: patches \"")
                            + fps[i]->name + "\" and \"" + fps[j]->name
                            + "\" overlap");
                }
            }

            long long faceCells = 1;
            for (int t = 0; t < 3; ++t)
                if (t != a) faceCells *= n[t];

            if (covered != faceCells)
                throw std::runtime_error(
                    std::string("Mesh::validatePatches: face ")
                    + defaultPatchName(axis, side)
                    + " not fully covered (covered=" + std::to_string(covered)
                    + ", expected=" + std::to_string(faceCells) + ")");
        }
    }
}

} // namespace PhiX
