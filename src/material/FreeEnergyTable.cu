#include "material/FreeEnergyTable.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace PhiX {
namespace Material {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

FreeEnergyTable::FreeEnergyTable(double c_min, double c_max, int nc,
                                 double T_min, double T_max, int nT,
                                 std::vector<double> data)
    : nc_(nc), nT_(nT),
      c_min_(c_min), c_max_(c_max),
      T_min_(T_min), T_max_(T_max),
      data_(std::move(data))
{
    if (nc_ < 2 || nT_ < 2)
        throw std::invalid_argument("FreeEnergyTable: nc and nT must be >= 2");
    if (c_max_ <= c_min_)
        throw std::invalid_argument("FreeEnergyTable: c_max must be > c_min");
    if (T_max_ <= T_min_)
        throw std::invalid_argument("FreeEnergyTable: T_max must be > T_min");
    if (static_cast<int>(data_.size()) != nc_ * nT_)
        throw std::invalid_argument(
            "FreeEnergyTable: data size mismatch (expected " +
            std::to_string(nc_ * nT_) + ", got " +
            std::to_string(data_.size()) + ")");

    dc_ = (c_max_ - c_min_) / (nc_ - 1);
    dT_ = (T_max_ - T_min_) / (nT_ - 1);
}

// ---------------------------------------------------------------------------
// Move semantics / destructor
// ---------------------------------------------------------------------------

FreeEnergyTable::FreeEnergyTable(FreeEnergyTable&& o) noexcept
    : nc_(o.nc_), nT_(o.nT_),
      c_min_(o.c_min_), c_max_(o.c_max_), dc_(o.dc_),
      T_min_(o.T_min_), T_max_(o.T_max_), dT_(o.dT_),
      data_(std::move(o.data_)),
      d_data_(o.d_data_)
{
    o.d_data_ = nullptr;
}

FreeEnergyTable& FreeEnergyTable::operator=(FreeEnergyTable&& o) noexcept
{
    if (this != &o) {
        freeDevice();
        nc_    = o.nc_;    nT_    = o.nT_;
        c_min_ = o.c_min_; c_max_ = o.c_max_; dc_ = o.dc_;
        T_min_ = o.T_min_; T_max_ = o.T_max_; dT_ = o.dT_;
        data_   = std::move(o.data_);
        d_data_ = o.d_data_;
        o.d_data_ = nullptr;
    }
    return *this;
}

FreeEnergyTable::~FreeEnergyTable()
{
    freeDevice();
}

// ---------------------------------------------------------------------------
// File loader — internal helpers
// ---------------------------------------------------------------------------
namespace {

// Return true if `str` is a comment or blank line
bool isSkippable(const std::string& line)
{
    auto first = line.find_first_not_of(" \t\r\n");
    return first == std::string::npos || line[first] == '#';
}

// Split a string by `delim`, trim whitespace from each token, return doubles.
// Empty tokens (e.g. trailing comma) are skipped.
std::vector<double> splitLine(const std::string& line, char delim)
{
    std::vector<double> vals;
    std::istringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, delim)) {
        // trim
        auto a = tok.find_first_not_of(" \t\r\n");
        auto b = tok.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        tok = tok.substr(a, b - a + 1);
        if (tok.empty() || tok[0] == '#') break;  // inline comment
        vals.push_back(std::stod(tok));
    }
    return vals;
}

// Detect format from extension: ".csv" → CSV, everything else → FETAB
FileFormat detectFormat(const std::string& path)
{
    auto dot = path.rfind('.');
    if (dot != std::string::npos) {
        std::string ext = path.substr(dot);
        // lowercase
        for (auto& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ext == ".csv") return FileFormat::CSV;
    }
    return FileFormat::FETAB;
}

// Core parser: works for both formats, `delim` selects whitespace vs comma
FreeEnergyTable parseStream(std::ifstream& ifs, const std::string& path, char delim)
{
    int    nc = 0, nT = 0;
    double c_min = 0, c_max = 0, T_min = 0, T_max = 0;
    bool   header_found = false;

    std::string line;
    while (std::getline(ifs, line)) {
        if (isSkippable(line)) continue;

        if (delim == ' ') {
            std::istringstream ss(line);
            if (!(ss >> nc >> nT >> c_min >> c_max >> T_min >> T_max))
                throw std::runtime_error(
                    "FreeEnergyTable::fromFile: invalid header in '" + path + "'");
        } else {
            auto v = splitLine(line, delim);
            if (v.size() < 6)
                throw std::runtime_error(
                    "FreeEnergyTable::fromFile: header needs 6 fields in '" + path + "'");
            nc = static_cast<int>(v[0]);  nT    = static_cast<int>(v[1]);
            c_min = v[2]; c_max = v[3];   T_min = v[4]; T_max = v[5];
        }
        header_found = true;
        break;
    }

    if (!header_found)
        throw std::runtime_error(
            "FreeEnergyTable::fromFile: no header found in '" + path + "'");

    std::vector<double> data;
    data.reserve(static_cast<std::size_t>(nc) * nT);

    while (std::getline(ifs, line)) {
        if (isSkippable(line)) continue;

        if (delim == ' ') {
            std::istringstream ss(line);
            double v;
            while (ss >> v) data.push_back(v);
        } else {
            auto row = splitLine(line, delim);
            data.insert(data.end(), row.begin(), row.end());
        }
    }

    if (static_cast<int>(data.size()) != nc * nT)
        throw std::runtime_error(
            "FreeEnergyTable::fromFile: expected " + std::to_string(nc * nT) +
            " values, got " + std::to_string(data.size()) +
            " in '" + path + "'");

    // Degenerate c axis (nc == 1): a stoichiometric phase whose f depends on
    // T only.  The bilinear view needs nc >= 2, so duplicate the single row
    // and widen the c range to [0,1] — interpolation between two identical
    // rows returns the c-independent value exactly, for any queried c.
    if (nc == 1) {
        data.insert(data.end(), data.begin(), data.end());
        nc = 2;  c_min = 0.0;  c_max = 1.0;
    }

    return FreeEnergyTable(c_min, c_max, nc, T_min, T_max, nT, std::move(data));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// File loader — public API
// ---------------------------------------------------------------------------

FreeEnergyTable FreeEnergyTable::fromFile(const std::string& path, FileFormat fmt)
{
    if (fmt == FileFormat::AUTO) fmt = detectFormat(path);

    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error(
            "FreeEnergyTable::fromFile: cannot open '" + path + "'");

    char delim = (fmt == FileFormat::CSV) ? ',' : ' ';
    return parseStream(ifs, path, delim);
}

// ---------------------------------------------------------------------------
// Host evaluation — delegate to hostView() so logic lives in one place
// ---------------------------------------------------------------------------

double FreeEnergyTable::f(double c, double T) const    { return hostView().f(c, T); }
double FreeEnergyTable::dfdc(double c, double T) const { return hostView().dfdc(c, T); }
double FreeEnergyTable::dfdT(double c, double T) const { return hostView().dfdT(c, T); }

// ---------------------------------------------------------------------------
// GPU memory management
// ---------------------------------------------------------------------------

void FreeEnergyTable::allocDevice()
{
    if (d_data_) return;   // idempotent
    std::size_t bytes = static_cast<std::size_t>(nc_) * nT_ * sizeof(double);
    cudaError_t err = cudaMalloc(&d_data_, bytes);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("FreeEnergyTable::allocDevice cudaMalloc failed: ") +
            cudaGetErrorString(err));
}

void FreeEnergyTable::uploadToDevice()
{
    if (!d_data_)
        throw std::runtime_error(
            "FreeEnergyTable::uploadToDevice: call allocDevice() first");
    std::size_t bytes = static_cast<std::size_t>(nc_) * nT_ * sizeof(double);
    cudaError_t err = cudaMemcpy(d_data_, data_.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("FreeEnergyTable::uploadToDevice cudaMemcpy failed: ") +
            cudaGetErrorString(err));
}

void FreeEnergyTable::freeDevice()
{
    if (d_data_) {
        cudaFree(d_data_);
        d_data_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// View factories
// ---------------------------------------------------------------------------

FreeEnergyTableView FreeEnergyTable::hostView() const
{
    return FreeEnergyTableView{
        data_.data(),
        nc_, nT_,
        c_min_, dc_, c_max_,
        T_min_, dT_, T_max_
    };
}

FreeEnergyTableView FreeEnergyTable::deviceView() const
{
    if (!d_data_)
        throw std::runtime_error(
            "FreeEnergyTable::deviceView: device data not allocated/uploaded");
    return FreeEnergyTableView{
        d_data_,
        nc_, nT_,
        c_min_, dc_, c_max_,
        T_min_, dT_, T_max_
    };
}

} // namespace Material
} // namespace PhiX
