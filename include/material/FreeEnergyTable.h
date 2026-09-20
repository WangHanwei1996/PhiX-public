#pragma once

#include <string>
#include <vector>
#include <stdexcept>

namespace PhiX {
namespace Material {

// ---------------------------------------------------------------------------
// FileFormat  —  选择 FreeEnergyTable::fromFile 读取格式
//
//   AUTO  : 根据文件扩展名自动识别（.fetab → FETAB，.csv → CSV，其余→ FETAB）
//   FETAB : 原生空白分隔格式（.fetab）
//   CSV   : 逗号分隔格式（.csv），结构与 FETAB 相同，可直接用 Excel 打开
// ---------------------------------------------------------------------------
enum class FileFormat { AUTO, FETAB, CSV };

// ---------------------------------------------------------------------------
// FreeEnergyTableView  —  lightweight device-safe view of a FreeEnergyTable
//
// Holds a raw pointer (host or device) plus grid parameters.
// All methods are tagged __host__ __device__ so they can be called from both
// CPU code and CUDA kernels.
//
// Do NOT construct directly; obtain via FreeEnergyTable::hostView() or
// FreeEnergyTable::deviceView() (the latter requires a prior allocDevice() +
// uploadToDevice() call).
// ---------------------------------------------------------------------------

struct FreeEnergyTableView {
    const double* data;   ///< flat row-major array, size nc*nT
    int    nc,  nT;
    double c_min, dc, c_max;
    double T_min, dT, T_max;

    /// Bilinear interpolation — f(c, T).  Out-of-range values are clamped.
    __host__ __device__
    double f(double c, double T) const
    {
        c = clamp(c, c_min, c_max);
        T = clamp(T, T_min, T_max);

        double fc = (c - c_min) / dc;
        double fT = (T - T_min) / dT;

        int ic = min_int(static_cast<int>(fc), nc - 2);
        int iT = min_int(static_cast<int>(fT), nT - 2);

        double wc = fc - ic;
        double wT = fT - iT;

        return (1.0 - wc) * (1.0 - wT) * at(ic,     iT    )
             + (1.0 - wc) *        wT  * at(ic,     iT + 1)
             +        wc  * (1.0 - wT) * at(ic + 1, iT    )
             +        wc  *        wT  * at(ic + 1, iT + 1);
    }

    /// ∂f/∂c via central finite differences.
    __host__ __device__
    double dfdc(double c, double T) const
    {
        double h    = 0.5 * dc;
        double c_lo = clamp(c - h, c_min, c_max);
        double c_hi = clamp(c + h, c_min, c_max);
        double span = c_hi - c_lo;
        if (span == 0.0) return 0.0;
        return (f(c_hi, T) - f(c_lo, T)) / span;
    }

    /// ∂f/∂T via central finite differences.
    __host__ __device__
    double dfdT(double c, double T) const
    {
        double h    = 0.5 * dT;
        double T_lo = clamp(T - h, T_min, T_max);
        double T_hi = clamp(T + h, T_min, T_max);
        double span = T_hi - T_lo;
        if (span == 0.0) return 0.0;
        return (f(c, T_hi) - f(c, T_lo)) / span;
    }

private:
    __host__ __device__
    double at(int ic, int iT) const {
#ifdef __CUDA_ARCH__
        return __ldg(&data[ic * nT + iT]);  // read-only cache (texture path)
#else
        return data[ic * nT + iT];
#endif
    }

    __host__ __device__
    static double clamp(double v, double lo, double hi)
    { return v < lo ? lo : (v > hi ? hi : v); }

    __host__ __device__
    static int min_int(int a, int b) { return a < b ? a : b; }
};

// ---------------------------------------------------------------------------
// FreeEnergyTable  —  2-D look-up table  f(c, T)  with bilinear interpolation
//
// Grid layout
// -----------
//   concentration : nc  uniformly spaced points in [c_min, c_max]
//   temperature   : nT  uniformly spaced points in [T_min, T_max]
//
// Internal storage (row-major, c varies slowly):
//   data_[ic * nT_ + iT]  =  f(c_min + ic*dc, T_min + iT*dT)
//
// GPU usage
// ---------
//   table.allocDevice();
//   table.uploadToDevice();
//   // pass the view by value into kernels:
//   auto view = table.deviceView();
//   myKernel<<<...>>>(view, ...);
//
// File formats
// ------------
// FETAB (.fetab)  — space/tab separated:
//   Lines beginning with '#' are ignored (comments anywhere in file).
//   First non-comment line:   nc  nT  c_min  c_max  T_min  T_max
//   Followed by nc rows, each containing nT whitespace-separated values.
//
// CSV (.csv)  — comma separated, same structure, opens directly in Excel:
//   Comment lines beginning with '#' are ignored.
//   First non-comment line:   nc,nT,c_min,c_max,T_min,T_max
//   Followed by nc rows, each containing nT comma-separated values.
//   Whitespace around commas is trimmed automatically.
//
// Example (FETAB):
//     # Fe-B binary alloy free energy table
//     40 100 0.0 1.0 300.0 1800.0
//     -1.23e4  -1.22e4  ...   (nT values, ic=0)
//     ...
//
// Example (CSV):
//     # Fe-B binary alloy free energy table
//     40,100,0.0,1.0,300.0,1800.0
//     -1.23e4,-1.22e4,...
//     ...
// ---------------------------------------------------------------------------

class FreeEnergyTable {
public:
    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------

    /// Construct from grid parameters and a flat row-major data vector.
    /// data must have exactly nc * nT elements.
    FreeEnergyTable(double c_min, double c_max, int nc,
                    double T_min, double T_max, int nT,
                    std::vector<double> data);

    /// Load from a file.  Supported formats: FETAB, CSV.
    /// fmt = FileFormat::AUTO  →  format inferred from file extension.
    static FreeEnergyTable fromFile(const std::string& path,
                                    FileFormat fmt = FileFormat::AUTO);

    // Non-copyable (may own GPU memory)
    FreeEnergyTable(const FreeEnergyTable&)            = delete;
    FreeEnergyTable& operator=(const FreeEnergyTable&) = delete;

    // Movable
    FreeEnergyTable(FreeEnergyTable&&) noexcept;
    FreeEnergyTable& operator=(FreeEnergyTable&&) noexcept;

    ~FreeEnergyTable();

    // -----------------------------------------------------------------------
    // Host evaluation  (convenience wrappers — use hostView() for hot paths)
    // -----------------------------------------------------------------------

    /// Bilinear interpolation — returns f at (c, T).
    double f(double c, double T) const;

    /// Partial derivative ∂f/∂c via central finite differences on the grid.
    double dfdc(double c, double T) const;

    /// Partial derivative ∂f/∂T via central finite differences on the grid.
    double dfdT(double c, double T) const;

    // -----------------------------------------------------------------------
    // GPU memory management
    // -----------------------------------------------------------------------

    /// Allocate device-side flat data array (idempotent).
    void allocDevice();

    /// Copy host data to the device buffer (requires prior allocDevice()).
    void uploadToDevice();

    /// Free the device buffer.  Safe to call even if never allocated.
    void freeDevice();

    /// Returns true if device memory has been allocated.
    bool isOnDevice() const { return d_data_ != nullptr; }

    // -----------------------------------------------------------------------
    // View accessors (for passing into CUDA kernels by value)
    // -----------------------------------------------------------------------

    /// View backed by host memory — usable on CPU and with __host__ methods.
    FreeEnergyTableView hostView() const;

    /// View backed by device memory — pass into GPU kernels.
    /// Requires allocDevice() + uploadToDevice() to have been called.
    FreeEnergyTableView deviceView() const;

    // -----------------------------------------------------------------------
    // Grid accessors
    // -----------------------------------------------------------------------
    int    nc()   const { return nc_; }
    int    nT()   const { return nT_; }
    double cMin() const { return c_min_; }
    double cMax() const { return c_max_; }
    double TMin() const { return T_min_; }
    double TMax() const { return T_max_; }
    double dc()   const { return dc_; }
    double dT()   const { return dT_; }

    /// Read-only access to the raw flat host data (row-major [ic*nT+iT]).
    const std::vector<double>& data() const { return data_; }

private:
    int    nc_,  nT_;
    double c_min_, c_max_, dc_;
    double T_min_, T_max_, dT_;
    std::vector<double> data_;     // host storage, size nc_ * nT_
    double*             d_data_ = nullptr;  // device storage (lazy)
};

} // namespace Material
} // namespace PhiX
