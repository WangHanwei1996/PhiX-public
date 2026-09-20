#pragma once

#include "material/IMaterial.h"
#include "material/FreeEnergyTable.h"
#include <string>
#include <memory>

namespace PhiX {
namespace Material {

// ---------------------------------------------------------------------------
// BinaryAlloy  —  thermodynamic model for a two-component alloy
//
// Stores a FreeEnergyTable and exposes the Gibbs free energy as a function of
// composition c ∈ [0,1] and temperature T:
//
//   double f    = alloy.freeEnergy(c, T);
//   double dfdc = alloy.dfdc(c, T);
//   double dfdT = alloy.dfdT(c, T);
//
// All three calls use bilinear interpolation over the internal table; out-of-
// range (c, T) values are silently clamped to the table bounds.
//
// Construction
// ------------
//   // From a pre-built table:
//   BinaryAlloy alloy("Fe-B", std::move(table));
//
//   // Directly from a .fetab file:
//   auto alloy = BinaryAlloy::fromFile("data/FeB.fetab", "Fe-B");
// ---------------------------------------------------------------------------

class BinaryAlloy : public IMaterial {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Construct with a human-readable label and a pre-built table.
    BinaryAlloy(std::string label, FreeEnergyTable table);

    /// Convenience factory: load the table from a file.
    /// Supported formats: FETAB (.fetab), CSV (.csv).
    /// fmt = FileFormat::AUTO  →  format inferred from file extension.
    static BinaryAlloy fromFile(const std::string& path,
                                std::string        label = "",
                                FileFormat         fmt   = FileFormat::AUTO);

    // -----------------------------------------------------------------------
    // IMaterial interface
    // -----------------------------------------------------------------------
    std::string name() const override { return label_; }

    // -----------------------------------------------------------------------
    // Thermodynamic properties
    // -----------------------------------------------------------------------

    /// Molar (or volumetric, depending on the table units) free energy f(c, T).
    double freeEnergy(double c, double T) const;

    /// ∂f/∂c at (c, T).
    double dfdc(double c, double T) const;

    /// ∂f/∂T at (c, T).
    double dfdT(double c, double T) const;

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    const FreeEnergyTable& table() const { return table_; }

    /// Shorthand: host view of the internal table (CPU / kernel testing).
    FreeEnergyTableView hostView()   const { return table_.hostView(); }

    /// Shorthand: device view — requires table().allocDevice() + uploadToDevice().
    FreeEnergyTableView deviceView() const { return table_.deviceView(); }

private:
    std::string     label_;
    FreeEnergyTable table_;
};

} // namespace Material
} // namespace PhiX
