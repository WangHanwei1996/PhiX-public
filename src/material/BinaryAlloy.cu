#include "material/BinaryAlloy.h"

namespace PhiX {
namespace Material {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

BinaryAlloy::BinaryAlloy(std::string label, FreeEnergyTable table)
    : label_(std::move(label)), table_(std::move(table))
{}

BinaryAlloy BinaryAlloy::fromFile(const std::string& path, std::string label,
                                   FileFormat fmt)
{
    FreeEnergyTable tbl = FreeEnergyTable::fromFile(path, fmt);
    if (label.empty()) label = path;
    return BinaryAlloy(std::move(label), std::move(tbl));
}

// ---------------------------------------------------------------------------
// Thermodynamic properties  (thin wrappers over FreeEnergyTable)
// ---------------------------------------------------------------------------

double BinaryAlloy::freeEnergy(double c, double T) const
{
    return table_.f(c, T);
}

double BinaryAlloy::dfdc(double c, double T) const
{
    return table_.dfdc(c, T);
}

double BinaryAlloy::dfdT(double c, double T) const
{
    return table_.dfdT(c, T);
}

} // namespace Material
} // namespace PhiX
