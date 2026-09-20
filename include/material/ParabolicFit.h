#pragma once

// ---------------------------------------------------------------------------
// ParabolicFit.h — second-order (parabolic) approximation of a tabulated
// free energy at fixed temperature:
//
//     f(c) ≈ f_min + ½ k (c − c_eq)²
//
// This is the STANDARD bridge between CALPHAD data and grand-potential
// phase-field models (Choudhury & Nestler; the Hötzer et al. Calphad→
// second-order-polynomial workflow; MOOSE grand-potential practice): the
// per-phase parabola keeps c(μ) analytically invertible with constant
// susceptibility χ = 1/k, which is what makes the μ-primary formulation
// cheap on the GPU.  Fidelity is local to the expansion point — far from
// c_eq the CALPHAD landscape is not represented (use the WBM family for
// full-range table fidelity).
//
// Numerical recipe (tables are bilinear ⇒ at fixed T, f is piecewise
// linear in c): locate the minimum node on the table's own c-grid inside
// [cLo, cHi], then fit the exact parabola through the three surrounding
// nodes — vertex position/value give c_eq/f_min, the second difference
// gives k.  Exact for exactly-parabolic data.
//
// Requires nvcc (FreeEnergyTable.h is CUDA-touching).
// ---------------------------------------------------------------------------

#include "material/FreeEnergyTable.h"
#include "core/Error.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace PhiX {
namespace Material {

struct ParabolicPhase {
    double k;      // curvature f''  (χ = 1/k)
    double c_eq;   // well position (vertex)
    double f_min;  // f at the vertex
};

// Fit around the free-energy minimum of `tab` at temperature T, searching
// within [cLo, cHi] (clamped to the table range).  Throws NumericsError if
// the landscape is not convex at the minimum (k ≤ 0) and ValidationError
// on a degenerate search window.
inline ParabolicPhase fitParabola(const FreeEnergyTable& tab, double T,
                                  double cLo, double cHi)
{
    const int    nc = tab.nc();
    const double dc = (tab.cMax() - tab.cMin()) / (nc - 1);

    cLo = std::max(cLo, tab.cMin());
    cHi = std::min(cHi, tab.cMax());
    // Interior nodes only — the three-point stencil needs both neighbours.
    int iLo = std::max(1,      static_cast<int>(std::ceil ((cLo - tab.cMin()) / dc)));
    int iHi = std::min(nc - 2, static_cast<int>(std::floor((cHi - tab.cMin()) / dc)));
    if (iLo > iHi)
        throw ValidationError("Material::fitParabola",
            "search window [" + std::to_string(cLo) + ", " + std::to_string(cHi)
            + "] holds no interior table node",
            "widen the window or refine the table's composition grid");

    // 1. Minimum node in the window.
    int    iMin = iLo;
    double fMin = tab.f(tab.cMin() + iLo * dc, T);
    for (int i = iLo + 1; i <= iHi; ++i) {
        const double fi = tab.f(tab.cMin() + i * dc, T);
        if (fi < fMin) { fMin = fi; iMin = i; }
    }

    // 2. Exact parabola through the three surrounding nodes.
    const double c0 = tab.cMin() + iMin * dc;
    const double fm = tab.f(c0 - dc, T);
    const double f0 = tab.f(c0,      T);
    const double fp = tab.f(c0 + dc, T);

    const double curv = (fp - 2.0 * f0 + fm) / (dc * dc);   // f''
    if (!(curv > 0.0))
        throw NumericsError("Material::fitParabola",
            "free energy is not convex at its minimum (f'' = "
            + std::to_string(curv) + " at c = " + std::to_string(c0) + ")",
            "the window may straddle a miscibility gap or spinodal region — "
            "narrow [cLo, cHi] to a single-phase branch");

    ParabolicPhase p;
    p.k     = curv;
    p.c_eq  = c0 - 0.5 * dc * (fp - fm) / (fp - 2.0 * f0 + fm);  // vertex
    p.f_min = f0 - 0.125 * (fp - fm) * (fp - fm) / (fp - 2.0 * f0 + fm);
    return p;
}

// Second-order Taylor expansion at a GIVEN composition c0 (matching f, f',
// f'' there) — for line compounds / dilute expansions where the operating
// point, not the global minimum, defines the parabola.  Note c_eq is the
// vertex implied by the local slope: c_eq = c0 − f'(c0)/f''(c0).
inline ParabolicPhase fitParabolaAt(const FreeEnergyTable& tab, double T,
                                    double c0)
{
    const int    nc = tab.nc();
    const double dc = (tab.cMax() - tab.cMin()) / (nc - 1);
    const double lo = tab.cMin() + dc, hi = tab.cMax() - dc;
    if (c0 < lo || c0 > hi)
        throw ValidationError("Material::fitParabolaAt",
            "expansion point c0 = " + std::to_string(c0)
            + " too close to the table edge",
            "need one grid cell of margin on each side");

    const double fm = tab.f(c0 - dc, T);
    const double f0 = tab.f(c0,      T);
    const double fp = tab.f(c0 + dc, T);
    const double curv  = (fp - 2.0 * f0 + fm) / (dc * dc);
    const double slope = (fp - fm) / (2.0 * dc);
    if (!(curv > 0.0))
        throw NumericsError("Material::fitParabolaAt",
            "f'' <= 0 at the expansion point c0 = " + std::to_string(c0),
            "pick an expansion point on a convex branch");

    ParabolicPhase p;
    p.k     = curv;
    p.c_eq  = c0 - slope / curv;
    p.f_min = f0 - 0.5 * slope * slope / curv;
    return p;
}

} // namespace Material
} // namespace PhiX
