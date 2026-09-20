#pragma once
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"

namespace PhiX {
// Mirrors BCBatch's ordered corner composition for built-in 2D closures.
// This preserves each BC's existing ghost values; it does not raise boundary order.
inline void applyBCsCPU2D(ScalarField &f, const std::vector<BoundaryCondition *> &bcs) {
    for (auto *bc : bcs)
        bc->applyOnCPU(f);
    if (f.mesh.dim != 2)
        return;
    const int nx = f.mesh.n[0], ny = f.mesh.n[1], g = f.ghost;
    for (int j = -g; j < ny + g; ++j)
        for (int i = -g; i < nx + g; ++i) {
            if (!(i < 0 || i >= nx) || !(j < 0 || j >= ny))
                continue;
            for (auto *bc : bcs)
                if (bc->axis() == Axis::X) {
                    const auto *periodic = dynamic_cast<const PeriodicBC *>(bc);
                    if (!periodic && int(bc->side()) != (i < 0 ? 0 : 1))
                        continue;
                    if (periodic)
                        f.curr[f.index(i, j)] = f.curr[f.index((i % nx + nx) % nx, j)];
                    else if (const auto *fixed = dynamic_cast<const FixedBC *>(bc))
                        f.curr[f.index(i, j)] = Real(fixed->value);
                    else if (const auto *nf = dynamic_cast<const NoFluxBC *>(bc)) {
                        int source = i < 0 ? 0 : nx - 1;
                        if (nf->closure() == NoFluxBC::Closure::Reflect)
                            source = i < 0 ? -i - 1 : 2 * nx - i - 1;
                        f.curr[f.index(i, j)] = f.curr[f.index(source, j)];
                    }
                }
        }
}
} // namespace PhiX
