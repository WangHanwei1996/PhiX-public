#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static bool near(double a, double b) {
    return std::abs(a - b) < 1e-12;
}

int main() {
    {
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        4, 1.0, 0.0,
                                        6, 1.0, 0.0);
        mesh.removeFacePatches(Axis::X, Side::LOW);
        mesh.addPatch(Patch{"left_low", Axis::X, Side::LOW,
                            IndexBox{{0, 0, 0}, {1, 3, 1}}, PatchKind::PHYSICAL});
        mesh.addPatch(Patch{"left_high", Axis::X, Side::LOW,
                            IndexBox{{0, 3, 0}, {1, 6, 1}}, PatchKind::PHYSICAL});
        mesh.validatePatches();

        ScalarField f(mesh, "phi", 1);
        for (int j = 0; j < mesh.n[1]; ++j) {
            for (int i = 0; i < mesh.n[0]; ++i) {
                f.curr[f.index(i, j)] = 10.0 * j + i;
            }
        }

        NoFluxBC bc_lo(mesh.patch("left_low"));
        FixedBC  bc_hi(mesh.patch("left_high"), -7.0);
        bc_lo.applyOnCPU(f);
        bc_hi.applyOnCPU(f);

        for (int j = 0; j < 3; ++j)
            require(near(f.curr[f.index(-1, j)], f.curr[f.index(0, j)]),
                    "NoFlux low patch fill mismatch");
        for (int j = 3; j < 6; ++j)
            require(near(f.curr[f.index(-1, j)], -7.0),
                    "Fixed low patch fill mismatch");
    }

    {
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        5, 1.0, 0.0,
                                        4, 1.0, 0.0);
        ScalarField f(mesh, "phi", 1);
        for (int j = 0; j < mesh.n[1]; ++j) {
            for (int i = 0; i < mesh.n[0]; ++i) {
                f.curr[f.index(i, j)] = 100.0 * j + i;
            }
        }

        PeriodicBC pbc(mesh.patch("xmin"));
        pbc.applyOnCPU(f);

        for (int j = 0; j < mesh.n[1]; ++j) {
            require(near(f.curr[f.index(-1, j)], f.curr[f.index(mesh.n[0] - 1, j)]),
                    "Periodic low ghost mismatch");
            require(near(f.curr[f.index(mesh.n[0], j)], f.curr[f.index(0, j)]),
                    "Periodic high ghost mismatch");
        }
    }

    return 0;
}
