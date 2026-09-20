// ---------------------------------------------------------------------------
// module_storage — padded-storage geometry API (v2.42.0, PHIX_IMPROVEMENTS P-1)
//
// The 2D-field footgun: a 2D mesh has n[2]=1, so fields store 1+2*ghost
// z-planes and the physical plane is the MIDDLE one.  Raw buffers sized
// sx*sy instead of storedSize are out of bounds.  This test pins down the
// contract of storedSize / storedBytes() / planeOffset():
//
//   1. storedSize == product(storedDims) for 1D/2D/3D and ghost 1/2.
//   2. planeOffset() == storedDims[0]*storedDims[1]*ghost == index(-g,-g,0).
//   3. Every physical index(i,j,0) of a 2D field equals
//      planeOffset() + (i+g) + sx*(j+g)  (the documented raw-buffer formula).
//   4. A raw buffer of storedSize elements addressed via the formula holds
//      exactly the field's physical cells (bounds-checked with .at()).
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static void checkGeometry(const ScalarField& f, const std::string& tag) {
    const std::size_t sx = f.storedDims[0];
    const std::size_t sy = f.storedDims[1];
    const std::size_t sz = f.storedDims[2];
    const int g = f.ghost;

    require(f.storedSize == sx * sy * sz, tag + ": storedSize != sx*sy*sz");
    require(f.storedBytes() == f.storedSize * sizeof(Real),
            tag + ": storedBytes mismatch");
    require(f.curr.size() == f.storedSize, tag + ": curr.size != storedSize");

    require(f.planeOffset() == sx * sy * static_cast<std::size_t>(g),
            tag + ": planeOffset != sx*sy*g");
    require(static_cast<std::size_t>(f.index(-g, -g, 0)) == f.planeOffset(),
            tag + ": planeOffset != index(-g,-g,0)");

    // Documented raw-buffer addressing formula must agree with index().
    for (int j = 0; j < f.mesh.n[1]; ++j)
    for (int i = 0; i < f.mesh.n[0]; ++i) {
        const std::size_t raw = f.planeOffset() + (i + g) + sx * (j + g);
        require(static_cast<std::size_t>(f.index(i, j, 0)) == raw,
                tag + ": index(i,j,0) != planeOffset()+(i+g)+sx*(j+g)");
    }
}

int main() {
    for (int g : {1, 2}) {
        const std::string gt = " (ghost " + std::to_string(g) + ")";

        Mesh m1 = Mesh::makeUniform1D(CoordSys::CARTESIAN, 7, 0.5);
        ScalarField f1(m1, "f1", g);
        checkGeometry(f1, "1D" + gt);
        require(f1.storedDims[2] == 1 + 2 * g, "1D: storedDims[2]" + gt);

        Mesh m2 = Mesh::makeUniform2D(CoordSys::CARTESIAN, 6, 0.5, 0.0,
                                      4, 0.25, 0.0);
        ScalarField f2(m2, "f2", g);
        checkGeometry(f2, "2D" + gt);
        // The trap this API guards against: 2D storage is NOT one plane.
        require(f2.storedDims[2] == 1 + 2 * g, "2D: storedDims[2]" + gt);
        require(f2.storedSize ==
                    std::size_t(6 + 2 * g) * (4 + 2 * g) * (1 + 2 * g),
                "2D: full 3-plane storedSize" + gt);

        Mesh m3 = Mesh::makeUniform3D(CoordSys::CARTESIAN, 5, 0.1, 0.0,
                                      4, 0.2, 0.0, 3, 0.4, 0.0);
        ScalarField f3(m3, "f3", g);
        checkGeometry(f3, "3D" + gt);
    }

    // Raw-buffer walkthrough (2D, ghost 1): copy physical cells through the
    // documented formula into a storedSize buffer; .at() throws on any
    // out-of-bounds — this is exactly the access pattern of shadow buffers.
    {
        Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN, 6, 1.0, 0.0,
                                     4, 1.0, 0.0);
        ScalarField f(m, "raw", 1);
        f.fillCurr(-7.0);
        for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 6; ++i)
            f.curr[static_cast<std::size_t>(f.index(i, j))] = 10.0 * j + i;

        std::vector<Real> raw(f.storedSize, 0.0);
        const std::size_t sx = f.storedDims[0];
        for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 6; ++i)
            raw.at(f.planeOffset() + (i + 1) + sx * (j + 1)) =
                f.curr.at(static_cast<std::size_t>(f.index(i, j)));

        for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 6; ++i)
            require(raw[static_cast<std::size_t>(f.index(i, j))] ==
                        10.0 * j + i,
                    "raw-buffer roundtrip mismatch");
    }

    std::printf("module_storage: ALL PASSED\n");
    return 0;
}
