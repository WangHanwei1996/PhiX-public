#include "field/FieldLayout.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"

#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    Mesh mesh = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                    5, 1.0, 0.0,
                                    4, 2.0, 0.0,
                                    3, 3.0, 0.0);

    FieldLayout layout(mesh, 2);
    require(layout.ghost == 2, "layout ghost mismatch");
    require(layout.storedDims[0] == 9, "layout sx mismatch");
    require(layout.storedDims[1] == 8, "layout sy mismatch");
    require(layout.storedDims[2] == 7, "layout sz mismatch");
    require(layout.index(0, 0, 0) == 2 + 9 * (2 + 8 * 2), "layout index mismatch");

    ScalarField sf(layout, "phi");
    require(sf.layout.ghost == sf.ghost, "scalar layout ghost mirror mismatch");
    require(sf.layout.storedSize == sf.storedSize, "scalar layout size mirror mismatch");
    require(sf.index(1, 1, 1) == sf.layout.index(1, 1, 1), "scalar index mismatch");

    VectorField vf(layout, "u", 3);
    require(vf.layout.ghost == vf.ghost, "vector layout ghost mirror mismatch");
    require(vf[0].layout.storedSize == layout.storedSize, "component layout mismatch");
    require(vf[2].name == "u_z", "vector component naming mismatch");

    return 0;
}
