#include "mesh/Mesh.h"

#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    Mesh mesh = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                    8, 1.0, 0.0,
                                    6, 1.0, 0.0,
                                    4, 1.0, 0.0);

    require(mesh.patches().size() == 6, "default patch count mismatch");
    require(mesh.patch("xmin").region.hi[0] == 1, "xmin region mismatch");
    require(mesh.patch("xmax").region.lo[0] == 7, "xmax region mismatch");

    mesh.removeFacePatches(Axis::X, Side::LOW);
    mesh.addPatch(Patch{"inlet", Axis::X, Side::LOW,
                        IndexBox{{0, 0, 0}, {1, 2, 4}}, PatchKind::PHYSICAL});
    mesh.addPatch(Patch{"wall", Axis::X, Side::LOW,
                        IndexBox{{0, 2, 0}, {1, 5, 4}}, PatchKind::PHYSICAL});
    mesh.addPatch(Patch{"outlet", Axis::X, Side::LOW,
                        IndexBox{{0, 5, 0}, {1, 6, 4}}, PatchKind::PHYSICAL});
    mesh.validatePatches();

    require(mesh.facePatches(Axis::X, Side::LOW).size() == 3,
            "split face patch count mismatch");
    require(mesh.patch("wall").region.lo[1] == 2, "wall split mismatch");

    Mesh bad = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                   8, 1.0, 0.0,
                                   6, 1.0, 0.0);
    bad.removeFacePatches(Axis::X, Side::LOW);
    bad.addPatch(Patch{"left_a", Axis::X, Side::LOW,
                       IndexBox{{0, 0, 0}, {1, 4, 1}}, PatchKind::PHYSICAL});
    bad.addPatch(Patch{"left_b", Axis::X, Side::LOW,
                       IndexBox{{0, 3, 0}, {1, 6, 1}}, PatchKind::PHYSICAL});

    bool overlap_caught = false;
    try {
        bad.validatePatches();
    } catch (const std::runtime_error&) {
        overlap_caught = true;
    }
    require(overlap_caught, "overlap validation not triggered");

    return 0;
}
