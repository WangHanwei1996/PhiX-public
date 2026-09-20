#include "mesh/Patch.h"

namespace PhiX {

const char* defaultPatchName(Axis axis, Side side) {
    switch (axis) {
        case Axis::X: return side == Side::LOW ? "xmin" : "xmax";
        case Axis::Y: return side == Side::LOW ? "ymin" : "ymax";
        case Axis::Z: return side == Side::LOW ? "zmin" : "zmax";
    }
    return "";
}

} // namespace PhiX
