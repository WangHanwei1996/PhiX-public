#include "field/FieldLayout.h"
#include "core/Error.h"

namespace PhiX {

FieldLayout::FieldLayout(const Mesh& mesh_, int ghost_, Centering centering_)
    : mesh(&mesh_)
    , ghost(ghost_)
    , centering(centering_)
{
    if (ghost_ < 0)
        throw std::invalid_argument("FieldLayout: ghost must be >= 0");
    // Every field passes through here — reject a default-constructed /
    // corrupted Mesh (n = 0 would silently produce an empty simulation).
    if (!mesh_.isValid())
        throw ValidationError("FieldLayout",
            "mesh is not valid (default-constructed or bad dims/spacing)",
            "build meshes with Mesh::makeUniform1D/2D/3D, which validate "
            "their parameters");

    int normalAxis = faceAxis(centering_);

    for (int ax = 0; ax < 3; ++ax) {
        if (ax == normalAxis) {
            // Face-normal direction: n+1 face values, no ghost padding.
            storedDims[ax] = mesh_.n[ax] + 1;
        } else {
            storedDims[ax] = mesh_.n[ax] + 2 * ghost_;
        }
    }

    storedSize = static_cast<std::size_t>(storedDims[0])
               * storedDims[1]
               * storedDims[2];
}

} // namespace PhiX
