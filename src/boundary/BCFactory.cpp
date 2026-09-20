#include "boundary/BCFactory.h"
#include "core/Error.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"

#include <stdexcept>
#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// Helper: extract BC type string from a JSON value (string or object)
// ---------------------------------------------------------------------------
static std::string getBCType(const nlohmann::json& v)
{
    if (v.is_string()) return v.get<std::string>();
    if (v.is_object()) return v.at("type").get<std::string>();
    throw std::runtime_error("buildBCs: BC entry must be a string or an object");
}

// ---------------------------------------------------------------------------
// Helper: process one axis pair (lo/hi) from the JSON config
// ---------------------------------------------------------------------------
static void addAxisBCs(const Mesh& mesh, BCSet& set, Axis axis,
                       const nlohmann::json& lo, const nlohmann::json& hi)
{
    std::string lo_type = getBCType(lo);
    std::string hi_type = getBCType(hi);

    bool lo_periodic = (lo_type == "Periodic");
    bool hi_periodic = (hi_type == "Periodic");

    if (lo_periodic != hi_periodic) {
        const char* name = (axis == Axis::X) ? "X" : (axis == Axis::Y) ? "Y" : "Z";
        throw std::runtime_error(
            std::string("buildBCs: Periodic BC must be set on both sides of axis ")
            + name + ", got \"" + lo_type + "\" / \"" + hi_type + "\"");
    }

    if (lo_periodic) {
        set.storage.push_back(std::make_unique<PeriodicBC>(mesh.facePatch(axis, Side::LOW)));
        set.ptrs.push_back(set.storage.back().get());
        return;
    }

    // Non-periodic: handle each side independently
    auto addSide = [&](const nlohmann::json& cfg, Side side) {
        std::string type = getBCType(cfg);
        const Patch& patch = mesh.facePatch(axis, side);
        if (type == "NoFlux") {
            auto closure=cfg.is_object()?cfg.value("closure",std::string("constant")):"constant";
            if(closure!="constant" && closure!="reflect")
                throw std::invalid_argument("buildBCs: NoFlux closure must be constant or reflect");
            if(closure=="reflect" && mesh.dim!=2)
                throw std::invalid_argument("buildBCs: reflected NoFlux currently supports 2D");
            set.storage.push_back(std::make_unique<NoFluxBC>(patch,
                closure=="reflect"?NoFluxBC::Closure::Reflect:NoFluxBC::Closure::Constant));
            set.ptrs.push_back(set.storage.back().get());
        } else if (type == "Fixed") {
            double val = (cfg.is_object() && cfg.contains("value"))
                         ? cfg.at("value").get<double>() : 0.0;
            set.storage.push_back(std::make_unique<FixedBC>(patch, val));
            set.ptrs.push_back(set.storage.back().get());
        } else {
            throw std::runtime_error(
                "buildBCs: unsupported BC type \"" + type + "\"");
        }
    };

    addSide(lo, Side::LOW);
    addSide(hi, Side::HIGH);
}

// ---------------------------------------------------------------------------
// buildBCs
// ---------------------------------------------------------------------------
BCSet buildBCs(const Mesh& mesh, const nlohmann::json& bc_config)
{
    BCSet set;

    // X axis (required)
    addAxisBCs(mesh, set, Axis::X,
               bc_config.at("x_min"),
               bc_config.at("x_max"));

    // Y axis: both keys or neither — one alone used to silently skip the
    // whole axis, leaving its ghost cells permanently stale.
    const bool yLo = bc_config.contains("y_min"), yHi = bc_config.contains("y_max");
    if (yLo != yHi)
        throw ConfigError("buildBCs",
            std::string(yLo ? "y_min" : "y_max") + " supplied without "
            + (yLo ? "y_max" : "y_min"),
            "provide both y_min and y_max, or neither");
    if (yLo && yHi) {
        addAxisBCs(mesh, set, Axis::Y,
                   bc_config.at("y_min"),
                   bc_config.at("y_max"));
    } else if (mesh.dim >= 2) {
        // Legit when the app applies hand-written Y BC kernels — warn, don't throw.
        warnOnce("buildBCs-no-y",
                 "buildBCs: 2D/3D mesh but no y_min/y_max in the config — "
                 "Y-face ghost cells will NOT be refreshed by these BCs");
    }

    // Z axis (optional, for 3D) — same both-or-neither rule.
    const bool zLo = bc_config.contains("z_min"), zHi = bc_config.contains("z_max");
    if (zLo != zHi)
        throw ConfigError("buildBCs",
            std::string(zLo ? "z_min" : "z_max") + " supplied without "
            + (zLo ? "z_max" : "z_min"),
            "provide both z_min and z_max, or neither");
    if (zLo && zHi) {
        addAxisBCs(mesh, set, Axis::Z,
                   bc_config.at("z_min"),
                   bc_config.at("z_max"));
    }

    return set;
}

} // namespace PhiX
