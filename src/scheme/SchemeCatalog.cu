#include "scheme/SchemeCatalog.h"
#include "solver/Solver.h"

namespace PhiX::scheme {
namespace {
template<class Factory>
std::vector<Descriptor> descriptions(const std::vector<detail::Entry<Factory>>& entries) {
    std::vector<Descriptor> result;
    result.reserve(entries.size());
    for (const auto& e : entries) result.push_back(e.descriptor);
    return result;
}

TimeScheme euler() { return TimeScheme::EULER; }
TimeScheme rk4()   { return TimeScheme::RK4; }
} // namespace

const std::vector<detail::Entry<detail::TimeFactory>>& detail::timeFactories() {
    // These select the existing time enum, not a promise that every execution
    // path implements RK4. Time/backend binding is a separate validation step.
    static const std::vector<Entry<TimeFactory>> entries = {
        {{"EULER", 1, 0, false, Grid2D::NotSpatial, "EULER", "forward Euler"}, &euler},
        {{"RK4", 4, 0, false, Grid2D::NotSpatial, "RK4", "classical explicit RK4"}, &rk4}
    };
    return entries;
}

namespace {
template <FaceMethod M> FaceMethod faceMethod() {
    return M;
}
} // namespace
const std::vector<detail::Entry<detail::FaceFactory>> &detail::faceFactories(Family family) {
    static const std::vector<Entry<FaceFactory>> interpolation = {
        {{"LinearGhost", 2, 1, false, Grid2D::Rectangular, "LinearGhost",
          "mean of the two cells, including boundary ghosts"},
         &faceMethod<FaceMethod::GhostLinear>},
        {{"NearestBoundary", 2, 0, false, Grid2D::Rectangular, "NearestBoundary",
          "legacy boundary-face nearest-cell interpolation"},
         &faceMethod<FaceMethod::NearestCell>}};
    static const std::vector<Entry<FaceFactory>> normal = {
        {{"CD2", 2, 1, false, Grid2D::Rectangular, "CD2", "two-point face-normal gradient"},
         &faceMethod<FaceMethod::NormalCD2>}};
    static const std::vector<Entry<FaceFactory>> vector = {
        {{"CD2", 2, 1, true, Grid2D::Rectangular, "CD2",
          "normal difference and four-cell tangential reconstruction"},
         &faceMethod<FaceMethod::VectorCD2>}};
    static const std::vector<Entry<FaceFactory>> div = {
        {{"Difference", 2, 0, false, Grid2D::Rectangular, "Difference",
          "oriented face-flux difference"},
         &faceMethod<FaceMethod::Difference>}};
    switch (family) {
    case Family::Interpolation:
        return interpolation;
    case Family::SnGrad:
        return normal;
    case Family::FaceGrad:
        return vector;
    case Family::FluxDiv:
        return div;
    default:
        throw std::invalid_argument("not a face scheme family");
    }
}

const std::vector<FamilyInfo>& families() {
    static const std::vector<FamilyInfo> entries = {{Family::Ddt, "ddt"},
                                                    {Family::Laplacian, "laplacian"},
                                                    {Family::Gradient, "gradient"},
                                                    {Family::GradSq, "gradSq"},
                                                    {Family::Advection, "advection"},
                                                    {Family::AnisoDiv, "anisoDiv"},
                                                    {Family::Interpolation, "interpolation"},
                                                    {Family::SnGrad, "snGrad"},
                                                    {Family::FaceGrad, "faceGrad"},
                                                    {Family::FluxDiv, "fluxDiv"},
                                                    {Family::SecondDerivative, "secondDerivative"},
                                                    {Family::DivGrad, "divGrad"},
                                                    {Family::DivNormal, "divNormal"}};
    return entries;
}

const FamilyInfo* findFamily(const std::string& section) {
    for (const auto& f : families())
        if (section == f.section) return &f;
    return nullptr;
}

const std::vector<Descriptor>& catalog(Family family) {
    switch (family) {
    case Family::DivGrad: {
        static const auto d = descriptions(detail::divGradFactories()); return d;
    }
    case Family::DivNormal: {
        static const auto d = descriptions(detail::divNormalFactories()); return d;
    }
    case Family::SecondDerivative: {
        static const auto d = descriptions(detail::secondDerivativeFactories()); return d;
    }
    case Family::Interpolation: {
        static const auto d = descriptions(detail::faceFactories(Family::Interpolation));
        return d;
    }
    case Family::SnGrad: {
        static const auto d = descriptions(detail::faceFactories(Family::SnGrad));
        return d;
    }
    case Family::FaceGrad: {
        static const auto d = descriptions(detail::faceFactories(Family::FaceGrad));
        return d;
    }
    case Family::FluxDiv: {
        static const auto d = descriptions(detail::faceFactories(Family::FluxDiv));
        return d;
    }
    case Family::Ddt: {
        static const auto d = descriptions(detail::timeFactories()); return d;
    }
    case Family::Laplacian: {
        static const auto d = descriptions(detail::laplacianFactories()); return d;
    }
    case Family::Gradient: {
        static const auto d = descriptions(detail::gradientFactories()); return d;
    }
    case Family::GradSq: {
        static const auto d = descriptions(detail::gradSqFactories()); return d;
    }
    case Family::Advection: {
        static const auto d = descriptions(detail::advectionFactories()); return d;
    }
    case Family::AnisoDiv: {
        static const auto d = descriptions(detail::anisoFactories()); return d;
    }
    }
    throw std::invalid_argument("scheme::catalog: invalid operator family");
}

const Descriptor* find(Family family, const std::string& name) {
    for (const auto& d : catalog(family))
        if (name == d.name) return &d;
    return nullptr;
}

std::string names(Family family) {
    std::string out;
    for (const auto& d : catalog(family)) {
        if (!out.empty()) out += ", ";
        out += d.name;
    }
    return out;
}

std::string sectionNames() {
    std::string out;
    for (const auto& f : families()) {
        if (!out.empty()) out += ", ";
        out += f.section;
    }
    return out;
}
} // namespace PhiX::scheme
