#pragma once

// Host-side catalogue. Each operator owns a typed table pairing its numerical
// description with its executable factory. Configuration reads those tables;
// it never maintains another list of accepted scheme names.
#include <stdexcept>
#include <string>
#include <vector>

namespace PhiX {
class ScalarField;
class VectorField;
struct Term;
enum class TimeScheme;
enum class AnisoScheme;

namespace scheme {

enum class Family {
    Ddt,
    Laplacian,
    Gradient,
    GradSq,
    Advection,
    AnisoDiv,
    Interpolation,
    SnGrad,
    FaceGrad,
    FluxDiv,
    SecondDerivative,
    DivGrad,
    DivNormal
};

// Describes the existing 2D implementation, not certification of a PDE or BC.
enum class Grid2D {
    Rectangular,
    SquareRequired,
    SquareForIsotropy,  // rectangular evaluation exists; isotropic error needs dx=dy
    NotSpatial
};

struct Descriptor {
    const char* name;
    int interiorOrder;       // smooth interior / formal temporal order only
    int primaryHalo;         // main scalar input; NOT every input's access radius
    bool readsCorners2D;
    Grid2D grid2D;
    const char* effective2D;  // e.g. legacy lap(Iso27) actually uses CD2 in 2D
    const char* meaning;      // operator-specific mathematical form
};

struct FamilyInfo {
    Family id;
    const char* section;
};

const std::vector<FamilyInfo>& families();
const FamilyInfo* findFamily(const std::string& section);
// Metadata is projected from the executable factory tables, in stable order.
// The first entry is the historical built-in default.
const std::vector<Descriptor>& catalog(Family family);
const Descriptor* find(Family family, const std::string& name);
std::string names(Family family);
std::string sectionNames();

enum class FaceMethod { GhostLinear, NearestCell, NormalCD2, VectorCD2, Difference };

namespace detail {

template<class Factory>
struct Entry {
    Descriptor descriptor;
    Factory factory;
};

enum class DerivativeMethod { CD2, CD4, CD6 };
enum class CompositeMethod { Axial, Isotropic, IsotropicNormal, IsotropicNormalVector };
using CompositeFactory = CompositeMethod (*)();
const std::vector<Entry<CompositeFactory>>& divGradFactories();
const std::vector<Entry<CompositeFactory>>& divNormalFactories();
using DerivativeFactory = DerivativeMethod (*)();
const std::vector<Entry<DerivativeFactory>>& secondDerivativeFactories();

using FaceFactory = FaceMethod (*)();
const std::vector<Entry<FaceFactory>> &faceFactories(Family family);

using ScalarFactory = Term (*)(const ScalarField&, double);
using GradientFactory = Term (*)(const ScalarField&, int, double);
using AdvectionFactory = Term (*)(const VectorField&, const ScalarField&, double);
using AnisoFactory = AnisoScheme (*)();
using TimeFactory = TimeScheme (*)();

const std::vector<Entry<ScalarFactory>>& laplacianFactories();
const std::vector<Entry<GradientFactory>>& gradientFactories();
const std::vector<Entry<ScalarFactory>>& gradSqFactories();
const std::vector<Entry<AdvectionFactory>>& advectionFactories();
const std::vector<Entry<AnisoFactory>>& anisoFactories();
const std::vector<Entry<TimeFactory>>& timeFactories();

template<class Scheme, class Factory>
Entry<Factory> entry(Factory factory, Grid2D grid, bool corners,
                     const char* meaning, const char* effective2D = nullptr) {
    return {{Scheme::name(), Scheme::order(), Scheme::ghostRequired(), corners,
             grid, effective2D ? effective2D : Scheme::name(), meaning}, factory};
}

template<class Factory>
const Entry<Factory>& lookup(const std::vector<Entry<Factory>>& entries,
                             const std::string& name, const char* where,
                             bool emptyMeansDefault = true) {
    if (emptyMeansDefault && name.empty()) return entries.front();
    for (const auto& e : entries)
        if (name == e.descriptor.name) return e;
    std::string supported;
    for (const auto& e : entries) {
        if (!supported.empty()) supported += ", ";
        supported += e.descriptor.name;
    }
    // Preserve the operator string APIs' historical exception class.
    throw std::invalid_argument(std::string(where) + ": unknown scheme '"
                                + name + "'. Supported: " + supported);
}

} // namespace detail
} // namespace scheme
} // namespace PhiX
