#include "field/VectorField.h"
#include "IO/FieldIO.h"

#include <iostream>
#include <stdexcept>

namespace PhiX {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string makeComponentName(const std::string& base, int c, int nComp) {
    if (nComp == 3) {
        switch (c) {
            case 0: return base + "_x";
            case 1: return base + "_y";
            case 2: return base + "_z";
        }
    }
    return base + "_" + std::to_string(c);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

std::string VectorField::componentName(const std::string& baseName,
                                        int c, int nComp) {
    return makeComponentName(baseName, c, nComp);
}

VectorField::VectorField(const Mesh& mesh_,
                          const std::string& name_,
                          int nComponents,
                          int ghost_)
    : VectorField(FieldLayout(mesh_, ghost_), name_, nComponents)
{}

VectorField::VectorField(const FieldLayout& layout_,
                          const std::string& name_,
                          int nComponents)
    : layout(layout_)
    , mesh(layout.meshRef())
    , ghost(layout.ghost)
    , name(name_)
{
    if (nComponents <= 0)
        throw std::invalid_argument("VectorField: nComponents must be > 0");

    components_.reserve(nComponents);
    for (int c = 0; c < nComponents; ++c) {
        components_.emplace_back(layout, makeComponentName(name_, c, nComponents));
    }
}

// ---------------------------------------------------------------------------
// Move semantics
// ---------------------------------------------------------------------------

VectorField::VectorField(VectorField&& other) noexcept
    : layout(other.layout)
    , mesh(other.mesh)
    , ghost(other.ghost)
    , name(std::move(other.name))
    , components_(std::move(other.components_))
{}

VectorField& VectorField::operator=(VectorField&& other) noexcept {
    if (this == &other) return *this;
    // mesh is const ref — cannot rebind; caller must ensure same mesh lifetime
    layout     = other.layout;
    ghost      = other.ghost;
    name       = std::move(other.name);
    components_ = std::move(other.components_);
    return *this;
}

// ---------------------------------------------------------------------------
// Component access
// ---------------------------------------------------------------------------

ScalarField& VectorField::operator[](int c) {
    if (c < 0 || c >= static_cast<int>(components_.size()))
        throw std::out_of_range("VectorField: component index out of range");
    return components_[c];
}

const ScalarField& VectorField::operator[](int c) const {
    if (c < 0 || c >= static_cast<int>(components_.size()))
        throw std::out_of_range("VectorField: component index out of range");
    return components_[c];
}

// ---------------------------------------------------------------------------
// Initialisation helpers
// ---------------------------------------------------------------------------

void VectorField::fill(double value) {
    for (auto& comp : components_) comp.fill(value);
}

void VectorField::fillCurr(double value) {
    for (auto& comp : components_) comp.fillCurr(value);
}

void VectorField::fillPrev(double value) {
    for (auto& comp : components_) comp.fillPrev(value);
}

// ---------------------------------------------------------------------------
// Time-stepping
// ---------------------------------------------------------------------------

void VectorField::advanceTimeLevelCPU() {
    for (auto& comp : components_) comp.advanceTimeLevelCPU();
}

void VectorField::advanceTimeLevelGPU() {
    for (auto& comp : components_) comp.advanceTimeLevelGPU();
}

// ---------------------------------------------------------------------------
// GPU management
// ---------------------------------------------------------------------------

bool VectorField::deviceAllocated() const {
    for (const auto& comp : components_)
        if (!comp.deviceAllocated()) return false;
    return !components_.empty();
}

void VectorField::allocDevice() {
    for (auto& comp : components_) comp.allocDevice();
}

void VectorField::freeDevice() {
    for (auto& comp : components_) comp.freeDevice();
}

void VectorField::uploadCurrToDevice() const {
    for (const auto& comp : components_) comp.uploadCurrToDevice();
}

void VectorField::uploadPrevToDevice() const {
    for (const auto& comp : components_) comp.uploadPrevToDevice();
}

void VectorField::uploadAllToDevice() const {
    for (const auto& comp : components_) comp.uploadAllToDevice();
}

void VectorField::downloadCurrFromDevice() {
    for (auto& comp : components_) comp.downloadCurrFromDevice();
}

void VectorField::downloadPrevFromDevice() {
    for (auto& comp : components_) comp.downloadPrevFromDevice();
}

void VectorField::downloadAllFromDevice() {
    for (auto& comp : components_) comp.downloadAllFromDevice();
}

// ---------------------------------------------------------------------------
// IO  (delegated to IO module)
// ---------------------------------------------------------------------------

void VectorField::write(const std::string& path, FieldFormat fmt,
                        double coordScale) const {
    IO::writeField(*this, path, fmt, coordScale);
}

VectorField VectorField::readFromFile(const Mesh& mesh,
                                       const std::string& path,
                                       int ghost) {
    return IO::readVectorField(mesh, path, ghost);
}

// ---------------------------------------------------------------------------
// print
// ---------------------------------------------------------------------------

void VectorField::print() const {
    std::cout << "=== VectorField ===\n";
    std::cout << "  name        : " << name << "\n";
    std::cout << "  nComponents : " << nComponents() << "\n";
    std::cout << "  ghost       : " << ghost << "\n";
    for (const auto& comp : components_) {
        comp.print();
    }
}

} // namespace PhiX
