#include "equation/VectorEquation.h"
#include "equation/Equation.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace PhiX {

VectorEquation::VectorEquation(VectorField& unknown_,
                                const std::string& name_)
    : name(name_)
    , unknown(unknown_)
{
    const int N = unknown_.nComponents();
    equations_.reserve(N);
    for (int c = 0; c < N; ++c) {
        equations_.push_back(
            std::make_unique<Equation>(unknown_[c],
                                       name_ + "_c" + std::to_string(c)));
    }
}

void VectorEquation::setRHS(const VectorRHSExpr& expr) {
    const int N = static_cast<int>(equations_.size());
    if (expr.nComponents() != N)
        throw std::invalid_argument(
            "VectorEquation::setRHS: VectorRHSExpr has "
            + std::to_string(expr.nComponents())
            + " components but unknown has " + std::to_string(N));
    for (int c = 0; c < N; ++c)
        equations_[c]->setRHS(expr[c]);
}

void VectorEquation::setRHSComponent(int c, const ExprTree& tree) {
    equations_.at(c)->setRHS(tree);
}

void VectorEquation::setStream(cudaStream_t s) {
    for (auto& eq : equations_)
        eq->setStream(s);
}

cudaStream_t VectorEquation::stream() const {
    return equations_.empty() ? nullptr : equations_[0]->stream();
}

void VectorEquation::registerBC(const ScalarField& field,
                                 std::vector<BoundaryCondition*> bcs) {
    for (auto& eq : equations_)
        eq->registerBC(field, bcs);
}

void VectorEquation::computeRHS(VectorField& rhs) const {
    if (rhs.nComponents() != static_cast<int>(equations_.size()))
        throw std::invalid_argument(
            "VectorEquation::computeRHS: rhs component count mismatch");
    for (int c = 0; c < static_cast<int>(equations_.size()); ++c)
        equations_[c]->computeRHS(rhs[c]);
}

void VectorEquation::computeRHSCPU(VectorField& rhs) const {
    if (rhs.nComponents() != static_cast<int>(equations_.size()))
        throw std::invalid_argument(
            "VectorEquation::computeRHSCPU: rhs component count mismatch");
    for (int c = 0; c < static_cast<int>(equations_.size()); ++c)
        equations_[c]->computeRHSCPU(rhs[c]);
}

Equation& VectorEquation::componentEquation(int c) {
    return *equations_.at(c);
}

const Equation& VectorEquation::componentEquation(int c) const {
    return *equations_.at(c);
}

bool VectorEquation::hasRHS() const {
    for (const auto& eq : equations_)
        if (!eq->hasRHS()) return false;
    return !equations_.empty();
}

void VectorEquation::advanceTransient(
        const std::vector<BoundaryCondition*>& bcs, double dt) {
    for (auto& eq : equations_)
        eq->advanceTransient(bcs, dt);
}

} // namespace PhiX
