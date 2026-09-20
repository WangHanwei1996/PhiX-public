#pragma once
#include "numerics/Expression.h"
#include "equation/Equation.h"
#include "boundary/BCBatch.h"
#include "solver/HealthCheck.h"
#include <cstdint>
#include <map>
#include <memory>
#include <ostream>

namespace PhiX::numerics {
namespace symbolic {
enum class ExpansionMode { Direct, Structured };
class Expr;
class Expanded;
class OperatorInventory;
class Preparation;
struct PreparationOptions;
struct Equation;
} // namespace symbolic
enum class Backend { CPU, CUDA };
enum class Coupling { SameLevel, Sequential };
struct ExecutionOptions {
    Backend backend = Backend::CUDA;
    Fusion fusion = Fusion::Auto;
    Coupling coupling = Coupling::SameLevel;
};
struct Evolution {
    ScalarField *state;
    std::function<Term(Fusion)> lower;
};
struct Ddt {
    ScalarField *state;
    template <class Expr> Evolution operator==(Expr expr) const {
        return {state, [expr](Fusion mode) { return expr.lower(mode); }};
    }
    Evolution operator==(Term t) const { return *this == adapted(std::move(t)); }
    Evolution operator==(const RHSExpr &e) const { return *this == adapted(e); }
};
inline Ddt ddt(ScalarField &state) {
    return {&state};
}

// External effects are explicit scheduling barriers. Captured objects must outlive the plan.
struct ExternalStep {
    std::string name;
    OperationInfo reads;
    std::vector<ScalarField *> writes;
    std::vector<FaceField *> faceWrites;
    std::function<void(double, double)> cpu, gpu; // current time, chosen dt
    std::function<void(Backend, Fusion)> configure;
    std::function<std::string()> describe;
};

class System {
  public:
    explicit System(const Schemes &schemes,
                    symbolic::ExpansionMode mode = symbolic::ExpansionMode::Direct)
        : schemes_(schemes), expansionMode_(mode) {}
    ~System();
    HealthCheck health;
    System(const System &) = delete;
    System &operator=(const System &) = delete;
    void bc(ScalarField &f, std::vector<BoundaryCondition *> bcs);
    // For custom BCs: caller certifies the callback fills all required edges/corners.
    void halo(ScalarField &f, std::function<void()> cpu, std::function<void()> gpu, int width);
    template <class Expr> void define(ScalarField &f, Expr expr) {
        addDefinition(f, [expr](Fusion m) { return expr.lower(m); });
    }
    // Original formulas: expand now for inspection, bind schemes in compile().
    void define(ScalarField &f, const symbolic::Expr &expr);
    void define(ScalarField &f, Term t) { define(f, adapted(std::move(t))); }
    void define(ScalarField &f, const RHSExpr &e) { define(f, adapted(e)); }
    void define(ScalarField &f, const ExprTree &e) { define(f, adapted(e)); }
    // Declared producers (including face operations) take part in dependency sorting.
    void producer(ExternalStep step);
    // Replace captured parameters/coefficients while retaining the compiled graph.
    template <class Expr> void replace(ScalarField &f, Expr e) {
        replaceExpression(f, [e](Fusion m) { return e.lower(m); });
    }
    void replace(ScalarField &f, const symbolic::Expr &expr);
    void replace(ScalarField &f, Term t) { replace(f, adapted(std::move(t))); }
    void replace(ScalarField &f, const RHSExpr &e) { replace(f, adapted(e)); }
    void add(Evolution e);
    void add(const symbolic::Equation &equation);
    // Symbolic registrations only; available before compile (include Symbolic.h).
    symbolic::OperatorInventory requiredOperators() const;
    void reportEquations(std::ostream &out) const;
    // Read-only host preparation; no RHS evaluation, GPU compilation or field writes.
    symbolic::Preparation prepare(const symbolic::PreparationOptions &options) const;
    symbolic::Preparation prepare() const;
    void beforeStep(ExternalStep step);
    void afterStep(ExternalStep step);
    System &compile(ExecutionOptions options = {});
    void advance(double dt);
    // Adaptive dt is selected after every RHS is evaluated and before any update.
    void advanceAdaptive(
        const std::function<double(const std::vector<const ScalarField *> &)> &chooseDt);
    void refresh(ScalarField &auxiliary);
    void refresh(FaceField &auxiliary);
    // Required after external edits, uploads, parameter changes, mapping, or restart.
    void invalidate();
    void touch(ScalarField &f);
    double time() const { return time_; }
    std::uint64_t step() const { return step_; }
    void restoreClock(double time, std::uint64_t step);
    void report(std::ostream &out) const;
    std::size_t persistentBytes() const;

  private:
    struct Node {
        ScalarField *cell = nullptr;
        std::vector<const void *> outputs;
        std::function<Term(Fusion)> lower;
        ExternalStep external;
        bool evolution = false;
        std::shared_ptr<const symbolic::Expanded> expanded;
        Term term;
        OperationInfo info;
        std::unique_ptr<Equation> equation;
        std::unique_ptr<ScalarField> rhs;
        std::vector<std::uint64_t> stamp;
        bool valid = false;
    };
    struct Halo {
        std::vector<BoundaryCondition *> bcs;
        std::unique_ptr<BCBatch> batch;
        std::function<void()> cpu, gpu;
        int width = 0;
        std::uint64_t stamp = 0;
        bool valid = false;
    };
    Schemes schemes_;
    symbolic::ExpansionMode expansionMode_;
    ExecutionOptions options_;
    bool compiled_ = false;
    double time_ = 0;
    std::uint64_t step_ = 0, version_ = 1;
    std::vector<Node> nodes_;
    std::vector<ExternalStep> before_, after_;
    std::map<const void *, std::size_t> producers_;
    std::map<const void *, std::uint64_t> versions_;
    std::map<const ScalarField *, Halo> halos_;
    std::vector<std::size_t> order_;
    std::vector<std::function<bool()>> layouts_;
    void addDefinition(ScalarField &, std::function<Term(Fusion)>);
    void editable() const;
    void replaceExpression(ScalarField &, std::function<Term(Fusion)>,
                           std::shared_ptr<const symbolic::Expanded> = {});
    void ensure(std::size_t);
    void prepareReads(const OperationInfo &);
    void validate(const OperationInfo &, const ScalarField *output = nullptr,
                  bool hostInspection = false) const;
    void runExternal(const ExternalStep &, double dt);
    void evaluateRHS(std::size_t);
    void update(std::size_t, double dt);
    void finish(double dt);
    void start(double dt);
    void checkCompiled() const;
};
} // namespace PhiX::numerics
