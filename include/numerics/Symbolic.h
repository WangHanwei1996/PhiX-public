#pragma once
#include "numerics/System.h"
#include <memory>
#include <string>
#include <vector>

// Continuous 2D expressions. Direct expands product/chain rules before scheme
// binding. Structured additionally preserves recognized composite operators;
// selected recipes then lower them to one scalar stencil DAG.
namespace PhiX::numerics::symbolic {
namespace detail { struct Node; struct Plan; }
class Expanded;
class Discrete;
class Expr {
public:
    Expr(double value = 0);
    explicit Expr(std::shared_ptr<const detail::Node> node) : node_(std::move(node)) {}
    const std::shared_ptr<const detail::Node>& node() const { return node_; }
    Expanded expand(const std::string& name = "expression",
                    ExpansionMode mode = ExpansionMode::Direct) const;
    std::string str() const;
private:
    std::shared_ptr<const detail::Node> node_;
};
Expr field(const ScalarField& f);
// Pointwise data whose spatial derivatives are explicitly zero, e.g. frozen
// grain orientation. It still participates in dependency and pointer binding.
Expr parameter(const ScalarField& f);
Expr coordinate(int axis);
Expr named(const std::string& name, Expr definition);
Expr operator+(Expr a, Expr b);
Expr operator-(Expr a, Expr b);
Expr operator-(Expr a);
Expr operator*(Expr a, Expr b);
Expr operator/(Expr a, Expr b);
Expr sin(Expr a); Expr cos(Expr a); Expr tanh(Expr a);
Expr exp(Expr a); Expr log(Expr a); Expr sqrt(Expr a); Expr abs(Expr a);
Expr pow(Expr a, double exponent);
Expr atan2(Expr y, Expr x);
Expr operator<(Expr a, Expr b); Expr operator<=(Expr a, Expr b);
Expr operator>(Expr a, Expr b); Expr operator>=(Expr a, Expr b);
Expr operator&&(Expr a, Expr b); Expr operator||(Expr a, Expr b);
Expr where(Expr condition, Expr yes, Expr no);
Expr min(Expr a, Expr b); Expr max(Expr a, Expr b);
Expr derivative(Expr a, int axis);
Expr dx(Expr a); Expr dy(Expr a);
Expr dxx(Expr a); Expr dyy(Expr a); Expr dxy(Expr a);
Expr lap(Expr a);
// Explicitly keep or expand a mathematical structure before scheme binding.
Expr gradSq(Expr a);
Expr divGrad(Expr coefficient, Expr potential);
Expr divNormal(Expr coefficient, Expr potential, double normFloor = 0);
Expr analytic(Expr a);
// Discrete building block: evaluate an expression at a grid offset. Its halo
// is inferred after nested operators have been lowered to their chosen stencils.
Expr sample(Expr a, int x, int y);
Expr hypot(Expr x, Expr y);
// Symbolic partial derivative: 'variable' is treated as an independent argument.
Expr partial(Expr expression, Expr variable);
struct Vector {
    const Expr x, y;
    Vector(Expr x, Expr y) : x(std::move(x)), y(std::move(y)) {}
private:
    // Immutable components keep this continuous divergence identity valid.
    std::shared_ptr<const Expr> divergence_;
    std::shared_ptr<const Expr> potential_, coefficient_;
    bool normalized_ = false;
    double normFloor_ = 0;
    friend Vector grad(Expr);
    friend Vector operator+(Vector, Vector);
    friend Vector operator*(Expr, Vector);
    friend Expr div(Vector);
    friend Expr normSquared(Vector);
    friend Vector normalized(Vector, double);
};
Vector grad(Expr a);
// Zero gradient -> zero normal flux. A positive floor is an explicit
// regularization choice; no equilibrium-profile approximation is inserted.
Vector normalized(Vector gradient, double normFloor = 0);
Vector operator+(Vector a, Vector b); Vector operator-(Vector a, Vector b);
Vector operator*(Expr s, Vector v); Vector operator*(Vector v, Expr s);
Vector operator/(Vector v, Expr s);
Vector perpendicular(Vector v);
Vector rotate(Vector v, Expr angle); // lab -> local frame
Expr dot(Vector a, Vector b); Expr normSquared(Vector v);
Expr div(Vector v);
Vector partial(Expr a, Vector arguments);
struct Hessian { Expr xx, xy, yy; };
Hessian hessian(Expr a);
Hessian rotate(Hessian h, Expr angle);

struct RequiredOperator {
    std::string section, key, field, direction;
    std::string location = "cell", frame = "Cartesian", timeLevel = "current";
    int derivativeOrder = 0;
    std::vector<std::string> sources, supportedSchemes;
};
class OperatorInventory {
public:
    std::vector<RequiredOperator> entries;
    void print(std::ostream& out) const;
    OperatorInventory& merge(const OperatorInventory& other);
    // Valid strict Schemes input; every required key is explicit and editable.
    nlohmann::json schemeTemplate() const;
    void writeSchemeTemplate(const std::string& path) const;
};
class Expanded {
public:
    Expanded(Expr original, Expr result, std::string name, std::vector<std::string> trace,
             ScalarField* state = nullptr);
    const Expr& expression() const { return result_; }
    OperatorInventory requiredOperators() const;
    void print(std::ostream& out) const;
    Discrete discretize(const Schemes& schemes) const;
    Discrete discretize(const Schemes& schemes, const ScalarField& layout) const;
    // Host-only footprint inspection. Returns metadata even when the allocated
    // halo is insufficient; does not compile or expose an executable plan.
    OperationInfo inspect(const Schemes& schemes, const ScalarField& layout) const;
private:
    Expr original_, result_;
    std::string name_;
    std::vector<std::string> trace_;
    ScalarField* state_;
    friend struct Equation;
};
class Discrete {
public:
    explicit Discrete(std::shared_ptr<detail::Plan> plan) : plan_(std::move(plan)) {}
    Term lower(Fusion mode) const;
    operator Evolution() const; // only for an expanded ddt equation
    void report(std::ostream& out) const;
    std::string cudaSource(Fusion mode = Fusion::Auto) const;
    std::size_t kernelLaunches(Fusion mode) const;
private:
    std::shared_ptr<detail::Plan> plan_;
};
struct Equation {
    ScalarField* state;
    Expr rhs;
    Expanded expand(const std::string& name = "",
                    ExpansionMode mode = ExpansionMode::Direct) const;
};
struct Ddt {
    ScalarField* state;
    Expr coefficient = 1;
    Equation operator==(Expr rhs) const { return {state, rhs/coefficient}; }
};
inline Ddt ddt(ScalarField& f) { return {&f}; }
inline Ddt operator*(Expr coefficient, Ddt d) { return {d.state, coefficient*d.coefficient}; }
inline Ddt operator*(Ddt d, Expr coefficient) { return coefficient*d; }
} // namespace PhiX::numerics::symbolic
