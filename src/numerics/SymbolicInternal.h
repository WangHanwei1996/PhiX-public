#pragma once
#include "numerics/Symbolic.h"
#include <map>
#include <mutex>
#include <set>
namespace PhiX::numerics::symbolic::detail {
enum class Op { Constant, Field, Parameter, Coordinate, Alias, Add, Mul, Divide,
    Negate, Sin, Cos, Tanh, Exp, Log, Sqrt, Abs, Pow, Atan2, Less, LessEqual,
    Greater, GreaterEqual, And, Or, Where, Derivative, Laplacian, Divergence,
    Partial, Jet, GradientSquare, DivGrad, DivNormal, Analytic, Shift, Sample, Hypot };
struct Node {
    Op op;
    std::vector<Expr> args;
    double value = 0;
    const ScalarField* field = nullptr;
    int x = 0, y = 0; // derivative multi-index; Jet(-1,0) is Laplacian
    std::string name;
};
Expr make(Op op, std::vector<Expr> args = {}, double value = 0,
          const ScalarField* field = nullptr, int x = 0, int y = 0,
          std::string name = "");
std::string key(const Node& n);
std::string section(const Node& n);
std::vector<Expr> nodes(const Expr& expression);
bool composite(const Node& n);
Expr canonicalize(Expr expression);
Expr lowerComposites(Expr expression, const Schemes& schemes,
                     const ScalarField& layout, OperationInfo& info);
struct Weight { int x, y; double value; };
struct Primitive {
    const Node* node;
    const ScalarField* field;
    Schemes::Selection selection;
    std::vector<Weight> weights;
    int halo = 0;
    bool corners = false;
};
struct GpuProgram;
struct Plan {
    Expr expression;
    std::string name;
    const ScalarField* layout;
    ScalarField* state = nullptr;
    std::vector<const ScalarField*> fields;
    std::vector<Primitive> primitives;
    OperationInfo info;
    int nx, ny, sx, sy, ghost;
    std::size_t storedSize;
    std::vector<std::function<void()>> layoutChecks;
    std::mutex mutex;
    std::map<std::pair<int,int>, std::shared_ptr<GpuProgram>> gpuPrograms;
    ~Plan();
};
std::shared_ptr<Plan> bind(Expr expression, const std::string& name,
                         const Schemes& schemes, const ScalarField& layout,
                         ScalarField* state, bool checkHalos = true);
Term lower(const std::shared_ptr<Plan>& plan, Fusion mode);
std::string source(const Plan& plan, Fusion mode);
} // namespace PhiX::numerics::symbolic::detail
