#pragma once
#include "equation/Term.h"
namespace PhiX {
struct TermCapture {
    Term term;
};
template <typename Fn> ExprTree expr_pw(const ScalarField &f, Fn fn, double coeff) {
    auto n = std::make_shared<ExprPointwise1>();
    n->child = ExprTree(f).node;
    n->termCapture = std::make_shared<TermCapture>(TermCapture{pw(f, fn, coeff)});
    return ExprTree(n);
}
template <typename Fn>
ExprTree expr_pw(const ScalarField &a, const ScalarField &b, Fn fn, double coeff) {
    auto n = std::make_shared<ExprPointwise1>();
    n->child = ExprTree(a).node;
    n->child2 = ExprTree(b).node;
    n->termCapture = std::make_shared<TermCapture>(TermCapture{pw(a, b, fn, coeff)});
    return ExprTree(n);
}
template <typename Fn>
ExprTree expr_pw(const ScalarField &a, const ScalarField &b, const ScalarField &c, Fn fn,
                 double coeff) {
    auto n = std::make_shared<ExprPointwise1>();
    n->child = ExprTree(a).node;
    n->child2 = ExprTree(b).node;
    n->child3 = ExprTree(c).node;
    n->termCapture = std::make_shared<TermCapture>(TermCapture{pw(a, b, c, fn, coeff)});
    return ExprTree(n);
}
} // namespace PhiX
