// ---------------------------------------------------------------------------
// rd_mms — method of manufactured solutions for a COUPLED two-field system.
//
// The convergence suite in test/convergence verifies operators and time
// integrators in isolation.  This case closes the remaining gap: it verifies
// the full pipeline -- boundary conditions, two simultaneously advanced
// equations, a nonlinear cross term between them, and the time integrator --
// against an exact solution.
//
// System (Gray-Scott operator structure, with manufactured sources):
//
//   du/dt = Du lap(u) - u v^2 + F (1 - u)     + S_u(x,y,t)
//   dv/dt = Dv lap(v) + u v^2 - (F + k) v     + S_v(x,y,t)
//
// Manufactured solution on the unit square with periodic boundaries:
//
//   u_ex = 1 + a U(x,y) E(t)      U = sin(2 pi x) cos(2 pi y)
//   v_ex = b + c V(x,y) E(t)      V = cos(2 pi x) sin(2 pi y)
//   E(t) = exp(-lambda t)
//
// Using  lap U = -8 pi^2 U  and  lap V = -8 pi^2 V,
//
//   S_u = (-lambda + 8 pi^2 Du) a U E + u_ex v_ex^2 - F (1 - u_ex)
//   S_v = (-lambda + 8 pi^2 Dv) c V E - u_ex v_ex^2 + (F + k) v_ex
//
// The sources are evaluated analytically per cell on the host at the current
// time level, uploaded, and injected as an identity pointwise term.  Forward
// Euler evaluates every right-hand side at t^n, so this is consistent.
//
// dt is refined as dx^3 so the O(dt) temporal error stays an order below the
// O(dx^2) spatial error and the measured order is that of the spatial
// discretisation and the coupling, uncontaminated.
//
// Usage:  ./rd_mms [Nbase] [T_end]      (default 32, 0.02)
//         refines Nbase, 2*Nbase, 4*Nbase and prints observed orders.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "equation/Equation.h"
#include "equation/EquationSystem.h"
#include "boundary/PeriodicBC.h"
#include "operators/Laplacian.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace PhiX;

namespace {
constexpr double Du = 1.0, Dv = 0.5;
constexpr double Fc = 0.05, kc = 0.06;
constexpr double aA = 0.30, bB = 0.50, cC = 0.20;
constexpr double lam = 1.0;
const     double PI2 = 2.0 * M_PI;
const     double L8  = 8.0 * M_PI * M_PI;

inline double Ufun(double x, double y) { return std::sin(PI2 * x) * std::cos(PI2 * y); }
inline double Vfun(double x, double y) { return std::cos(PI2 * x) * std::sin(PI2 * y); }
inline double uEx(double x, double y, double t) { return 1.0 + aA * Ufun(x, y) * std::exp(-lam * t); }
inline double vEx(double x, double y, double t) { return bB + cC * Vfun(x, y) * std::exp(-lam * t); }

inline double sU(double x, double y, double t) {
    const double E = std::exp(-lam * t), ue = uEx(x, y, t), ve = vEx(x, y, t);
    return (-lam + L8 * Du) * aA * Ufun(x, y) * E + ue * ve * ve - Fc * (1.0 - ue);
}
inline double sV(double x, double y, double t) {
    const double E = std::exp(-lam * t), ue = uEx(x, y, t), ve = vEx(x, y, t);
    return (-lam + L8 * Dv) * cC * Vfun(x, y) * E - ue * ve * ve + (Fc + kc) * ve;
}

struct Err { double u, v; };

Err runOnce(int N, double tEnd, double dt) {
    const double dx = 1.0 / N;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);

    ScalarField u(mesh, "u", 1), v(mesh, "v", 1);
    ScalarField Su(mesh, "Su", 1), Sv(mesh, "Sv", 1);
    u.initialize([](double x, double y, double) { return uEx(x, y, 0.0); });
    v.initialize([](double x, double y, double) { return vEx(x, y, 0.0); });
    Su.fill(0.0); Sv.fill(0.0);
    for (ScalarField* f : {&u, &v, &Su, &Sv}) { f->allocDevice(); f->uploadAllToDevice(); }

    PeriodicBC bx(mesh.facePatch(Axis::X, Side::LOW));
    PeriodicBC by(mesh.facePatch(Axis::Y, Side::LOW));
    std::vector<BoundaryCondition*> bcs = {&bx, &by};

    Equation eqU(u, "mms_u");
    eqU.setRHS(lap(u, Du)
             + pw(u, v, PHIX_FN (Real uu, Real vv) {
                   return -uu * vv * vv + Real(Fc) * (Real(1) - uu); })
             + pw(Su, PHIX_FN (Real s) { return s; }));

    Equation eqV(v, "mms_v");
    eqV.setRHS(lap(v, Dv)
             + pw(u, v, PHIX_FN (Real uu, Real vv) {
                   return uu * vv * vv - Real(Fc + kc) * vv; })
             + pw(Sv, PHIX_FN (Real s) { return s; }));

    EquationSystem sys(dt, TimeScheme::EULER);
    sys.add(eqU, bcs);
    sys.add(eqV, bcs);

    const int nSteps = static_cast<int>(tEnd / dt + 0.5);
    for (int step = 0; step < nSteps; ++step) {
        const double t = step * dt;                 // forward Euler: RHS at t^n
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const double x = mesh.coord(0, i), y = mesh.coord(1, j);
                Su.curr[Su.index(i, j, 0)] = static_cast<Real>(sU(x, y, t));
                Sv.curr[Sv.index(i, j, 0)] = static_cast<Real>(sV(x, y, t));
            }
        Su.uploadAllToDevice();
        Sv.uploadAllToDevice();
        sys.advance();
    }

    const double tFin = nSteps * dt;
    u.downloadCurrFromDevice();
    v.downloadCurrFromDevice();
    double e2u = 0.0, e2v = 0.0;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double x = mesh.coord(0, i), y = mesh.coord(1, j);
            const double du = static_cast<double>(u.curr[u.index(i, j, 0)]) - uEx(x, y, tFin);
            const double dv = static_cast<double>(v.curr[v.index(i, j, 0)]) - vEx(x, y, tFin);
            e2u += du * du; e2v += dv * dv;
        }
    const double nc = static_cast<double>(N) * N;
    return { std::sqrt(e2u / nc), std::sqrt(e2v / nc) };
}
}  // namespace

int main(int argc, char** argv) {
try {
    const int    Nb   = (argc > 1) ? std::atoi(argv[1]) : 32;
    const double tEnd = (argc > 2) ? std::atof(argv[2]) : 0.02;

    // Base step: a fifth of the explicit diffusion limit on the coarse mesh,
    // then refined as dx^3 so the temporal error stays subdominant.
    const double dxb = 1.0 / Nb;
    const double dt0 = 0.2 * dxb * dxb / (4.0 * Du);

    std::printf("rd_mms — manufactured solution for a coupled 2-field "
                "reaction-diffusion system\n");
    std::printf("  Du=%.3g Dv=%.3g F=%.3g k=%.3g lambda=%.3g  t_end=%.4g\n",
                Du, Dv, Fc, kc, lam, tEnd);
    std::printf("  dt refined as dx^3 (base dt=%.3e at N=%d)\n\n", dt0, Nb);
    std::printf("%6s %12s %14s %10s %14s %10s\n",
                "N", "dt", "L2 err u", "order u", "L2 err v", "order v");

    std::vector<int> Ns = {Nb, 2 * Nb, 4 * Nb};
    std::vector<Err> errs;
    for (std::size_t i = 0; i < Ns.size(); ++i) {
        const double r  = double(Ns[i]) / Nb;
        const double dt = dt0 / (r * r * r);
        Err e = runOnce(Ns[i], tEnd, dt);
        errs.push_back(e);
        if (i == 0) {
            std::printf("%6d %12.3e %14.6e %10s %14.6e %10s\n",
                        Ns[i], dt, e.u, "-", e.v, "-");
        } else {
            const double pu = std::log(errs[i - 1].u / e.u) / std::log(2.0);
            const double pv = std::log(errs[i - 1].v / e.v) / std::log(2.0);
            std::printf("%6d %12.3e %14.6e %10.3f %14.6e %10.3f\n",
                        Ns[i], dt, e.u, pu, e.v, pv);
        }
        std::fflush(stdout);
    }

    const double pu = std::log(errs[1].u / errs[2].u) / std::log(2.0);
    const double pv = std::log(errs[1].v / errs[2].v) / std::log(2.0);
    std::printf("\nfinest-pair observed order:  u = %.3f   v = %.3f"
                "   (formal 2)\n", pu, pv);
    return (pu > 1.8 && pv > 1.8) ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "rd_mms FAILED: %s\n", e.what());
    return 1;
}
}
