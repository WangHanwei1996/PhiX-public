// ---------------------------------------------------------------------------
// module_schemes_ext — CD4 stencils + first-order upwind advection
//
// 1. CD4 accuracy: on sin(kx)·cos(ky) with analytically filled ghost cells
//    (incl. corners), the CD4 Laplacian/gradient error must be far below the
//    CD2 error on the same grid (the strict order measurement lives in the
//    convergence suite).
// 2. Ghost validation: a CD4 term on a ghost-1 field must throw at setRHS.
// 3. Upwind advection:
//    a) direction check against a hand-computed reference on a 2D field with
//       mixed-sign velocity;
//    b) 1D periodic transport of a step profile (Euler, CFL 0.5, 100 steps):
//       GPU result must match an exact host replication of the same upwind
//       update to 1e-12, stay monotone (no over/undershoot), and conserve
//       the discrete sum exactly (telescoping upwind flux, constant u).
// ---------------------------------------------------------------------------

#include "equation/Equation.h"
#include "solver/Solver.h"
#include "boundary/PeriodicBC.h"
#include "operators/Laplacian.h"
#include "operators/Gradient.h"
#include "operators/Advection.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// Fill curr at ALL stored cells (physical + ghost) from fn(x, y).
template<typename Fn>
static void fillWithGhost2D(ScalarField& f, Fn fn) {
    const int g = f.ghost;
    for (int j = -g; j < f.mesh.n[1] + g; ++j)
    for (int i = -g; i < f.mesh.n[0] + g; ++i)
        f.curr[static_cast<std::size_t>(f.index(i, j))] =
            fn(f.mesh.coord(0, i), f.mesh.coord(1, j));
}

// Max |rhs - ref| over physical cells after computing `term` into rhs.
static double stencilMaxErr(ScalarField& src, const Term& term,
                            const std::vector<double>& ref) {
    Equation eq(src, "op");
    eq.setRHS(term);
    ScalarField rhs(src.mesh, "rhs", src.ghost);
    rhs.allocDevice();
    eq.computeRHS(rhs);
    rhs.downloadCurrFromDevice();
    double err = 0.0;
    std::size_t m = 0;
    for (int j = 0; j < src.mesh.n[1]; ++j)
    for (int i = 0; i < src.mesh.n[0]; ++i, ++m)
        err = std::max(err, std::fabs(
            rhs.curr[static_cast<std::size_t>(rhs.index(i, j))] - ref[m]));
    return err;
}

int main() {
    // =======================================================================
    // 1. CD4 vs CD2 accuracy on sin(kx)·cos(ky)
    // =======================================================================
    {
        const int    N  = 48;
        const double L  = 2.0 * M_PI;
        const double dx = L / N;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        N, dx, 0.0, N, dx, 0.0);
        ScalarField f(mesh, "f", 2);
        const double kx = 1.0, ky = 2.0;
        fillWithGhost2D(f, [=](double x, double y) {
            return std::sin(kx * x) * std::cos(ky * y);
        });
        f.allocDevice();
        f.uploadAllToDevice();

        // Analytic references at physical cells
        std::vector<double> refLap, refGx;
        refLap.reserve(N * N);
        refGx.reserve(N * N);
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double x = mesh.coord(0, i), y = mesh.coord(1, j);
            refLap.push_back(-(kx*kx + ky*ky) * std::sin(kx*x) * std::cos(ky*y));
            refGx.push_back(kx * std::cos(kx*x) * std::cos(ky*y));
        }

        const double errLap2 = stencilMaxErr(f, lap(f, "CD2", 1.0), refLap);
        const double errLap4 = stencilMaxErr(f, lap(f, "CD4", 1.0), refLap);
        const double errGx2  = stencilMaxErr(f, grad(f, 0, "CD2", 1.0), refGx);
        const double errGx4  = stencilMaxErr(f, grad(f, 0, "CD4", 1.0), refGx);

        // At N=48: CD2 error ~ (k*dx)^2/6, CD4 ~ (k*dx)^4/30 → ratio ~ 1e-2
        require(errLap4 < 0.05 * errLap2,
                "CD4 laplacian not clearly more accurate than CD2 ("
                + std::to_string(errLap4) + " vs " + std::to_string(errLap2) + ")");
        require(errGx4 < 0.05 * errGx2,
                "CD4 gradient not clearly more accurate than CD2");
        require(errLap4 < 5e-3 && errGx4 < 5e-4,
                "CD4 absolute error unexpectedly large");
    }

    // =======================================================================
    // 2. CD4 on a ghost-1 field must throw at setRHS
    // =======================================================================
    {
        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, 16, 0.1);
        ScalarField f(mesh, "g1", 1);
        f.allocDevice();
        Equation eq(f, "bad");
        bool threw = false;
        try { eq.setRHS(lap(f, "CD4", 1.0)); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "CD4 term on ghost-1 field did not throw");
    }

    // =======================================================================
    // 3a. Upwind direction check against a hand-built reference
    // =======================================================================
    {
        const int N = 12;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        N, 0.5, 0.0, N, 0.25, 0.0);
        ScalarField f(mesh, "f", 1);
        fillWithGhost2D(f, [](double x, double y) {
            return std::sin(1.3 * x) + 0.7 * std::cos(2.1 * y) + 0.1 * x * y;
        });
        VectorField u(mesh, "u", 2, 1);   // 2 components, ghost 1
        // Mixed-sign velocity: ux flips sign mid-domain, uy < 0 everywhere
        for (int j = -1; j < N + 1; ++j)
        for (int i = -1; i < N + 1; ++i) {
            const double x = mesh.coord(0, i);
            u[0].curr[static_cast<std::size_t>(u[0].index(i, j))] = (x < 3.0) ? 1.5 : -2.0;
            u[1].curr[static_cast<std::size_t>(u[1].index(i, j))] = -0.8;
        }
        f.allocDevice();    f.uploadAllToDevice();
        u[0].allocDevice(); u[0].uploadAllToDevice();
        u[1].allocDevice(); u[1].uploadAllToDevice();

        // Hand-computed upwind reference
        std::vector<double> ref;
        ref.reserve(N * N);
        const double idx = 1.0 / mesh.d[0], idy = 1.0 / mesh.d[1];
        auto F = [&](int i, int j) {
            return f.curr[static_cast<std::size_t>(f.index(i, j))];
        };
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double vx = (mesh.coord(0, i) < 3.0) ? 1.5 : -2.0;
            const double vy = -0.8;
            const double dfdx = (vx > 0.0) ? (F(i, j) - F(i-1, j)) * idx
                                           : (F(i+1, j) - F(i, j)) * idx;
            const double dfdy = (vy > 0.0) ? (F(i, j) - F(i, j-1)) * idy
                                           : (F(i, j+1) - F(i, j)) * idy;
            ref.push_back(vx * dfdx + vy * dfdy);
        }
        const double err = stencilMaxErr(f, adv(u, f, 1.0), ref);
        require(err < 1e-13, "upwind advection mismatch vs hand reference ("
                             + std::to_string(err) + ")");
    }

    // =======================================================================
    // 3b. 1D periodic step transport: exact replication + monotone + conservative
    // =======================================================================
    {
        const int    N  = 64;
        const double dx = 1.0 / N;
        const double u0 = 1.0;
        const double dt = 0.5 * dx / u0;   // CFL 0.5
        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, dx);

        ScalarField phi(mesh, "phi", 1);
        for (int i = -1; i < N + 1; ++i)
            phi.curr[static_cast<std::size_t>(phi.index(i))] = 0.0;
        for (int i = N / 4; i < N / 2; ++i)
            phi.curr[static_cast<std::size_t>(phi.index(i))] = 1.0;
        phi.allocDevice();
        phi.uploadAllToDevice();

        VectorField u(mesh, "u", 1, 1);
        for (int i = -1; i < N + 1; ++i)
            u[0].curr[static_cast<std::size_t>(u[0].index(i))] = u0;
        u[0].allocDevice();
        u[0].uploadAllToDevice();

        Equation eq(phi, "transport");
        eq.setRHS(adv(u, phi, -1.0));      // dphi/dt = -u·∇phi

        PeriodicBC bc(mesh.facePatch(Axis::X, Side::LOW));
        Solver solver(eq, {&bc}, dt, TimeScheme::EULER);

        // Host replication of the identical scheme
        std::vector<double> h(N);
        for (int i = 0; i < N; ++i)
            h[static_cast<std::size_t>(i)] = (i >= N/4 && i < N/2) ? 1.0 : 0.0;

        const int nSteps = 100;
        double sum0 = 0.0;
        for (double v : h) sum0 += v;

        solver.run(nSteps);

        std::vector<double> hn(N);
        for (int s = 0; s < nSteps; ++s) {
            for (int i = 0; i < N; ++i) {
                const int im = (i + N - 1) % N;
                hn[static_cast<std::size_t>(i)] =
                    h[static_cast<std::size_t>(i)]
                    - dt * u0 * (h[static_cast<std::size_t>(i)]
                                 - h[static_cast<std::size_t>(im)]) / dx;
            }
            h.swap(hn);
        }

        phi.downloadCurrFromDevice();
        double sumG = 0.0, minG = 1e300, maxG = -1e300;
        for (int i = 0; i < N; ++i) {
            const double v = phi.curr[static_cast<std::size_t>(phi.index(i))];
            require(std::fabs(v - h[static_cast<std::size_t>(i)]) < 1e-12,
                    "transport: GPU result differs from exact host replication");
            sumG += v;
            minG = std::min(minG, v);
            maxG = std::max(maxG, v);
        }
        require(minG >= -1e-12 && maxG <= 1.0 + 1e-12,
                "transport: upwind produced over/undershoot");
        require(std::fabs(sumG - sum0) < 1e-11,
                "transport: discrete sum not conserved");
    }

    // =======================================================================
    // 4. Iso27: 3D isotropic Laplacian is EXACT on quadratics
    // =======================================================================
    {
        const int N = 12;
        Mesh mesh = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                        N, 0.1, 0.0, N, 0.1, 0.0, N, 0.1, 0.0);
        ScalarField f(mesh, "f", 1);
        for (int k = -1; k <= N; ++k)
        for (int j = -1; j <= N; ++j)
        for (int i = -1; i <= N; ++i) {
            const double x = mesh.coord(0, i), y = mesh.coord(1, j),
                         z = mesh.coord(2, k);
            f.curr[static_cast<std::size_t>(f.index(i, j, k))] =
                x * x + 2.0 * y * y + 3.0 * z * z + 0.5 * x - y + 0.1;
        }
        f.allocDevice();
        f.uploadAllToDevice();

        Equation eq(f, "iso27");
        eq.setRHS(lap(f, "Iso27", 1.0));
        ScalarField rhs(mesh, "rhs", 1);
        rhs.allocDevice();
        eq.computeRHS(rhs);
        rhs.downloadCurrFromDevice();

        for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            require(std::fabs(rhs.curr[static_cast<std::size_t>(
                        rhs.index(i, j, k))] - 12.0) < 1e-9,
                    "Iso27 not exact on quadratic (lap should be 12)");
    }

    // =======================================================================
    // 5. WENO5 step transport: essentially non-oscillatory
    // =======================================================================
    {
        const int    N  = 64;
        const double dx = 1.0 / N;
        const double u0 = 1.0, dt = 0.4 * dx / u0;
        Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, N, dx);

        ScalarField phi(mesh, "phi", 3);
        for (int i = -3; i < N + 3; ++i)
            phi.curr[static_cast<std::size_t>(phi.index(i))] =
                (i >= N / 4 && i < N / 2) ? 1.0 : 0.0;
        phi.allocDevice();
        phi.uploadAllToDevice();

        VectorField u(mesh, "u", 1, 3);
        for (int i = -3; i < N + 3; ++i)
            u[0].curr[static_cast<std::size_t>(u[0].index(i))] = u0;
        u[0].allocDevice();
        u[0].uploadAllToDevice();

        Equation eq(phi, "weno");
        eq.setRHS(adv(u, phi, "WENO5", -1.0));
        PeriodicBC bcW(mesh.facePatch(Axis::X, Side::LOW));
        Solver solver(eq, {&bcW}, dt, TimeScheme::EULER);
        solver.run(80);

        phi.downloadCurrFromDevice();
        double mn = 1e300, mx = -1e300, sum = 0.0;
        for (int i = 0; i < N; ++i) {
            const double v = phi.curr[static_cast<std::size_t>(phi.index(i))];
            mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
        }
        require(mn > -5e-3 && mx < 1.0 + 5e-3,
                "WENO5 produced significant over/undershoot");
        // The HJ-WENO derivative form is NOT a conservative flux difference
        // (nonlinear weights break telescoping) — mass is preserved only
        // approximately, at the scheme's truncation level.
        require(std::fabs(sum - N / 4.0) < 5e-3 * (N / 4.0),
                "WENO5 mass drift beyond truncation level: "
                + std::to_string(sum - N / 4.0));
    }

    return 0;
}
