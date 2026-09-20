// ---------------------------------------------------------------------------
// module_lbm — D2Q9 BGK lattice-Boltzmann (lbm/LBM.h)
//
// 1. Poiseuille channel: body force fx between two no-slip walls (halfway
//    bounce-back), periodic in x.  Steady profile vs the analytic parabola
//      u(y) = fx/(2ν) · h·(H − h),  h = j + ½,  H = ny
//    (halfway BB places the walls half a cell outside the outer rows).
//    BGK+BB carries a small τ-dependent slip → 2% tolerance.
// 2. Viscosity dynamics: the Poiseuille start-up centreline deficit decays
//    at the slowest mode rate ν·(π/H)² — measured vs analytic within 5%.
// 3. Mass conservation: Σρ constant to machine precision through collide,
//    stream, and bounce-back.
// ---------------------------------------------------------------------------

#include "lbm/LBM.h"
#include "field/ScalarField.h"
#include "field/Reduce.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    // =======================================================================
    // 1. Poiseuille flow
    // =======================================================================
    {
        const int    nx = 16, ny = 32;
        const double tau = 0.9, fx = 1e-6;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        nx, 1.0, 0.0, ny, 1.0, 0.0);
        LBMParams p;
        p.tau = tau;
        p.fx  = fx;
        LBM2D lbm(mesh, p);
        lbm.setWall(Axis::Y, Side::LOW);
        lbm.setWall(Axis::Y, Side::HIGH);
        lbm.initialize(1.0);
        lbm.run(20000);                       // steady state

        ScalarField ux(mesh, "ux", 1), rho(mesh, "rho", 1);
        ux.allocDevice();
        rho.allocDevice();
        lbm.macroscopics(&rho, &ux, nullptr);
        ux.downloadCurrFromDevice();

        const double nu = lbm.latticeViscosity();
        double maxRel = 0.0, uMax = 0.0;
        for (int j = 0; j < ny; ++j) {
            const double h  = j + 0.5;
            const double ua = fx / (2.0 * nu) * h * (ny - h);
            const double um = ux.curr[static_cast<std::size_t>(
                ux.index(nx / 2, j))];
            maxRel = std::max(maxRel, std::fabs(um - ua)
                                      / (fx / (8.0 * nu) * ny * ny));
            uMax = std::max(uMax, um);
        }
        std::printf("  poiseuille: u_max = %.4e (analytic %.4e), "
                    "max rel dev %.2e\n",
                    uMax, fx / (8.0 * nu) * ny * ny, maxRel);
        require(maxRel < 0.02,
                "Poiseuille profile deviates from the parabola: "
                + std::to_string(maxRel));

        // x-invariance (periodic direction)
        double xVar = 0.0;
        for (int i = 0; i < nx; ++i)
            xVar = std::max(xVar, std::fabs(
                ux.curr[static_cast<std::size_t>(ux.index(i, ny / 2))]
                - ux.curr[static_cast<std::size_t>(ux.index(0, ny / 2))]));
        require(xVar < 1e-12 * uMax + 1e-15,
                "Poiseuille profile not x-invariant");
    }

    // =======================================================================
    // 2. Viscosity via the Poiseuille START-UP transient
    //    The centreline deficit u_steady − u(t) decays with the slowest
    //    diffusive mode rate ν·(π/H)²; measuring it between two late times
    //    (higher modes long gone) validates ν = (τ−½)/3 dynamically.
    // =======================================================================
    {
        const double tau = 0.8;
        const int nx = 8, ny = 32;
        Mesh meshC = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                         nx, 1.0, 0.0, ny, 1.0, 0.0);
        LBMParams pc;
        pc.tau = tau;
        pc.fx  = 1e-6;
        LBM2D chan(meshC, pc);
        chan.setWall(Axis::Y, Side::LOW);
        chan.setWall(Axis::Y, Side::HIGH);
        chan.initialize(1.0);

        ScalarField ux(meshC, "ux", 1);
        ux.allocDevice();

        // centreline deficit d(t) = u_st − u(t) ∝ e^{−ν k₁² t}, k₁ = π/H
        const double nu  = chan.latticeViscosity();
        const double ust = pc.fx / (8.0 * nu) * ny * ny;
        auto centre = [&]() {
            chan.macroscopics(nullptr, &ux, nullptr);
            ux.downloadCurrFromDevice();
            return static_cast<double>(ux.curr[static_cast<std::size_t>(
                ux.index(nx / 2, ny / 2))]);
        };
        const int t1 = 2000, dtScan = 2000;
        chan.run(t1);
        const double d1 = ust - centre();
        chan.run(dtScan);
        const double d2 = ust - centre();

        const double rateMeas = std::log(d1 / d2) / dtScan;
        const double rateAna  = nu * (M_PI / ny) * (M_PI / ny);
        std::printf("  startup decay: measured rate %.4e vs analytic %.4e "
                    "(dev %.2f%%)\n", rateMeas, rateAna,
                    100.0 * (rateMeas - rateAna) / rateAna);
        require(std::fabs(rateMeas - rateAna) < 0.05 * rateAna,
                "viscous decay rate off analytic: "
                + std::to_string(rateMeas) + " vs "
                + std::to_string(rateAna));
    }

    // =======================================================================
    // 3. Mass conservation through collide/stream/bounce-back
    // =======================================================================
    {
        const int nx = 24, ny = 24;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        nx, 1.0, 0.0, ny, 1.0, 0.0);
        LBMParams p;
        p.tau = 0.7;
        p.fx  = 2e-6;
        p.fy  = -1e-6;
        LBM2D lbm(mesh, p);
        lbm.setWall(Axis::X, Side::LOW);
        lbm.setWall(Axis::X, Side::HIGH);
        lbm.initialize(1.0);

        ScalarField rho(mesh, "rho", 1);
        rho.allocDevice();
        lbm.macroscopics(&rho, nullptr, nullptr);
        const double m0 = reduce::fieldSum(rho);

        lbm.run(5000);
        lbm.macroscopics(&rho, nullptr, nullptr);
        const double m1 = reduce::fieldSum(rho);

        std::printf("  mass drift after 5000 steps: %.2e (rel %.2e)\n",
                    m1 - m0, (m1 - m0) / m0);
        require(std::fabs(m1 - m0) < 1e-9 * m0,
                "LBM mass not conserved: " + std::to_string(m1 - m0));

        // parameter validation
        bool threw = false;
        LBMParams bad;
        bad.tau = 0.5;
        try { bad.validate(); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "tau <= 0.5 did not throw");
    }

    // =======================================================================
    // 4. Velocity inlet + outflow: Poiseuille without body force
    // =======================================================================
    {
        const int nx = 96, ny = 32;
        const double tau = 0.8, uMax = 0.02;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        nx, 1.0, 0.0, ny, 1.0, 0.0);
        LBMParams p;
        p.tau = tau;
        LBM2D lbm(mesh, p);
        lbm.setWall(Axis::Y, Side::LOW);
        lbm.setWall(Axis::Y, Side::HIGH);
        std::vector<double> prof(ny);
        for (int j = 0; j < ny; ++j) {
            const double h = j + 0.5;
            prof[static_cast<std::size_t>(j)] =
                4.0 * uMax * h * (ny - h) / (double(ny) * ny);
        }
        lbm.setVelocityInlet(Axis::X, Side::LOW, prof);
        lbm.setOutflow(Axis::X, Side::HIGH);
        lbm.initialize(1.0);
        lbm.run(20000);

        ScalarField ux(mesh, "ux", 1), rho(mesh, "rho", 1);
        ux.allocDevice(); rho.allocDevice();
        lbm.macroscopics(&rho, &ux, nullptr);
        ux.downloadCurrFromDevice();
        rho.downloadCurrFromDevice();

        // downstream (3/4 length) profile vs the inlet parabola; the
        // conserved quantity along the channel is the MASS flux Σρu
        // (Σu varies with the driving pressure gradient).
        double dev = 0.0, fluxIn = 0.0, fluxMid = 0.0;
        const int xm = 3 * nx / 4;
        auto at = [&](ScalarField& f, int i, int j) {
            return static_cast<double>(
                f.curr[static_cast<std::size_t>(f.index(i, j))]);
        };
        for (int j = 0; j < ny; ++j) {
            const double um = at(ux, xm, j);
            dev = std::max(dev, std::fabs(
                um - prof[static_cast<std::size_t>(j)]) / uMax);
            fluxIn  += at(rho, 1, j) * at(ux, 1, j);
            fluxMid += at(rho, xm, j) * um;
        }
        std::printf("  inlet/outflow poiseuille: max profile dev %.2e,"
                    " mass-flux drift %.2e\n", dev,
                    std::fabs(fluxMid - fluxIn) / fluxIn);
        require(dev < 0.02, "downstream profile deviates: "
                            + std::to_string(dev));
        require(std::fabs(fluxMid - fluxIn) / fluxIn < 1e-3,
                "mass flux not conserved along the channel");
    }

    // =======================================================================
    // 5. Interior obstacle: solid cells pinned, symmetric blockage flow
    // =======================================================================
    {
        const int nx = 96, ny = 48;
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        nx, 1.0, 0.0, ny, 1.0, 0.0);
        LBMParams p;
        p.tau = 0.8;
        LBM2D lbm(mesh, p);
        lbm.setWall(Axis::Y, Side::LOW);
        lbm.setWall(Axis::Y, Side::HIGH);
        std::vector<double> prof(ny);
        for (int j = 0; j < ny; ++j) {
            const double h = j + 0.5;
            prof[static_cast<std::size_t>(j)] =
                4.0 * 0.02 * h * (ny - h) / (double(ny) * ny);
        }
        lbm.setVelocityInlet(Axis::X, Side::LOW, prof);
        lbm.setOutflow(Axis::X, Side::HIGH);

        // centred square block 8×8
        ScalarField mask(mesh, "mask", 1);
        mask.initialize([&](double x, double y, double) {
            return (std::fabs(x - nx / 2.0) < 4.0
                    && std::fabs(y - ny / 2.0) < 4.0) ? 1.0 : 0.0;
        });
        lbm.setObstacleMask(mask);
        lbm.initialize(1.0);
        lbm.run(20000);

        ScalarField ux(mesh, "ux", 1), uy(mesh, "uy", 1);
        ux.allocDevice(); uy.allocDevice();
        lbm.macroscopics(nullptr, &ux, &uy);
        ux.downloadCurrFromDevice();
        uy.downloadCurrFromDevice();

        require(!reduce::fieldHasNonFinite(ux), "obstacle run produced NaN");

        // solid cells report zero velocity
        require(ux.curr[static_cast<std::size_t>(
                    ux.index(nx / 2, ny / 2))] == 0.0,
                "solid cell reports nonzero velocity");

        // blockage accelerates the bypass; symmetric geometry → mirror flow
        const double uSide = ux.curr[static_cast<std::size_t>(
            ux.index(nx / 2, ny / 4))];
        require(uSide > 0.02, "no bypass acceleration beside the block");
        double asym = 0.0;
        for (int i = 0; i < nx; ++i) {
            const double a = ux.curr[static_cast<std::size_t>(
                ux.index(i, ny / 4))];
            const double b = ux.curr[static_cast<std::size_t>(
                ux.index(i, ny - 1 - ny / 4))];
            asym = std::max(asym, std::fabs(a - b));
        }
        require(asym < 1e-10, "flow not symmetric about the channel axis: "
                              + std::to_string(asym));
    }

    return 0;
}
