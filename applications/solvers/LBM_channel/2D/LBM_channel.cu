/***********************************************************************\
 *
 *  LBM Channel Flow Solver (2D, D2Q9)  --  PFHub Benchmark 5 geometry
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  Description
 *  -----------
 *  Steady incompressible channel flow (Stokes regime) via the LBM module
 *  (lbm/LBM.h, v2.32.0): BGK + Guo body force, halfway bounce-back
 *  walls, Zou-He parabolic velocity inlet (x-min), Zou-He pressure
 *  outlet (x-max), optional elliptical obstacle mask.
 *
 *  All physical <-> lattice unit conversions are derived from the
 *  config (dx = lattice spacing, dt = physical time per step):
 *      u_lat  = u_phys * dt/dx
 *      nu_lat = nu_phys * dt/dx^2        -> tau = 3 nu_lat + 1/2
 *      a_lat  = g_phys * dt^2/dx         (body acceleration)
 *      p_phys = (rho_lat - rho_ref) c_s^2 * rho_phys * (dx/dt)^2
 *  The pressure reference is the config-given point (PFHub: p(30,6)=0).
 *
 *  Outputs (PFHub BM5): two cross-section CSVs
 *      x-cut:  y, velocity_x, velocity_y, pressure   (at x = x_cut)
 *      y-cut:  x, velocity_x, velocity_y, pressure   (at y = y_cut)
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "core/RunGuard.h"
#include "field/ScalarField.h"
#include "lbm/LBM.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/PFHubWriter.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

static int phixMain(int argc, char* argv[])
{
    using namespace PhiX;

    IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);

    // === 1. Mesh (dx = dy enforced: LBM lattice) =============================
    const int    nx = cfg["mesh"]["nx"];
    const int    ny = cfg["mesh"]["ny"];
    const double dx = cfg["mesh"]["dx"];
    const double x0 = cfg["mesh"]["x0"];
    const double y0 = cfg["mesh"]["y0"];

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    nx, dx, x0, ny, dx, y0);
    mesh.print();

    // === 2. Physical parameters & unit conversion ============================
    const double dt       = cfg["lbm"]["dt"];        // phys time per step
    const int    nSteps   = cfg["lbm"]["nSteps"];
    const double rho_phys = cfg["lbm"]["rho"];       // kg/m^3
    const double mu_phys  = cfg["lbm"]["mu"];        // Pa s
    const double gx       = cfg["lbm"]["gx"];        // m/s^2
    const double gy       = cfg["lbm"]["gy"];

    const double nu_lat = (mu_phys / rho_phys) * dt / (dx * dx);
    LBMParams p;
    p.tau = 3.0 * nu_lat + 0.5;
    p.fx  = gx * dt * dt / dx;
    p.fy  = gy * dt * dt / dx;

    std::cout << "  lattice: tau = " << p.tau
              << "  (nu_lat = " << nu_lat << ")\n";

    LBM2D lbm(mesh, p);

    // === 3. Boundaries =======================================================
    lbm.setWall(Axis::Y, Side::LOW);
    lbm.setWall(Axis::Y, Side::HIGH);

    // Parabolic inlet u_x(y): zero at both walls, u_max at mid-height
    const double uMax = cfg["inlet"]["umax"];        // m/s
    const double H    = ny * dx;
    std::vector<double> uN(ny);
    for (int j = 0; j < ny; ++j) {
        const double y = mesh.coord(1, j) - y0;
        const double uPhys = uMax * (1.0 - std::pow((y - 0.5 * H)
                                                    / (0.5 * H), 2));
        uN[j] = uPhys * dt / dx;                     // lattice units
    }
    lbm.setVelocityInlet(Axis::X, Side::LOW, uN);
    lbm.setOutflow(Axis::X, Side::HIGH, 1.0);

    // Optional elliptical obstacle
    ScalarField maskF(mesh, "mask", 1);
    maskF.fill(0.0);
    if (cfg.has("obstacle")) {
        const double cx = cfg["obstacle"]["cx"];
        const double cy = cfg["obstacle"]["cy"];
        const double rx = cfg["obstacle"]["rx"];
        const double ry = cfg["obstacle"]["ry"];
        long long nSolid = 0;
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double ex = (mesh.coord(0, i) - cx) / rx;
            const double ey = (mesh.coord(1, j) - cy) / ry;
            if (ex * ex + ey * ey <= 1.0) {
                maskF.curr[static_cast<std::size_t>(maskF.index(i, j))] = 1.0;
                ++nSolid;
            }
        }
        maskF.allocDevice();
        maskF.uploadAllToDevice();
        lbm.setObstacleMask(maskF);
        std::cout << "  obstacle: ellipse (" << cx << "," << cy
                  << ") rx=" << rx << " ry=" << ry
                  << "  -> " << nSolid << " solid cells\n";
    }

    // === 4. Run to steady state ==============================================
    lbm.initialize(1.0, 0.0, 0.0);

    ScalarField rho(mesh, "rho", 1), ux(mesh, "ux", 1), uy(mesh, "uy", 1);
    for (ScalarField* f : {&rho, &ux, &uy}) { f->fill(0); f->allocDevice(); }

    const int checkEvery = cfg["lbm"]["check_interval"];
    std::vector<double> prevU;
    for (int s = 0; s < nSteps; s += checkEvery) {
        lbm.run(checkEvery);
        lbm.macroscopics(&rho, &ux, &uy);
        ux.downloadCurrFromDevice();

        // steady-state residual: max |du| between samples (physical units)
        double res = 0.0, umax = 0.0;
        std::vector<double> cur(static_cast<std::size_t>(nx) * ny);
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double v = ux.curr[static_cast<std::size_t>(
                ux.index(i, j))] * dx / dt;
            cur[static_cast<std::size_t>(j) * nx + i] = v;
            umax = std::max(umax, std::fabs(v));
            if (!prevU.empty())
                res = std::max(res, std::fabs(
                    v - prevU[static_cast<std::size_t>(j) * nx + i]));
        }
        std::cout << "  step " << lbm.stepCount()
                  << "  max|u_x| = " << umax << " m/s"
                  << "  residual = " << (prevU.empty() ? 1.0 : res)
                  << "\n" << std::flush;
        prevU.swap(cur);
        if (!prevU.empty() && res < umax * 1e-8 && s > 0) {
            std::cout << "  converged (residual < 1e-8 * umax)\n";
            break;
        }
    }

    // === 5. Cross-section CSVs ===============================================
    lbm.macroscopics(&rho, &ux, &uy);
    for (ScalarField* f : {&rho, &ux, &uy}) f->downloadCurrFromDevice();

    const double xCut = cfg["cuts"]["x_cut"];
    const double yCut = cfg["cuts"]["y_cut"];
    const double pRefX = cfg["cuts"]["p_ref_x"];
    const double pRefY = cfg["cuts"]["p_ref_y"];
    const std::string xCsv = cfg["cuts"]["x_csv"];
    const std::string yCsv = cfg["cuts"]["y_csv"];

    auto cellAt = [&](double xq, int axis) {
        const double o = (axis == 0) ? x0 : y0;
        int i = static_cast<int>(std::floor((xq - o) / dx));
        const int n = (axis == 0) ? nx : ny;
        return std::min(std::max(i, 0), n - 1);
    };
    auto V = [&](ScalarField& f, int i, int j) {
        return static_cast<double>(f.curr[static_cast<std::size_t>(
            f.index(i, j))]);
    };
    const double uConv = dx / dt;
    const double pConv = rho_phys * uConv * uConv / 3.0;   // c_s^2 = 1/3
    const int iRef = cellAt(pRefX, 0), jRef = cellAt(pRefY, 1);
    const double rhoRef = V(rho, iRef, jRef);

    // Optional analytic hydrostatic superposition.  A constant body force
    // in incompressible flow is EXACTLY balanced by p_h = rho*g.(r - r_ref)
    // and leaves the velocity field unchanged — so channel cases run the
    // LBM without gravity (a uniform-density Zou-He outlet is inconsistent
    // with a hydrostatic gradient) and add p_h at output time.
    double hgx = 0.0, hgy = 0.0;
    if (cfg.has("hydrostatic")) {
        hgx = cfg["hydrostatic"]["gx"];
        hgy = cfg["hydrostatic"]["gy"];
    }
    auto pHydro = [&](double x, double y) {
        return rho_phys * (hgx * (x - pRefX) + hgy * (y - pRefY));
    };

    {
        IO::PFHubWriter wx(xCsv, {"y", "velocity_x", "velocity_y", "pressure"});
        const int i = cellAt(xCut, 0);
        for (int j = 0; j < ny; ++j)
            wx.addRow({mesh.coord(1, j),
                       V(ux, i, j) * uConv, V(uy, i, j) * uConv,
                       (V(rho, i, j) - rhoRef) * pConv
                           + pHydro(xCut, mesh.coord(1, j))});
    }
    {
        IO::PFHubWriter wy(yCsv, {"x", "velocity_x", "velocity_y", "pressure"});
        const int j = cellAt(yCut, 1);
        for (int i = 0; i < nx; ++i)
            wy.addRow({mesh.coord(0, i),
                       V(ux, i, j) * uConv, V(uy, i, j) * uConv,
                       (V(rho, i, j) - rhoRef) * pConv
                           + pHydro(mesh.coord(0, i), yCut)});
    }
    std::cout << "  wrote " << xCsv << ", " << yCsv << "\n";

    // === 6. Steady-state field snapshots (physical units, for ParaView) =====
    std::filesystem::create_directories("output");
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t id = static_cast<std::size_t>(ux.index(i, j));
        const double pv = (static_cast<double>(rho.curr[id]) - rhoRef) * pConv
                        + pHydro(mesh.coord(0, i), mesh.coord(1, j));
        ux.curr[id]  = static_cast<Real>(static_cast<double>(ux.curr[id])
                                         * uConv);
        uy.curr[id]  = static_cast<Real>(static_cast<double>(uy.curr[id])
                                         * uConv);
        rho.curr[id] = static_cast<Real>(pv);   // reuse rho storage for p
    }
    rho.name = "p";
    const std::string tag = std::to_string(lbm.stepCount());
    ux.write ("output/ux_"  + tag + ".vts", FieldFormat::VTS);
    uy.write ("output/uy_"  + tag + ".vts", FieldFormat::VTS);
    rho.write("output/p_"   + tag + ".vts", FieldFormat::VTS);
    std::cout << "  wrote output/{ux,uy,p}_" << tag << ".vts\nDone.\n";
    return 0;
}

int main(int argc, char* argv[]) {
    return PhiX::runGuarded(phixMain, argc, argv);
}
