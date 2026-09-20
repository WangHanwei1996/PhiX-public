// ---------------------------------------------------------------------------
// singlegrain3d — curvature-driven shrinkage of a spherical grain in 3D.
//
// PhiX counterpart of OpenPhase 1.0.1 `benchmarks/SingleGrain` (51^3, the only
// minimal genuinely-3D benchmark in that suite).  Both codes solve the same
// sharp-interface problem through different diffuse-interface regularisations
// -- OpenPhase uses a multi-phase-field double-obstacle potential, PhiX the
// classical Allen-Cahn double well -- so the common ground for a fair
// comparison is the analytical law both must reproduce.
//
//   PDE      dphi/dt = M [ kappa lap(phi) - W g'(phi) ],  g = phi^2 (1-phi)^2
//
//   Sharp-interface limit: an interface moves at  v = -(M kappa) K  with K the
//   sum of principal curvatures.  For a sphere K = 2/R, hence
//
//       dR/dt = -2 M kappa / R      =>      R(t)^2 = R0^2 - 4 M kappa t
//
//   which is exactly OpenPhase's  R^2 = R0^2 - 4 mu sigma t  once the physical
//   interface mobility and energy are identified as  mu sigma = M kappa.
//
// Calibration (double well, profile phi = 1/2 (1 - tanh((r-R)/delta))):
//       sigma = sqrt(2 kappa W) / 6 ,   delta = sqrt(2 kappa / W)
//   =>  kappa = 3 sigma delta ,         W = 6 sigma / delta
//   The interface then spans roughly 4*delta cells.
//
// Defaults reproduce the OpenPhase benchmark configuration:
//   N = 51, dx = 1e-6 m, dt = 1e-4 s, 1000 steps, R0 = 0.4*N*dx,
//   sigma = 1.0 J/m^2, mu = 1.0e-9 m^4/(J s)  =>  M kappa = 1e-9 m^2/s.
//
// The measured radius comes from a device-side reduction of the phase volume,
//       V = sum(phi) dx^3 ,   R = (3V / 4pi)^(1/3),
// so the check is global and needs no interface reconstruction.
//
// Usage:
//   ./singlegrain3d [N] [nSteps] [scheme] [--out]
//     N       cubic grid size            (default 51)
//     nSteps  number of Euler steps      (default 1000)
//     scheme  CD2 | Iso27                (default CD2)
//     --out   also write VTS fields (off by default so timing runs are clean)
//
// Writes radius_history.csv: step, time, R_num, R_analytic, rel_error.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "equation/Equation.h"
#include "solver/Solver.h"
#include "boundary/NoFluxBC.h"
#include "operators/Laplacian.h"
#include "perf/Perf.h"
#include "IO/FieldIO.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace PhiX;

int main(int argc, char** argv) {
try {
    // ---- configuration (OpenPhase SingleGrain defaults) -------------------
    const int   N       = (argc > 1) ? std::atoi(argv[1]) : 51;
    int         nSteps  = (argc > 2) ? std::atoi(argv[2]) : 1000;
    const std::string scheme = (argc > 3) ? argv[3] : "CD2";
    bool writeFields = false, fixDomain = false, fixDelta = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--out")       == 0) writeFields = true;
        if (std::strcmp(argv[i], "--fixdomain") == 0) fixDomain   = true;
        if (std::strcmp(argv[i], "--fixdelta")  == 0) { fixDomain = true;
                                                        fixDelta  = true; }
    }

    const double sigma = 1.0;                 // J/m^2   (OpenPhase $Sigma_0_1)
    const double mu    = 1.0e-9;              // m^4/(J s) (OpenPhase $Mu_0_1)
    const double Mkap  = mu * sigma;          // m^2/s — the only combination
                                              // the sharp-interface law sees

    // Two run modes.
    //
    //   default      dx fixed at the OpenPhase value; growing N grows the box.
    //                The sphere is always 0.4 N cells across, so the physics
    //                per cell is identical — this is the throughput sweep.
    //
    //   --fixdomain  the physical box and R0 are pinned to the OpenPhase 51^3
    //                configuration and dx = L/N.  The diffuse interface then
    //                shrinks with the mesh, so delta/R0 -> 0 and the result
    //                must converge to the sharp-interface law — this is the
    //                accuracy study.  dt follows the explicit diffusion limit
    //                and nSteps is set to reach the same physical end time.
    const double dxRef   = 1.0e-6;                    // OpenPhase $dx
    const double Lref    = 51 * dxRef;                // OpenPhase box edge
    const double dx      = fixDomain ? (Lref / N) : dxRef;
    const double Lbox    = N * dx;
    const double R0      = fixDomain ? (0.4 * Lref) : (0.4 * N * dx);
    const double cx = 0.5 * Lbox, cy = cx, cz = cx;

    // Interface thickness.  OpenPhase uses IWidth = 5 cells; the tanh profile
    // spans about 4*delta, so delta = 1.25 dx gives a comparable interface.
    //
    // --fixdelta pins delta to its physical value on the 51^3 reference mesh,
    // so refining the grid raises delta/dx and the discrete solution converges
    // to the continuum diffuse-interface answer.  Without it delta follows dx
    // and the interface stays 1.25 cells wide however fine the mesh, which
    // converges to a *fixed* discretisation bias instead.
    const double delta = fixDelta ? (1.25 * dxRef) : (1.25 * dx);
    const double kappa = 3.0 * sigma * delta;         // J/m
    const double W     = 6.0 * sigma / delta;         // J/m^3
    const double Mmob  = Mkap / kappa;                // m^3/(J s)

    // Time step: the OpenPhase value in default mode, the explicit diffusion
    // limit with a safety factor when the mesh is refined.
    const double tEnd = 0.08;                         // s — stop while the
                                                      // sphere is still large
    const double dt   = fixDomain ? (0.6 * dx * dx / (6.0 * Mkap)) : 1.0e-4;
    if (fixDomain) nSteps = static_cast<int>(tEnd / dt + 0.5);

    // Stability of the explicit step (3D: 6 neighbours) and of the reaction
    // term; report both so a changed N is self-diagnosing.
    const double dtDiff = dx * dx / (6.0 * Mkap);
    const double dtReact = 1.0 / (2.0 * Mmob * W);

    std::printf("singlegrain3d  N=%d^3  scheme=%s  steps=%d\n",
                N, scheme.c_str(), nSteps);
    std::printf("  dx=%.3e m  dt=%.3e s  (diff limit %.3e, react limit %.3e)\n",
                dx, dt, dtDiff, dtReact);
    std::printf("  sigma=%.3g J/m^2  mu=%.3g m^4/(J s)  M*kappa=%.3e m^2/s\n",
                sigma, mu, Mkap);
    std::printf("  delta=%.3e m (%.2f dx)  kappa=%.3e  W=%.3e  M=%.3e\n",
                delta, delta / dx, kappa, W, Mmob);
    std::printf("  R0=%.4e m (%.2f cells);  analytic vanish time %.4e s\n",
                R0, R0 / dx, R0 * R0 / (4.0 * Mkap));
    if (dt > dtDiff || dt > dtReact)
        std::printf("  [warning] dt exceeds a stability limit — expect blow-up\n");

    // ---- fields ------------------------------------------------------------
    const int ghost = (scheme == "CD6") ? 3 : ((scheme == "CD4") ? 2 : 1);
    Mesh mesh = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                    N, dx, 0.0, N, dx, 0.0, N, dx, 0.0);
    ScalarField phi(mesh, "phi", ghost);
    phi.initialize([&](double x, double y, double z) {
        const double r = std::sqrt((x - cx) * (x - cx)
                                 + (y - cy) * (y - cy)
                                 + (z - cz) * (z - cz));
        return 0.5 * (1.0 - std::tanh((r - R0) / delta));
    });
    phi.allocDevice();
    phi.uploadAllToDevice();

    std::vector<NoFluxBC> bcStore;
    bcStore.reserve(6);
    for (int ax = 0; ax < 3; ++ax)
        for (int sd = 0; sd < 2; ++sd)
            bcStore.emplace_back(mesh.facePatch(static_cast<Axis>(ax),
                                                sd == 0 ? Side::LOW : Side::HIGH));
    std::vector<BoundaryCondition*> bcs;
    for (auto& b : bcStore) bcs.push_back(&b);

    // ---- equation ----------------------------------------------------------
    //   RHS = M kappa lap(phi) - M W g'(phi),   g'(phi) = 2 phi (1-phi)(1-2phi)
    Equation eq(phi, "allen_cahn_3d");
    eq.setRHS(lap(phi, scheme, Mmob * kappa)
            + pw(phi, PHIX_FN (Real p) {
                  return Real(2) * p * (Real(1) - p) * (Real(1) - Real(2) * p);
              }, -Mmob * W));

    Solver solver(eq, bcs, dt, TimeScheme::EULER);

    // ---- radius measurement ------------------------------------------------
    const double cellVol = dx * dx * dx;
    auto measureR = [&]() {
        phi.downloadCurrFromDevice();
        const double V = reduce::fieldSum(phi) * cellVol;
        return (V > 0.0) ? std::cbrt(3.0 * V / (4.0 * M_PI)) : 0.0;
    };
    auto analyticR = [&](double t) {
        const double r2 = R0 * R0 - 4.0 * Mkap * t;
        return (r2 > 0.0) ? std::sqrt(r2) : 0.0;
    };

    std::filesystem::create_directories("output");
    std::FILE* csv = std::fopen("output/radius_history.csv", "w");
    std::fprintf(csv, "step,time_s,R_num_m,R_analytic_m,rel_error\n");

    const int sampleEvery = (nSteps >= 20) ? nSteps / 20 : 1;
    std::printf("\n%8s %12s %14s %14s %10s\n",
                "step", "time [s]", "R_num [m]", "R_ana [m]", "rel err");

    // The sharp-interface law only applies while the radius is large compared
    // with the diffuse interface.  Samples with R > fitCut are accumulated for
    // a least-squares fit of R^2 against t, whose slope is the quantity to
    // compare with the analytic -4 M kappa (OpenPhase reports the same ratio
    // in R_2_graph.dat).
    const double fitCut = 5.0 * delta;
    double sN = 0.0, sT = 0.0, sR2 = 0.0, sTT = 0.0, sTR2 = 0.0;

    auto sample = [&](int step) {
        const double t  = step * dt;
        const double Rn = measureR();
        const double Ra = analyticR(t);
        const double e  = (Ra > 0.0) ? (Rn - Ra) / Ra : 0.0;
        std::fprintf(csv, "%d,%.6e,%.6e,%.6e,%.6e\n", step, t, Rn, Ra, e);
        std::printf("%8d %12.4e %14.6e %14.6e %9.2f%%%s\n",
                    step, t, Rn, Ra, 100.0 * e, (Rn > fitCut) ? "" : "   (excl.)");
        std::fflush(stdout);
        if (Rn > fitCut) {
            const double R2 = Rn * Rn;
            sN += 1.0; sT += t; sR2 += R2; sTT += t * t; sTR2 += t * R2;
        }
    };

    sample(0);
    if (writeFields) IO::writeField(phi, "output/phi_0", FieldFormat::VTS);

    // ---- time loop (timed) -------------------------------------------------
    perf::WallTimer wall;
    for (int step = 1; step <= nSteps; ++step) {
        solver.advance();
        if (step % sampleEvery == 0 || step == nSteps) {
            cudaDeviceSynchronize();
            sample(step);
            if (writeFields)
                IO::writeField(phi, "output/phi_" + std::to_string(step),
                               FieldFormat::VTS);
        }
    }
    cudaDeviceSynchronize();
    const double sec = wall.seconds();
    std::fclose(csv);

    // ---- quantitative verdict: slope of R^2 vs t ---------------------------
    if (sN >= 3.0) {
        const double det   = sN * sTT - sT * sT;
        const double slope = (sN * sTR2 - sT * sR2) / det;
        const double icept = (sR2 * sTT - sT * sTR2) / det;
        const double slopeAna = -4.0 * Mkap;
        std::printf("\nR^2 vs t least squares over %d samples with R > %.2f dx"
                    " (%.1f delta):\n", static_cast<int>(sN), fitCut / dx, 5.0);
        std::printf("  measured slope  = %.6e m^2/s\n", slope);
        std::printf("  analytic slope  = %.6e m^2/s   (-4 M kappa)\n", slopeAna);
        std::printf("  slope ratio     = %.4f        deviation %+.2f %%\n",
                    slope / slopeAna, 100.0 * (slope / slopeAna - 1.0));
        std::printf("  intercept R0    = %.6e m   (set %.6e, %+.2f %%)\n",
                    std::sqrt(std::max(icept, 0.0)), R0,
                    100.0 * (std::sqrt(std::max(icept, 0.0)) / R0 - 1.0));
    } else {
        std::printf("\n[warning] too few samples above the fit cut-off"
                    " — increase nSteps or the grid\n");
    }

    const double cells = static_cast<double>(N) * N * N;
    std::printf("\nwall %.3f s   %.4f ms/step   %.1f Mcell-updates/s"
                "   (%.3e cells)\n",
                sec, sec * 1e3 / nSteps, cells * nSteps / (sec * 1e6), cells);
    std::printf("history -> output/radius_history.csv\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "singlegrain3d FAILED: %s\n", e.what());
    return 1;
}
}
