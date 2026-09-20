// ---------------------------------------------------------------------------
// gray_scott — Gray-Scott reaction-diffusion, 2D and 3D.
//
// A deliberately NON-phase-field problem, solved with the same Mesh / Field /
// BoundaryCondition / Equation / Solver stack and the same equation DSL:
//
//   du/dt = Du lap(u) - u v^2 + F (1 - u)
//   dv/dt = Dv lap(v) + u v^2 - (F + k) v
//
// The two equations are coupled through the cubic autocatalytic term u v^2 and
// must be advanced from the same time level, which is exactly what
// EquationSystem provides (every RHS is evaluated before any field updates).
// Each right-hand side is one stencil term plus one two-field pointwise term:
//
//   eqU.setRHS(lap(u, Du) + pw(u, v, [](u,v){ return -u*v*v + F*(1-u); }));
//
// Classical Pearson (Science 261, 189 (1993)) parameter sets are selected by
// name; the lattice-scaled form (dx = 1, Du = 0.16, Dv = 0.08) is used so the
// familiar (F, k) values apply directly.
//
// Usage:
//   ./gray_scott [pattern] [N] [nSteps] [--3d] [--out]
//     pattern  spots | coral | worms | uskate   (default coral)
//     N        grid size per side               (default 256, 3D default 128)
//     nSteps   number of Euler steps            (default 20000)
//     --3d     run on an N^3 mesh instead of N^2
//     --out    write VTS fields (off by default so timing runs stay clean)
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/Reduce.h"
#include "equation/Equation.h"
#include "equation/EquationSystem.h"
#include "boundary/PeriodicBC.h"
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

struct Pattern { const char* name; double F, k; };
static const Pattern kPatterns[] = {
    {"spots",  0.0367, 0.0649},
    {"coral",  0.0545, 0.0620},
    {"worms",  0.0780, 0.0610},
    {"uskate", 0.0620, 0.0609},
};

int main(int argc, char** argv) {
try {
    std::string want = (argc > 1 && argv[1][0] != '-') ? argv[1] : "coral";
    bool threeD = false, writeFields = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--3d") == 0) threeD      = true;
        if (std::strcmp(argv[i], "--out") == 0) writeFields = true;
    }
    const int N      = (argc > 2 && argv[2][0] != '-') ? std::atoi(argv[2])
                                                       : (threeD ? 128 : 256);
    const int nSteps = (argc > 3 && argv[3][0] != '-') ? std::atoi(argv[3]) : 20000;

    Pattern pat = kPatterns[1];
    for (const auto& p : kPatterns)
        if (want == p.name) pat = p;

    // Lattice-scaled Gray-Scott: dx = 1 so the textbook (F, k) apply directly.
    const double dx = 1.0;
    const double Du = 0.16, Dv = 0.08;
    const double F = pat.F, k = pat.k;
    // Explicit diffusion limit: dx^2 / (2 d Du);  stay comfortably inside it.
    const int    dim = threeD ? 3 : 2;
    const double dtLimit = dx * dx / (2.0 * dim * Du);
    const double dt = 0.5 * dtLimit;

    std::printf("gray_scott  pattern=%s  F=%.4f  k=%.4f  %dD  N=%d  steps=%d\n",
                pat.name, F, k, dim, N, nSteps);
    std::printf("  Du=%.3g Dv=%.3g dx=%.3g  dt=%.4f (limit %.4f)\n",
                Du, Dv, dx, dt, dtLimit);

    Mesh mesh = threeD
        ? Mesh::makeUniform3D(CoordSys::CARTESIAN,
                              N, dx, 0.0, N, dx, 0.0, N, dx, 0.0)
        : Mesh::makeUniform2D(CoordSys::CARTESIAN,
                              N, dx, 0.0, N, dx, 0.0);

    // u = 1, v = 0 everywhere; a seeded block at the centre with u = 1/2,
    // v = 1/4 plus a deterministic hash perturbation to break symmetry.
    const double c = 0.5 * N * dx;
    const double half = 0.05 * N * dx;          // seed block half-width
    ScalarField u(mesh, "u", 1), v(mesh, "v", 1);
    auto hash = [](double a, double b, double cc) {
        double s = std::sin(12.9898 * a + 78.233 * b + 37.719 * cc) * 43758.5453;
        return s - std::floor(s);
    };
    u.initialize([&](double x, double y, double z) {
        const bool in = std::fabs(x - c) < half && std::fabs(y - c) < half
                     && (!threeD || std::fabs(z - c) < half);
        return in ? 0.50 + 0.02 * (hash(x, y, z) - 0.5) : 1.0;
    });
    v.initialize([&](double x, double y, double z) {
        const bool in = std::fabs(x - c) < half && std::fabs(y - c) < half
                     && (!threeD || std::fabs(z - c) < half);
        return in ? 0.25 + 0.02 * (hash(y, x, z) - 0.5) : 0.0;
    });
    for (ScalarField* f : {&u, &v}) { f->allocDevice(); f->uploadAllToDevice(); }

    std::vector<PeriodicBC> bcStore;
    bcStore.reserve(dim);
    for (int ax = 0; ax < dim; ++ax)
        bcStore.emplace_back(mesh.facePatch(static_cast<Axis>(ax), Side::LOW));
    std::vector<BoundaryCondition*> bcs;
    for (auto& b : bcStore) bcs.push_back(&b);

    // The whole model, in the DSL: one stencil term + one two-field pointwise
    // term per equation.
    Equation eqU(u, "gs_u");
    eqU.setRHS(lap(u, Du)
             + pw(u, v, PHIX_FN (Real uu, Real vv) {
                   return -uu * vv * vv + Real(F) * (Real(1) - uu);
               }));

    Equation eqV(v, "gs_v");
    eqV.setRHS(lap(v, Dv)
             + pw(u, v, PHIX_FN (Real uu, Real vv) {
                   return uu * vv * vv - Real(F + k) * vv;
               }));

    // Simultaneous update: both RHS come from the same time level.
    EquationSystem sys(dt, TimeScheme::EULER);
    sys.add(eqU, bcs);
    sys.add(eqV, bcs);

    std::filesystem::create_directories("output");
    const int sampleEvery = (nSteps >= 10) ? nSteps / 10 : 1;
    std::printf("\n%10s %14s %14s %14s\n", "step", "mean u", "mean v", "max v");

    const double nCells = threeD ? double(N) * N * N : double(N) * N;
    auto report = [&](int step) {
        u.downloadCurrFromDevice();
        v.downloadCurrFromDevice();
        std::printf("%10d %14.6f %14.6f %14.6f\n", step,
                    reduce::fieldSum(u) / nCells,
                    reduce::fieldSum(v) / nCells,
                    reduce::fieldMax(v));
        std::fflush(stdout);
    };
    report(0);
    if (writeFields) {
        IO::writeField(u, "output/u_0", FieldFormat::VTS);
        IO::writeField(v, "output/v_0", FieldFormat::VTS);
    }

    perf::WallTimer wall;
    for (int step = 1; step <= nSteps; ++step) {
        sys.advance();
        if (step % sampleEvery == 0 || step == nSteps) {
            cudaDeviceSynchronize();
            report(step);
            if (writeFields) {
                IO::writeField(u, "output/u_" + std::to_string(step),
                               FieldFormat::VTS);
                IO::writeField(v, "output/v_" + std::to_string(step),
                               FieldFormat::VTS);
            }
        }
    }
    cudaDeviceSynchronize();
    const double sec = wall.seconds();

    std::printf("\nwall %.3f s   %.4f ms/step   %.1f Mcell-updates/s"
                "  (2 fields x %.3e cells)\n",
                sec, sec * 1e3 / nSteps,
                2.0 * nCells * nSteps / (sec * 1e6), nCells);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "gray_scott FAILED: %s\n", e.what());
    return 1;
}
}
