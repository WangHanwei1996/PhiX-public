// PFHub Benchmark 1a (spinodal decomposition) — OpenPhase-idiom CPU driver.
//
// Manuscript v4 Block B (PLAN.md D4): like-for-like GPU/CPU throughput
// comparison against the PhiX CH solver. OpenPhase 1.0.1 ships no pure
// Cahn-Hilliard module, so this driver implements BM1a directly on
// OpenPhase's Storage3D field container with the same OpenMP loop structure
// the library's own examples use (collapse(2) parallel sweeps), compiled
// with the library's recommended flags. Numerics mirror the PhiX solver
// exactly: double precision, CD2 5-point Laplacians, explicit Euler,
// periodic BCs via halo exchange.
//
// PFHub BM1a: f = rho (c-ca)^2 (cb-c)^2, rho=5, ca=0.3, cb=0.7, kappa=2,
// M=5, domain 200x200 (dx=1), c0=0.5 + prescribed cosine perturbation.
//
// Build: g++ -O3 -march=native -fopenmp -I$OPENPHASE/include bm1_spinodal.cpp -o bm1_op
// Run  : OMP_NUM_THREADS=32 ./bm1_op [nsteps=50000] [dt=1e-3]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "Base/Storage3D.h"

using openphase::Storage3D;

static const long   Nx = NX_DEF, Ny = NX_DEF;
static const double rho = 5.0, ca = 0.3, cb = 0.7, kappa = 2.0, M = 5.0;
static const double dx = 1.0;

static inline void wrap(Storage3D<double, 0>& f)
{
    for (long j = 0; j < Ny; ++j)
    {
        f(-1, j, 0) = f(Nx - 1, j, 0);
        f(Nx, j, 0) = f(0, j, 0);
    }
    for (long i = -1; i <= Nx; ++i)
    {
        f(i, -1, 0) = f(i, Ny - 1, 0);
        f(i, Ny, 0) = f(i, 0, 0);
    }
}

int main(int argc, char** argv)
{
    const long   nsteps = (argc > 1) ? atol(argv[1]) : 50000;
    const double dt     = (argc > 2) ? atof(argv[2]) : 1.0e-3;

    Storage3D<double, 0> c, mu;
    c.Allocate(Nx, Ny, 1, 1, 1, 0, 1);
    mu.Allocate(Nx, Ny, 1, 1, 1, 0, 1);

    // BM1a initial condition
    for (long i = 0; i < Nx; ++i)
        for (long j = 0; j < Ny; ++j)
        {
            const double x = i * dx, y = j * dx;
            c(i, j, 0) = 0.5
                + 0.01 * (std::cos(0.105 * x) * std::cos(0.11 * y)
                          + std::pow(std::cos(0.13 * x) * std::cos(0.087 * y), 2)
                          + std::cos(0.025 * x - 0.15 * y)
                            * std::cos(0.07 * x - 0.02 * y));
        }

    const double inv_dx2 = 1.0 / (dx * dx);
    const auto t0 = std::chrono::steady_clock::now();
    for (long step = 0; step < nsteps; ++step)
    {
        wrap(c);
#pragma omp parallel for collapse(2) schedule(static)
        for (long i = 0; i < Nx; ++i)
            for (long j = 0; j < Ny; ++j)
            {
                const double v = c(i, j, 0);
                const double lap = (c(i + 1, j, 0) + c(i - 1, j, 0)
                                    + c(i, j + 1, 0) + c(i, j - 1, 0)
                                    - 4.0 * v) * inv_dx2;
                mu(i, j, 0) = 2.0 * rho * (v - ca) * (v - cb) * (2.0 * v - ca - cb)
                              - kappa * lap;
            }
        wrap(mu);
#pragma omp parallel for collapse(2) schedule(static)
        for (long i = 0; i < Nx; ++i)
            for (long j = 0; j < Ny; ++j)
            {
                const double lap = (mu(i + 1, j, 0) + mu(i - 1, j, 0)
                                    + mu(i, j + 1, 0) + mu(i, j - 1, 0)
                                    - 4.0 * mu(i, j, 0)) * inv_dx2;
                c(i, j, 0) += dt * M * lap;
            }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();

    // free energy + final field for the cross-code sanity check
    double F = 0.0, csum = 0.0;
#pragma omp parallel for collapse(2) reduction(+ : F, csum)
    for (long i = 0; i < Nx; ++i)
        for (long j = 0; j < Ny; ++j)
        {
            const double v = c(i, j, 0);
            const double gx = (c(i + 1 == Nx ? 0 : i + 1, j, 0)
                               - c(i == 0 ? Nx - 1 : i - 1, j, 0)) / (2 * dx);
            const double gy = (c(i, j + 1 == Ny ? 0 : j + 1, 0)
                               - c(i, j == 0 ? Ny - 1 : j - 1, 0)) / (2 * dx);
            F += rho * std::pow(v - ca, 2) * std::pow(cb - v, 2)
                 + 0.5 * kappa * (gx * gx + gy * gy);
            csum += v;
        }
    F *= dx * dx;

    FILE* f = std::fopen("bm1_op_final.csv", "w");
    for (long j = 0; j < Ny; ++j)
        for (long i = 0; i < Nx; ++i)
            std::fprintf(f, "%ld,%ld,%.10e\n", i, j, c(i, j, 0));
    std::fclose(f);

    std::printf("OpenPhase-idiom BM1a: %ld steps (dt=%g)  wall=%.2f s  "
                "%.3e cell-updates/s  F(t_end)=%.6f  <c>=%.8f\n",
                nsteps, dt, wall, (double)Nx * Ny * nsteps / wall, F,
                csum / (Nx * Ny));
    return 0;
}
