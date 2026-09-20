/***********************************************************************\
 *
 *  diffusion_compare — 1-D Transient Diffusion
 *  Side-by-side comparison: staggered face-flux vs. collocated two-step
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  PDE (both cases)
 *  ----------------
 *      ∂u/∂t = D ∂²u/∂x²
 *
 *  ┌─────────────────────────────────────────────────────────────────────┐
 *  │  Staggered  (face-centred flux, 2-step)                            │
 *  │    Step 1: faceGrad(u, x, flux)                                    │
 *  │               flux[i+½] = (u[i+1] − u[i]) / dx                    │
 *  │    Step 2: divFace(flux)                                           │
 *  │               rhs[i] = (flux[i+½] − flux[i-½]) / dx               │
 *  │    ⇒  rhs[i] = (u[i+1] − 2u[i] + u[i-1]) / dx²                   │
 *  │    Checker mode (-1)^i: eigenvalue = 4/dx² → strongly damped ✓     │
 *  ├─────────────────────────────────────────────────────────────────────┤
 *  │  Collocated  (CD2 applied twice, skip-one-neighbour)               │
 *  │    Step 1: g[i] = grad(u)[i] = (u[i+1] − u[i-1]) / (2dx)         │
 *  │    Step 2: rhs[i] = grad(g)[i] = (g[i+1] − g[i-1]) / (2dx)       │
 *  │    ⇒  rhs[i] = (u[i+2] − 2u[i] + u[i-2]) / (4dx²)               │
 *  │    Checker mode (-1)^i: u[i+2]=u[i] → eigenvalue = 0 → undamped ✗ │
 *  └─────────────────────────────────────────────────────────────────────┘
 *
 *  Initial condition:
 *      u₀(xᵢ) = sin(2π xᵢ / L) + 0.25 (−1)^i
 *
 *  Analytic solution:
 *      u(x,t) = exp(−D(2π/L)² t) sin(2πx/L)
 *             + 0.25 exp(−D(π/dx)² t) (−1)^i
 *
 *  Boundary conditions: Periodic
 *
 \***********************************************************************/

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "field/FaceField.h"
#include "boundary/BCFactory.h"
#include "equation/Equation.h"
#include "operators/FaceOps.h"
#include "operators/Gradient.h"
#include "IO/ConfigFile.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char* argv[])
{
    using namespace PhiX;

    IO::ConfigFile cfg = IO::ConfigFile::fromArgs(
        argc, argv, "settings/settings.jsonc");

    // =========================================================================
    // 1. Mesh
    // =========================================================================
    const int    nx = cfg["mesh"]["nx"];
    const double dx = cfg["mesh"]["dx"];
    const double x0 = cfg["mesh"]["x0"];

    Mesh mesh = Mesh::makeUniform1D(CoordSys::CARTESIAN, nx, dx, x0);
    mesh.print();

    const double L = nx * dx;   // physical domain length

    // =========================================================================
    // 2. Time parameters & stability check
    // =========================================================================
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];
    const double D      = cfg["constants"]["D"];

    // Forward-Euler stability: λ = D dt / dx² ≤ 0.5
    const double lambda = D * dt / (dx * dx);
    if (lambda > 0.5)
        std::cerr << "[WARNING] Fourier number λ = " << lambda
                  << " > 0.5 — forward-Euler diffusion will be unstable!\n";

    std::cout << "1-D diffusion comparison: staggered vs. collocated\n"
              << "  nx=" << nx << "  dx=" << dx << "  L=" << L
              << "  D=" << D << "  dt=" << dt
              << "  λ=" << lambda << "  nSteps=" << nSteps << "\n\n"
              << "  Staggered:  rhs[i] = (u[i+1]-2u[i]+u[i-1]) / dx²\n"
              << "  Collocated: rhs[i] = (u[i+2]-2u[i]+u[i-2]) / (4dx²)\n\n";

    // =========================================================================
    // 3. Fields
    // =========================================================================
    //   u_stag  — staggered scheme solution
    //   u_coloc — collocated scheme solution
    //   g       — intermediate: cell-centred gradient  g = grad(u_coloc, 0)
    //   flux    — x-face fluxes for staggered scheme

    ScalarField u_stag (mesh, "u_stag",  /*ghost=*/1);
    ScalarField u_coloc(mesh, "u_coloc", /*ghost=*/1);
    ScalarField g      (mesh, "g",       /*ghost=*/1);
    FaceField   flux   (mesh, /*axis=*/0, "flux_x");

    // Shared initial condition: sin(2πx/L) + 0.25·(-1)^i
    auto ic = [L, &mesh](double x, double /*y*/, double /*z*/) -> double {
        const int    i   = static_cast<int>((x - mesh.origin[0]) / mesh.d[0]);
        const double chk = (i % 2 == 0) ? 1.0 : -1.0;   // (-1)^i
        return std::sin(2.0 * M_PI * x / L) + 0.25 * chk;
    };

    u_stag .initialize(ic);
    u_coloc.initialize(ic);

    u_stag .allocDevice();  u_stag .uploadAllToDevice();
    u_coloc.allocDevice();  u_coloc.uploadAllToDevice();
    g      .allocDevice();  // filled each step by eq_g; no upload needed
    flux   .allocDevice();  // filled each step by faceGradGPU

    // =========================================================================
    // 4. Boundary conditions (periodic, applied to all fields on this mesh)
    // =========================================================================
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // =========================================================================
    // 5a. Staggered equation
    //
    //   ∂u_stag/∂t = D · divFace( faceGrad(u_stag) )
    //
    //   We embed faceGradGPU inside the Term's gpu_launcher so the face flux
    //   is always recomputed AFTER advanceTransient applies BCs to u_stag.
    // =========================================================================
    Term div_t = D * divFace(flux);

    Term face_diff;
    face_diff.type          = TermType::COMPOSITE;
    face_diff.coeff         = 1.0;
    face_diff.field         = &u_stag;
    face_diff.ghostRequired = 1;

    face_diff.gpu_launcher =
        [&u_stag, &flux, div_t](double* d_rhs, double c, ScratchPool& pool) {
            faceGradGPU(u_stag, 0, flux);
            div_t.gpu_launcher(d_rhs, c * div_t.coeff, pool);
        };

    face_diff.cpu_kernel =
        [&u_stag, &flux, div_t](double* rhs, double c, ScratchPool& pool) {
            faceGrad(u_stag, 0, flux);
            div_t.cpu_kernel(rhs, c * div_t.coeff, pool);
        };

    Equation eq_stag(u_stag, "staggered");
    eq_stag.setRHS(face_diff);

    // =========================================================================
    // 5b. Collocated equation (two-step)
    //
    //   Step 1: g = grad(u_coloc, 0)   ← advanceSteady each step
    //             g[i] = (u_coloc[i+1] − u_coloc[i-1]) / (2dx)
    //
    //   Step 2: u_coloc += dt · D · grad(g, 0)
    //             rhs[i] = (g[i+1] − g[i-1]) / (2dx)
    //
    //   Combined: rhs[i] = (u[i+2] − 2u[i] + u[i-2]) / (4dx²)
    //
    //   The checkerboard mode (-1)^i satisfies u[i+2]=u[i], so rhs=0.
    //   This mode is INVISIBLE to the collocated scheme and is not damped.
    // =========================================================================
    Equation eq_g    (g,       "grad_u");
    Equation eq_coloc(u_coloc, "collocated");

    eq_g    .setRHS(grad(u_coloc, 0));
    eq_coloc.setRHS(D * grad(g, 0));

    // =========================================================================
    // 6. Output & time loop
    // =========================================================================
    IO::OutputWriter writer(cfg["output"]);

    // Write initial state for both fields
    writer.writeFields(u_stag,  0, 0.0);
    writer.writeFields(u_coloc, 0, 0.0);
    writer.resetTimer();

    for (int s = 0; s < nSteps; ++s) {

        // --- Staggered step ---
        // advanceTransient: applies BCs → u_stag, then face_diff.gpu_launcher
        //   calls faceGradGPU then divFace, accumulates into u_stag.
        eq_stag.advanceTransient(bcs, dt);

        // --- Collocated two-step ---
        // Step 1: apply BCs → u_coloc, compute g = grad(u_coloc)
        eq_g.advanceSteady(bcs, &u_coloc);
        // Step 2: apply BCs → g,       compute u_coloc += dt * D * grad(g)
        eq_coloc.advanceTransient(bcs, dt, &g);

        const bool do_print = writer.shouldPrint(eq_stag.step);
        const bool do_write = writer.shouldWrite(eq_stag.step);

        if (do_print)
            writer.printProgress(eq_stag.step, eq_stag.time);

        if (do_write) {
            writer.writeFields(u_stag,  eq_stag.step,  eq_stag.time);
            writer.writeFields(u_coloc, eq_coloc.step, eq_coloc.time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}
