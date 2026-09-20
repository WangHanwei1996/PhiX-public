/***********************************************************************\
 *
 *  diffusion_1D — 1-D Transient Diffusion  (face-flux form)
 *
 *  Author : Wang Hanwei
 *  Email  : wanghanweibnds2015@gmail.com
 *
 *  PDE
 *  ---
 *      ∂u/∂t = D ∂/∂x( ∂u/∂x )
 *
 *  Discretisation (two-step, face-flux):
 *      Step 1 — faceGrad(u, x, flux):
 *                  flux[i+½] = (u[i+1] − u[i]) / dx   (CD2 at face)
 *      Step 2 — divFace(flux):
 *                  rhs[i] = (flux[i+½] − flux[i-½]) / dx
 *
 *  This two-step form is equivalent to the standard 3-point Laplacian for
 *  uniform D, but the explicit face-flux layer allows easy extension to
 *  spatially-varying or nonlinear diffusion coefficients (D = D(u, x, ...)).
 *
 *  Initial condition (oscillating sine + Nyquist checkerboard):
 *      u₀(xᵢ) = sin(2π xᵢ / L)  +  0.25 (−1)^i
 *
 *  The fast Nyquist mode is strongly damped by diffusion in just a few steps,
 *  while the slow sine decays as exp(−D k² t) and remains visible longer.
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

    // =========================================================================
    // 3. Fields
    // =========================================================================
    ScalarField u   (mesh, "u",      /*ghost=*/1);
    FaceField   flux(mesh, /*axis=*/0, "flux_x");  // x-face fluxes

    const std::string start_from  = cfg["initialize"]["start_from"];
    const int         start_step  = IO::resolveStartStep(start_from, "u");

    if (start_step == 0) {
        // Cold start: oscillating sine + high-frequency checkerboard mode.
        // Cell centre position: x_i = x0 + (i + 0.5) * dx
        u.initialize([L, &mesh](double x, double /*y*/, double /*z*/) {
            const int    i   = static_cast<int>((x - mesh.origin[0]) / mesh.d[0]);
            const double chk = (i % 2 == 0) ? 1.0 : -1.0;   // (-1)^i
            return std::sin(2.0 * M_PI * x / L) + 0.25 * chk;
        });
    } else {
        IO::initField(u, start_step);
    }

    u.allocDevice();
    u.uploadAllToDevice();
    flux.allocDevice();   // face field lives on device

    // =========================================================================
    // 4. Boundary conditions
    // =========================================================================
    auto  bcSet = buildBCs(mesh, cfg["boundary_conditions"]);
    auto& bcs   = bcSet.ptrs;

    // =========================================================================
    // 5. Equation  ∂u/∂t = D · div( grad(u) )
    //
    // We build a composite Term whose gpu_launcher:
    //   (a) calls faceGradGPU(u, 0, flux) — runs AFTER advanceTransient has
    //       applied BCs to u, so ghost cells are valid.
    //   (b) calls the divFace launcher — reads flux.d_data updated in (a).
    //
    // This approach requires no external intervention inside the time loop.
    // =========================================================================

    // Base divergence Term capturing flux by reference (pointer stays valid).
    Term div_t = D * divFace(flux);

    // Combined face-diffusion Term.
    Term face_diff;
    face_diff.type          = TermType::COMPOSITE;
    face_diff.coeff         = 1.0;
    face_diff.field         = &u;      // representative field for sanity checks
    face_diff.ghostRequired = 1;

    // GPU path: faceGradGPU then div_t
    face_diff.gpu_launcher =
        [&u, &flux, div_t](double* d_rhs, double c, ScratchPool& pool) {
            faceGradGPU(u, 0, flux);                          // update face flux
            div_t.gpu_launcher(d_rhs, c * div_t.coeff, pool); // accumulate div
        };

    // CPU fallback path
    face_diff.cpu_kernel =
        [&u, &flux, div_t](double* rhs, double c, ScratchPool& pool) {
            faceGrad(u, 0, flux);
            div_t.cpu_kernel(rhs, c * div_t.coeff, pool);
        };

    Equation eq(u, "diffusion");
    eq.setRHS(face_diff);

    // =========================================================================
    // 6. Output & time loop
    // =========================================================================
    IO::OutputWriter writer(cfg["output"]);
    eq.step = start_step;
    eq.time = start_step * dt;

    if (start_step == 0)
        writer.writeFields(u, 0, eq.time);

    std::cout << "1-D diffusion (face-flux)\n"
              << "  nx=" << nx << "  dx=" << dx
              << "  L="  << L  << "  D="  << D
              << "  dt=" << dt << "  λ="  << lambda
              << "  nSteps=" << nSteps << "\n";

    writer.resetTimer();

    for (int s = start_step; s < nSteps; ++s) {
        // advanceTransient:
        //   1. applies BCs to u  (fills ghost cells)
        //   2. computeRHS  →  face_diff.gpu_launcher:
        //        faceGradGPU(u, 0, flux)  then  divFace kernel
        //   3. u.d_curr += dt * rhs
        //   4. u.advanceTimeLevelGPU()
        eq.advanceTransient(bcs, dt);

        if (writer.shouldPrint(eq.step))
            writer.printProgress(eq.step, eq.time);

        if (writer.shouldWrite(eq.step))
            writer.writeFields(u, eq.step, eq.time);
    }

    std::cout << "Done.\n";
    return 0;
}
