/***********************************************************************\
 *
 *  MPF_AC.cu — Multi-Phase Field Allen-Cahn solver (2D)
 *
 *  Benchmark: static triple junction with equal dihedral angles (120°)
 *
 *  Three-phase system: phi0, phi_a, phi_b   (sum constraint: phi0+phi_a+phi_b=1)
 *
 *  Free energy (pairwise double-well + grad2 weighted-gradient energy):
 *    f_grad = Σ_{i<j} (ε²_ij/2) |φ_i ∇φ_j − φ_j ∇φ_i|²
 *    f_dw   = Σ_{i<j} W_ij φ_i² φ_j²
 *
 *  Chemical potential (functional derivative μ_i = δF/δφ_i):
 *    μ_i = 2φ_i Σ_{j≠i} W_ij φ_j²
 *        + Σ_{j≠i} ε²_ij [ 2φ_i|∇φ_j|² − 2φ_j(∇φ_i·∇φ_j)
 *                          + φ_i φ_j ∇²φ_j − φ_j² ∇²φ_i ]
 *
 *  Pairwise Allen-Cahn (conserves Σφ_i automatically):
 *    ∂φ_i/∂t = − Σ_{j≠i} L_ij (μ_i − μ_j)
 *    φ_i^{n+1} = φ_i^n + dt · ∂φ_i/∂t      (explicit Euler)
 *
 *  Parameter relations (ℓ = diffuse interface width):
 *    ε²_ij = (3/2)·γ_ij·ℓ
 *    W_ij  = 12·γ_ij / ℓ
 *
 *  Boundary conditions (static triple junction benchmark, matching MATLAB ref):
 *    Top  wall (y=H): Neumann (zero normal gradient)
 *    Bottom wall (y=0): Neumann (zero normal gradient)
 *    Left  wall (x=0): non-uniform Dirichlet — upper ny_top rows: phi0=1;
 *                       lower rows: phi_a=1
 *    Right wall (x=W): non-uniform Dirichlet — upper ny_top rows: phi0=1;
 *                       lower rows: phi_b=1
 *
 *  Reference: doc/static_triple_junction_pairwise_double_well_v2.md
 *
\***********************************************************************/

#include "mesh/Mesh.h"
#include "core/RunGuard.h"
#include "core/Error.h"
#include "field/ScalarField.h"
#include "boundary/NoFluxBC.h"
#include "IO/ConfigFile.h"
#include "scheme/Schemes.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"
#include "field/GibbsSimplex.h"
#include "equation/Equation.h"
#include "equation/Term.h"
#include "equation/EquationSystem.h"
#include "equation/FusedTerm.h"

#include <cmath>
#include <iostream>
#include <string>

// ===========================================================================
// Kernel: Initialize fields for static triple-junction IC (matching MATLAB ref)
//   Upper ny_top rows (iy >= ny - ny_top): phi0 = 1
//   Lower rows, left  half (ix < nx/2)  : phi_a = 1
//   Lower rows, right half (ix >= nx/2) : phi_b = 1
// ===========================================================================
__global__ void k_init_fields(
    double* d_phi0, double* d_phi_a, double* d_phi_b,
    int nx, int ny, int ny_top, int sx, int sy, int g)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;

    int idx = (ix + g) + sx * ((iy + g) + sy * g);

    if (iy >= ny - ny_top) {
        // Top region: phi0 = 1
        d_phi0[idx] = 1.0;  d_phi_a[idx] = 0.0;  d_phi_b[idx] = 0.0;
    } else if (ix < nx / 2) {
        // Bottom-left: phi_a = 1
        d_phi0[idx] = 0.0;  d_phi_a[idx] = 1.0;  d_phi_b[idx] = 0.0;
    } else {
        // Bottom-right: phi_b = 1
        d_phi0[idx] = 0.0;  d_phi_a[idx] = 0.0;  d_phi_b[idx] = 1.0;
    }
}


// ===========================================================================
// Kernel: Non-uniform Dirichlet BC on left (x=0) and right (x=W) ghost cells.
//
//   Upper ny_top rows (iy >= ny - ny_top):
//     Left ghost + Right ghost → phi0=1, phi_a=0, phi_b=0
//   Lower rows:
//     Left  ghost → phi0=0, phi_a=1, phi_b=0
//     Right ghost → phi0=0, phi_a=0, phi_b=1
//
//   Each thread handles one iy, setting both left and right ghost columns.
// ===========================================================================
__global__ void k_apply_xbc(
    double* d_phi0, double* d_phi_a, double* d_phi_b,
    int nx, int ny, int sx, int sy, int g, int ny_top)
{
    int iy = blockIdx.x * blockDim.x + threadIdx.x;
    if (iy >= ny) return;

    bool in_top = (iy >= ny - ny_top);

    // Left ghost: ix = -1  →  stored index ix_stored = g - 1
    int idxL = (g - 1) + sx * ((iy + g) + sy * g);
    // Right ghost: ix = nx  →  stored index ix_stored = nx + g
    int idxR = (nx + g) + sx * ((iy + g) + sy * g);

    if (in_top) {
        d_phi0[idxL] = 1.0;  d_phi_a[idxL] = 0.0;  d_phi_b[idxL] = 0.0;
        d_phi0[idxR] = 1.0;  d_phi_a[idxR] = 0.0;  d_phi_b[idxR] = 0.0;
    } else {
        d_phi0[idxL] = 0.0;  d_phi_a[idxL] = 1.0;  d_phi_b[idxL] = 0.0;  // left: phi_a
        d_phi0[idxR] = 0.0;  d_phi_a[idxR] = 0.0;  d_phi_b[idxR] = 1.0;  // right: phi_b
    }
}

// ===========================================================================
// main
// ===========================================================================
static int phixMain(int argc, char* argv[])
{
    using namespace PhiX;

    IO::ConfigFile cfg = IO::ConfigFile::fromArgs(argc, argv);
    // Numerical schemes from settings/schemes.jsonc (missing file → the
    // defaults below, i.e. what this solver used before v3.8.8).
    Schemes sch = Schemes::load(cfg, {});

    // -----------------------------------------------------------------------
    // 1. Mesh
    // -----------------------------------------------------------------------
    const int    nx = cfg["mesh"]["nx"];
    const double dx = cfg["mesh"]["dx"];
    const double x0 = cfg["mesh"]["x0"];
    const int    ny = cfg["mesh"]["ny"];
    const double dy = cfg["mesh"]["dy"];
    const double y0 = cfg["mesh"]["y0"];

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, nx, dx, x0, ny, dy, y0);
    mesh.print();

    // -----------------------------------------------------------------------
    // 2. Time parameters
    // -----------------------------------------------------------------------
    const double dt     = cfg["initialize"]["dt"];
    const int    nSteps = cfg["initialize"]["nSteps"];
    const int    ny_top = cfg["initialize"]["ny_top"];

    // -----------------------------------------------------------------------
    // 3. Physical parameters
    // -----------------------------------------------------------------------
    // Pairwise interface energies
    const double gamma_0a = cfg["constants"]["gamma_0a"];
    const double gamma_0b = cfg["constants"]["gamma_0b"];
    const double gamma_ab = cfg["constants"]["gamma_ab"];

    // Diffuse interface width (physical units, same as dx units)
    const double ell = cfg["constants"]["ell"];

    // Pairwise kinetic coefficients (equal for all pairs, from settings)
    const double L     = cfg["constants"]["L"];
    const double L_0a  = L;
    const double L_0b  = L;
    const double L_ab  = L;

    // Derived parameters (double-well + grad2)
    const double eps2_0a = 1.5 * gamma_0a * ell;
    const double eps2_0b = 1.5 * gamma_0b * ell;
    const double eps2_ab = 1.5 * gamma_ab * ell;
    const double W_0a    = 12.0 * gamma_0a / ell;
    const double W_0b    = 12.0 * gamma_0b / ell;
    const double W_ab    = 12.0 * gamma_ab / ell;

    // Analytical equilibrium angle (for verification output)
    const double cos_half = gamma_ab / (2.0 * gamma_0a);
    const double theta_deg = (std::abs(cos_half) <= 1.0)
                           ? 2.0 * std::acos(cos_half) * 180.0 / M_PI
                           : -1.0;
    // Analytical grain boundary rise above y_b
    const double h_GB = (gamma_0a > 0.0 && std::abs(gamma_ab) < 2.0 * gamma_0a)
                      ? (nx * dx) * gamma_ab
                        / (2.0 * std::sqrt(4.0 * gamma_0a * gamma_0a
                                           - gamma_ab * gamma_ab))
                      : -1.0;

    std::cout << "=== MPF_AC_DW — Static Triple Junction Benchmark (double-well) ===\n"
              << "  γ_0α=" << gamma_0a << "  γ_0β=" << gamma_0b
              << "  γ_αβ=" << gamma_ab << "  ℓ=" << ell << "\n"
              << "  ε²_0α=" << eps2_0a << "  W_0α=" << W_0a << "\n"
              << "  ε²_αβ=" << eps2_ab << "  W_αβ=" << W_ab << "\n"
              << "  L=" << L << "\n"
              << "  Analytical equilibrium angle θ ≈ " << theta_deg << "°\n"
              << "  Analytical GB drop below top wall: h_GB ≈ " << h_GB << "\n"
              << "  dt=" << dt << "  nSteps=" << nSteps << "\n";

    // -----------------------------------------------------------------------
    // 4. Fields
    // -----------------------------------------------------------------------
    ScalarField phi0 (mesh, "phi0",  /*ghost=*/1);
    ScalarField phi_a(mesh, "phi_a", /*ghost=*/1);
    ScalarField phi_b(mesh, "phi_b", /*ghost=*/1);

    // Scratch fields for chemical potentials (interior cells only, no BCs)
    ScalarField mu0 (mesh, "mu0",  /*ghost=*/1);
    ScalarField mu_a(mesh, "mu_a", /*ghost=*/1);
    ScalarField mu_b(mesh, "mu_b", /*ghost=*/1);

    phi0.fill(0.0);   phi_a.fill(0.0);   phi_b.fill(0.0);
    mu0.fill(0.0);    mu_a.fill(0.0);    mu_b.fill(0.0);

    // -----------------------------------------------------------------------
    // 5. Restart / initialization
    // -----------------------------------------------------------------------
    const std::string start_from = cfg["initialize"]["start_from"];
    const int start_step = IO::resolveStartStep(start_from, "phi0");

    if (start_step == 0) {
        // Cold start: initialize programmatically
        phi0.allocDevice();   phi0.uploadAllToDevice();
        phi_a.allocDevice();  phi_a.uploadAllToDevice();
        phi_b.allocDevice();  phi_b.uploadAllToDevice();
        mu0.allocDevice();    mu_a.allocDevice();   mu_b.allocDevice();

        const dim3 blk(16, 16);
        const dim3 grd((nx + 15) / 16, (ny + 15) / 16);
        k_init_fields<<<grd, blk>>>(
            phi0.d_curr, phi_a.d_curr, phi_b.d_curr,
            nx, ny, ny_top,
            phi0.storedDims[0], phi0.storedDims[1], phi0.ghost);
        cudaDeviceSynchronize();
    } else {
        // Warm restart: read from output/
        IO::initField(phi0,  start_step);
        IO::initField(phi_a, start_step);
        IO::initField(phi_b, start_step);

        phi0.allocDevice();   phi0.uploadAllToDevice();
        phi_a.allocDevice();  phi_a.uploadAllToDevice();
        phi_b.allocDevice();  phi_b.uploadAllToDevice();
        mu0.allocDevice();    mu_a.allocDevice();   mu_b.allocDevice();
    }

    // -----------------------------------------------------------------------
    // 6. Boundary conditions (matching MATLAB reference)
    //    Top  (y=H): Neumann (zero flux)
    //    Bottom(y=0): Neumann (zero flux)
    //    Left/Right: non-uniform Dirichlet via k_apply_xbc
    //      upper ny_top rows → phi0=1; lower rows → phi_a=1 (left) / phi_b=1 (right)
    // -----------------------------------------------------------------------
    NoFluxBC bcTop(mesh.facePatch(Axis::Y, Side::HIGH));
    NoFluxBC bcBot(mesh.facePatch(Axis::Y, Side::LOW));

    // -----------------------------------------------------------------------
    // 6b. Equations expressed in DSL
    //
    // μ equations (STEADY): δF/δφ_i via double-well + grad2 model.
    // Each μ is computed from the current φ snapshot (before any φ update).
    //
    //   μ_0 = 2φ_0(W_0a φ_a² + W_0b φ_b²)
    //       + ε²_0a(2φ_0|∇φ_a|² - 2φ_a ∇φ_0·∇φ_a + φ_0 φ_a ∇²φ_a - φ_a² ∇²φ_0)
    //       + ε²_0b(2φ_0|∇φ_b|² - 2φ_b ∇φ_0·∇φ_b + φ_0 φ_b ∇²φ_b - φ_b² ∇²φ_0)
    //
    // φ equations (TRANSIENT, simultaneous via EquationSystem):
    //   dφ_0/dt = (-L_0a-L_0b)·μ_0 + L_0a·μ_a + L_0b·μ_b
    //   dφ_a/dt =  L_0a·μ_0 + (-L_0a-L_ab)·μ_a + L_ab·μ_b
    //   dφ_b/dt =  L_0b·μ_0 +  L_ab·μ_a + (-L_0b-L_ab)·μ_b
    // -----------------------------------------------------------------------
    // ---- Fused μ expressions: three outputs computed in one kernel launch ----
    // Build expression trees with `auto` — the full type is deduced at compile
    // time and passed by value to kernel_fused_multi3 via fuse_multi_compute().
    // d_curr pointers captured here remain valid (EULER advances in-place).
    using namespace PhiX::Fused;

    auto expr_mu0 =
        fpw2(phi0, phi_a, PHIX_FN(double p0, double pa) { return 2.0*W_0a*p0*pa*pa; }) +
        fpw2(phi0, phi_b, PHIX_FN(double p0, double pb) { return 2.0*W_0b*p0*pb*pb; }) +
        fmul(ffield(phi0),  fgrad_dot(phi_a, phi_a)) * ( 2.0*eps2_0a) +
        fmul(ffield(phi_a), fgrad_dot(phi0,  phi_a)) * (-2.0*eps2_0a) +
        fmul(fmul(ffield(phi0), ffield(phi_a)), flap(phi_a)) * ( eps2_0a) +
        fmul(fmul(ffield(phi_a), ffield(phi_a)), flap(phi0)) * (-eps2_0a) +
        fmul(ffield(phi0),  fgrad_dot(phi_b, phi_b)) * ( 2.0*eps2_0b) +
        fmul(ffield(phi_b), fgrad_dot(phi0,  phi_b)) * (-2.0*eps2_0b) +
        fmul(fmul(ffield(phi0), ffield(phi_b)), flap(phi_b)) * ( eps2_0b) +
        fmul(fmul(ffield(phi_b), ffield(phi_b)), flap(phi0)) * (-eps2_0b);

    auto expr_mu_a =
        fpw2(phi_a, phi0, PHIX_FN(double pa, double p0) { return 2.0*W_0a*pa*p0*p0; }) +
        fpw2(phi_a, phi_b, PHIX_FN(double pa, double pb) { return 2.0*W_ab*pa*pb*pb; }) +
        fmul(ffield(phi_a), fgrad_dot(phi0,  phi0)) * ( 2.0*eps2_0a) +
        fmul(ffield(phi0),  fgrad_dot(phi_a, phi0)) * (-2.0*eps2_0a) +
        fmul(fmul(ffield(phi_a), ffield(phi0)), flap(phi0))  * ( eps2_0a) +
        fmul(fmul(ffield(phi0),  ffield(phi0)), flap(phi_a)) * (-eps2_0a) +
        fmul(ffield(phi_a), fgrad_dot(phi_b, phi_b)) * ( 2.0*eps2_ab) +
        fmul(ffield(phi_b), fgrad_dot(phi_a, phi_b)) * (-2.0*eps2_ab) +
        fmul(fmul(ffield(phi_a), ffield(phi_b)), flap(phi_b)) * ( eps2_ab) +
        fmul(fmul(ffield(phi_b), ffield(phi_b)), flap(phi_a)) * (-eps2_ab);

    auto expr_mu_b =
        fpw2(phi_b, phi0, PHIX_FN(double pb, double p0) { return 2.0*W_0b*pb*p0*p0; }) +
        fpw2(phi_b, phi_a, PHIX_FN(double pb, double pa) { return 2.0*W_ab*pb*pa*pa; }) +
        fmul(ffield(phi_b), fgrad_dot(phi0,  phi0)) * ( 2.0*eps2_0b) +
        fmul(ffield(phi0),  fgrad_dot(phi_b, phi0)) * (-2.0*eps2_0b) +
        fmul(fmul(ffield(phi_b), ffield(phi0)), flap(phi0))  * ( eps2_0b) +
        fmul(fmul(ffield(phi0),  ffield(phi0)), flap(phi_b)) * (-eps2_0b) +
        fmul(ffield(phi_b), fgrad_dot(phi_a, phi_a)) * ( 2.0*eps2_ab) +
        fmul(ffield(phi_a), fgrad_dot(phi_b, phi_a)) * (-2.0*eps2_ab) +
        fmul(fmul(ffield(phi_b), ffield(phi_a)), flap(phi_a)) * ( eps2_ab) +
        fmul(fmul(ffield(phi_a), ffield(phi_a)), flap(phi_b)) * (-eps2_ab);

    Equation eq_phi0 (phi0,  "phi0");
    Equation eq_phi_a(phi_a, "phi_a");
    Equation eq_phi_b(phi_b, "phi_b");

    eq_phi0.setRHS(
        pw(mu0,  PHIX_FN(double m) { return (-L_0a - L_0b) * m; }) +
        pw(mu_a, PHIX_FN(double m) { return  L_0a           * m; }) +
        pw(mu_b, PHIX_FN(double m) { return  L_0b           * m; })
    );
    eq_phi_a.setRHS(
        pw(mu0,  PHIX_FN(double m) { return  L_0a           * m; }) +
        pw(mu_a, PHIX_FN(double m) { return (-L_0a - L_ab)  * m; }) +
        pw(mu_b, PHIX_FN(double m) { return  L_ab           * m; })
    );
    eq_phi_b.setRHS(
        pw(mu0,  PHIX_FN(double m) { return  L_0b           * m; }) +
        pw(mu_a, PHIX_FN(double m) { return  L_ab           * m; }) +
        pw(mu_b, PHIX_FN(double m) { return (-L_0b - L_ab)  * m; })
    );

    // Fused kernels are EULER-only (a fused RHS never sees RK4 stage buffers).
    if (sch.ddt() != TimeScheme::EULER)
        throw ConfigError("schemes", "this solver evaluates its fused RHS with EULER only",
                          "set \"ddt\": { \"default\": \"EULER\" } in settings/schemes.jsonc");
    EquationSystem sys(dt, sch.ddt());
    sch.warnUnused();   // per-term keys nobody looked up = misspelled field names
    sys.add(eq_phi0,  {&bcTop, &bcBot});
    sys.add(eq_phi_a, {&bcTop, &bcBot});
    sys.add(eq_phi_b, {&bcTop, &bcBot});
    // NaN/blow-up sentinel cadence (steps); 0 disables.  Default 100.
    sys.health.every = cfg["initialize"].value("nan_check_every", 100);

    // -----------------------------------------------------------------------
    // 7. Kernel launch config
    // -----------------------------------------------------------------------
    const int  g  = phi0.ghost;
    const int  sx = phi0.storedDims[0];
    const int  sy = phi0.storedDims[1];

    const dim3 blk1D(256);
    const dim3 grd1D((ny + 255) / 256);

    // -----------------------------------------------------------------------
    // 8. Output writer
    // -----------------------------------------------------------------------
    IO::OutputWriter writer(cfg["output"]);

    int    step = start_step;
    double time = start_step * dt;

    if (start_step == 0) {
        // Download and write initial state
        phi0.downloadCurrFromDevice();
        phi_a.downloadCurrFromDevice();
        phi_b.downloadCurrFromDevice();
        writer.writeFields(phi0,  0, 0.0);
        writer.writeFields(phi_a, 0, 0.0);
        writer.writeFields(phi_b, 0, 0.0);
        std::cout << "Starting MPF_AC simulation ("
                  << nSteps << " steps, dt=" << dt << ")\n";
    } else {
        std::cout << "Resuming MPF_AC simulation from step " << start_step
                  << " (t=" << time << "), "
                  << nSteps - start_step << " steps remaining, dt=" << dt
                  << "\n";
    }

    writer.resetTimer();

    // -----------------------------------------------------------------------
    // 9. Time loop
    // -----------------------------------------------------------------------
    for (int s = start_step; s < nSteps; ++s) {

        // (a) Apply boundary conditions
        bcTop.applyOnGPU(phi0);  bcTop.applyOnGPU(phi_a);  bcTop.applyOnGPU(phi_b);
        bcBot.applyOnGPU(phi0);  bcBot.applyOnGPU(phi_a);  bcBot.applyOnGPU(phi_b);
        k_apply_xbc<<<grd1D, blk1D>>>(
            phi0.d_curr, phi_a.d_curr, phi_b.d_curr,
            nx, ny, sx, sy, g, ny_top);

        // (c) Compute chemical potentials μ_0, μ_α, μ_β (STEADY, from current φ)
        //     Ghost cells of φ are already filled by step (a) → no BCs needed here.
        //     One kernel launch writes all three μ fields simultaneously.
        fuse_multi_compute(phi0,
            mu0,  expr_mu0,
            mu_a, expr_mu_a,
            mu_b, expr_mu_b);

        // (d) Explicit Euler advance — all φ updated simultaneously
        //     sys.advance() re-applies bcTop/bcBot (idempotent), evaluates all
        //     pw() RHS from the frozen μ fields, then updates φ_0/φ_a/φ_b.
        sys.advance();

        // (e) Gibbs simplex projection
        PhiX::gibbsSimplexOnGPU({&phi0, &phi_a, &phi_b});

        ++step;
        time += dt;

        // (f) Output
        if (writer.shouldPrint(step))
            writer.printProgress(step, time);

        if (writer.shouldWrite(step)) {
            writer.writeFields(phi0,  step, time);
            writer.writeFields(phi_a, step, time);
            writer.writeFields(phi_b, step, time);
        }
    }

    std::cout << "Done.\n";
    return 0;
}

int main(int argc, char* argv[]) {
    return PhiX::runGuarded(phixMain, argc, argv);
}
