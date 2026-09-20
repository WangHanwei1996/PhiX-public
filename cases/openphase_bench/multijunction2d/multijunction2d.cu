// ---------------------------------------------------------------------------
// multijunction2d — equilibrium triple-junction angles against Young's law.
//
// PhiX counterpart of OpenPhase 1.0.1 `benchmarks/MultiJunction2D`.  Reference
// solution is exact and purely geometric, which makes it one of the sharpest
// verification cases available for a multi-phase model: at a triple junction
// the three interface tensions must balance, so
//
//       sigma_12 / sin(theta_0) = sigma_02 / sin(theta_1) = sigma_01 / sin(theta_2)
//
// with theta_i the dihedral angle of the wedge occupied by phase i and
// theta_0 + theta_1 + theta_2 = 360 deg.  Equal energies give 120 deg each;
// unequal energies give a whole one-parameter family for free.
//
// With sigma_01 = sigma_02 = 1 and sigma_12 = s, symmetry gives
// theta_1 = theta_2 = theta and theta_0 = 360 - 2 theta, so Young's law
// collapses to the closed form
//
//       s = -2 cos(theta)        =>       theta = acos(-s/2),  s < 2
//
// which the sweep below checks point by point.
//
// Model (same pairwise double-well multi-phase field as
// applications/solvers/MPF_AC_DW, whose calibration this reuses):
//
//   f = sum_{i<j} [ W_ij phi_i^2 phi_j^2 + (eps2_ij/2) |phi_i grad phi_j - phi_j grad phi_i|^2 ]
//   mu_i = dF/dphi_i
//   dphi_i/dt = - sum_{j!=i} L_ij (mu_i - mu_j)        (conserves sum phi_i)
//   eps2_ij = 1.5 sigma_ij l ,   W_ij = 12 sigma_ij / l ,   l = interface width
//
// The three chemical potentials are produced by ONE fused kernel launch via
// fuse_multi_compute, exactly as in the production solver.
//
// Angle measurement: locate the junction as the cell minimising
// sum_i (phi_i - 1/3)^2, then sample the three fields on a circle of radius r
// around it and record the angular extent over which each phase is the largest.
//
// Usage:  ./multijunction2d [N] [nSteps] [s ...]
//           N       grid size          (default 256)
//           nSteps  steps per run      (default 20000)
//           s ...   sigma_12 values    (default 0.6 0.8 1.0 1.2 1.5)
// ---------------------------------------------------------------------------

#include "mesh/Mesh.h"
#include "field/ScalarField.h"
#include "IO/FieldIO.h"
#include "equation/Equation.h"
#include "equation/EquationSystem.h"
#include "equation/Term.h"
#include "equation/TermPW.inl"
#include "equation/FieldOps.inl"
#include "equation/FusedTerm.h"
#include "boundary/NoFluxBC.h"
#include "perf/Perf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdlib>

using namespace PhiX;
using namespace PhiX::Fused;

// Side walls are pinned to the far-field profile (non-uniform Dirichlet), the
// convention of applications/solvers/MPF_AC_DW and of the MATLAB reference in
// etc/temp/triple-junction-benchmark.  Without it the interfaces must meet a
// no-flux wall at 90 deg, the whole structure keeps drifting and no static
// equilibrium exists — the junction angles then never settle on Young's law.
__global__ void k_pin_side_walls(Real* p0, Real* p1, Real* p2,
                                 int sx, int sy, int g, int planeOff,
                                 double dx, double yTop, double wIfc) {
    const int jj = blockIdx.x * blockDim.x + threadIdx.x;
    if (jj >= sy) return;
    const double y  = (jj - g + 0.5) * dx;
    const double a0 = 0.5 * (1.0 + tanh((y - yTop) / wIfc));
    for (int ii = 0; ii <= g; ++ii) {                       // left ghosts + col 0
        const int f = planeOff + jj * sx + ii;
        p0[f] = static_cast<Real>(a0);
        p1[f] = static_cast<Real>(1.0 - a0);
        p2[f] = Real(0);
    }
    for (int ii = sx - 1 - g; ii < sx; ++ii) {              // col nx-1 + right ghosts
        const int f = planeOff + jj * sx + ii;
        p0[f] = static_cast<Real>(a0);
        p1[f] = Real(0);
        p2[f] = static_cast<Real>(1.0 - a0);
    }
}

namespace {

struct Angles { double t0, t1, t2, junctionX, junctionY; };

// Bilinear sample of a downloaded field at a physical position.
double sampleAt(const ScalarField& f, double x, double y, double dx) {
    const int nx = f.mesh.n[0], ny = f.mesh.n[1];
    double fi = x / dx - 0.5, fj = y / dx - 0.5;
    int i = static_cast<int>(std::floor(fi)), j = static_cast<int>(std::floor(fj));
    const double a = fi - i, b = fj - j;
    auto clampi = [&](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
    const int i0 = clampi(i, nx - 1), i1 = clampi(i + 1, nx - 1);
    const int j0 = clampi(j, ny - 1), j1 = clampi(j + 1, ny - 1);
    const double f00 = f.curr[f.index(i0, j0, 0)], f10 = f.curr[f.index(i1, j0, 0)];
    const double f01 = f.curr[f.index(i0, j1, 0)], f11 = f.curr[f.index(i1, j1, 0)];
    return (1 - a) * (1 - b) * f00 + a * (1 - b) * f10
         + (1 - a) * b * f01 + a * b * f11;
}

Angles measureAngles(const ScalarField& p0, const ScalarField& p1,
                     const ScalarField& p2, double dx, double radius) {
    const int nx = p0.mesh.n[0], ny = p0.mesh.n[1];
    // Junction position as the centroid of w = phi0 phi1 phi2, which peaks
    // sharply where all three phases coexist.  A sub-cell estimate matters:
    // picking the best single cell biases the sampling circle by up to half a
    // cell and breaks the mirror symmetry the answer must have.
    // Junction position from the PEAK of w = phi0 phi1 phi2, refined to sub-cell
    // accuracy by a parabolic fit in each direction.
    //
    // A centroid of w does not work here: the pairwise double-well potential
    // lets a small amount of the third phase survive along a binary interface
    // ("third-phase ghost"), so w has a long tail down the whole 1-2 interface
    // that drags the centroid tens of cells away from the junction.  A double
    // obstacle potential — what OpenPhase uses — suppresses this by construction.
    auto W = [&](int i, int j) {
        return p0.curr[p0.index(i, j, 0)] * p1.curr[p1.index(i, j, 0)]
             * p2.curr[p2.index(i, j, 0)];
    };
    double wmax = -1.0; int bi = 0, bj = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double w = W(i, j);
            if (w > wmax) { wmax = w; bi = i; bj = j; }
        }
    auto refine = [](double fm, double f0, double fp) {
        const double d = fm - 2.0 * f0 + fp;
        return (std::fabs(d) > 1e-30) ? 0.5 * (fm - fp) / d : 0.0;
    };
    double sx = 0.0, sy = 0.0;
    if (bi > 0 && bi < nx - 1) sx = refine(W(bi - 1, bj), wmax, W(bi + 1, bj));
    if (bj > 0 && bj < ny - 1) sy = refine(W(bi, bj - 1), wmax, W(bi, bj + 1));
    sx = std::max(-1.0, std::min(1.0, sx));
    sy = std::max(-1.0, std::min(1.0, sy));
    const double jx = (bi + 0.5 + sx) * dx, jy = (bj + 0.5 + sy) * dx;
    if (getenv("MJ_DEBUG"))
        std::printf("      [dbg] w_max=%.4e at cell (%d, %d) -> (%.3f, %.3f)\n",
                    wmax, bi, bj, jx, jy);

    // Angular extent over which each phase is the largest, on a circle of the
    // given radius around the junction.
    const int nAng = 3600;
    int count[3] = {0, 0, 0};
    for (int a = 0; a < nAng; ++a) {
        const double th = 2.0 * M_PI * a / nAng;
        const double x = jx + radius * std::cos(th);
        const double y = jy + radius * std::sin(th);
        const double v[3] = { sampleAt(p0, x, y, dx),
                              sampleAt(p1, x, y, dx),
                              sampleAt(p2, x, y, dx) };
        int k = (v[1] > v[0]) ? 1 : 0;
        if (v[2] > v[k]) k = 2;
        ++count[k];
    }
    const double deg = 360.0 / nAng;
    return { count[0] * deg, count[1] * deg, count[2] * deg, jx, jy };
}

Angles runOnce(int N, int nSteps, double s12, double* msPerStep, bool verbose = false) {
    // Lattice units: dx = 1, sigma_01 = sigma_02 = 1, sigma_12 = s12, L = 1.
    const double dx = 1.0;
    const char* ellEnv = getenv("MJ_ELL");
    const double ell = ellEnv ? atof(ellEnv) : 5.0;   // interface width, cells
    const double s01 = 1.0, s02 = 1.0;
    const double eps2_01 = 1.5 * s01 * ell, W_01 = 12.0 * s01 / ell;
    const double eps2_02 = 1.5 * s02 * ell, W_02 = 12.0 * s02 / ell;
    const double eps2_12 = 1.5 * s12 * ell, W_12 = 12.0 * s12 / ell;
    const double L01 = 1.0, L02 = 1.0, L12 = 1.0;

    const double eps2max = std::max({eps2_01, eps2_02, eps2_12});
    const double Wmax    = std::max({W_01, W_02, W_12});
    const double dt = 0.2 * std::min(dx * dx / (4.0 * eps2max), 1.0 / Wmax);

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN, N, dx, 0.0, N, dx, 0.0);

    // Initial layout: phase 0 on top, phases 1 and 2 side by side below,
    // giving a junction at (N/2, 0.6 N) with angles 180 / 90 / 90 that must
    // relax to the Young values.
    const double yTop = 0.60 * N, xMid = 0.50 * N, wIfc = 0.5 * ell;
    auto smooth = [&](double d) { return 0.5 * (1.0 + std::tanh(d / wIfc)); };

    ScalarField phi0(mesh, "phi0", 1), phi1(mesh, "phi1", 1), phi2(mesh, "phi2", 1);
    phi0.initialize([&](double x, double y, double) { return smooth(y - yTop); });
    phi1.initialize([&](double x, double y, double) {
        return smooth(yTop - y) * smooth(xMid - x); });
    phi2.initialize([&](double x, double y, double) {
        return smooth(yTop - y) * smooth(x - xMid); });
    // Normalise onto the Gibbs simplex.
    for (std::size_t i = 0; i < phi0.storedSize; ++i) {
        const double s = phi0.curr[i] + phi1.curr[i] + phi2.curr[i];
        if (s > 0) { phi0.curr[i] /= s; phi1.curr[i] /= s; phi2.curr[i] /= s; }
    }
    ScalarField mu0(mesh, "mu0", 1), mu1(mesh, "mu1", 1), mu2(mesh, "mu2", 1);
    mu0.fill(0.0); mu1.fill(0.0); mu2.fill(0.0);
    for (ScalarField* f : {&phi0, &phi1, &phi2, &mu0, &mu1, &mu2}) {
        f->allocDevice(); f->uploadAllToDevice();
    }

    // Top and bottom are no-flux; the side walls are pinned by k_pin_side_walls.
    std::vector<NoFluxBC> bcStore;
    bcStore.reserve(2);
    bcStore.emplace_back(mesh.facePatch(Axis::Y, Side::LOW));
    bcStore.emplace_back(mesh.facePatch(Axis::Y, Side::HIGH));
    std::vector<BoundaryCondition*> bcs;
    for (auto& b : bcStore) bcs.push_back(&b);

    const int sx = phi0.storedDims[0], sy = phi0.storedDims[1];
    const int gg = phi0.ghost, planeOff = static_cast<int>(phi0.planeOffset());
    const dim3 blk(128), grd((sy + 127) / 128);
    auto pinWalls = [&]() {
        k_pin_side_walls<<<grd, blk>>>(phi0.d_curr, phi1.d_curr, phi2.d_curr,
                                       sx, sy, gg, planeOff, dx, yTop, wIfc);
    };

    // mu_i = 2 phi_i sum_j W_ij phi_j^2
    //      + sum_j eps2_ij [ 2 phi_i |grad phi_j|^2 - 2 phi_j (grad phi_i . grad phi_j)
    //                        + phi_i phi_j lap phi_j - phi_j^2 lap phi_i ]
    auto muExpr = [&](const ScalarField& pi, const ScalarField& pj, const ScalarField& pk,
                      double Wij, double e2ij, double Wik, double e2ik) {
        return
            fpw2(pi, pj, PHIX_FN(double a, double b) { return 2.0 * a * b * b; }) * Wij +
            fpw2(pi, pk, PHIX_FN(double a, double b) { return 2.0 * a * b * b; }) * Wik +
            fmul(ffield(pi), fgrad_dot(pj, pj)) * ( 2.0 * e2ij) +
            fmul(ffield(pj), fgrad_dot(pi, pj)) * (-2.0 * e2ij) +
            fmul(fmul(ffield(pi), ffield(pj)), flap(pj)) * ( e2ij) +
            fmul(fmul(ffield(pj), ffield(pj)), flap(pi)) * (-e2ij) +
            fmul(ffield(pi), fgrad_dot(pk, pk)) * ( 2.0 * e2ik) +
            fmul(ffield(pk), fgrad_dot(pi, pk)) * (-2.0 * e2ik) +
            fmul(fmul(ffield(pi), ffield(pk)), flap(pk)) * ( e2ik) +
            fmul(fmul(ffield(pk), ffield(pk)), flap(pi)) * (-e2ik);
    };
    auto e_mu0 = muExpr(phi0, phi1, phi2, W_01, eps2_01, W_02, eps2_02);
    auto e_mu1 = muExpr(phi1, phi0, phi2, W_01, eps2_01, W_12, eps2_12);
    auto e_mu2 = muExpr(phi2, phi0, phi1, W_02, eps2_02, W_12, eps2_12);

    Equation eq0(phi0, "phi0"), eq1(phi1, "phi1"), eq2(phi2, "phi2");
    eq0.setRHS(pw(mu0, PHIX_FN(double m) { return (-L01 - L02) * m; })
             + pw(mu1, PHIX_FN(double m) { return   L01        * m; })
             + pw(mu2, PHIX_FN(double m) { return   L02        * m; }));
    eq1.setRHS(pw(mu0, PHIX_FN(double m) { return   L01        * m; })
             + pw(mu1, PHIX_FN(double m) { return (-L01 - L12) * m; })
             + pw(mu2, PHIX_FN(double m) { return   L12        * m; }));
    eq2.setRHS(pw(mu0, PHIX_FN(double m) { return   L02        * m; })
             + pw(mu1, PHIX_FN(double m) { return   L12        * m; })
             + pw(mu2, PHIX_FN(double m) { return (-L02 - L12) * m; }));

    EquationSystem sys(dt, TimeScheme::EULER);
    sys.add(eq0, bcs); sys.add(eq1, bcs); sys.add(eq2, bcs);

    pinWalls();
    cudaDeviceSynchronize();
    perf::WallTimer wall;
    const int probe = nSteps / 5;
    for (int step = 0; step < nSteps; ++step) {
        // Three chemical potentials, one fused kernel launch.
        fuse_multi_compute(phi0, mu0, e_mu0, mu1, e_mu1, mu2, e_mu2);
        sys.advance();
        pinWalls();
        if (verbose && probe > 0 && (step + 1) % probe == 0) {
            cudaDeviceSynchronize();
            phi0.downloadCurrFromDevice();
            phi1.downloadCurrFromDevice();
            phi2.downloadCurrFromDevice();
            const char* rr0 = getenv("MJ_RADIUS");
            Angles h = measureAngles(phi0, phi1, phi2, dx, (rr0?atof(rr0):2.5) * ell);
            std::printf("      [t] step %7d  th = %8.3f %8.3f %8.3f   "
                        "junction (%.2f, %.2f)\n",
                        step + 1, h.t0, h.t1, h.t2, h.junctionX, h.junctionY);
            std::fflush(stdout);
        }
    }
    cudaDeviceSynchronize();
    *msPerStep = wall.seconds() * 1e3 / nSteps;

    phi0.downloadCurrFromDevice();
    phi1.downloadCurrFromDevice();
    phi2.downloadCurrFromDevice();
    // MJ_DUMP=<prefix>: write the final phase fields (paper figure fig_junction)
    if (const char* dp = getenv("MJ_DUMP")) {
        IO::writeField(phi0, std::string(dp) + "_phi0.dat", FieldFormat::DAT);
        IO::writeField(phi1, std::string(dp) + "_phi1.dat", FieldFormat::DAT);
        IO::writeField(phi2, std::string(dp) + "_phi2.dat", FieldFormat::DAT);
    }
    { const char* rr = getenv("MJ_RADIUS");
        const double rmul = rr ? atof(rr) : 2.5;
        return measureAngles(phi0, phi1, phi2, dx, rmul * ell); }
}

}  // namespace

int main(int argc, char** argv) {
try {
    const int N      = (argc > 1) ? std::atoi(argv[1]) : 256;
    const int nSteps = (argc > 2) ? std::atoi(argv[2]) : 20000;
    std::vector<double> sweep;
    for (int i = 3; i < argc; ++i) sweep.push_back(std::atof(argv[i]));
    if (sweep.empty()) sweep = {0.6, 0.8, 1.0, 1.2, 1.5};

    std::printf("multijunction2d — triple-junction angles vs Young's law\n");
    std::printf("  N=%d  steps=%d  sigma_01 = sigma_02 = 1, sigma_12 = s\n", N, nSteps);
    std::printf("  closed form for this symmetric family:  theta_1 = theta_2 = "
                "acos(-s/2),  theta_0 = 360 - 2 theta\n\n");
    std::printf("%6s %11s %11s %11s %11s %11s %11s %9s\n",
                "s", "th0 meas", "th0 exact", "th1 meas", "th2 meas",
                "th exact", "max err", "ms/step");

    for (double s : sweep) {
        if (s >= 2.0) { std::printf("%6.2f   (s >= 2: no equilibrium junction)\n", s); continue; }
        double ms = 0.0;
        Angles a = runOnce(N, nSteps, s, &ms, true);
        const double thEx  = std::acos(-0.5 * s) * 180.0 / M_PI;
        const double th0Ex = 360.0 - 2.0 * thEx;
        const double err = std::max({std::fabs(a.t0 - th0Ex),
                                     std::fabs(a.t1 - thEx),
                                     std::fabs(a.t2 - thEx)});
        std::printf("%6.2f %11.3f %11.3f %11.3f %11.3f %11.3f %11.3f %9.4f\n",
                    s, a.t0, th0Ex, a.t1, a.t2, thEx, err, ms);
        std::fflush(stdout);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "multijunction2d FAILED: %s\n", e.what());
    return 1;
}
}
