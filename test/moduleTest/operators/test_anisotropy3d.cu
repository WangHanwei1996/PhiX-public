// ---------------------------------------------------------------------------
// module_anisotropy3d — 3D cubic anisotropy (operators/Anisotropy.h)
//
// 1. ε = 0 reduces exactly to W0²·∇²φ (CD2, 3D).
// 2. CROSS-VALIDATION: on a z-invariant field the 3D cubic form reduces
//    analytically to the 2D m=4 form with the same ε — anisoDiv3D must
//    match the (independently implemented) 2D anisoDiv per cell.
// 3. Conservation: divergence form + centred blob ⇒ domain sum ≈ 0.
// 4. GPU Term path == CPU Term path; factor3D GPU == CPU.
// 5. a(n) direction point checks: <100> → 1+ε, <110> → 1−ε,
//    <111> → 1−(5/3)ε, ε=0 → 1.
// ---------------------------------------------------------------------------

#include "operators/Anisotropy.h"
#include "operators/Laplacian.h"
#include "equation/Equation.h"
#include "field/ScalarField.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

template<typename Fn>
static void fillAll3D(ScalarField& f, Fn fn) {
    const int g = f.ghost;
    for (int k = -g; k < f.mesh.n[2] + g; ++k)
    for (int j = -g; j < f.mesh.n[1] + g; ++j)
    for (int i = -g; i < f.mesh.n[0] + g; ++i)
        f.curr[static_cast<std::size_t>(f.index(i, j, k))] =
            fn(f.mesh.coord(0, i), f.mesh.coord(1, j), f.mesh.coord(2, k));
    if (!f.deviceAllocated()) f.allocDevice();
    f.uploadCurrToDevice();
}

static double blob2(double x, double y) {
    const double r  = std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5));
    const double th = std::atan2(y - 0.5, x - 0.5);
    const double R  = 0.22 * (1.0 + 0.08 * std::cos(4.0 * th));
    return 0.5 * (1.0 - std::tanh((r - R) / 0.05));
}

int main() {
    const int    N  = 32;
    const double dx = 1.0 / N;

    // =======================================================================
    // 1. ε = 0  ≡  W0²·lap (CD2, 3D)
    // =======================================================================
    {
        Mesh mesh = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                        N, dx, 0.0, N, dx, 0.0, N, dx, 0.0);
        ScalarField phi(mesh, "phi", 1);
        fillAll3D(phi, [](double x, double y, double z) {
            return std::sin(6.28 * x) * std::cos(6.28 * y)
                 + 0.5 * std::sin(6.28 * z) + 0.2 * x * y;
        });

        Aniso3DParams p;
        p.W0 = 1.4;
        p.eps = 0.0;

        ScalarField rA(mesh, "rA", 1), rL(mesh, "rL", 1);
        rA.allocDevice(); rL.allocDevice();

        Equation eA(phi, "a");
        eA.setRHS(anisoDiv3D(phi, p));
        eA.computeRHS(rA);
        Equation eL(phi, "l");
        eL.setRHS(lap(phi, p.W0 * p.W0));
        eL.computeRHS(rL);

        rA.downloadCurrFromDevice();
        rL.downloadCurrFromDevice();
        double dev = 0.0;
        for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const std::size_t idx = static_cast<std::size_t>(rA.index(i, j, k));
            dev = std::max(dev, std::fabs(
                static_cast<double>(rA.curr[idx]) - rL.curr[idx]));
        }
        require(dev < 1e-9, "eps=0 does not reduce to W0^2 * laplacian (3D): "
                            + std::to_string(dev));
    }

    // =======================================================================
    // 2. z-invariant field: 3D cubic == 2D m=4 (same eps), cell by cell
    // =======================================================================
    {
        const double eps = 0.05, W0 = 1.2;
        const int    NZ  = 6;

        Mesh m3 = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                      N, dx, 0.0, N, dx, 0.0, NZ, dx, 0.0);
        Mesh m2 = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                      N, dx, 0.0, N, dx, 0.0);

        ScalarField p3(m3, "p3", 1), p2f(m2, "p2", 1);
        fillAll3D(p3, [](double x, double y, double) { return blob2(x, y); });
        const int g = p2f.ghost;
        for (int j = -g; j < N + g; ++j)
        for (int i = -g; i < N + g; ++i)
            p2f.curr[static_cast<std::size_t>(p2f.index(i, j))] =
                blob2(m2.coord(0, i), m2.coord(1, j));
        p2f.allocDevice();
        p2f.uploadCurrToDevice();

        Aniso3DParams a3;
        a3.W0 = W0; a3.eps = eps;
        AnisoParams a2;
        a2.W0 = W0; a2.eps = eps; a2.m = 4; a2.theta0 = 0.0;

        ScalarField r3(m3, "r3", 1), r2(m2, "r2", 1);
        r3.allocDevice(); r2.allocDevice();

        Equation e3(p3, "e3");
        e3.setRHS(anisoDiv3D(p3, a3));
        e3.computeRHS(r3);
        Equation e2(p2f, "e2");
        e2.setRHS(anisoDiv(p2f, a2));
        e2.computeRHS(r2);

        r3.downloadCurrFromDevice();
        r2.downloadCurrFromDevice();
        double dev = 0.0, scale = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double v2 = r2.curr[static_cast<std::size_t>(r2.index(i, j))];
            scale = std::max(scale, std::fabs(v2));
            for (int k = 0; k < NZ; ++k)
                dev = std::max(dev, std::fabs(
                    static_cast<double>(r3.curr[static_cast<std::size_t>(
                        r3.index(i, j, k))]) - v2));
        }
        std::printf("  3D-vs-2D (z-invariant) max dev: %.2e (scale %.2e)\n",
                    dev, scale);
        require(dev < 1e-9 * std::max(scale, 1.0),
                "3D cubic does not reduce to 2D m=4 on a z-invariant field: "
                + std::to_string(dev));
    }

    // =======================================================================
    // 3. Conservation (centred 3D blob, gradients vanish at the boundary)
    // =======================================================================
    {
        Mesh mesh = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                        N, dx, 0.0, N, dx, 0.0, N, dx, 0.0);
        // COMPACTLY supported C∞ bump: exactly zero for r >= 0.35, so every
        // boundary/ghost face flux is exactly zero and the domain sum tests
        // pure interior telescoping (a tanh tail would leave a legitimate
        // O(boundary-flux) residue instead).
        ScalarField phi(mesh, "phi", 1);
        fillAll3D(phi, [](double x, double y, double z) {
            const double r2 = ((x - 0.5) * (x - 0.5)
                               + (y - 0.5) * (y - 0.5)
                               + (z - 0.5) * (z - 0.5)) / (0.35 * 0.35);
            return (r2 < 1.0) ? std::exp(1.0 - 1.0 / (1.0 - r2)) : 0.0;
        });

        Aniso3DParams p;
        p.W0 = 1.0; p.eps = 0.05;

        ScalarField rhs(mesh, "rhs", 1);
        rhs.allocDevice();
        Equation e(phi, "cons");
        e.setRHS(anisoDiv3D(phi, p));
        e.computeRHS(rhs);
        rhs.downloadCurrFromDevice();

        double sum = 0.0, scale = 0.0;
        for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double v = rhs.curr[static_cast<std::size_t>(
                rhs.index(i, j, k))];
            sum += v;
            scale = std::max(scale, std::fabs(v));
        }
        require(std::fabs(sum) < 1e-9 * std::max(scale, 1.0) * N * N,
                "3D anisotropic divergence not conservative: sum = "
                + std::to_string(sum));
    }

    // =======================================================================
    // 4. GPU == CPU (Term paths + factor field)
    // =======================================================================
    {
        Mesh mesh = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                        20, dx, 0.0, 20, dx, 0.0, 20, dx, 0.0);
        ScalarField phi(mesh, "phi", 1);
        fillAll3D(phi, [](double x, double y, double z) {
            return std::sin(5.0 * x + 1.0) * std::cos(4.0 * y)
                 * std::cos(3.0 * z + 0.5);
        });

        Aniso3DParams p;
        p.W0 = 1.1; p.eps = 0.04;

        ScalarField rG(mesh, "rG", 1), rC(mesh, "rC", 1);
        rG.allocDevice();
        Equation e(phi, "x");
        e.setRHS(anisoDiv3D(phi, p));
        e.computeRHS(rG);
        rC.fillCurr(0.0);
        e.computeRHSCPU(rC);
        rG.downloadCurrFromDevice();
        double dev = 0.0;
        for (int k = 0; k < 20; ++k)
        for (int j = 0; j < 20; ++j)
        for (int i = 0; i < 20; ++i) {
            const std::size_t idx = static_cast<std::size_t>(rG.index(i, j, k));
            dev = std::max(dev, std::fabs(
                static_cast<double>(rG.curr[idx]) - rC.curr[idx]));
        }
        require(dev < 1e-11, "anisoDiv3D GPU != CPU: " + std::to_string(dev));

        ScalarField aG(mesh, "aG", 1), aC(mesh, "aC", 1);
        aG.allocDevice();
        anisoFactor3DOnGPU(phi, aG, p);
        anisoFactor3DOnCPU(phi, aC, p);
        aG.downloadCurrFromDevice();
        dev = 0.0;
        for (int k = 0; k < 20; ++k)
        for (int j = 0; j < 20; ++j)
        for (int i = 0; i < 20; ++i) {
            const std::size_t idx = static_cast<std::size_t>(aG.index(i, j, k));
            dev = std::max(dev, std::fabs(
                static_cast<double>(aG.curr[idx]) - aC.curr[idx]));
        }
        require(dev < 1e-13, "anisoFactor3D GPU != CPU");
    }

    // =======================================================================
    // 5. a(n) direction point checks + validation
    // =======================================================================
    {
        const double e = 0.07;
        require(std::fabs(aniso::factor3D(1, 0, 0, e) - (1.0 + e)) < 1e-14,
                "<100> factor wrong");
        require(std::fabs(aniso::factor3D(1, 1, 0, e) - (1.0 - e)) < 1e-14,
                "<110> factor wrong");
        require(std::fabs(aniso::factor3D(1, 1, 1, e)
                          - (1.0 - 5.0 / 3.0 * e)) < 1e-14,
                "<111> factor wrong");
        require(std::fabs(aniso::factor3D(0.3, -0.7, 0.2, 0.0) - 1.0) < 1e-15,
                "eps=0 factor != 1");

        bool threw = false;
        Aniso3DParams bad;
        bad.eps = 0.5;
        try { bad.validate(); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "eps >= 0.3 did not throw");

        // ---- rotation: cubic-group invariance + genuine effect ----------
        {
            Mesh mesh = Mesh::makeUniform3D(CoordSys::CARTESIAN,
                                            16, 1.0/16, 0.0, 16, 1.0/16, 0.0,
                                            16, 1.0/16, 0.0);
            ScalarField phi(mesh, "phi", 1);
            fillAll3D(phi, [](double x, double y, double z) {
                return std::sin(4.0*x + 0.3) * std::cos(3.0*y)
                     * std::cos(5.0*z + 0.7);
            });

            auto rhsFor = [&](const Aniso3DParams& ap) {
                ScalarField r(mesh, "r", 1);
                r.allocDevice();
                Equation eq(phi, "rot");
                eq.setRHS(anisoDiv3D(phi, ap));
                eq.computeRHS(r);
                r.downloadCurrFromDevice();
                return r;
            };

            Aniso3DParams id;   id.eps = 0.06;
            Aniso3DParams r90;  r90.eps = 0.06;
            r90.setEulerZXZ(M_PI / 2.0, 0.0, 0.0);   // 90° about z ∈ cubic group
            Aniso3DParams r45;  r45.eps = 0.06;
            r45.setEulerZXZ(M_PI / 4.0, 0.0, 0.0);   // 45° — NOT a symmetry

            ScalarField a = rhsFor(id), b = rhsFor(r90), c45 = rhsFor(r45);
            double devSym = 0.0, devRot = 0.0, scale = 0.0;
            for (int k = 0; k < 16; ++k)
            for (int j = 0; j < 16; ++j)
            for (int i = 0; i < 16; ++i) {
                const std::size_t idx =
                    static_cast<std::size_t>(a.index(i, j, k));
                scale = std::max(scale, std::fabs(
                    static_cast<double>(a.curr[idx])));
                devSym = std::max(devSym, std::fabs(
                    static_cast<double>(a.curr[idx]) - b.curr[idx]));
                devRot = std::max(devRot, std::fabs(
                    static_cast<double>(a.curr[idx]) - c45.curr[idx]));
            }
            require(devSym < 1e-10 * scale,
                    "90° z-rotation (cubic symmetry) changed the result: "
                    + std::to_string(devSym));
            require(devRot > 1e-4 * scale,
                    "45° rotation had no effect — rotation not applied?");

            // rotated factor point check: lab <110> at 45° z-rotation is
            // crystal <100> → a = 1 + ε
            Aniso3DParams pf;
            pf.eps = 0.07;
            pf.setEulerZXZ(M_PI / 4.0, 0.0, 0.0);
            const double inv = 1.0 / std::sqrt(2.0);
            const double px = inv, py = inv, pz = 0.0;
            const double cx = pf.R[0]*px + pf.R[1]*py + pf.R[2]*pz;
            const double cy = pf.R[3]*px + pf.R[4]*py + pf.R[5]*pz;
            const double cz = pf.R[6]*px + pf.R[7]*py + pf.R[8]*pz;
            require(std::fabs(aniso::factor3D(cx, cy, cz, pf.eps)
                              - (1.0 + pf.eps)) < 1e-12,
                    "rotated <110> is not crystal <100>");

            // garbage R must be rejected
            Aniso3DParams badR;
            badR.eps = 0.05;
            badR.R[0] = 2.0;
            bool threwR = false;
            try { badR.validate(); }
            catch (const std::invalid_argument&) { threwR = true; }
            require(threwR, "non-orthonormal R did not throw");
        }

        // 2D mesh must be rejected
        Mesh m2 = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                      8, 0.1, 0.0, 8, 0.1, 0.0);
        ScalarField f2(m2, "f2", 1);
        f2.allocDevice();
        Aniso3DParams ok;
        ok.eps = 0.05;
        threw = false;
        try { anisoDiv3D(f2, ok); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "anisoDiv3D accepted a 2D mesh");
    }

    return 0;
}
