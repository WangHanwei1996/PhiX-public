// ---------------------------------------------------------------------------
// module_mask — DomainMask: non-rectangular domains (PFHub BM1c T-shape)
//
// 1. T-shape cell count matches the analytic count; ghosts are inactive.
// 2. maskFaces: every face NOT between two active cells is zeroed; faces
//    between two active cells are untouched.
// 3. Conservation: explicit diffusion on the T-shape through the
//    conservative face-flux chain (faceGradGPU → maskFaces → divFace).
//    Σc over active cells is conserved to machine precision, inactive
//    cells stay exactly 0, and the field relaxes towards the active-region
//    mean.
// 4. applyClosure: a constant active field stays an equilibrium of the
//    CD2 laplacian at the mask boundary (mirror = zero normal gradient).
// 5. Masked sum ignores poisoned inactive/ghost values.
// ---------------------------------------------------------------------------
#include "field/DomainMask.h"
#include "field/ScalarField.h"
#include "field/FaceField.h"
#include "operators/FaceOps.h"
#include "equation/Equation.h"
#include "boundary/NoFluxBC.h"
#include "boundary/BoundaryCondition.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    const int    N  = 64;
    const double dx = 1.0 / N;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    N, dx, 0.0, N, dx, 0.0);

    // T-shape: horizontal bar y ∈ [0.75, 1] full width, vertical stem
    // x ∈ [0.375, 0.625], y ∈ [0, 0.75]
    auto inT = [](double x, double y, double) {
        if (y > 0.75) return true;
        return x > 0.375 && x < 0.625;
    };
    DomainMask mask(mesh, inT, 1);

    // ---- 1. cell count -----------------------------------------------------
    {
        // bar: 64×16 rows (y-index 48..63); stem: 16 columns × 48 rows
        const long long expect = 64LL * 16 + 16LL * 48;
        require(mask.activeCells() == expect,
                "active cell count " + std::to_string(mask.activeCells())
                + " != " + std::to_string(expect));
        const ScalarField& m = mask.cellMask();
        require(m.curr[static_cast<std::size_t>(m.index(-1, 0))] == Real(0)
                && m.curr[static_cast<std::size_t>(m.index(0, N))] == Real(0),
                "mask ghosts not inactive");
        std::printf("  [1] T-shape: %lld active cells\n", mask.activeCells());
    }

    // ---- 2. maskFaces zeroes exactly the right faces -----------------------
    {
        FaceField Fx(mesh, 0, "Fx", 1);
        for (Real& v : Fx.data) v = Real(7);
        Fx.allocDevice();
        Fx.uploadToDevice();
        mask.maskFaces(Fx);
        Fx.downloadFromDevice();

        const ScalarField& m = mask.cellMask();
        auto active = [&](int i, int j) {
            if (i < 0 || i >= N || j < 0 || j >= N) return false;
            return m.curr[static_cast<std::size_t>(m.index(i, j))]
                   >= Real(0.5);
        };
        int nZero = 0, nKept = 0;
        for (int j = 0; j < N; ++j)
        for (int fi = 0; fi <= N; ++fi) {
            const std::size_t idx = static_cast<std::size_t>(
                fi + Fx.storedDims[0] * (j + Fx.ghost
                     + Fx.storedDims[1] * Fx.ghost));
            const bool keep = active(fi - 1, j) && active(fi, j);
            const Real v = Fx.data[idx];
            require(v == (keep ? Real(7) : Real(0)),
                    "face (" + std::to_string(fi) + "," + std::to_string(j)
                    + ") wrong after maskFaces");
            keep ? ++nKept : ++nZero;
        }
        std::printf("  [2] maskFaces: %d kept, %d zeroed (exact)\n",
                    nKept, nZero);
    }

    // ---- 3. conservative diffusion on the T-shape ---------------------------
    {
        ScalarField c(mesh, "c", 1);
        c.fill(0.0);
        const ScalarField& m = mask.cellMask();
        double s0 = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            if (m.curr[static_cast<std::size_t>(m.index(i, j))] < Real(0.5))
                continue;
            const double v = 0.5 + 0.4 * std::sin(9.0 * i * dx)
                                 * std::cos(7.0 * j * dx);
            c.curr[static_cast<std::size_t>(c.index(i, j))] = v;
            s0 += v;
        }
        c.allocDevice();
        c.uploadAllToDevice();

        FaceField Fx(mesh, 0, "Fx", 1), Fy(mesh, 1, "Fy", 1);
        Fx.allocDevice();
        Fy.allocDevice();

        NoFluxBC bcXL(mesh.facePatch(Axis::X, Side::LOW));
        NoFluxBC bcXH(mesh.facePatch(Axis::X, Side::HIGH));
        NoFluxBC bcYL(mesh.facePatch(Axis::Y, Side::LOW));
        NoFluxBC bcYH(mesh.facePatch(Axis::Y, Side::HIGH));
        BoundaryCondition* bcs[4] = {&bcXL, &bcXH, &bcYL, &bcYH};
        Equation eq(c, "diff");
        eq.setRHS(divFace(Fx, Fy));

        const double D = 1.0, dt = 0.2 * dx * dx / D;
        for (int step = 0; step < 400; ++step) {
            for (BoundaryCondition* bc : bcs) bc->applyOnGPU(c);
            faceGradGPU(c, 0, Fx);
            faceGradGPU(c, 1, Fy);
            mask.maskFaces(Fx);
            mask.maskFaces(Fy);
            eq.advanceTransient({}, dt);
        }
        c.downloadCurrFromDevice();

        double s1 = 0.0, leak = 0.0, dev = 0.0;
        const double mean = s0 / static_cast<double>(mask.activeCells());
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const double v = c.curr[static_cast<std::size_t>(c.index(i, j))];
            if (m.curr[static_cast<std::size_t>(m.index(i, j))] >= Real(0.5)) {
                s1 += v;
                dev = std::max(dev, std::fabs(v - mean));
            } else {
                leak = std::max(leak, std::fabs(v));
            }
        }
        std::printf("  [3] diffusion: dSum %.2e, leak %.2e,"
                    " relax-to-mean dev %.2e\n",
                    std::fabs(s1 - s0), leak, dev);
        require(std::fabs(s1 - s0) < 1e-10 * std::fabs(s0),
                "mass not conserved on masked domain");
        require(leak == 0.0, "diffusion leaked into inactive cells");
        require(dev < 0.25, "field not relaxing towards active-region mean");
    }

    // ---- 4. closure keeps a constant in equilibrium -------------------------
    {
        ScalarField f(mesh, "f", 1);
        f.fill(0.0);
        const ScalarField& m = mask.cellMask();
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            if (m.curr[static_cast<std::size_t>(m.index(i, j))] >= Real(0.5))
                f.curr[static_cast<std::size_t>(f.index(i, j))] = Real(3.25);
        f.allocDevice();
        f.uploadAllToDevice();
        mask.applyClosure(f);
        NoFluxBC bcXL(mesh.facePatch(Axis::X, Side::LOW));
        NoFluxBC bcXH(mesh.facePatch(Axis::X, Side::HIGH));
        NoFluxBC bcYL(mesh.facePatch(Axis::Y, Side::LOW));
        NoFluxBC bcYH(mesh.facePatch(Axis::Y, Side::HIGH));
        for (BoundaryCondition* bc : std::initializer_list<BoundaryCondition*>{
                 &bcXL, &bcXH, &bcYL, &bcYH})
            bc->applyOnGPU(f);
        f.downloadCurrFromDevice();

        auto V = [&](int i, int j) {
            return static_cast<double>(f.curr[static_cast<std::size_t>(
                f.index(i, j))]);
        };
        double lapMax = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            if (m.curr[static_cast<std::size_t>(m.index(i, j))] < Real(0.5))
                continue;
            const double lap = (V(i + 1, j) + V(i - 1, j) + V(i, j + 1)
                                + V(i, j - 1) - 4.0 * V(i, j)) / (dx * dx);
            lapMax = std::max(lapMax, std::fabs(lap));
        }
        std::printf("  [4] closure: max|lap(const)| = %.2e\n", lapMax);
        require(lapMax < 1e-10,
                "constant field not an equilibrium with closure");
    }

    // ---- 5. masked sum with poisoned outside values -------------------------
    {
        ScalarField f(mesh, "f", 1);
        f.fill(1e150);                    // poison everything incl. ghosts
        const ScalarField& m = mask.cellMask();
        double ref = 0.0;
        for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            if (m.curr[static_cast<std::size_t>(m.index(i, j))] < Real(0.5))
                continue;
            const double v = 0.01 * i - 0.02 * j + 1.0;
            f.curr[static_cast<std::size_t>(f.index(i, j))] = v;
            ref += v;
        }
        f.allocDevice();
        f.uploadAllToDevice();
        const double s = mask.sum(f);
        std::printf("  [5] masked sum: %.12e (ref %.12e)\n", s, ref);
        require(std::fabs(s - ref) < 1e-9 * std::fabs(ref),
                "masked sum wrong");
    }

    std::printf("module_mask: ALL PASS\n");
    return 0;
}
