// ---------------------------------------------------------------------------
// test_face_pw.cu — Unit tests for facePW / facePWGPU
//
// Test strategy:
//   Each test has an analytic "expected" value computable by hand,
//   derived from known face-field inputs.  CPU and GPU paths are both
//   exercised and compared against each other.
//
// Tests:
//   1. facePW 1-field (CPU): f_out = 2*a
//   2. facePW 2-field (CPU): f_out = a + b
//   3. facePW 3-field (CPU): f_out = a*b + c
//   4. facePWGPU 1-field:    same as test 1, GPU path round-trip
//   5. facePWGPU 2-field:    same as test 2, GPU path round-trip
//   6. facePWGPU 3-field:    same as test 3, GPU path round-trip
//   7. 2D y-face, 2-field:   confirm correct index mapping for axis=1
//   8. Anisotropic flux (2D, x-face, 2-field):
//        simulate the dendrite J_x assembly:
//          J_x = W0^2 * a * (a*px + eps*m*sin(m*theta)*py)
//        with known inputs phi_x_face = cos(theta0), phi_y_face = sin(theta0)
//        where theta0 = pi/4 (45 deg) and eps=0.05, m=4 → a=1, sin_term=0
//        expected J_x = W0^2 * 1 * (1*cos(pi/4) + 0) = cos(pi/4)
// ---------------------------------------------------------------------------

#include "operators/FaceOps.h"   // pulls in FacePW.h
#include "field/FaceField.h"
#include "mesh/Mesh.h"
#include "equation/Term.h"       // PHIX_FN

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

using namespace PhiX;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void require(bool cond, const std::string& msg)
{
    if (!cond) throw std::runtime_error("FAIL: " + msg);
}

static bool near(double a, double b, double tol = 1e-10)
{
    return std::abs(a - b) < tol;
}

// Fill every element of a FaceField (physical + tangential ghost) with val.
static void fillFace(FaceField& f, double val)
{
    f.fill(val);
}

// Read a face value using physical indices; delegates to FaceField::index.
static double getFace(const FaceField& f, int i, int j = 0, int k = 0)
{
    return f.data[f.index(i, j, k)];
}

// ===========================================================================
// Test 1: facePW 1-field (CPU) — out = 2.0 * a
// ===========================================================================
static void test_cpu_1field_1d()
{
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 4, 1.0, 0.0);

    FaceField a(m, 0, "a"), out(m, 0, "out");
    fillFace(a, 3.0);
    fillFace(out, 0.0);

    facePW(out, a, [](double av) { return 2.0 * av; });

    for (int i = 0; i <= m.n[0]; ++i)
        require(near(getFace(out, i), 6.0),
                "cpu 1-field 1D face " + std::to_string(i));
}

// ===========================================================================
// Test 2: facePW 2-field (CPU) — out = a + b
// ===========================================================================
static void test_cpu_2field_1d()
{
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 4, 1.0, 0.0);

    FaceField a(m, 0, "a"), b(m, 0, "b"), out(m, 0, "out");
    fillFace(a, 5.0);
    fillFace(b, 3.0);
    fillFace(out, 0.0);

    facePW(out, a, b, [](double av, double bv) { return av + bv; });

    for (int i = 0; i <= m.n[0]; ++i)
        require(near(getFace(out, i), 8.0),
                "cpu 2-field 1D face " + std::to_string(i));
}

// ===========================================================================
// Test 3: facePW 3-field (CPU) — out = a*b + c
// ===========================================================================
static void test_cpu_3field_1d()
{
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 4, 1.0, 0.0);

    FaceField a(m, 0, "a"), b(m, 0, "b"), c(m, 0, "c"), out(m, 0, "out");
    fillFace(a, 2.0);
    fillFace(b, 3.0);
    fillFace(c, 1.0);
    fillFace(out, 0.0);

    facePW(out, a, b, c,
           [](double av, double bv, double cv) { return av * bv + cv; });

    for (int i = 0; i <= m.n[0]; ++i)
        require(near(getFace(out, i), 7.0),
                "cpu 3-field 1D face " + std::to_string(i));
}

// ===========================================================================
// Test 4: facePWGPU 1-field — out = 2.0 * a (GPU round-trip)
// ===========================================================================
static void test_gpu_1field_1d()
{
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 4, 1.0, 0.0);

    FaceField a(m, 0, "a"), out(m, 0, "out");
    fillFace(a,  3.0);
    fillFace(out, 0.0);

    a.allocDevice();   a.uploadToDevice();
    out.allocDevice(); out.uploadToDevice();

    facePWGPU(out, a, PHIX_FN (double av) { return 2.0 * av; });

    cudaDeviceSynchronize();
    out.downloadFromDevice();

    for (int i = 0; i <= m.n[0]; ++i)
        require(near(getFace(out, i), 6.0),
                "gpu 1-field 1D face " + std::to_string(i));
}

// ===========================================================================
// Test 5: facePWGPU 2-field — out = a + b (GPU round-trip)
// ===========================================================================
static void test_gpu_2field_1d()
{
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 4, 1.0, 0.0);

    FaceField a(m, 0, "a"), b(m, 0, "b"), out(m, 0, "out");
    fillFace(a, 5.0);
    fillFace(b, 3.0);
    fillFace(out, 0.0);

    a.allocDevice();   a.uploadToDevice();
    b.allocDevice();   b.uploadToDevice();
    out.allocDevice(); out.uploadToDevice();

    facePWGPU(out, a, b, PHIX_FN (double av, double bv) { return av + bv; });

    cudaDeviceSynchronize();
    out.downloadFromDevice();

    for (int i = 0; i <= m.n[0]; ++i)
        require(near(getFace(out, i), 8.0),
                "gpu 2-field 1D face " + std::to_string(i));
}

// ===========================================================================
// Test 6: facePWGPU 3-field — out = a*b + c (GPU round-trip)
// ===========================================================================
static void test_gpu_3field_1d()
{
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 4, 1.0, 0.0);

    FaceField a(m, 0, "a"), b(m, 0, "b"), c(m, 0, "c"), out(m, 0, "out");
    fillFace(a, 2.0);
    fillFace(b, 3.0);
    fillFace(c, 1.0);
    fillFace(out, 0.0);

    a.allocDevice();   a.uploadToDevice();
    b.allocDevice();   b.uploadToDevice();
    c.allocDevice();   c.uploadToDevice();
    out.allocDevice(); out.uploadToDevice();

    facePWGPU(out, a, b, c,
              PHIX_FN (double av, double bv, double cv) { return av * bv + cv; });

    cudaDeviceSynchronize();
    out.downloadFromDevice();

    for (int i = 0; i <= m.n[0]; ++i)
        require(near(getFace(out, i), 7.0),
                "gpu 3-field 1D face " + std::to_string(i));
}

// ===========================================================================
// Test 7: 2D y-face, 2-field (CPU) — confirm axis=1 index mapping
//   Mesh 3x3; a_face[i,j] = i+1, b_face[i,j] = j+1  (constant per column/row)
//   out = a + b → out[i,j] = (i+1) + (j+1)
// ===========================================================================
static void test_cpu_2d_yface_2field()
{
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                  3, 1.0, 0.0,
                                  3, 1.0, 0.0);

    FaceField a(m, 1, "a"), b(m, 1, "b"), out(m, 1, "out");
    fillFace(a, 0.0); fillFace(b, 0.0); fillFace(out, 0.0);

    // Fill a[i, j_face] = i + 1 (varies with x-cell i)
    for (int j = 0; j <= m.n[1]; ++j)
        for (int i = 0; i < m.n[0]; ++i)
            a.data[a.index(i, j)] = static_cast<double>(i + 1);

    // Fill b[i, j_face] = j + 1 (varies with y-face index j)
    for (int j = 0; j <= m.n[1]; ++j)
        for (int i = 0; i < m.n[0]; ++i)
            b.data[b.index(i, j)] = static_cast<double>(j + 1);

    facePW(out, a, b, [](double av, double bv) { return av + bv; });

    for (int j = 0; j <= m.n[1]; ++j)
        for (int i = 0; i < m.n[0]; ++i) {
            double expected = static_cast<double>((i + 1) + (j + 1));
            require(near(getFace(out, i, j), expected),
                    "cpu 2D y-face: i=" + std::to_string(i)
                    + " j=" + std::to_string(j));
        }
}

// ===========================================================================
// Test 8: Anisotropic x-flux assembly (CPU + GPU, 2D, 2-field)
//
//   Simulates the first step of dendrite J_x construction:
//     J_x = W0^2 * a * (a*px + sin_term*py)
//   where a = 1 + eps*cos(m*(theta-theta0)),  sin_term = eps*m*sin(m*(theta-theta0))
//
//   We choose inputs so the result is analytic:
//     phi_x_face = cos(pi/4) for all faces (uniform)
//     phi_y_face = sin(pi/4) for all faces (uniform)
//     theta = atan2(sin(pi/4), cos(pi/4)) = pi/4
//     With m=4, theta0=0:
//       m*(theta-theta0) = pi  →  cos(pi) = -1,  sin(pi) = 0
//       a = 1 + eps*(-1) = 1 - eps
//       sin_term = eps*4*0 = 0
//     J_x = W0^2 * (1-eps) * ((1-eps)*cos(pi/4) + 0)
//          = W0^2 * (1-eps)^2 * cos(pi/4)
// ===========================================================================
static void test_aniso_flux_xface()
{
    const double W0  = 1.0;
    const double eps = 0.05;
    const double m   = 4.0;
    const double th0 = 0.0;
    const double W0sq = W0 * W0;

    const double theta_in = M_PI / 4.0;     // 45 deg
    const double px_in    = std::cos(theta_in);
    const double py_in    = std::sin(theta_in);

    // Analytic expected value
    const double a_exp       = 1.0 + eps * std::cos(m * (theta_in - th0));
    const double sin_term_ex = eps * m * std::sin(m * (theta_in - th0));
    const double jx_exp      = W0sq * a_exp * (a_exp * px_in + sin_term_ex * py_in);

    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                     4, 1.0, 0.0,
                                     4, 1.0, 0.0);

    FaceField phi_x_xf(mesh, 0, "px_xf");
    FaceField phi_y_xf(mesh, 0, "py_xf");
    FaceField jx(mesh, 0, "jx");
    fillFace(phi_x_xf, px_in);
    fillFace(phi_y_xf, py_in);
    fillFace(jx, 0.0);

    // CPU path
    facePW(jx, phi_x_xf, phi_y_xf,
           [W0sq, eps, m, th0](double px, double py) {
               double theta    = atan2(py, px);
               double a        = 1.0 + eps * cos(m * (theta - th0));
               double sin_term = eps * m * sin(m * (theta - th0));
               return W0sq * a * (a * px + sin_term * py);
           });

    // Check all x-faces
    for (int j = 0; j < mesh.n[1]; ++j)
        for (int i = 0; i <= mesh.n[0]; ++i)
            require(near(getFace(jx, i, j), jx_exp, 1e-9),
                    "aniso flux CPU: i=" + std::to_string(i)
                    + " j=" + std::to_string(j));

    // GPU path
    fillFace(jx, 0.0);
    phi_x_xf.allocDevice(); phi_x_xf.uploadToDevice();
    phi_y_xf.allocDevice(); phi_y_xf.uploadToDevice();
    jx.allocDevice();       jx.uploadToDevice();

    facePWGPU(jx, phi_x_xf, phi_y_xf,
              PHIX_FN (double px, double py) {
                  double theta    = atan2(py, px);
                  double a        = 1.0 + eps * cos(m * (theta - th0));
                  double sin_term = eps * m * sin(m * (theta - th0));
                  return W0sq * a * (a * px + sin_term * py);
              });

    cudaDeviceSynchronize();
    jx.downloadFromDevice();

    for (int j = 0; j < mesh.n[1]; ++j)
        for (int i = 0; i <= mesh.n[0]; ++i)
            require(near(getFace(jx, i, j), jx_exp, 1e-9),
                    "aniso flux GPU: i=" + std::to_string(i)
                    + " j=" + std::to_string(j));
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    struct Test { void(*fn)(); const char* name; };
    Test tests[] = {
        { test_cpu_1field_1d,        "cpu 1-field 1D"          },
        { test_cpu_2field_1d,        "cpu 2-field 1D"          },
        { test_cpu_3field_1d,        "cpu 3-field 1D"          },
        { test_gpu_1field_1d,        "gpu 1-field 1D"          },
        { test_gpu_2field_1d,        "gpu 2-field 1D"          },
        { test_gpu_3field_1d,        "gpu 3-field 1D"          },
        { test_cpu_2d_yface_2field,  "cpu 2D y-face 2-field"   },
        { test_aniso_flux_xface,     "anisotropic flux x-face" },
    };

    int failed = 0;
    for (auto& t : tests) {
        try {
            t.fn();
            std::cout << "[PASS] " << t.name << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[FAIL] " << t.name << ": " << e.what() << "\n";
            ++failed;
        }
    }

    if (failed > 0) {
        std::cerr << failed << " test(s) failed.\n";
        return 1;
    }
    std::cout << "All facePW tests passed.\n";
    return 0;
}
