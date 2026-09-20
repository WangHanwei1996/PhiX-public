// Test face-centred operators: interp, faceGrad, divFace (CPU paths)

#include "operators/FaceOps.h"
#include "field/ScalarField.h"
#include "field/FaceField.h"
#include "mesh/Mesh.h"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static bool near(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) < tol;
}

// ---------------------------------------------------------------------------
// 1D test: interp on a linear field f(x) = x
//    cell centres: x = 0.5, 1.5, 2.5   (dx = 1.0, origin = 0.0, n = 3)
//    face positions: x = 0, 1, 2, 3
//    expected interp: f_face = { 0.5, 1.0, 2.0, 2.5 }
//        (boundaries clamped to nearest cell)
// ---------------------------------------------------------------------------
static void test_interp_1d() {
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 3, 1.0, 0.0);

    ScalarField phi(m, "phi", 1);
    // fill: physical cells and ghost cells
    for (int i = -1; i <= m.n[0]; ++i) {
        double x = m.origin[0] + (i + 0.5) * m.d[0];
        phi.curr[phi.index(i)] = x;
    }

    FaceField fface(m, 0, "fx");
    interp(phi, 0, fface);

    // Interior faces: 1, 2 — average of neighbours
    require(near(fface.data[fface.index(1)], 1.0), "interp 1D face 1");
    require(near(fface.data[fface.index(2)], 2.0), "interp 1D face 2");
    // Boundary faces: 0 and 3 — nearest cell value
    require(near(fface.data[fface.index(0)], 0.5), "interp 1D face 0 (clamped)");
    require(near(fface.data[fface.index(3)], 2.5), "interp 1D face 3 (clamped)");
}

// ---------------------------------------------------------------------------
// 1D test: faceGrad on f(x) = x
//    cell centres: 0.5, 1.5, 2.5  (dx=1.0)
//    ghost cells filled: x = -0.5 (left), 3.5 (right)
//    expected gradient at every face: 1.0
// ---------------------------------------------------------------------------
static void test_face_grad_1d() {
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 3, 1.0, 0.0);

    ScalarField phi(m, "phi", 1);
    for (int i = -1; i <= m.n[0]; ++i) {
        double x = m.origin[0] + (i + 0.5) * m.d[0];
        phi.curr[phi.index(i)] = x;
    }

    FaceField fface(m, 0, "gx");
    faceGrad(phi, 0, fface);

    for (int i = 0; i <= m.n[0]; ++i)
        require(near(fface.data[fface.index(i)], 1.0),
                "faceGrad 1D face " + std::to_string(i));
}

// ---------------------------------------------------------------------------
// 1D test: divFace on constant flux F = 5.0 → divergence = 0
// ---------------------------------------------------------------------------
static void test_div_face_zero_1d() {
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 4, 1.0, 0.0);

    FaceField fx(m, 0, "fx");
    fx.fill(5.0);

    Term dt = divFace(fx);

    ScalarField rhs(m, "rhs", 1);
    std::fill(rhs.curr.begin(), rhs.curr.end(), 0.0);
    ScratchPool pool;
    dt.cpu_kernel(rhs.curr.data(), dt.coeff, pool);

    for (int i = 0; i < m.n[0]; ++i)
        require(near(rhs.curr[rhs.index(i)], 0.0),
                "divFace constant flux → 0, cell " + std::to_string(i));
}

// ---------------------------------------------------------------------------
// 1D test: divFace on linear flux F(x_face) = x_face
//    F_face[i] = i * dx   (dx=1.0)
//    div = (F[i+1] - F[i]) / dx = ((i+1) - i) = 1.0 for all cells
// ---------------------------------------------------------------------------
static void test_div_face_linear_1d() {
    Mesh m = Mesh::makeUniform1D(CoordSys::CARTESIAN, 4, 1.0, 0.0);

    FaceField fx(m, 0, "fx");
    for (int i = 0; i <= m.n[0]; ++i)
        fx.data[fx.index(i)] = static_cast<double>(i) * m.d[0];

    Term dt = divFace(fx);

    ScalarField rhs(m, "rhs", 1);
    std::fill(rhs.curr.begin(), rhs.curr.end(), 0.0);
    ScratchPool pool;
    dt.cpu_kernel(rhs.curr.data(), dt.coeff, pool);

    for (int i = 0; i < m.n[0]; ++i)
        require(near(rhs.curr[rhs.index(i)], 1.0),
                "divFace linear flux → 1.0, cell " + std::to_string(i));
}

// ---------------------------------------------------------------------------
// 2D test: divFace — combine x and y fluxes; check specific cell
//   Mesh: 3x3, dx=dy=1.0
//   F_x(x_face, j) = x_face            → div_x = 1.0 everywhere
//   F_y(i, y_face) = y_face            → div_y = 1.0 everywhere
//   expected div = 2.0 at every cell
// ---------------------------------------------------------------------------
static void test_div_face_2d() {
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                  3, 1.0, 0.0,
                                  3, 1.0, 0.0);

    FaceField flux_x(m, 0, "fx"), flux_y(m, 1, "fy");

    // F_x[i_face, j] = i_face * dx
    for (int j = 0; j < m.n[1]; ++j)
        for (int i = 0; i <= m.n[0]; ++i)
            flux_x.data[flux_x.index(i, j)] = static_cast<double>(i) * m.d[0];

    // F_y[i, j_face] = j_face * dy
    for (int i = 0; i < m.n[0]; ++i)
        for (int j = 0; j <= m.n[1]; ++j)
            flux_y.data[flux_y.index(i, j)] = static_cast<double>(j) * m.d[1];

    Term dt = divFace(flux_x, flux_y);

    ScalarField rhs(m, "rhs", 1);
    std::fill(rhs.curr.begin(), rhs.curr.end(), 0.0);
    ScratchPool pool;
    dt.cpu_kernel(rhs.curr.data(), dt.coeff, pool);

    for (int j = 0; j < m.n[1]; ++j)
        for (int i = 0; i < m.n[0]; ++i)
            require(near(rhs.curr[rhs.index(i, j)], 2.0),
                    "divFace 2D: cell (" + std::to_string(i) + "," + std::to_string(j) + ")");
}

// ---------------------------------------------------------------------------
// 2D test: interp in y direction
//   f(x, y) = y_cell  (cell centre at j + 0.5 for origin=0, dy=1)
//   expected face value at j_face=1: 0.5*(0.5+1.5)=1.0
// ---------------------------------------------------------------------------
static void test_interp_y_2d() {
    Mesh m = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                  2, 1.0, 0.0,
                                  4, 1.0, 0.0);

    ScalarField phi(m, "phi", 1);
    for (int j = -1; j <= m.n[1]; ++j)
        for (int i = -1; i <= m.n[0]; ++i)
            phi.curr[phi.index(i, j)] = (j + 0.5) * m.d[1];

    FaceField fy(m, 1, "fy");
    interp(phi, 1, fy);

    // Interior face j=1: (0.5+1.5)/2 = 1.0
    require(near(fy.data[fy.index(0, 1)], 1.0), "interp Y face j=1, i=0");
    require(near(fy.data[fy.index(1, 1)], 1.0), "interp Y face j=1, i=1");
    // Boundary face j=0: clamped to j=-1... wait, j=0 → nearest right cell is j=0 value
    require(near(fy.data[fy.index(0, 0)], 0.5), "interp Y face j=0 (clamped)");
    require(near(fy.data[fy.index(0, 4)], 3.5), "interp Y face j=4 (clamped)");
}

int main() {
    test_interp_1d();
    test_face_grad_1d();
    test_div_face_zero_1d();
    test_div_face_linear_1d();
    test_div_face_2d();
    test_interp_y_2d();
    return 0;
}
