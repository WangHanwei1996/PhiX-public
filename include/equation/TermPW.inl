// ---------------------------------------------------------------------------
// TermPW.inl — Template definitions for pw<Functor>().
// Included automatically by Term.h.  Do NOT include directly.
// Requires nvcc (contains __global__ kernel template).
// ---------------------------------------------------------------------------

#pragma once

#include "core/Check.h"
#include "core/CudaCheck.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace PhiX {

// ---------------------------------------------------------------------------
// GPU kernel: rhs[idx] += coeff * Func(src[idx])
// One thread per physical cell, row-major (x fast).
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw_accumulate(
        Real*       rhs,
        const Real* src,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,          // storedDims[0], storedDims[1]
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src[idx]);
}

// ---------------------------------------------------------------------------
// GPU kernel (2-field): rhs[idx] += coeff * Func(src1[idx], src2[idx])
// Functor signature: __device__ double operator()(double, double) const
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw2_accumulate(
        Real*       rhs,
        const Real* src1,
        const Real* src2,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src1[idx], src2[idx]);
}

// ---------------------------------------------------------------------------
// GPU kernel (3-field): rhs[idx] += coeff * Func(src1[idx], src2[idx], src3[idx])
// Functor signature: __device__ double operator()(double, double, double) const
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw3_accumulate(
        Real*       rhs,
        const Real* src1,
        const Real* src2,
        const Real* src3,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src1[idx], src2[idx], src3[idx]);
}

// ---------------------------------------------------------------------------
// GPU kernel (4-field): rhs[idx] += coeff * Func(src1..src4 at idx)
// Functor signature: __device__ double operator()(double, double, double, double) const
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw4_accumulate(
        Real*       rhs,
        const Real* src1,
        const Real* src2,
        const Real* src3,
        const Real* src4,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src1[idx], src2[idx], src3[idx], src4[idx]);
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (single field)
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f, Functor func, double coeff) {
    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f;
    t.inputs = {&f};
    describeInputs(t);
    t.coeff = coeff;

    // Capture mesh layout at construction time
    int nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int sx = f.storedDims[0], sy = f.storedDims[1];
    int g  = f.ghost;

    // Capture pointer-to-field so RK4 d_curr swap remains effective.
    const ScalarField* pf = &f;

    // GPU launcher: host function that launches the templated kernel
    t.gpu_launcher = [func, pf, nx, ny, nz, sx, sy, g]
                     (Real* d_rhs, double c, ScratchPool& pool) mutable {
        const Real* d_src = pf->d_curr;
        if (!d_src)
            throw std::runtime_error(
                "pw GPU: source field not on device");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw_accumulate<Functor><<<blocks, threads, 0, pool.stream>>>(
            d_rhs, d_src, func, c, nx, ny, nz, sx, sy, g);
        PHIX_KERNEL_CHECK("pw");
    };

    // CPU fallback (Functor::operator() must also work on host)
    t.cpu_kernel = [func, pf, nx, ny, nz, sx, sy, g]
                   (Real* rhs, double c, ScratchPool&) mutable {
        const Real* src = pf->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (2 fields)
//   rhs[idx] += coeff * func(f1[idx], f2[idx])
//   Functor signature: (double f1_val, double f2_val) -> double
//   Both fields must share the same mesh dimensions and ghost width.
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, Functor func, double coeff) {
    check::checkSameMesh(f1, f2, "pw(f1,f2)");

    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f1;   // primary field (used by computeRHS for null-check)
    t.inputs = {&f1, &f2};
    describeInputs(t);
    t.coeff = coeff;

    int nx = f1.mesh.n[0], ny = f1.mesh.n[1], nz = f1.mesh.n[2];
    int sx = f1.storedDims[0], sy = f1.storedDims[1];
    int g  = f1.ghost;

    // Capture pointers (non-owning); read d_curr at launch time
    const ScalarField* pf1 = &f1;
    const ScalarField* pf2 = &f2;

    t.gpu_launcher = [func, nx, ny, nz, sx, sy, g, pf1, pf2]
                     (Real* d_rhs, double c, ScratchPool& pool) mutable {
        const Real* d_src1 = pf1->d_curr;
        const Real* d_src2 = pf2->d_curr;
        if (!d_src1 || !d_src2)
            throw std::runtime_error(
                "pw(f1,f2) GPU: a field not on device. "
                "Call allocDevice() and uploadToDevice() first.");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw2_accumulate<Functor><<<blocks, threads, 0, pool.stream>>>(
            d_rhs, d_src1, d_src2, func, c, nx, ny, nz, sx, sy, g);
        PHIX_KERNEL_CHECK("pw2");
    };

    t.cpu_kernel = [func, nx, ny, nz, sx, sy, g, pf1, pf2]
                   (Real* rhs, double c, ScratchPool&) mutable {
        const Real* src1 = pf1->curr.data();
        const Real* src2 = pf2->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src1[idx], src2[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (3 fields)
//   rhs[idx] += coeff * func(f1[idx], f2[idx], f3[idx])
//   Functor signature: (double, double, double) -> double
//   All fields must share the same mesh dimensions and ghost width.
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3,
        Functor func, double coeff) {
    check::checkSameMesh(f1, f2, "pw(f1,f2,f3)");
    check::checkSameMesh(f1, f3, "pw(f1,f2,f3)");

    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f1;
    t.inputs = {&f1, &f2, &f3};
    describeInputs(t);
    t.coeff = coeff;

    int nx = f1.mesh.n[0], ny = f1.mesh.n[1], nz = f1.mesh.n[2];
    int sx = f1.storedDims[0], sy = f1.storedDims[1];
    int g  = f1.ghost;

    const ScalarField* pf1 = &f1;
    const ScalarField* pf2 = &f2;
    const ScalarField* pf3 = &f3;

    t.gpu_launcher = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3]
                     (Real* d_rhs, double c, ScratchPool& pool) mutable {
        const Real* d_src1 = pf1->d_curr;
        const Real* d_src2 = pf2->d_curr;
        const Real* d_src3 = pf3->d_curr;
        if (!d_src1 || !d_src2 || !d_src3)
            throw std::runtime_error(
                "pw(f1,f2,f3) GPU: a field not on device.");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw3_accumulate<Functor><<<blocks, threads, 0, pool.stream>>>(
            d_rhs, d_src1, d_src2, d_src3, func, c, nx, ny, nz, sx, sy, g);
        PHIX_KERNEL_CHECK("pw3");
    };

    t.cpu_kernel = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3]
                   (Real* rhs, double c, ScratchPool&) mutable {
        const Real* src1 = pf1->curr.data();
        const Real* src2 = pf2->curr.data();
        const Real* src3 = pf3->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src1[idx], src2[idx], src3[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (4 fields)
//   rhs[idx] += coeff * func(f1[idx], f2[idx], f3[idx], f4[idx])
//   Functor signature: (double, double, double, double) -> double
//   All fields must share the same mesh dimensions and ghost width.
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3,
        const ScalarField& f4, Functor func, double coeff) {
    const ScalarField* fs[] = {&f2, &f3, &f4};
    for (const ScalarField* f : fs)
        check::checkSameMesh(f1, *f, "pw(f1,f2,f3,f4)");

    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f1;
    t.inputs = {&f1, &f2, &f3, &f4};
    describeInputs(t);
    t.coeff = coeff;

    int nx = f1.mesh.n[0], ny = f1.mesh.n[1], nz = f1.mesh.n[2];
    int sx = f1.storedDims[0], sy = f1.storedDims[1];
    int g  = f1.ghost;

    const ScalarField* pf1 = &f1;
    const ScalarField* pf2 = &f2;
    const ScalarField* pf3 = &f3;
    const ScalarField* pf4 = &f4;

    t.gpu_launcher = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3, pf4]
                     (Real* d_rhs, double c, ScratchPool& pool) mutable {
        const Real* d_src1 = pf1->d_curr;
        const Real* d_src2 = pf2->d_curr;
        const Real* d_src3 = pf3->d_curr;
        const Real* d_src4 = pf4->d_curr;
        if (!d_src1 || !d_src2 || !d_src3 || !d_src4)
            throw std::runtime_error(
                "pw(f1,f2,f3,f4) GPU: a field not on device.");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw4_accumulate<Functor><<<blocks, threads, 0, pool.stream>>>(
            d_rhs, d_src1, d_src2, d_src3, d_src4, func, c,
            nx, ny, nz, sx, sy, g);
        PHIX_KERNEL_CHECK("pw4");
    };

    t.cpu_kernel = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3, pf4]
                   (Real* rhs, double c, ScratchPool&) mutable {
        const Real* src1 = pf1->curr.data();
        const Real* src2 = pf2->curr.data();
        const Real* src3 = pf3->curr.data();
        const Real* src4 = pf4->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src1[idx], src2[idx], src3[idx], src4[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// GPU kernel (5-field): rhs[idx] += coeff * Func(src1..src5 at idx)   (v3.7.0)
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw5_accumulate(
        Real*       rhs,
        const Real* src1,
        const Real* src2,
        const Real* src3,
        const Real* src4,
        const Real* src5,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src1[idx], src2[idx], src3[idx], src4[idx], src5[idx]);
}

// ---------------------------------------------------------------------------
// GPU kernel (6-field): rhs[idx] += coeff * Func(src1..src6 at idx)   (v3.7.0)
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw6_accumulate(
        Real*       rhs,
        const Real* src1,
        const Real* src2,
        const Real* src3,
        const Real* src4,
        const Real* src5,
        const Real* src6,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src1[idx], src2[idx], src3[idx], src4[idx], src5[idx], src6[idx]);
}

// ---------------------------------------------------------------------------
// GPU kernel (7-field): rhs[idx] += coeff * Func(src1..src7 at idx)   (v3.7.0)
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw7_accumulate(
        Real*       rhs,
        const Real* src1,
        const Real* src2,
        const Real* src3,
        const Real* src4,
        const Real* src5,
        const Real* src6,
        const Real* src7,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src1[idx], src2[idx], src3[idx], src4[idx], src5[idx], src6[idx], src7[idx]);
}

// ---------------------------------------------------------------------------
// GPU kernel (8-field): rhs[idx] += coeff * Func(src1..src8 at idx)   (v3.7.0)
// ---------------------------------------------------------------------------
template<typename Functor>
__global__ void kernel_pw8_accumulate(
        Real*       rhs,
        const Real* src1,
        const Real* src2,
        const Real* src3,
        const Real* src4,
        const Real* src5,
        const Real* src6,
        const Real* src7,
        const Real* src8,
        Functor       func,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * func(src1[idx], src2[idx], src3[idx], src4[idx], src5[idx], src6[idx], src7[idx], src8[idx]);
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (5 fields)   (v3.7.0)
//   rhs[idx] += coeff * func(f1[idx], ..., f5[idx])
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3, const ScalarField& f4, const ScalarField& f5,
        Functor func, double coeff) {
    check::checkSameMesh(f1, f2, "pw(5 fields)");
    check::checkSameMesh(f1, f3, "pw(5 fields)");
    check::checkSameMesh(f1, f4, "pw(5 fields)");
    check::checkSameMesh(f1, f5, "pw(5 fields)");

    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f1;
    t.inputs = {&f1, &f2, &f3, &f4, &f5};
    describeInputs(t);
    t.coeff = coeff;

    int nx = f1.mesh.n[0], ny = f1.mesh.n[1], nz = f1.mesh.n[2];
    int sx = f1.storedDims[0], sy = f1.storedDims[1];
    int g  = f1.ghost;

    const ScalarField* pf1 = &f1;
    const ScalarField* pf2 = &f2;
    const ScalarField* pf3 = &f3;
    const ScalarField* pf4 = &f4;
    const ScalarField* pf5 = &f5;

    t.gpu_launcher = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3, pf4, pf5]
                     (Real* d_rhs, double c, ScratchPool& pool) mutable {
        const Real* d_src1 = pf1->d_curr;
        const Real* d_src2 = pf2->d_curr;
        const Real* d_src3 = pf3->d_curr;
        const Real* d_src4 = pf4->d_curr;
        const Real* d_src5 = pf5->d_curr;
        if (!d_src1 || !d_src2 || !d_src3 || !d_src4 || !d_src5)
            throw std::runtime_error(
                "pw(5 fields) GPU: a field not on device. "
                "Call allocDevice() and uploadToDevice() first.");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw5_accumulate<Functor><<<blocks, threads, 0, pool.stream>>>(
            d_rhs, d_src1, d_src2, d_src3, d_src4, d_src5, func, c, nx, ny, nz, sx, sy, g);
        PHIX_KERNEL_CHECK("pw5");
    };

    t.cpu_kernel = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3, pf4, pf5]
                   (Real* rhs, double c, ScratchPool&) mutable {
        const Real* src1 = pf1->curr.data();
        const Real* src2 = pf2->curr.data();
        const Real* src3 = pf3->curr.data();
        const Real* src4 = pf4->curr.data();
        const Real* src5 = pf5->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src1[idx], src2[idx], src3[idx], src4[idx], src5[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (6 fields)   (v3.7.0)
//   rhs[idx] += coeff * func(f1[idx], ..., f6[idx])
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3, const ScalarField& f4, const ScalarField& f5, const ScalarField& f6,
        Functor func, double coeff) {
    check::checkSameMesh(f1, f2, "pw(6 fields)");
    check::checkSameMesh(f1, f3, "pw(6 fields)");
    check::checkSameMesh(f1, f4, "pw(6 fields)");
    check::checkSameMesh(f1, f5, "pw(6 fields)");
    check::checkSameMesh(f1, f6, "pw(6 fields)");

    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f1;
    t.inputs = {&f1, &f2, &f3, &f4, &f5, &f6};
    describeInputs(t);
    t.coeff = coeff;

    int nx = f1.mesh.n[0], ny = f1.mesh.n[1], nz = f1.mesh.n[2];
    int sx = f1.storedDims[0], sy = f1.storedDims[1];
    int g  = f1.ghost;

    const ScalarField* pf1 = &f1;
    const ScalarField* pf2 = &f2;
    const ScalarField* pf3 = &f3;
    const ScalarField* pf4 = &f4;
    const ScalarField* pf5 = &f5;
    const ScalarField* pf6 = &f6;

    t.gpu_launcher = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3, pf4, pf5, pf6]
                     (Real* d_rhs, double c, ScratchPool& pool) mutable {
        const Real* d_src1 = pf1->d_curr;
        const Real* d_src2 = pf2->d_curr;
        const Real* d_src3 = pf3->d_curr;
        const Real* d_src4 = pf4->d_curr;
        const Real* d_src5 = pf5->d_curr;
        const Real* d_src6 = pf6->d_curr;
        if (!d_src1 || !d_src2 || !d_src3 || !d_src4 || !d_src5 || !d_src6)
            throw std::runtime_error(
                "pw(6 fields) GPU: a field not on device. "
                "Call allocDevice() and uploadToDevice() first.");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw6_accumulate<Functor><<<blocks, threads, 0, pool.stream>>>(
            d_rhs, d_src1, d_src2, d_src3, d_src4, d_src5, d_src6, func, c, nx, ny, nz, sx, sy, g);
        PHIX_KERNEL_CHECK("pw6");
    };

    t.cpu_kernel = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3, pf4, pf5, pf6]
                   (Real* rhs, double c, ScratchPool&) mutable {
        const Real* src1 = pf1->curr.data();
        const Real* src2 = pf2->curr.data();
        const Real* src3 = pf3->curr.data();
        const Real* src4 = pf4->curr.data();
        const Real* src5 = pf5->curr.data();
        const Real* src6 = pf6->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src1[idx], src2[idx], src3[idx], src4[idx], src5[idx], src6[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (7 fields)   (v3.7.0)
//   rhs[idx] += coeff * func(f1[idx], ..., f7[idx])
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3, const ScalarField& f4, const ScalarField& f5, const ScalarField& f6, const ScalarField& f7,
        Functor func, double coeff) {
    check::checkSameMesh(f1, f2, "pw(7 fields)");
    check::checkSameMesh(f1, f3, "pw(7 fields)");
    check::checkSameMesh(f1, f4, "pw(7 fields)");
    check::checkSameMesh(f1, f5, "pw(7 fields)");
    check::checkSameMesh(f1, f6, "pw(7 fields)");
    check::checkSameMesh(f1, f7, "pw(7 fields)");

    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f1;
    t.inputs = {&f1, &f2, &f3, &f4, &f5, &f6, &f7};
    describeInputs(t);
    t.coeff = coeff;

    int nx = f1.mesh.n[0], ny = f1.mesh.n[1], nz = f1.mesh.n[2];
    int sx = f1.storedDims[0], sy = f1.storedDims[1];
    int g  = f1.ghost;

    const ScalarField* pf1 = &f1;
    const ScalarField* pf2 = &f2;
    const ScalarField* pf3 = &f3;
    const ScalarField* pf4 = &f4;
    const ScalarField* pf5 = &f5;
    const ScalarField* pf6 = &f6;
    const ScalarField* pf7 = &f7;

    t.gpu_launcher = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3, pf4, pf5, pf6, pf7]
                     (Real* d_rhs, double c, ScratchPool& pool) mutable {
        const Real* d_src1 = pf1->d_curr;
        const Real* d_src2 = pf2->d_curr;
        const Real* d_src3 = pf3->d_curr;
        const Real* d_src4 = pf4->d_curr;
        const Real* d_src5 = pf5->d_curr;
        const Real* d_src6 = pf6->d_curr;
        const Real* d_src7 = pf7->d_curr;
        if (!d_src1 || !d_src2 || !d_src3 || !d_src4 || !d_src5 || !d_src6 || !d_src7)
            throw std::runtime_error(
                "pw(7 fields) GPU: a field not on device. "
                "Call allocDevice() and uploadToDevice() first.");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw7_accumulate<Functor><<<blocks, threads, 0, pool.stream>>>(
            d_rhs, d_src1, d_src2, d_src3, d_src4, d_src5, d_src6, d_src7, func, c, nx, ny, nz, sx, sy, g);
        PHIX_KERNEL_CHECK("pw7");
    };

    t.cpu_kernel = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3, pf4, pf5, pf6, pf7]
                   (Real* rhs, double c, ScratchPool&) mutable {
        const Real* src1 = pf1->curr.data();
        const Real* src2 = pf2->curr.data();
        const Real* src3 = pf3->curr.data();
        const Real* src4 = pf4->curr.data();
        const Real* src5 = pf5->curr.data();
        const Real* src6 = pf6->curr.data();
        const Real* src7 = pf7->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src1[idx], src2[idx], src3[idx], src4[idx], src5[idx], src6[idx], src7[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// pw<Functor> definition (8 fields)   (v3.7.0)
//   rhs[idx] += coeff * func(f1[idx], ..., f8[idx])
// ---------------------------------------------------------------------------
template<typename Functor>
Term pw(const ScalarField& f1, const ScalarField& f2, const ScalarField& f3, const ScalarField& f4, const ScalarField& f5, const ScalarField& f6, const ScalarField& f7, const ScalarField& f8,
        Functor func, double coeff) {
    check::checkSameMesh(f1, f2, "pw(8 fields)");
    check::checkSameMesh(f1, f3, "pw(8 fields)");
    check::checkSameMesh(f1, f4, "pw(8 fields)");
    check::checkSameMesh(f1, f5, "pw(8 fields)");
    check::checkSameMesh(f1, f6, "pw(8 fields)");
    check::checkSameMesh(f1, f7, "pw(8 fields)");
    check::checkSameMesh(f1, f8, "pw(8 fields)");

    Term t;
    t.type  = TermType::POINTWISE;
    t.field = &f1;
    t.inputs = {&f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8};
    describeInputs(t);
    t.coeff = coeff;

    int nx = f1.mesh.n[0], ny = f1.mesh.n[1], nz = f1.mesh.n[2];
    int sx = f1.storedDims[0], sy = f1.storedDims[1];
    int g  = f1.ghost;

    const ScalarField* pf1 = &f1;
    const ScalarField* pf2 = &f2;
    const ScalarField* pf3 = &f3;
    const ScalarField* pf4 = &f4;
    const ScalarField* pf5 = &f5;
    const ScalarField* pf6 = &f6;
    const ScalarField* pf7 = &f7;
    const ScalarField* pf8 = &f8;

    t.gpu_launcher = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3, pf4, pf5, pf6, pf7, pf8]
                     (Real* d_rhs, double c, ScratchPool& pool) mutable {
        const Real* d_src1 = pf1->d_curr;
        const Real* d_src2 = pf2->d_curr;
        const Real* d_src3 = pf3->d_curr;
        const Real* d_src4 = pf4->d_curr;
        const Real* d_src5 = pf5->d_curr;
        const Real* d_src6 = pf6->d_curr;
        const Real* d_src7 = pf7->d_curr;
        const Real* d_src8 = pf8->d_curr;
        if (!d_src1 || !d_src2 || !d_src3 || !d_src4 || !d_src5 || !d_src6 || !d_src7 || !d_src8)
            throw std::runtime_error(
                "pw(8 fields) GPU: a field not on device. "
                "Call allocDevice() and uploadToDevice() first.");
        int total   = nx * ny * nz;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        kernel_pw8_accumulate<Functor><<<blocks, threads, 0, pool.stream>>>(
            d_rhs, d_src1, d_src2, d_src3, d_src4, d_src5, d_src6, d_src7, d_src8, func, c, nx, ny, nz, sx, sy, g);
        PHIX_KERNEL_CHECK("pw8");
    };

    t.cpu_kernel = [func, nx, ny, nz, sx, sy, g, pf1, pf2, pf3, pf4, pf5, pf6, pf7, pf8]
                   (Real* rhs, double c, ScratchPool&) mutable {
        const Real* src1 = pf1->curr.data();
        const Real* src2 = pf2->curr.data();
        const Real* src3 = pf3->curr.data();
        const Real* src4 = pf4->curr.data();
        const Real* src5 = pf5->curr.data();
        const Real* src6 = pf6->curr.data();
        const Real* src7 = pf7->curr.data();
        const Real* src8 = pf8->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int idx = (i+g) + sx*((j+g) + sy*(k+g));
            rhs[idx] += c * func(src1[idx], src2[idx], src3[idx], src4[idx], src5[idx], src6[idx], src7[idx], src8[idx]);
        }
    };

    return t;
}

// ---------------------------------------------------------------------------
// pw(VectorField, Functor) — pointwise per-component (single scalar field)
//   Returns VectorRHSExpr; component c is pw(vf[c], func, coeff)
//   NOTE: Term.h must be included before this file (VectorRHSExpr must exist).
// ---------------------------------------------------------------------------
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf, Functor func, double coeff) {
    VectorRHSExpr expr(vf.nComponents());
    for (int c = 0; c < vf.nComponents(); ++c)
        expr[c] = RHSExpr(pw(vf[c], func, coeff));
    return expr;
}

// ---------------------------------------------------------------------------
// pw(VectorField, ScalarField, Functor) — per-component binary operation
//   component c: rhs_c[idx] += coeff * func(vf[c][idx], sf[idx])
//   Functor signature: (double vf_val, double sf_val) -> double
// ---------------------------------------------------------------------------
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf, const ScalarField& sf,
                 Functor func, double coeff) {
    VectorRHSExpr expr(vf.nComponents());
    for (int c = 0; c < vf.nComponents(); ++c)
        expr[c] = RHSExpr(pw(vf[c], sf, func, coeff));
    return expr;
}

// ---------------------------------------------------------------------------
// pw(VectorField, VectorField, Functor) — component-wise binary operation
//   component c: rhs_c[idx] += coeff * func(vf1[c][idx], vf2[c][idx])
//   Both VectorFields must have the same number of components.
//   Functor signature: (double v1_val, double v2_val) -> double
// ---------------------------------------------------------------------------
template<typename Functor>
VectorRHSExpr pw(const VectorField& vf1, const VectorField& vf2,
                 Functor func, double coeff) {
    if (vf1.nComponents() != vf2.nComponents())
        throw std::invalid_argument(
            "pw(vf1, vf2): VectorFields must have the same number of components");
    VectorRHSExpr expr(vf1.nComponents());
    for (int c = 0; c < vf1.nComponents(); ++c)
        expr[c] = RHSExpr(pw(vf1[c], vf2[c], func, coeff));
    return expr;
}

} // namespace PhiX

// Field arithmetic operator overloads — enables DSL syntax like:
//   eq.setRHS(c * eta - 2.0 * c + lap(c))
#include "equation/FieldOps.inl"
