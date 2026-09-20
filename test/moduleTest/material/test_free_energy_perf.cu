// ---------------------------------------------------------------------------
// test_free_energy_perf.cu
//
// 效率对比：在 GPU kernel 中使用查表插值 vs 使用预设常数
//
// 测试设计
// --------
//   每个 thread 处理一个网格点 (i,j,k)：
//     1. 从显存读取场数据 c[idx], T[idx]
//     2. 用 coeff 计算输出：out[idx] = coeff * c[idx]
//
//   coeff 的来源分两种：
//     - baseline  : 硬编码常数 1.0
//     - table_ldg : FreeEnergyTableView::f(c, T)  (__ldg 只读缓存路径)
//
//   每种重复运行若干次，用 CUDA Event 计时，输出平均时间和相对开销。
//
// 表格参数
// --------
//   nc=40, nT=100  =>  32 KB，能装进单个 SM 的 L1 只读缓存
// ---------------------------------------------------------------------------

#include "material/FreeEnergyTable.h"
#include "mesh/Mesh.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

using namespace PhiX;
using namespace PhiX::Material;

// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            printf("CUDA error %s:%d — %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(_e));                                     \
            std::terminate();                                                   \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Baseline：coeff 是编译期/调用期常数，不查表
__global__ void k_baseline(const double* __restrict__ c,
                            const double* __restrict__ T_field,
                            double* __restrict__ out,
                            int N,
                            double coeff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = coeff * c[idx];
}

// 查表：coeff = fe.f(c[idx], T_field[idx])
__global__ void k_table(const double* __restrict__ c,
                         const double* __restrict__ T_field,
                         double* __restrict__ out,
                         int N,
                         FreeEnergyTableView fe)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    double coeff = fe.f(c[idx], T_field[idx]);
    out[idx] = coeff * c[idx];
}

// ---------------------------------------------------------------------------
// 计时辅助：运行 kernel 若干次，返回平均毫秒
// ---------------------------------------------------------------------------
template<typename LaunchFn>
float bench(LaunchFn fn, int warmup, int repeat)
{
    // warmup
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < repeat; ++i) fn();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / repeat;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    // -----------------------------------------------------------------------
    // 网格 & 数据规模
    // -----------------------------------------------------------------------
    const int NX = 256, NY = 256, NZ = 1;
    const int N  = NX * NY * NZ;

    // -----------------------------------------------------------------------
    // 构造 FreeEnergyTable：40×100，f(c,T) = (c - 0.5)^2 * T / 1000
    // -----------------------------------------------------------------------
    const int NC = 40, NT = 100;
    const double C_MIN = 0.0, C_MAX = 1.0;
    const double T_MIN = 300.0, T_MAX = 1800.0;

    std::vector<double> tableData(NC * NT);
    for (int ic = 0; ic < NC; ++ic) {
        double cv = C_MIN + ic * (C_MAX - C_MIN) / (NC - 1);
        for (int iT = 0; iT < NT; ++iT) {
            double Tv = T_MIN + iT * (T_MAX - T_MIN) / (NT - 1);
            tableData[ic * NT + iT] = (cv - 0.5) * (cv - 0.5) * Tv / 1000.0;
        }
    }

    FreeEnergyTable table(C_MIN, C_MAX, NC, T_MIN, T_MAX, NT,
                          std::move(tableData));
    table.allocDevice();
    table.uploadToDevice();
    FreeEnergyTableView fe_view = table.deviceView();

    printf("Table size: %d × %d = %.1f KB\n",
           NC, NT, NC * NT * 8.0 / 1024.0);

    // -----------------------------------------------------------------------
    // 初始化 c 和 T 场：c 均匀分布在 [0,1]，T 均匀分布在 [300,1800]
    // -----------------------------------------------------------------------
    std::vector<double> h_c(N), h_T(N), h_out(N);
    for (int i = 0; i < N; ++i) {
        h_c[i] = static_cast<double>(i) / N;                    // [0,1)
        h_T[i] = 300.0 + 1500.0 * static_cast<double>(i) / N;  // [300,1800)
    }

    double *d_c, *d_T, *d_out;
    CUDA_CHECK(cudaMalloc(&d_c,   N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_T,   N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T, h_T.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    // -----------------------------------------------------------------------
    // 基准测试
    // -----------------------------------------------------------------------
    const int WARMUP = 10, REPEAT = 200;

    float t_base = bench([&]{
        k_baseline<<<GRID, BLOCK>>>(d_c, d_T, d_out, N, 1.0);
    }, WARMUP, REPEAT);

    float t_table = bench([&]{
        k_table<<<GRID, BLOCK>>>(d_c, d_T, d_out, N, fe_view);
    }, WARMUP, REPEAT);

    // -----------------------------------------------------------------------
    // 输出结果
    // -----------------------------------------------------------------------
    printf("\n");
    printf("Grid: %d × %d  (%d cells)\n", NX, NY, N);
    printf("Warmup: %d   Repeat: %d\n\n", WARMUP, REPEAT);
    printf("  %-28s  %8.4f ms\n", "baseline (constant coeff)", t_base);
    printf("  %-28s  %8.4f ms\n", "table lookup (f(c,T))",    t_table);
    printf("\n");
    printf("  overhead: %+.4f ms  (×%.2f)\n",
           t_table - t_base, t_table / t_base);

    // -----------------------------------------------------------------------
    // 正确性简检：手算一个点和查表结果对比
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // 最后一次运行的是 k_table，检查 idx=0
    // f(c=0, T=300) = (0-0.5)^2 * 300/1000 = 0.075
    // out[0] = 0.075 * c[0] = 0.075 * 0 = 0
    double ref0 = 0.0;
    assert(std::fabs(h_out[0] - ref0) < 1e-10);

    // idx = N/2: c = 0.5, T = 1050, f = (0.5-0.5)^2 * 1050/1000 = 0
    double ref_half = 0.0;
    assert(std::fabs(h_out[N/2] - ref_half) < 1e-10);

    printf("  Correctness check: PASSED\n");

    // -----------------------------------------------------------------------
    // 清理
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_T));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
