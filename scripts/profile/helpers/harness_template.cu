// ============================================================================
// harness_template.cu — SM89 独立 profiling harness 模板
//
// 当不想通过完整 CMake 重编来 profile 单个 kernel 时使用。
// 复制到 profile/<run_name>/harness/ 下，填写 TODO(you) 部分。
//
// 编译：
//     nvcc -O2 -std=c++17 --generate-line-info ^
//          -gencode=arch=compute_89,code=sm_89 ^
//          my_kernel_harness.cu -o my_kernel_harness.exe
//
// 用法：
//     my_kernel_harness.exe --workload <path.safetensors>     # 真实数据
//     my_kernel_harness.exe <shape_arg1> <shape_arg2> ...     # 合成数据
//
// 然后 profile：
//     python scripts/profile/run_ncu.py ^
//         --run-name my_kernel_v1 ^
//         --exe profile\my_kernel_v1\harness\my_kernel_harness.exe ^
//         --kernel-regex "my_kernel.*" ^
//         --set both --count 1
//
// 参考：scripts/profile/reference/workflow.md
// ============================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>       // SM89 支持 FP8 (e4m3/e5m2)
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// safetensors_loader.h 和本文件放在同一目录，或指定 include 路径
// #include "safetensors_loader.h"

// ---------------------------------------------------------------------------
// CUDA error helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                               \
    do {                                                                               \
        cudaError_t err = (expr);                                                      \
        if (err != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                           \
            std::exit(1);                                                               \
        }                                                                              \
    } while (0)

template <typename T>
static T* alloc_device(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

// ---------------------------------------------------------------------------
// 合成填充辅助函数
// ---------------------------------------------------------------------------
static void fill_half_random(std::vector<__half>& h, uint64_t seed, float scale = 0.5f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (auto& x : h) x = __float2half(d(rng));
}
static void fill_bf16_random(std::vector<__nv_bfloat16>& h, uint64_t seed, float scale = 0.5f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (auto& x : h) x = __float2bfloat16(d(rng));
}
static void fill_f32_random(std::vector<float>& h, uint64_t seed, float scale = 0.5f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (auto& x : h) x = d(rng);
}

// ============================================================================
// TODO(you): 在此粘贴或 include 你的 kernel 源码
//
//   - 包含所有 __device__ helper、intrinsics 和常量
//   - 如果 kernel 是模板的，添加显式实例化
//   - 示例：
//
//     template<int TILE_M, int TILE_N>
//     __global__ void my_kernel(const half* A, const half* B, half* C,
//                               int M, int N, int K) {
//         // ... kernel body ...
//     }
//
//     // 显式实例化 — 没有这些，未用到的变体会被链接器 strip
//     template __global__ void my_kernel<64, 64>(const half*, const half*, half*, int, int, int);
//     template __global__ void my_kernel<128, 64>(const half*, const half*, half*, int, int, int);
// ============================================================================

// ... your kernel goes here ...

// ============================================================================
// TODO(you): launch 辅助函数
// ============================================================================
static void launch_kernel(/* input pointers, shape params, cudaStream_t stream */) {
    // dim3 grid(...);
    // dim3 block(...);
    // my_kernel<TILE_M, TILE_N><<<grid, block, smem_bytes, stream>>>(A, B, C, M, N, K);
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// CLI 解析 + 输入准备
// ============================================================================

static void usage(const char* argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --workload <safetensors_path>\n"
            "      Load real tensor values from a workload file.\n"
            "  %s <shape_arg1> <shape_arg2> ...\n"
            "      Shape-match mode (synthetic at given shape).\n",
            argv0, argv0);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 2; }

    bool use_workload = false;
    std::string workload_path;

    // --- 参数解析 ---
    if (std::string(argv[1]) == "--workload") {
        if (argc < 3) { usage(argv[0]); return 2; }
        use_workload = true;
        workload_path = argv[2];
    } else {
        // TODO(you): 从 argv 解析你的 shape 参数
        // 如: int M = std::atoi(argv[1]); int N = std::atoi(argv[2]); ...
    }

    // --- 如果使用 safetensors，取消注释以下代码 ---
    // SafetensorsFile st;
    // if (use_workload) {
    //     st = SafetensorsFile::load(workload_path);
    //     // 从 header 中读取 shape
    //     // const auto& a_entry = st.entry("A");
    //     // int M = (int)a_entry.shape[0];
    //     fprintf(stderr, "[harness] loaded workload: %s\n", workload_path.c_str());
    // }

    // --- TODO(you): 分配 host & device 内存 ---
    // std::vector<half> h_A(M * K), h_B(K * N);
    // std::vector<half> h_C(M * N, __float2half(0.f));
    // half *d_A = alloc_device<half>(h_A.size());
    // half *d_B = alloc_device<half>(h_B.size());
    // half *d_C = alloc_device<half>(h_C.size());

    // --- TODO(you): 填充输入 ---
    // if (use_workload) {
    //     std::memcpy(h_A.data(), st.tensor_bytes("A"), h_A.size() * sizeof(half));
    // } else {
    //     fill_half_random(h_A, 0xA0A0ULL);
    //     fill_half_random(h_B, 0xB0B0ULL);
    // }

    // --- TODO(you): 拷贝到 device ---
    // CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), cudaMemcpyHostToDevice));

    // --- 启动 kernel ---
    fprintf(stderr, "[harness] launching kernel...\n");
    launch_kernel(/* d_A, d_B, d_C, M, N, K */);
    fprintf(stderr, "[harness] done.\n");

    // --- 清理 ---
    // CUDA_CHECK(cudaFree(d_A));
    // CUDA_CHECK(cudaFree(d_B));
    // CUDA_CHECK(cudaFree(d_C));

    return 0;
}
