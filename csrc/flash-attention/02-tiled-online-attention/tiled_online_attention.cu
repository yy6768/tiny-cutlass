/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief CUTLASS tiled online Multi-Head Attention.

    This example keeps the fixed-seqlen BMHK scope from 00-naive-attention but
    changes the device dataflow:

      00: MM0 writes full P -> Softmax reads/writes P -> MM1 reads P
      02: one kernel scans K/V tiles and writes O only

    The device path uses the CUTLASS example 41 fused forward kernel:
      MM0 MMA -> online softmax -> shared-memory handoff -> MM1 MMA.
*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "kernel_forward.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  double gflops;
  cudaError_t error;
  bool passed;

  Result(double runtime_ms = 0, double gflops = 0, cudaError_t error = cudaSuccess)
      : runtime_ms(runtime_ms), gflops(gflops), error(error), passed(true) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help;
  bool error;
  bool reference_check;

  int head_number;
  int batch_size;
  int head_size;
  int head_size_v;
  int seq_length;
  int seq_length_kv;
  int iterations;

  float alpha;

  Options()
      : help(false),
        error(false),
        reference_check(true),
        head_number(12),
        batch_size(16),
        head_size(64),
        head_size_v(64),
        seq_length(1024),
        seq_length_kv(1024),
        iterations(20),
        alpha(0.f) {}

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("head_number", head_number, 12);
    cmd.get_cmd_line_argument("batch_size", batch_size, 16);
    cmd.get_cmd_line_argument("head_size", head_size, 64);
    cmd.get_cmd_line_argument("head_size_v", head_size_v, head_size);
    cmd.get_cmd_line_argument("seq_length", seq_length, 1024);
    cmd.get_cmd_line_argument("seq_length_kv", seq_length_kv, seq_length);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "02_tiled_online_attention_test\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement.\n\n"
        << "  --head_number=<int>         Head number (default: 12)\n"
        << "  --batch_size=<int>          Batch size (default: 16)\n"
        << "  --head_size=<int>           Head dim for Q/K (default: 64)\n"
        << "  --head_size_v=<int>         Head dim for V/O (default: head_size)\n"
        << "  --seq_length=<int>          Sequence length for Q (default: 1024)\n"
        << "  --seq_length_kv=<int>       Sequence length for K/V (default: seq_length)\n"
        << "  --iterations=<int>          Number of profiling iterations (default: 20)\n"
        << "  --reference-check=<bool>    If true, run host reference check (default: true)\n";
    return out;
  }

  double gflops(double runtime_s) const {
    int64_t fops = int64_t(2) * batch_size * head_number * seq_length * seq_length_kv * head_size
                 + int64_t(2) * batch_size * head_number * seq_length * seq_length_kv * head_size_v
                 + int64_t(3) * batch_size * head_number * seq_length * seq_length_kv;
    return double(fops) / 1.0e9 / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int kQueriesPerBlock, int kKeysPerBlock, int kMaxK>
class TestbedTiledOnlineAttention {
public:
  using Attention = AttentionKernel<
      cutlass::half_t,
      cutlass::arch::Sm80,
      true,
      kQueriesPerBlock,
      kKeysPerBlock,
      kMaxK,
      false,
      false>;

  using scalar_t = typename Attention::scalar_t;
  using accum_t = typename Attention::accum_t;
  using output_accum_t = typename Attention::output_accum_t;

private:
  Options& options;
  uint32_t seed;

  cutlass::DeviceAllocation<scalar_t> block_Q; // [B, Sq, H, d]
  cutlass::DeviceAllocation<scalar_t> block_K; // [B, Sk, H, d]
  cutlass::DeviceAllocation<scalar_t> block_V; // [B, Sk, H, dv]
  cutlass::DeviceAllocation<scalar_t> block_O; // [B, Sq, H, dv]
  cutlass::DeviceAllocation<output_accum_t> block_O_accum;

public:
  TestbedTiledOnlineAttention(Options& options_, uint32_t seed_ = 3080)
      : options(options_), seed(seed_) {}

private:
  void initialize_() {
    options.alpha = 1.0f / std::sqrt(float(options.head_size));

    int B = options.batch_size;
    int Sq = options.seq_length;
    int Sk = options.seq_length_kv;
    int H = options.head_number;
    int d = options.head_size;
    int dv = options.head_size_v;

    int64_t total_Q = int64_t(B) * Sq * H * d;
    int64_t total_K = int64_t(B) * Sk * H * d;
    int64_t total_V = int64_t(B) * Sk * H * dv;
    int64_t total_O = int64_t(B) * Sq * H * dv;

    block_Q.reset(total_Q);
    block_K.reset(total_K);
    block_V.reset(total_V);
    block_O.reset(total_O);
    if (Attention::kNeedsOutputAccumulatorBuffer) {
      block_O_accum.reset(total_O);
    }

    cutlass::reference::device::BlockFillRandomUniform(
        block_Q.get(), total_Q, seed + 1, scalar_t(2), scalar_t(-2), 0);
    cutlass::reference::device::BlockFillRandomUniform(
        block_K.get(), total_K, seed + 2, scalar_t(2), scalar_t(-2), 0);
    cutlass::reference::device::BlockFillRandomUniform(
        block_V.get(), total_V, seed + 3, scalar_t(2), scalar_t(-2), 0);
    cudaMemset(block_O.get(), 0, total_O * sizeof(scalar_t));
  }

  void compute_reference_host_(std::vector<scalar_t>& host_ref_O) {
    int B = options.batch_size;
    int Sq = options.seq_length;
    int Sk = options.seq_length_kv;
    int H = options.head_number;
    int d = options.head_size;
    int dv = options.head_size_v;
    float scale = options.alpha;

    int64_t total_Q = int64_t(B) * Sq * H * d;
    int64_t total_K = int64_t(B) * Sk * H * d;
    int64_t total_V = int64_t(B) * Sk * H * dv;

    std::vector<scalar_t> host_Q(total_Q), host_K(total_K), host_V(total_V);
    cutlass::device_memory::copy_to_host(host_Q.data(), block_Q.get(), total_Q);
    cutlass::device_memory::copy_to_host(host_K.data(), block_K.get(), total_K);
    cutlass::device_memory::copy_to_host(host_V.data(), block_V.get(), total_V);

    int q_strideM = H * d;
    int k_strideM = H * d;
    int v_strideM = H * dv;
    int o_strideM = H * dv;

    std::vector<float> P(Sq * Sk);

    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
        for (int i = 0; i < Sq; ++i) {
          for (int j = 0; j < Sk; ++j) {
            float dot = 0.f;
            for (int k = 0; k < d; ++k) {
              float q = float(host_Q[int64_t(b) * Sq * q_strideM + i * q_strideM + h * d + k]);
              float kk = float(host_K[int64_t(b) * Sk * k_strideM + j * k_strideM + h * d + k]);
              dot += q * kk;
            }
            P[i * Sk + j] = dot * scale;
          }
        }

        for (int i = 0; i < Sq; ++i) {
          float m = P[i * Sk];
          for (int j = 1; j < Sk; ++j) {
            m = std::max(m, P[i * Sk + j]);
          }

          float s = 0.f;
          for (int j = 0; j < Sk; ++j) {
            P[i * Sk + j] = std::exp(P[i * Sk + j] - m);
            s += P[i * Sk + j];
          }

          float inv = 1.f / s;
          for (int j = 0; j < Sk; ++j) {
            P[i * Sk + j] *= inv;
          }
        }

        for (int i = 0; i < Sq; ++i) {
          for (int j = 0; j < dv; ++j) {
            float sum = 0.f;
            for (int k = 0; k < Sk; ++k) {
              float v = float(host_V[int64_t(b) * Sk * v_strideM + k * v_strideM + h * dv + j]);
              sum += P[i * Sk + k] * v;
            }
            host_ref_O[int64_t(b) * Sq * o_strideM + i * o_strideM + h * dv + j] = scalar_t(sum);
          }
        }
      }
    }
  }

  bool verify_() {
    int B = options.batch_size;
    int Sq = options.seq_length;
    int H = options.head_number;
    int dv = options.head_size_v;
    int64_t total_O = int64_t(B) * Sq * H * dv;

    std::vector<scalar_t> host_O(total_O), host_ref_O(total_O);
    cutlass::device_memory::copy_to_host(host_O.data(), block_O.get(), total_O);

    compute_reference_host_(host_ref_O);

    float abs_tol = 6e-2f;
    float rel_tol = 1.2e-1f;
    for (int64_t i = 0; i < total_O; ++i) {
      float a = float(host_O[i]);
      float r = float(host_ref_O[i]);
      float diff = std::fabs(a - r);
      float rel = diff / (std::fabs(r) + 1e-5f);
      if (std::isnan(a) || std::isinf(a) || (diff > abs_tol && rel > rel_tol)) {
        printf("[%lld] computed=%f ref=%f diff=%f rel=%f\n",
               (long long)i, a, r, diff, rel);
        return false;
      }
    }
    return true;
  }

  cudaError_t launch_kernel_() {
    int B = options.batch_size;
    int Sq = options.seq_length;
    int Sk = options.seq_length_kv;
    int H = options.head_number;
    int d = options.head_size;
    int dv = options.head_size_v;

    int32_t q_strideH = d;
    int32_t k_strideH = d;
    int32_t v_strideH = dv;
    int32_t q_strideM = H * d;
    int32_t k_strideM = H * d;
    int32_t v_strideM = H * dv;
    int32_t o_strideM = H * dv;
    int64_t q_strideB = int64_t(Sq) * q_strideM;
    int64_t k_strideB = int64_t(Sk) * k_strideM;
    int64_t v_strideB = int64_t(Sk) * v_strideM;
    typename Attention::Params p;
    p.query_ptr = block_Q.get();
    p.key_ptr = block_K.get();
    p.value_ptr = block_V.get();
    p.output_ptr = block_O.get();
    p.output_accum_ptr = Attention::kNeedsOutputAccumulatorBuffer
        ? block_O_accum.get()
        : nullptr;
    p.logsumexp_ptr = nullptr;
    p.scale = options.alpha;
    p.num_queries = Sq;
    p.num_keys = Sk;
    p.head_dim = d;
    p.head_dim_value = dv;
    p.num_heads = H;
    p.num_batches = B;
    p.q_strideM = q_strideM;
    p.k_strideM = k_strideM;
    p.v_strideM = v_strideM;
    p.o_strideM = o_strideM;
    p.q_strideH = q_strideH;
    p.k_strideH = k_strideH;
    p.v_strideH = v_strideH;
    p.q_strideB = q_strideB;
    p.k_strideB = k_strideB;
    p.v_strideB = v_strideB;

    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = int(sizeof(typename Attention::SharedStorage));
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
      std::cerr << "Kernel does not support these inputs\n";
      return cudaErrorInvalidValue;
    }

    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    return cudaGetLastError();
  }

public:
  Result profile() {
    Result result;
    result.passed = false;

    initialize_();

    cudaError_t launch_err = launch_kernel_();
    if (launch_err != cudaSuccess) {
      std::cerr << "Kernel launch error: " << cudaGetErrorString(launch_err) << std::endl;
      result.error = launch_err;
      return result;
    }

    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess) {
      std::cerr << "Kernel exec error: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    result.passed = true;
    if (options.reference_check) {
      result.passed = verify_();
      if (!result.passed) {
        return result;
      }
    }

    launch_kernel_();
    cudaDeviceSynchronize();

    cudaEvent_t events[2];
    for (auto& e : events) {
      result.error = cudaEventCreate(&e);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate failed\n";
        return result;
      }
    }

    cudaEventRecord(events[0]);
    for (int i = 0; i < options.iterations; ++i) {
      launch_kernel_();
    }
    cudaEventRecord(events[1]);
    cudaEventSynchronize(events[1]);

    float runtime_ms = 0;
    cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    for (auto e : events) {
      cudaEventDestroy(e);
    }

    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    std::cout << "\nCUTLASS Tiled Online Attention (fused MMA):\n"
              << "====================================================\n"
              << "    {Sq, Sk, d, d_v, H, B} = {"
              << options.seq_length << ", " << options.seq_length_kv << ", "
              << options.head_size << ", " << options.head_size_v << ", "
              << options.head_number << ", " << options.batch_size << "}\n"
              << "    tile {queries, keys, maxK} = {"
              << kQueriesPerBlock << ", " << kKeysPerBlock << ", " << kMaxK << "}\n\n"
              << "    Runtime: " << result.runtime_ms << " ms\n"
              << "    GFLOPs : " << result.gflops << "\n";

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int kQueriesPerBlock, int kKeysPerBlock, int kMaxK>
int run_attention_config(Options& options) {
  TestbedTiledOnlineAttention<kQueriesPerBlock, kKeysPerBlock, kMaxK> testbed(options);
  Result r = testbed.profile();
  if (!r.passed) {
    std::cout << "\nFailed\n";
    return -1;
  }
  std::cout << "\nPassed\n";
  return 0;
}

int run_attention(Options& options) {
  int const max_head_size = std::max(options.head_size, options.head_size_v);

  if (max_head_size > 64) {
    if (max_head_size <= 128) {
      static int const kQueriesPerBlock = 32;
      static int const kKeysPerBlock = 128;
      return run_attention_config<kQueriesPerBlock, kKeysPerBlock, 128>(options);
    }
    if (max_head_size <= 256) {
      static int const kQueriesPerBlock = 64;
      static int const kKeysPerBlock = 64;
      return run_attention_config<kQueriesPerBlock, kKeysPerBlock, 65536>(options);
    }
    static int const kQueriesPerBlock = 32;
    static int const kKeysPerBlock = 128;
    return run_attention_config<kQueriesPerBlock, kKeysPerBlock, 65536>(options);
  }

  static int const kQueriesPerBlock = 64;
  static int const kKeysPerBlock = 64;
  static int const kMaxK = 64;
  return run_attention_config<kQueriesPerBlock, kKeysPerBlock, kMaxK>(options);
}

int main(int argc, char const** args) {
  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }
  std::cout << "Device: " << props.name << " (SM" << props.major << props.minor << ")\n";

  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
    std::cout << "This example requires Ampere (SM80) or later.\n";
    return 0;
  }

  Options options;
  options.parse(argc, args);
  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  if (options.error) {
    std::cerr << "Aborting.\n";
    return -1;
  }

  return run_attention(options);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
