/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Tiny Swin CUTLASS path.

    This file keeps one FasterTransformer-style Swin implementation path:

      WindowPartition
      QKV projection
      WindowAttention
      output projection
      WindowReverse

    QK, softmax and PV are deliberately not public interfaces here. They live
    inside WindowAttention, backed by the tiled online CUTLASS attention core.
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "../flash-attention/02-tiled-online-attention/kernel_forward.h"
#include "kernels/swin_kernels.h"

namespace tiny_cutlass {
namespace swin {

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms = 0.0;
  double gflops = 0.0;
  cudaError_t error = cudaSuccess;
  cutlass::Status status = cutlass::Status::kSuccess;
  bool passed = true;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help = false;
  bool error = false;
  bool reference_check = true;
  bool use_mask = true;

  int batch_size = 2;
  int image_size = 14;
  int window_size = 7;
  int shift_size = 3;
  int head_number = 3;
  int head_size = 32;
  int iterations = 20;

  float alpha = 0.0f;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("batch_size", batch_size, batch_size);
    cmd.get_cmd_line_argument("image_size", image_size, image_size);
    cmd.get_cmd_line_argument("window_size", window_size, window_size);
    cmd.get_cmd_line_argument("shift_size", shift_size, window_size / 2);
    cmd.get_cmd_line_argument("head_number", head_number, head_number);
    cmd.get_cmd_line_argument("head_size", head_size, head_size);
    cmd.get_cmd_line_argument("iterations", iterations, iterations);
    cmd.get_cmd_line_argument("reference-check", reference_check, reference_check);
    cmd.get_cmd_line_argument("mask", use_mask, use_mask);

    alpha = 1.0f / std::sqrt(float(head_size));
    validate();
  }

  void validate() {
    if (batch_size <= 0 || image_size <= 0 || window_size <= 0 ||
        head_number <= 0 || head_size <= 0 || iterations <= 0) {
      std::cerr << "All dimensions and iterations must be positive.\n";
      error = true;
    }
    if (image_size % window_size != 0) {
      std::cerr << "--image_size must be divisible by --window_size.\n";
      error = true;
    }
    if (image_size % 2 != 0) {
      std::cerr << "--image_size must be even for PatchMerge.\n";
      error = true;
    }
    if (shift_size < 0 || shift_size >= window_size) {
      std::cerr << "--shift_size must satisfy 0 <= shift_size < window_size.\n";
      error = true;
    }
    if (channels() % 8 != 0 || head_size % 8 != 0) {
      std::cerr << "TensorOp path requires channels and head_size to be multiples of 8.\n";
      error = true;
    }
    if (window_len() > 64) {
      std::cerr << "This teaching path currently supports window_len <= 64.\n";
      error = true;
    }
  }

  int channels() const {
    return head_number * head_size;
  }
  int window_len() const {
    return window_size * window_size;
  }
  int window_len_padded() const {
    int l = window_len();
    return ((l + 7) / 8) * 8;
  }
  int num_windows() const {
    int windows_per_side = image_size / window_size;
    return windows_per_side * windows_per_side;
  }
  int batched_windows() const {
    return batch_size * num_windows();
  }
  int rows() const {
    return batched_windows() * window_len();
  }

  double gflops(double runtime_s) const {
    int64_t rows_ = rows();
    int64_t c = channels();
    int64_t bw = batched_windows();
    int64_t h = head_number;
    int64_t l = window_len();
    int64_t d = head_size;
    int64_t qkv = int64_t(2) * rows_ * c * (3 * c);
    int64_t attention = int64_t(4) * bw * h * l * l * d
                      + int64_t(3) * bw * h * l * l;
    int64_t projection = int64_t(2) * rows_ * c * c;
    return double(qkv + attention + projection) / 1.0e9 / runtime_s;
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "swin_cutlass_test\n\n"
        << "Options:\n\n"
        << "  --help                      Displays this usage statement.\n\n"
        << "  --batch_size=<int>          Batch size (default: 2)\n"
        << "  --image_size=<int>          Square image resolution H=W (default: 14)\n"
        << "  --window_size=<int>         Swin local window size (default: 7)\n"
        << "  --shift_size=<int>          Cyclic shift size (default: window_size/2)\n"
        << "  --head_number=<int>         Number of attention heads (default: 3)\n"
        << "  --head_size=<int>           Per-head dimension (default: 32)\n"
        << "  --iterations=<int>          Profiling iterations (default: 20)\n"
        << "  --mask=<bool>               Apply shifted-window mask (default: true)\n"
        << "  --reference-check=<bool>    Run host reference check (default: true)\n";
    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
class SwinCutlassTestbed {
public:
  using ElementAccumulator = float;
  using ElementCompute = float;
  using Layout = cutlass::layout::RowMajor;
  using ArchTag = cutlass::arch::Sm80;

  using WindowPartition = WindowPartitionKernel<Element, 256>;
  using WindowReverse = WindowReverseKernel<Element, 256>;
  using PatchMerge = PatchMergeKernel<Element, 256>;
  using AddQkvBiasSplit = AddQkvBiasSplitKernel<Element, 256>;
  using AddBias = AddBiasKernel<Element, 256>;

  using ProjectionGemm = cutlass::gemm::device::Gemm<
      Element,
      Layout,
      Element,
      Layout,
      Element,
      Layout,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      cutlass::gemm::GemmShape<128, 64, 64>,
      cutlass::gemm::GemmShape<64, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          Element,
          128 / cutlass::sizeof_bits<Element>::value,
          ElementAccumulator,
          ElementCompute>>;

  using WindowAttention = AttentionKernel<
      Element,
      ArchTag,
      true,
      64,
      64,
      64,
      false,
      true>;

private:
  Options& options;
  uint32_t seed;

  std::vector<Element> host_qkv_weight;
  std::vector<Element> host_qkv_bias;
  std::vector<Element> host_output_weight;
  std::vector<Element> host_output_bias;
  std::vector<Element> host_rel_bias;
  std::vector<Element> host_attention_mask;

  cutlass::DeviceAllocation<Element> input;           // [B, H, W, C]
  cutlass::DeviceAllocation<Element> windows;         // [BW, L, C]
  cutlass::DeviceAllocation<Element> qkv;             // [BW * L, 3C]
  cutlass::DeviceAllocation<Element> q;               // [BW, L, heads, D]
  cutlass::DeviceAllocation<Element> k;               // [BW, L, heads, D]
  cutlass::DeviceAllocation<Element> v;               // [BW, L, heads, D]
  cutlass::DeviceAllocation<Element> attention_out;   // [BW, L, C]
  cutlass::DeviceAllocation<Element> projected;       // [BW, L, C]
  cutlass::DeviceAllocation<Element> output;          // [B, H, W, C]
  cutlass::DeviceAllocation<Element> patch_merged;    // [B, H/2, W/2, 4C]
  cutlass::DeviceAllocation<Element> qkv_weight;      // [C, 3C]
  cutlass::DeviceAllocation<Element> qkv_bias;        // [3C]
  cutlass::DeviceAllocation<Element> output_weight;   // [C, C]
  cutlass::DeviceAllocation<Element> output_bias;     // [C]
  cutlass::DeviceAllocation<Element> rel_mask_bias;   // [BW, heads, L, L_pad]

public:
  explicit SwinCutlassTestbed(Options& options_, uint32_t seed_ = 2026)
      : options(options_), seed(seed_) {}

private:
  void init_weights_() {
    int C = options.channels();
    int L = options.window_len();
    int Lp = options.window_len_padded();
    int Hh = options.head_number;
    int BW = options.batched_windows();
    int nW = options.num_windows();

    host_qkv_weight.resize(int64_t(C) * 3 * C);
    host_qkv_bias.resize(3 * C);
    host_output_weight.resize(int64_t(C) * C);
    host_output_bias.resize(C);
    host_rel_bias.resize(int64_t(Hh) * L * L);

    for (int k_col = 0; k_col < C; ++k_col) {
      for (int n = 0; n < 3 * C; ++n) {
        host_qkv_weight[int64_t(k_col) * (3 * C) + n] =
            Element(0.015f * std::sin(float((k_col + 1) * (n + 3)) * 0.011f));
      }
      for (int n = 0; n < C; ++n) {
        host_output_weight[int64_t(k_col) * C + n] =
            Element(0.013f * std::cos(float((k_col + 5) * (n + 1)) * 0.009f));
      }
    }
    for (int n = 0; n < 3 * C; ++n) {
      host_qkv_bias[n] = Element(0.01f * std::sin(float(n + 1) * 0.017f));
    }
    for (int n = 0; n < C; ++n) {
      host_output_bias[n] = Element(0.01f * std::cos(float(n + 1) * 0.019f));
    }
    for (int h = 0; h < Hh; ++h) {
      for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
          host_rel_bias[(int64_t(h) * L + i) * L + j] =
              Element(0.01f * std::sin(float((h + 1) * (i - j))));
        }
      }
    }

    build_shift_attention_mask_();

    std::vector<Element> rel_mask_bias_host(int64_t(BW) * Hh * L * Lp, Element(0));
    for (int bw = 0; bw < BW; ++bw) {
      int window_id = bw % nW;
      for (int h = 0; h < Hh; ++h) {
        for (int i = 0; i < L; ++i) {
          for (int j = 0; j < L; ++j) {
            float bias = float(host_rel_bias[(int64_t(h) * L + i) * L + j]);
            if (options.use_mask) {
              bias += float(host_attention_mask[(int64_t(window_id) * L + i) * L + j]);
            }
            rel_mask_bias_host[((int64_t(bw) * Hh + h) * L + i) * Lp + j] =
                Element(bias);
          }
        }
      }
    }
    rel_mask_bias.copy_from_host(rel_mask_bias_host.data());
  }

  void initialize_() {
    int B = options.batch_size;
    int I = options.image_size;
    int C = options.channels();
    int L = options.window_len();
    int Lp = options.window_len_padded();
    int BW = options.batched_windows();

    int64_t input_elements = int64_t(B) * I * I * C;
    int64_t window_elements = int64_t(BW) * L * C;
    int64_t qkv_elements = int64_t(BW) * L * 3 * C;
    int64_t patch_elements = int64_t(B) * (I / 2) * (I / 2) * (4 * C);

    input.reset(input_elements);
    windows.reset(window_elements);
    qkv.reset(qkv_elements);
    q.reset(window_elements);
    k.reset(window_elements);
    v.reset(window_elements);
    attention_out.reset(window_elements);
    projected.reset(window_elements);
    output.reset(input_elements);
    patch_merged.reset(patch_elements);
    qkv_weight.reset(int64_t(C) * 3 * C);
    qkv_bias.reset(3 * C);
    output_weight.reset(int64_t(C) * C);
    output_bias.reset(C);
    rel_mask_bias.reset(int64_t(BW) * options.head_number * L * Lp);

    cutlass::reference::device::BlockFillRandomUniform(
        input.get(), input_elements, seed, Element(1), Element(-1), 0);

    init_weights_();
    qkv_weight.copy_from_host(host_qkv_weight.data());
    qkv_bias.copy_from_host(host_qkv_bias.data());
    output_weight.copy_from_host(host_output_weight.data());
    output_bias.copy_from_host(host_output_bias.data());

    cudaMemset(windows.get(), 0, window_elements * sizeof(Element));
    cudaMemset(qkv.get(), 0, qkv_elements * sizeof(Element));
    cudaMemset(q.get(), 0, window_elements * sizeof(Element));
    cudaMemset(k.get(), 0, window_elements * sizeof(Element));
    cudaMemset(v.get(), 0, window_elements * sizeof(Element));
    cudaMemset(attention_out.get(), 0, window_elements * sizeof(Element));
    cudaMemset(projected.get(), 0, window_elements * sizeof(Element));
    cudaMemset(output.get(), 0, input_elements * sizeof(Element));
    cudaMemset(patch_merged.get(), 0, patch_elements * sizeof(Element));
  }

  void build_shift_attention_mask_() {
    int I = options.image_size;
    int W = options.window_size;
    int S = options.shift_size;
    int L = options.window_len();
    int nW = options.num_windows();

    host_attention_mask.assign(int64_t(nW) * L * L, Element(0));
    if (!options.use_mask || S == 0) {
      return;
    }

    std::vector<int> image_mask(I * I, 0);
    int cnt = 0;
    for (int y_region = 0; y_region < 3; ++y_region) {
      int y0 = (y_region == 0) ? 0 : ((y_region == 1) ? I - W : I - S);
      int y1 = (y_region == 0) ? I - W : ((y_region == 1) ? I - S : I);
      for (int x_region = 0; x_region < 3; ++x_region) {
        int x0 = (x_region == 0) ? 0 : ((x_region == 1) ? I - W : I - S);
        int x1 = (x_region == 0) ? I - W : ((x_region == 1) ? I - S : I);
        for (int y = y0; y < y1; ++y) {
          for (int x = x0; x < x1; ++x) {
            image_mask[y * I + x] = cnt;
          }
        }
        ++cnt;
      }
    }

    std::vector<int> window_regions(nW * L, 0);
    int windows_per_row = I / W;
    for (int y = 0; y < I; ++y) {
      for (int x = 0; x < I; ++x) {
        int shifted_y = (y - S + I) % I;
        int shifted_x = (x - S + I) % I;
        int window_y = shifted_y / W;
        int window_x = shifted_x / W;
        int window_idx = window_y * windows_per_row + window_x;
        int idx_in_window = (shifted_y % W) * W + (shifted_x % W);
        window_regions[window_idx * L + idx_in_window] = image_mask[y * I + x];
      }
    }

    for (int w = 0; w < nW; ++w) {
      for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
          bool same_region =
              window_regions[w * L + i] == window_regions[w * L + j];
          host_attention_mask[(int64_t(w) * L + i) * L + j] =
              same_region ? Element(0) : Element(-100);
        }
      }
    }
  }

  static cudaError_t check_stage_(char const* stage) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << stage << " error: " << cudaGetErrorString(err) << "\n";
    }
    return err;
  }

  cutlass::Status launch_qkv_projection_() {
    int C = options.channels();
    typename ProjectionGemm::Arguments args(
        {options.rows(), 3 * C, C},
        {windows.get(), C},
        {qkv_weight.get(), 3 * C},
        {qkv.get(), 3 * C},
        {qkv.get(), 3 * C},
        {ElementCompute(1), ElementCompute(0)});
    cutlass::Status status = ProjectionGemm::can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    ProjectionGemm gemm;
    return gemm(args);
  }

  cudaError_t launch_qkv_split_() {
    typename AddQkvBiasSplit::Params p;
    p.qkv = qkv.get();
    p.bias = qkv_bias.get();
    p.q = q.get();
    p.k = k.get();
    p.v = v.get();
    p.elements = options.rows() * options.channels();
    p.channels = options.channels();
    swin_add_qkv_bias_split_kernel<AddQkvBiasSplit>
        <<<p.getBlocksGrid(), p.getThreadsGrid()>>>(p);
    return check_stage_("qkv_bias_split");
  }

  cudaError_t launch_window_attention_() {
    int BW = options.batched_windows();
    int L = options.window_len();
    int Lp = options.window_len_padded();
    int Hh = options.head_number;
    int D = options.head_size;
    int C = options.channels();

    typename WindowAttention::Params p;
    p.query_ptr = q.get();
    p.key_ptr = k.get();
    p.value_ptr = v.get();
    p.attn_bias_ptr = rel_mask_bias.get();
    p.output_ptr = attention_out.get();
    p.output_accum_ptr = nullptr;
    p.logsumexp_ptr = nullptr;
    p.scale = options.alpha;
    p.num_queries = L;
    p.num_keys = L;
    p.head_dim = D;
    p.head_dim_value = D;
    p.num_heads = Hh;
    p.num_batches = BW;
    p.q_strideM = C;
    p.k_strideM = C;
    p.v_strideM = C;
    p.bias_strideM = Lp;
    p.o_strideM = C;
    p.q_strideH = D;
    p.k_strideH = D;
    p.v_strideH = D;
    p.bias_strideH = int64_t(L) * Lp;
    p.q_strideB = int64_t(L) * C;
    p.k_strideB = int64_t(L) * C;
    p.v_strideB = int64_t(L) * C;
    p.bias_strideB = int64_t(Hh) * L * Lp;
    p.custom_mask_type = WindowAttention::NoCustomMask;

    constexpr auto kernel_fn = attention_kernel_batched_impl<WindowAttention>;
    int smem_bytes = int(sizeof(typename WindowAttention::SharedStorage));
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!WindowAttention::check_supported(p)) {
      return cudaErrorInvalidValue;
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
    return check_stage_("window_attention");
  }

  cutlass::Status launch_output_projection_() {
    int C = options.channels();
    typename ProjectionGemm::Arguments args(
        {options.rows(), C, C},
        {attention_out.get(), C},
        {output_weight.get(), C},
        {projected.get(), C},
        {projected.get(), C},
        {ElementCompute(1), ElementCompute(0)});
    cutlass::Status status = ProjectionGemm::can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    ProjectionGemm gemm;
    return gemm(args);
  }

  cudaError_t launch_output_bias_() {
    typename AddBias::Params p;
    p.output = projected.get();
    p.bias = output_bias.get();
    p.elements = options.rows() * options.channels();
    p.channels = options.channels();
    swin_add_bias_kernel<AddBias><<<p.getBlocksGrid(), p.getThreadsGrid()>>>(p);
    return check_stage_("output_bias");
  }

  Result launch_once_() {
    Result result;

    {
      typename WindowPartition::Params p;
      p.input = input.get();
      p.output = windows.get();
      p.batch = options.batch_size;
      p.height = options.image_size;
      p.width = options.image_size;
      p.channels = options.channels();
      p.shift_size = options.shift_size;
      p.window_size = options.window_size;
      swin_window_partition_kernel<WindowPartition>
          <<<p.getBlocksGrid(), p.getThreadsGrid()>>>(p);
      result.error = check_stage_("window_partition");
      if (result.error != cudaSuccess) {
        result.passed = false;
        return result;
      }
    }

    result.status = launch_qkv_projection_();
    if (result.status != cutlass::Status::kSuccess) {
      result.passed = false;
      return result;
    }
    result.error = launch_qkv_split_();
    if (result.error != cudaSuccess) {
      result.passed = false;
      return result;
    }

    result.error = launch_window_attention_();
    if (result.error != cudaSuccess) {
      result.passed = false;
      return result;
    }

    result.status = launch_output_projection_();
    if (result.status != cutlass::Status::kSuccess) {
      result.passed = false;
      return result;
    }
    result.error = launch_output_bias_();
    if (result.error != cudaSuccess) {
      result.passed = false;
      return result;
    }

    {
      typename WindowReverse::Params p;
      p.input = projected.get();
      p.output = output.get();
      p.batch = options.batch_size;
      p.height = options.image_size;
      p.width = options.image_size;
      p.channels = options.channels();
      p.shift_size = options.shift_size;
      p.window_size = options.window_size;
      swin_window_reverse_kernel<WindowReverse>
          <<<p.getBlocksGrid(), p.getThreadsGrid()>>>(p);
      result.error = check_stage_("window_reverse");
      if (result.error != cudaSuccess) {
        result.passed = false;
        return result;
      }
    }

    {
      typename PatchMerge::Params p;
      p.input = input.get();
      p.output = patch_merged.get();
      p.batch = options.batch_size;
      p.height = options.image_size;
      p.width = options.image_size;
      p.channels = options.channels();
      swin_patch_merge_kernel<PatchMerge>
          <<<p.getBlocksGrid(), p.getThreadsGrid()>>>(p);
      result.error = check_stage_("patch_merge");
      if (result.error != cudaSuccess) {
        result.passed = false;
        return result;
      }
    }

    return result;
  }

  void window_partition_host_(std::vector<Element> const& src,
                              std::vector<Element>& dst) const {
    int B = options.batch_size;
    int I = options.image_size;
    int W = options.window_size;
    int S = options.shift_size;
    int C = options.channels();
    int L = options.window_len();
    int windows_per_row = I / W;
    int tokens_per_batch = I * I;

    dst.assign(int64_t(B) * tokens_per_batch * C, Element(0));
    for (int b = 0; b < B; ++b) {
      for (int y = 0; y < I; ++y) {
        for (int x = 0; x < I; ++x) {
          int shifted_y = S != 0 ? ((y - S + I) % I) : y;
          int shifted_x = S != 0 ? ((x - S + I) % I) : x;
          int window_y = shifted_y / W;
          int window_x = shifted_x / W;
          int window_idx = window_y * windows_per_row + window_x;
          int idx_in_window = (shifted_y % W) * W + (shifted_x % W);
          int64_t input_token = int64_t(b) * I * I + y * I + x;
          int64_t output_token = int64_t(b) * tokens_per_batch
                               + int64_t(window_idx) * L
                               + idx_in_window;
          for (int c = 0; c < C; ++c) {
            dst[output_token * C + c] = src[input_token * C + c];
          }
        }
      }
    }
  }

  void window_reverse_host_(std::vector<Element> const& src,
                            std::vector<Element>& dst) const {
    int B = options.batch_size;
    int I = options.image_size;
    int W = options.window_size;
    int S = options.shift_size;
    int C = options.channels();
    int L = options.window_len();
    int windows_per_row = I / W;
    int tokens_per_batch = I * I;

    dst.assign(int64_t(B) * I * I * C, Element(0));
    for (int b = 0; b < B; ++b) {
      for (int y = 0; y < I; ++y) {
        for (int x = 0; x < I; ++x) {
          int shifted_y = S != 0 ? ((y - S + I) % I) : y;
          int shifted_x = S != 0 ? ((x - S + I) % I) : x;
          int window_y = shifted_y / W;
          int window_x = shifted_x / W;
          int window_idx = window_y * windows_per_row + window_x;
          int idx_in_window = (shifted_y % W) * W + (shifted_x % W);
          int64_t input_token = int64_t(b) * tokens_per_batch
                              + int64_t(window_idx) * L
                              + idx_in_window;
          int64_t output_token = int64_t(b) * I * I + y * I + x;
          for (int c = 0; c < C; ++c) {
            dst[output_token * C + c] = src[input_token * C + c];
          }
        }
      }
    }
  }

  void patch_merge_host_(std::vector<Element> const& src,
                         std::vector<Element>& dst) const {
    int B = options.batch_size;
    int I = options.image_size;
    int C = options.channels();
    int OH = I / 2;
    int OW = I / 2;

    dst.assign(int64_t(B) * OH * OW * (4 * C), Element(0));
    for (int b = 0; b < B; ++b) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          for (int col = 0; col < 4 * C; ++col) {
            int part = col / C;
            int c = col - part * C;
            int offset_w = part / 2;
            int offset_h = part % 2;
            int y = 2 * oh + offset_h;
            int x = 2 * ow + offset_w;
            int64_t input_idx =
                ((int64_t(b) * I * I + y * I + x) * C) + c;
            int64_t output_idx =
                ((int64_t(b) * OH * OW + oh * OW + ow) * (4 * C)) + col;
            dst[output_idx] = src[input_idx];
          }
        }
      }
    }
  }

  void qkv_projection_host_(std::vector<Element> const& src,
                            std::vector<Element>& q_host,
                            std::vector<Element>& k_host,
                            std::vector<Element>& v_host) const {
    int rows = options.rows();
    int C = options.channels();
    q_host.assign(int64_t(rows) * C, Element(0));
    k_host.assign(int64_t(rows) * C, Element(0));
    v_host.assign(int64_t(rows) * C, Element(0));

    for (int row = 0; row < rows; ++row) {
      for (int n = 0; n < 3 * C; ++n) {
        float acc = float(host_qkv_bias[n]);
        for (int c = 0; c < C; ++c) {
          acc += float(src[int64_t(row) * C + c])
               * float(host_qkv_weight[int64_t(c) * (3 * C) + n]);
        }
        Element value = Element(acc);
        if (n < C) {
          q_host[int64_t(row) * C + n] = value;
        } else if (n < 2 * C) {
          k_host[int64_t(row) * C + (n - C)] = value;
        } else {
          v_host[int64_t(row) * C + (n - 2 * C)] = value;
        }
      }
    }
  }

  void attention_host_(std::vector<Element> const& q_host,
                       std::vector<Element> const& k_host,
                       std::vector<Element> const& v_host,
                       std::vector<Element>& dst) const {
    int BW = options.batched_windows();
    int Hh = options.head_number;
    int D = options.head_size;
    int C = options.channels();
    int L = options.window_len();
    int nW = options.num_windows();
    float scale = options.alpha;

    dst.assign(int64_t(BW) * L * C, Element(0));
    std::vector<float> row(L);

    for (int bw = 0; bw < BW; ++bw) {
      int window_id = bw % nW;
      for (int h = 0; h < Hh; ++h) {
        for (int i = 0; i < L; ++i) {
          float row_max = -std::numeric_limits<float>::infinity();
          for (int j = 0; j < L; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
              float qv = float(q_host[(int64_t(bw) * L + i) * C + h * D + d]);
              float kv = float(k_host[(int64_t(bw) * L + j) * C + h * D + d]);
              dot += qv * kv;
            }
            float bias = float(host_rel_bias[(int64_t(h) * L + i) * L + j]);
            if (options.use_mask) {
              bias += float(host_attention_mask[(int64_t(window_id) * L + i) * L + j]);
            }
            row[j] = dot * scale + bias;
            row_max = std::max(row_max, row[j]);
          }

          float row_sum = 0.0f;
          for (int j = 0; j < L; ++j) {
            row[j] = std::exp(row[j] - row_max);
            row_sum += row[j];
          }
          float inv_sum = 1.0f / row_sum;
          for (int j = 0; j < L; ++j) {
            row[j] *= inv_sum;
          }

          for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < L; ++j) {
              float vv = float(v_host[(int64_t(bw) * L + j) * C + h * D + d]);
              acc += row[j] * vv;
            }
            dst[(int64_t(bw) * L + i) * C + h * D + d] = Element(acc);
          }
        }
      }
    }
  }

  void output_projection_host_(std::vector<Element> const& src,
                               std::vector<Element>& dst) const {
    int rows = options.rows();
    int C = options.channels();
    dst.assign(int64_t(rows) * C, Element(0));
    for (int row = 0; row < rows; ++row) {
      for (int n = 0; n < C; ++n) {
        float acc = float(host_output_bias[n]);
        for (int c = 0; c < C; ++c) {
          acc += float(src[int64_t(row) * C + c])
               * float(host_output_weight[int64_t(c) * C + n]);
        }
        dst[int64_t(row) * C + n] = Element(acc);
      }
    }
  }

  bool compare_(std::vector<Element> const& computed,
                std::vector<Element> const& reference,
                std::string const& label,
                float abs_tol,
                float rel_tol) const {
    if (computed.size() != reference.size()) {
      std::cerr << label << ": size mismatch\n";
      return false;
    }
    for (size_t i = 0; i < computed.size(); ++i) {
      float a = float(computed[i]);
      float r = float(reference[i]);
      float diff = std::fabs(a - r);
      float rel = diff / (std::fabs(r) + 1e-5f);
      if (std::isnan(a) || std::isinf(a) || (diff > abs_tol && rel > rel_tol)) {
        std::cerr << label << " mismatch at " << i
                  << ": computed=" << a
                  << " reference=" << r
                  << " diff=" << diff
                  << " rel=" << rel << "\n";
        return false;
      }
    }
    return true;
  }

  bool verify_() {
    int B = options.batch_size;
    int I = options.image_size;
    int C = options.channels();
    int L = options.window_len();
    int BW = options.batched_windows();

    int64_t input_elements = int64_t(B) * I * I * C;
    int64_t window_elements = int64_t(BW) * L * C;
    int64_t patch_elements = int64_t(B) * (I / 2) * (I / 2) * (4 * C);

    std::vector<Element> host_input(input_elements);
    std::vector<Element> host_windows(window_elements);
    std::vector<Element> host_attention(window_elements);
    std::vector<Element> host_projected(window_elements);
    std::vector<Element> host_output(input_elements);
    std::vector<Element> host_patch(patch_elements);

    cutlass::device_memory::copy_to_host(host_input.data(), input.get(), input_elements);
    cutlass::device_memory::copy_to_host(host_windows.data(), windows.get(), window_elements);
    cutlass::device_memory::copy_to_host(host_attention.data(), attention_out.get(), window_elements);
    cutlass::device_memory::copy_to_host(host_projected.data(), projected.get(), window_elements);
    cutlass::device_memory::copy_to_host(host_output.data(), output.get(), input_elements);
    cutlass::device_memory::copy_to_host(host_patch.data(), patch_merged.get(), patch_elements);

    std::vector<Element> ref_windows;
    std::vector<Element> ref_q;
    std::vector<Element> ref_k;
    std::vector<Element> ref_v;
    std::vector<Element> ref_attention;
    std::vector<Element> ref_projected;
    std::vector<Element> ref_output;
    std::vector<Element> ref_patch;

    window_partition_host_(host_input, ref_windows);
    qkv_projection_host_(ref_windows, ref_q, ref_k, ref_v);
    attention_host_(ref_q, ref_k, ref_v, ref_attention);
    output_projection_host_(ref_attention, ref_projected);
    window_reverse_host_(ref_projected, ref_output);
    patch_merge_host_(host_input, ref_patch);

    return compare_(host_windows, ref_windows, "WindowPartition", 0.0f, 0.0f)
        && compare_(host_attention, ref_attention, "WindowAttention", 1.4e-1f, 2.5e-1f)
        && compare_(host_projected, ref_projected, "OutputProjection", 1.6e-1f, 3.0e-1f)
        && compare_(host_output, ref_output, "WindowReverse", 1.6e-1f, 3.0e-1f)
        && compare_(host_patch, ref_patch, "PatchMerge", 0.0f, 0.0f);
  }

public:
  Result profile() {
    Result result;
    result.passed = false;

    initialize_();

    result = launch_once_();
    if (!result.passed) {
      return result;
    }
    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess) {
      std::cerr << "Kernel exec error: " << cudaGetErrorString(result.error) << "\n";
      result.passed = false;
      return result;
    }
    result.passed = !options.reference_check || verify_();
    if (!result.passed) {
      return result;
    }

    result = launch_once_();
    if (!result.passed) {
      return result;
    }
    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess) {
      result.passed = false;
      return result;
    }

    cudaEvent_t events[2];
    result.error = cudaEventCreate(&events[0]);
    if (result.error != cudaSuccess) {
      return result;
    }
    result.error = cudaEventCreate(&events[1]);
    if (result.error != cudaSuccess) {
      cudaEventDestroy(events[0]);
      return result;
    }

    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      cudaEventDestroy(events[0]);
      cudaEventDestroy(events[1]);
      return result;
    }
    for (int i = 0; i < options.iterations; ++i) {
      result = launch_once_();
      if (!result.passed) {
        cudaEventDestroy(events[0]);
        cudaEventDestroy(events[1]);
        return result;
      }
    }
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      cudaEventDestroy(events[0]);
      cudaEventDestroy(events[1]);
      return result;
    }
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      cudaEventDestroy(events[0]);
      cudaEventDestroy(events[1]);
      return result;
    }

    float ms = 0.0f;
    result.error = cudaEventElapsedTime(&ms, events[0], events[1]);
    cudaEventDestroy(events[0]);
    cudaEventDestroy(events[1]);
    if (result.error != cudaSuccess) {
      return result;
    }

    result.runtime_ms = double(ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);
    return result;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace swin
} // namespace tiny_cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const** args) {
  using namespace tiny_cutlass::swin;

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties: " << cudaGetErrorString(error) << "\n";
    return -1;
  }

  Options options;
  options.parse(argc, args);
  if (options.help) {
    options.print_usage(std::cout) << "\n";
    return 0;
  }
  if (options.error) {
    std::cerr << "Aborting.\n";
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
    std::cout << "This example requires Ampere (SM80) or later.\n";
    return 0;
  }

  SwinCutlassTestbed<cutlass::half_t> testbed(options);
  Result result = testbed.profile();

  std::cout << "\nSwin CUTLASS path:\n"
            << "====================================================\n"
            << "Device: " << props.name << " (SM" << props.major << props.minor << ")\n"
            << "  {B, I, window, shift, heads, head_dim} = {"
            << options.batch_size << ", "
            << options.image_size << ", "
            << options.window_size << ", "
            << options.shift_size << ", "
            << options.head_number << ", "
            << options.head_size << "}\n"
            << "  Path: WindowPartition -> QKV -> WindowAttention -> OutputProjection -> WindowReverse\n"
            << "  num_windows=" << options.num_windows()
            << ", window_len=" << options.window_len()
            << ", bias_stride=" << options.window_len_padded()
            << ", mask=" << (options.use_mask ? "true" : "false") << "\n"
            << "  Runtime: " << result.runtime_ms << " ms\n"
            << "  GFLOPs : " << result.gflops << "\n";

  if (!result.passed || result.status != cutlass::Status::kSuccess ||
      result.error != cudaSuccess) {
    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "CUTLASS status: " << cutlassGetStatusString(result.status) << "\n";
    }
    if (result.error != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(result.error) << "\n";
    }
    std::cout << "\nFailed\n";
    return -1;
  }

  std::cout << "\nPassed\n";
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
