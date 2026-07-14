// Full fused SwinBlock verification vs a
// self-contained host reference implementing the complete Microsoft
// SwinTransformerBlock (v1 pre-norm): norm1 -> shift/partition -> window
// attention -> reverse -> residual1 -> norm2 -> MLP(fc1,GELU,fc2) -> residual2.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"

#include "../../swin/swin.h"

namespace tiny_cutlass {
namespace swin {
namespace {

using Runner = device::SwinBlock<cutlass::arch::Sm80, cutlass::half_t>;
using Element = typename Runner::Element;
using Tensors = typename Runner::Tensors;

struct Options {
  bool help = false;
  bool error = false;
  bool use_mask = true;

  int batch_size = 2;
  int image_size = 14;
  int window_size = 7;
  int shift_size = 3;
  int head_number = 3;
  int head_size = 32;
  int mlp_ratio = 4;
  int iterations = 20;
  int seed = 2026;

  float mae_tolerance = 3.0e-2f;

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
    cmd.get_cmd_line_argument("mlp_ratio", mlp_ratio, mlp_ratio);
    cmd.get_cmd_line_argument("iterations", iterations, iterations);
    cmd.get_cmd_line_argument("seed", seed, seed);
    cmd.get_cmd_line_argument("mask", use_mask, use_mask);
    cmd.get_cmd_line_argument("mae-tolerance", mae_tolerance, mae_tolerance);

    if (batch_size <= 0 || image_size <= 0 || window_size <= 0 ||
        head_number <= 0 || head_size <= 0 || mlp_ratio <= 0 ||
        iterations <= 0 || mae_tolerance <= 0.0f) {
      error = true;
    }
  }

  SwinBlockProblem problem() const {
    SwinBlockProblem p;
    p.batch_size = batch_size;
    p.image_size = image_size;
    p.window_size = window_size;
    p.shift_size = shift_size;
    p.head_number = head_number;
    p.head_size = head_size;
    p.mlp_ratio = mlp_ratio;
    p.scale = 1.0f / std::sqrt(float(head_size));
    return p;
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "swin_block\n\n"
        << "Options:\n\n"
        << "  --help                    Display this usage statement.\n"
        << "  --batch_size=<int>        Batch size (default: 2).\n"
        << "  --image_size=<int>        Square resolution H=W (default: 14).\n"
        << "  --window_size=<int>       Local window size (default: 7).\n"
        << "  --shift_size=<int>        Cyclic shift (default: window_size/2).\n"
        << "  --head_number=<int>       Attention heads (default: 3).\n"
        << "  --head_size=<int>         Per-head dim (default: 32).\n"
        << "  --mlp_ratio=<int>         MLP hidden ratio (default: 4).\n"
        << "  --mask=<bool>             Apply shifted-window mask (default: true).\n"
        << "  --mae-tolerance=<float>   Full-block MAE tolerance (default: 3e-2).\n"
        << "  --iterations=<int>        Timed iterations (default: 20).\n"
        << "  --seed=<int>              Random seed (default: 2026).\n";
    return out;
  }
};

template <typename T>
void copy_to_device(cutlass::DeviceAllocation<T>& dst, std::vector<T> const& src) {
  cutlass::device_memory::copy_to_device(dst.get(), src.data(), src.size());
}

void fill_random_uniform(std::vector<Element>& dst, int seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& v : dst) {
    v = Element(dist(rng));
  }
}

// ---- weight / bias generators (deterministic, small magnitude) -------------

void fill_block_weights(
    SwinBlockProblem const& p,
    std::vector<Element>& qkv_weight,
    std::vector<Element>& qkv_bias,
    std::vector<Element>& output_weight,
    std::vector<Element>& output_bias,
    std::vector<Element>& gamma1,
    std::vector<Element>& beta1,
    std::vector<Element>& gamma2,
    std::vector<Element>& beta2,
    std::vector<Element>& fc1_weight,
    std::vector<Element>& fc1_bias,
    std::vector<Element>& fc2_weight,
    std::vector<Element>& fc2_bias) {
  int c = swin_channels(p);
  int h = swin_mlp_hidden(p);
  qkv_weight.resize(swin_qkv_weight_elements(p));
  qkv_bias.resize(3 * c);
  output_weight.resize(swin_output_weight_elements(p));
  output_bias.resize(c);
  gamma1.resize(c);
  beta1.resize(c);
  gamma2.resize(c);
  beta2.resize(c);
  fc1_weight.resize(swin_mlp_weight1_elements(p));
  fc1_bias.resize(h);
  fc2_weight.resize(swin_mlp_weight2_elements(p));
  fc2_bias.resize(c);

  for (int k_col = 0; k_col < c; ++k_col) {
    for (int n = 0; n < 3 * c; ++n) {
      qkv_weight[int64_t(k_col) * (3 * c) + n] =
          Element(0.015f * std::sin(float((k_col + 1) * (n + 3)) * 0.011f));
    }
    for (int n = 0; n < c; ++n) {
      output_weight[int64_t(k_col) * c + n] =
          Element(0.013f * std::cos(float((k_col + 5) * (n + 1)) * 0.009f));
    }
    for (int n = 0; n < h; ++n) {
      fc1_weight[int64_t(k_col) * h + n] =
          Element(0.012f * std::sin(float((k_col + 2) * (n + 1)) * 0.007f));
    }
  }
  for (int k_col = 0; k_col < h; ++k_col) {
    for (int n = 0; n < c; ++n) {
      fc2_weight[int64_t(k_col) * c + n] =
          Element(0.010f * std::cos(float((k_col + 3) * (n + 2)) * 0.006f));
    }
  }
  for (int n = 0; n < 3 * c; ++n) {
    qkv_bias[n] = Element(0.01f * std::sin(float(n + 1) * 0.017f));
  }
  for (int n = 0; n < c; ++n) {
    output_bias[n] = Element(0.01f * std::cos(float(n + 1) * 0.019f));
    gamma1[n] = Element(1.0f + 0.02f * std::sin(float(n + 1) * 0.013f));
    beta1[n] = Element(0.01f * std::cos(float(n + 1) * 0.021f));
    gamma2[n] = Element(1.0f + 0.02f * std::cos(float(n + 1) * 0.015f));
    beta2[n] = Element(0.01f * std::sin(float(n + 1) * 0.023f));
    fc2_bias[n] = Element(0.01f * std::sin(float(n + 3) * 0.011f));
  }
  for (int n = 0; n < h; ++n) {
    fc1_bias[n] = Element(0.01f * std::cos(float(n + 1) * 0.009f));
  }
}

void build_shift_attention_mask(
    SwinBlockProblem const& p,
    bool use_mask,
    std::vector<Element>& mask) {
  int image = p.image_size;
  int window = p.window_size;
  int shift = p.shift_size;
  int l = swin_window_len(p);
  int nW = swin_num_windows(p);

  mask.assign(int64_t(nW) * l * l, Element(0));
  if (!use_mask || shift == 0) {
    return;
  }

  std::vector<int> image_mask(image * image, 0);
  int cnt = 0;
  for (int y_region = 0; y_region < 3; ++y_region) {
    int y0 = (y_region == 0) ? 0 : ((y_region == 1) ? image - window : image - shift);
    int y1 = (y_region == 0) ? image - window : ((y_region == 1) ? image - shift : image);
    for (int x_region = 0; x_region < 3; ++x_region) {
      int x0 = (x_region == 0) ? 0 : ((x_region == 1) ? image - window : image - shift);
      int x1 = (x_region == 0) ? image - window : ((x_region == 1) ? image - shift : image);
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          image_mask[y * image + x] = cnt;
        }
      }
      ++cnt;
    }
  }

  std::vector<int> window_regions(nW * l, 0);
  int windows_per_row = image / window;
  for (int y = 0; y < image; ++y) {
    for (int x = 0; x < image; ++x) {
      int shifted_y = (y - shift + image) % image;
      int shifted_x = (x - shift + image) % image;
      int window_idx = (shifted_y / window) * windows_per_row + (shifted_x / window);
      int idx_in_window = (shifted_y % window) * window + (shifted_x % window);
      window_regions[window_idx * l + idx_in_window] = image_mask[y * image + x];
    }
  }

  for (int w = 0; w < nW; ++w) {
    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < l; ++j) {
        bool same = window_regions[w * l + i] == window_regions[w * l + j];
        mask[(int64_t(w) * l + i) * l + j] = same ? Element(0) : Element(-100);
      }
    }
  }
}

void build_attention_bias(
    SwinBlockProblem const& p,
    bool use_mask,
    std::vector<Element>& attention_bias) {
  int bw = swin_batched_windows(p);
  int h = p.head_number;
  int l = swin_window_len(p);
  int lp = swin_window_len_padded(p);
  int nW = swin_num_windows(p);

  std::vector<Element> rel_bias(int64_t(h) * l * l);
  for (int head = 0; head < h; ++head) {
    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < l; ++j) {
        rel_bias[(int64_t(head) * l + i) * l + j] =
            Element(0.01f * std::sin(float((head + 1) * (i - j))));
      }
    }
  }

  std::vector<Element> mask;
  build_shift_attention_mask(p, use_mask, mask);

  attention_bias.assign(swin_attention_bias_elements(p), Element(0));
  for (int batch_window = 0; batch_window < bw; ++batch_window) {
    int window_id = batch_window % nW;
    for (int head = 0; head < h; ++head) {
      for (int i = 0; i < l; ++i) {
        for (int j = 0; j < l; ++j) {
          float v = float(rel_bias[(int64_t(head) * l + i) * l + j]);
          if (use_mask) {
            v += float(mask[(int64_t(window_id) * l + i) * l + j]);
          }
          attention_bias[((int64_t(batch_window) * h + head) * l + i) * lp + j] =
              Element(v);
        }
      }
    }
  }
}

// ---- full-block host reference ---------------------------------------------

void layernorm_host(
    std::vector<Element> const& in,
    std::vector<Element> const& gamma,
    std::vector<Element> const& beta,
    int tokens,
    int channels,
    float eps,
    std::vector<Element>& out) {
  out.assign(int64_t(tokens) * channels, Element(0));
  for (int t = 0; t < tokens; ++t) {
    double sum = 0.0, sq = 0.0;
    for (int c = 0; c < channels; ++c) {
      float v = float(in[int64_t(t) * channels + c]);
      sum += v;
      sq += double(v) * v;
    }
    float mean = float(sum / channels);
    float var = float(sq / channels) - mean * mean;
    float inv = 1.0f / std::sqrt(var + eps);
    for (int c = 0; c < channels; ++c) {
      float v = float(in[int64_t(t) * channels + c]);
      float norm = (v - mean) * inv;
      out[int64_t(t) * channels + c] =
          Element(norm * float(gamma[c]) + float(beta[c]));
    }
  }
}

void shift_partition_host(
    SwinBlockProblem const& p,
    std::vector<Element> const& in,
    std::vector<Element>& out) {
  int bsz = p.batch_size, image = p.image_size, window = p.window_size;
  int shift = p.shift_size, c = swin_channels(p), l = swin_window_len(p);
  int wpr = image / window;
  int tpb = image * image;
  out.assign(int64_t(bsz) * tpb * c, Element(0));
  for (int b = 0; b < bsz; ++b) {
    for (int y = 0; y < image; ++y) {
      for (int x = 0; x < image; ++x) {
        int sy = shift ? ((y - shift + image) % image) : y;
        int sx = shift ? ((x - shift + image) % image) : x;
        int widx = (sy / window) * wpr + (sx / window);
        int iw = (sy % window) * window + (sx % window);
        int64_t src = int64_t(b) * tpb + y * image + x;
        int64_t dst = int64_t(b) * tpb + int64_t(widx) * l + iw;
        for (int ch = 0; ch < c; ++ch) {
          out[dst * c + ch] = in[src * c + ch];
        }
      }
    }
  }
}

void reverse_host(
    SwinBlockProblem const& p,
    std::vector<Element> const& in,
    std::vector<Element>& out) {
  int bsz = p.batch_size, image = p.image_size, window = p.window_size;
  int shift = p.shift_size, c = swin_channels(p), l = swin_window_len(p);
  int wpr = image / window;
  int tpb = image * image;
  out.assign(int64_t(bsz) * tpb * c, Element(0));
  for (int b = 0; b < bsz; ++b) {
    for (int y = 0; y < image; ++y) {
      for (int x = 0; x < image; ++x) {
        int sy = shift ? ((y - shift + image) % image) : y;
        int sx = shift ? ((x - shift + image) % image) : x;
        int widx = (sy / window) * wpr + (sx / window);
        int iw = (sy % window) * window + (sx % window);
        int64_t src = int64_t(b) * tpb + int64_t(widx) * l + iw;
        int64_t dst = int64_t(b) * tpb + y * image + x;
        for (int ch = 0; ch < c; ++ch) {
          out[dst * c + ch] = in[src * c + ch];
        }
      }
    }
  }
}

// windows[rows,C] @ W[C,3C] + bias -> q,k,v each [rows,C]
void qkv_host(
    SwinBlockProblem const& p,
    std::vector<Element> const& windows,
    std::vector<Element> const& weight,
    std::vector<Element> const& bias,
    std::vector<Element>& q,
    std::vector<Element>& k,
    std::vector<Element>& v) {
  int rows = swin_rows(p), c = swin_channels(p);
  q.assign(int64_t(rows) * c, Element(0));
  k.assign(int64_t(rows) * c, Element(0));
  v.assign(int64_t(rows) * c, Element(0));
  for (int row = 0; row < rows; ++row) {
    for (int n = 0; n < 3 * c; ++n) {
      float acc = float(bias[n]);
      for (int ch = 0; ch < c; ++ch) {
        acc += float(windows[int64_t(row) * c + ch]) *
               float(weight[int64_t(ch) * (3 * c) + n]);
      }
      if (n < c) {
        q[int64_t(row) * c + n] = Element(acc);
      } else if (n < 2 * c) {
        k[int64_t(row) * c + (n - c)] = Element(acc);
      } else {
        v[int64_t(row) * c + (n - 2 * c)] = Element(acc);
      }
    }
  }
}

// Host window attention with pre-packed bias [BW, heads, L, L_pad].
void attention_host(
    SwinBlockProblem const& p,
    std::vector<Element> const& q,
    std::vector<Element> const& k,
    std::vector<Element> const& v,
    std::vector<Element> const& bias,
    std::vector<Element>& out) {
  int bw = swin_batched_windows(p);
  int l = swin_window_len(p);
  int lp = swin_window_len_padded(p);
  int heads = p.head_number, d = p.head_size, c = swin_channels(p);
  float scale = p.scale;
  out.assign(int64_t(bw) * l * c, Element(0));

  std::vector<float> scores(l);
  for (int w = 0; w < bw; ++w) {
    for (int h = 0; h < heads; ++h) {
      int64_t bias_base = (int64_t(w) * heads + h) * l * lp;
      for (int i = 0; i < l; ++i) {
        int64_t qbase = (int64_t(w) * l + i) * c + h * d;
        float maxv = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < l; ++j) {
          int64_t kbase = (int64_t(w) * l + j) * c + h * d;
          float dot = 0.0f;
          for (int e = 0; e < d; ++e) {
            dot += float(q[qbase + e]) * float(k[kbase + e]);
          }
          float s = dot * scale + float(bias[bias_base + int64_t(i) * lp + j]);
          scores[j] = s;
          if (s > maxv) maxv = s;
        }
        float denom = 0.0f;
        for (int j = 0; j < l; ++j) {
          scores[j] = std::exp(scores[j] - maxv);
          denom += scores[j];
        }
        float inv = 1.0f / denom;
        int64_t obase = (int64_t(w) * l + i) * c + h * d;
        for (int e = 0; e < d; ++e) {
          float acc = 0.0f;
          for (int j = 0; j < l; ++j) {
            int64_t vbase = (int64_t(w) * l + j) * c + h * d;
            acc += scores[j] * inv * float(v[vbase + e]);
          }
          out[obase + e] = Element(acc);
        }
      }
    }
  }
}

// generic row-major GEMM: in[rows,K] @ W[K,N] + bias[N] -> out[rows,N]
void linear_host(
    std::vector<Element> const& in,
    std::vector<Element> const& weight,
    std::vector<Element> const& bias,
    int rows,
    int K,
    int N,
    bool gelu,
    std::vector<Element>& out) {
  out.assign(int64_t(rows) * N, Element(0));
  for (int r = 0; r < rows; ++r) {
    for (int n = 0; n < N; ++n) {
      float acc = bias.empty() ? 0.0f : float(bias[n]);
      for (int kk = 0; kk < K; ++kk) {
        acc += float(in[int64_t(r) * K + kk]) *
               float(weight[int64_t(kk) * N + n]);
      }
      if (gelu) {
        acc = acc * 0.5f * (1.0f + std::erf(acc * 0.7071067811865476f));
      }
      out[int64_t(r) * N + n] = Element(acc);
    }
  }
}

void block_host_reference(
    SwinBlockProblem const& p,
    bool use_mask,
    std::vector<Element> const& input,
    std::vector<Element> const& qkv_weight,
    std::vector<Element> const& qkv_bias,
    std::vector<Element> const& output_weight,
    std::vector<Element> const& output_bias,
    std::vector<Element> const& gamma1,
    std::vector<Element> const& beta1,
    std::vector<Element> const& gamma2,
    std::vector<Element> const& beta2,
    std::vector<Element> const& fc1_weight,
    std::vector<Element> const& fc1_bias,
    std::vector<Element> const& fc2_weight,
    std::vector<Element> const& fc2_bias,
    std::vector<Element>& output) {
  int c = swin_channels(p);
  int image_tokens = p.batch_size * p.image_size * p.image_size;
  int rows = swin_rows(p);
  int hidden = swin_mlp_hidden(p);

  std::vector<Element> normed1, windows, q, k, v, attn, proj, reversed;
  std::vector<Element> bias, residual, normed2, mlp_hidden, mlp_out;

  layernorm_host(input, gamma1, beta1, image_tokens, c, p.layernorm_eps, normed1);
  shift_partition_host(p, normed1, windows);
  qkv_host(p, windows, qkv_weight, qkv_bias, q, k, v);
  build_attention_bias(p, use_mask, bias);
  attention_host(p, q, k, v, bias, attn);
  linear_host(attn, output_weight, output_bias, rows, c, c, false, proj);
  reverse_host(p, proj, reversed);

  residual.assign(int64_t(image_tokens) * c, Element(0));
  for (int64_t i = 0; i < int64_t(image_tokens) * c; ++i) {
    residual[i] = Element(float(input[i]) + float(reversed[i]));
  }
  layernorm_host(residual, gamma2, beta2, image_tokens, c, p.layernorm_eps, normed2);
  linear_host(normed2, fc1_weight, fc1_bias, image_tokens, c, hidden, true, mlp_hidden);
  linear_host(mlp_hidden, fc2_weight, fc2_bias, image_tokens, hidden, c, false, mlp_out);

  output.assign(int64_t(image_tokens) * c, Element(0));
  for (int64_t i = 0; i < int64_t(image_tokens) * c; ++i) {
    output[i] = Element(float(residual[i]) + float(mlp_out[i]));
  }
}

struct CompareResult {
  double mae = 0.0;
  float max_abs = 0.0f;
  int64_t max_index = 0;
  bool passed = false;
};

CompareResult compare_mae(
    cutlass::DeviceAllocation<Element> const& output,
    std::vector<Element> const& reference,
    int64_t elements,
    float tol) {
  std::vector<Element> host(elements);
  cutlass::device_memory::copy_to_host(host.data(), output.get(), elements);
  CompareResult r;
  long double abs_sum = 0.0;
  for (int64_t i = 0; i < elements; ++i) {
    float a = float(host[i]);
    float e = float(reference[i]);
    float diff = std::fabs(a - e);
    if (!std::isfinite(a) || !std::isfinite(e)) {
      r.mae = std::numeric_limits<double>::infinity();
      r.max_abs = diff;
      r.max_index = i;
      r.passed = false;
      return r;
    }
    abs_sum += diff;
    if (diff > r.max_abs) { r.max_abs = diff; r.max_index = i; }
  }
  r.mae = double(abs_sum / (long double)elements);
  r.passed = r.mae <= double(tol);
  return r;
}

int run_block(Options const& options) {
  SwinBlockProblem problem = options.problem();
  cutlass::Status status = Runner::can_implement(problem);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Unsupported Swin block problem: "
              << cutlassGetStatusString(status) << "\n";
    return -1;
  }

  int c = swin_channels(problem);
  int hidden = swin_mlp_hidden(problem);
  int image_tokens = problem.batch_size * problem.image_size * problem.image_size;

  std::vector<Element> host_input(swin_input_elements(problem));
  std::vector<Element> qkv_weight, qkv_bias, output_weight, output_bias;
  std::vector<Element> gamma1, beta1, gamma2, beta2;
  std::vector<Element> fc1_weight, fc1_bias, fc2_weight, fc2_bias;
  std::vector<Element> attention_bias;

  fill_random_uniform(host_input, options.seed, -1.0f, 1.0f);
  fill_block_weights(problem, qkv_weight, qkv_bias, output_weight, output_bias,
                     gamma1, beta1, gamma2, beta2,
                     fc1_weight, fc1_bias, fc2_weight, fc2_bias);
  build_attention_bias(problem, options.use_mask, attention_bias);

  // Device allocations.
  cutlass::DeviceAllocation<Element> input(swin_input_elements(problem));
  cutlass::DeviceAllocation<Element> d_qkv_weight(swin_qkv_weight_elements(problem));
  cutlass::DeviceAllocation<Element> d_qkv_bias(3 * c);
  cutlass::DeviceAllocation<Element> d_output_weight(swin_output_weight_elements(problem));
  cutlass::DeviceAllocation<Element> d_output_bias(c);
  cutlass::DeviceAllocation<Element> d_attention_bias(swin_attention_bias_elements(problem));
  cutlass::DeviceAllocation<Element> d_gamma1(c), d_beta1(c), d_gamma2(c), d_beta2(c);
  cutlass::DeviceAllocation<Element> d_fc1_weight(swin_mlp_weight1_elements(problem));
  cutlass::DeviceAllocation<Element> d_fc1_bias(hidden);
  cutlass::DeviceAllocation<Element> d_fc2_weight(swin_mlp_weight2_elements(problem));
  cutlass::DeviceAllocation<Element> d_fc2_bias(c);

  cutlass::DeviceAllocation<Element> windows(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> qkv(swin_qkv_elements(problem));
  cutlass::DeviceAllocation<Element> query(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> key(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> value(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> attention_output(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> projected(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> output(swin_input_elements(problem));
  cutlass::DeviceAllocation<Element> residual(swin_input_elements(problem));
  cutlass::DeviceAllocation<Element> normed2(swin_input_elements(problem));
  cutlass::DeviceAllocation<Element> mlp_hidden(swin_mlp_hidden_elements(problem));
  cutlass::DeviceAllocation<Element> mlp_output(swin_input_elements(problem));

  copy_to_device(input, host_input);
  copy_to_device(d_qkv_weight, qkv_weight);
  copy_to_device(d_qkv_bias, qkv_bias);
  copy_to_device(d_output_weight, output_weight);
  copy_to_device(d_output_bias, output_bias);
  copy_to_device(d_attention_bias, attention_bias);
  copy_to_device(d_gamma1, gamma1);
  copy_to_device(d_beta1, beta1);
  copy_to_device(d_gamma2, gamma2);
  copy_to_device(d_beta2, beta2);
  copy_to_device(d_fc1_weight, fc1_weight);
  copy_to_device(d_fc1_bias, fc1_bias);
  copy_to_device(d_fc2_weight, fc2_weight);
  copy_to_device(d_fc2_bias, fc2_bias);

  Tensors t;
  t.input = input.get();
  t.attention.qkv_weight = d_qkv_weight.get();
  t.attention.qkv_bias = d_qkv_bias.get();
  t.attention.output_weight = d_output_weight.get();
  t.attention.output_bias = d_output_bias.get();
  t.attention.attention_bias = d_attention_bias.get();
  t.attention.windows = windows.get();
  t.attention.qkv = qkv.get();
  t.attention.query = query.get();
  t.attention.key = key.get();
  t.attention.value = value.get();
  t.attention.attention_output = attention_output.get();
  t.attention.projected = projected.get();
  t.output = output.get();
  t.gamma1 = d_gamma1.get();
  t.beta1 = d_beta1.get();
  t.gamma2 = d_gamma2.get();
  t.beta2 = d_beta2.get();
  t.fc1_weight = d_fc1_weight.get();
  t.fc1_bias = d_fc1_bias.get();
  t.fc2_weight = d_fc2_weight.get();
  t.fc2_bias = d_fc2_bias.get();
  t.residual = residual.get();
  t.normed2 = normed2.get();
  t.mlp_hidden = mlp_hidden.get();
  t.mlp_output = mlp_output.get();

  status = Runner::run(problem, t, nullptr);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "SwinBlock launch failed: "
              << cutlassGetStatusString(status) << "\n";
    return -1;
  }
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "SwinBlock sync failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }

  std::vector<Element> reference;
  block_host_reference(problem, options.use_mask, host_input,
                       qkv_weight, qkv_bias, output_weight, output_bias,
                       gamma1, beta1, gamma2, beta2,
                       fc1_weight, fc1_bias, fc2_weight, fc2_bias, reference);

  CompareResult cmp = compare_mae(output, reference, swin_input_elements(problem), options.mae_tolerance);
  std::cout << "Full fused SwinBlock:\n"
            << "====================================================\n"
            << "    {B, I, window, shift, heads, head_dim, mlp} = {"
            << problem.batch_size << ", " << problem.image_size << ", "
            << problem.window_size << ", " << problem.shift_size << ", "
            << problem.head_number << ", " << problem.head_size << ", "
            << problem.mlp_ratio << "}\n"
            << "    Launches : 10 (norm1+shift+part | QKV | bias/split | attn | proj"
            << " | reverse+res1+norm2 | fc1 | gelu | fc2 | res2)\n"
            << "    Reference: self-contained host full block\n"
            << "    MAE      : " << cmp.mae << " (tolerance " << options.mae_tolerance << ")\n"
            << "    Max abs  : " << cmp.max_abs << " at index " << cmp.max_index << "\n";
  if (!cmp.passed) {
    std::cout << "\nFailed\n";
    return -1;
  }

  // Timing.
  status = Runner::run(problem, t, nullptr);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "SwinBlock warmup failed: "
              << cutlassGetStatusString(status) << "\n";
    return -1;
  }
  cudaDeviceSynchronize();
  cudaEvent_t start = nullptr, stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < options.iterations; ++i) {
    status = Runner::run(problem, t, nullptr);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Timed launch failed: "
                << cutlassGetStatusString(status) << "\n";
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return -1;
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  double runtime_ms = double(elapsed_ms) / double(options.iterations);

  (void)image_tokens;
  std::cout << "    Runtime  : " << runtime_ms << " ms\n\nPassed\n";
  return 0;
}

} // namespace
} // namespace swin
} // namespace tiny_cutlass

int main(int argc, char const** args) {
  using namespace tiny_cutlass::swin;
  cudaDeviceProp props;
  cudaError_t err = cudaGetDeviceProperties(&props, 0);
  if (err != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties: " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  std::cout << "Device: " << props.name << " (SM" << props.major << props.minor << ")\n";
  if (CUDART_VERSION < 11000 || props.major < 8) {
    std::cout << "This path requires Ampere or later.\n";
    return 0;
  }
  Options options;
  options.parse(argc, args);
  if (options.help) {
    options.print_usage(std::cout) << "\n";
    return 0;
  }
  if (options.error) {
    std::cerr << "Invalid options.\n";
    return -1;
  }
  return run_block(options);
}
