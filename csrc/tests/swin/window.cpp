#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"

#include "../../swin/swin.h"
#include "reference.h"

namespace tiny_cutlass {
namespace swin {
namespace {

using Runner = device::SwinAttention<cutlass::arch::Sm80, cutlass::half_t>;
using Element = typename Runner::Element;
using Tensors = typename Runner::Tensors;

struct Options {
  bool help = false;
  bool error = false;
  bool reference_check = true;
  bool use_mask = true;
  bool profile_once = false;

  int batch_size = 2;
  int image_size = 14;
  int window_size = 7;
  int shift_size = 3;
  int head_number = 3;
  int head_size = 32;
  int iterations = 20;
  int seed = 2026;

  float mae_tolerance = 2.0e-2f;
  float host_abs_tolerance = 1.6e-1f;
  float host_rel_tolerance = 3.0e-1f;
  std::string input_file;
  std::string qkv_weight_file;
  std::string qkv_bias_file;
  std::string output_weight_file;
  std::string output_bias_file;
  std::string rel_bias_file;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    auto get_string_arg = [&](char const* name, std::string& value) {
      std::string prefix = std::string("--") + name + "=";
      for (int i = 1; i < argc; ++i) {
        std::string arg(args[i]);
        if (arg.rfind(prefix, 0) == 0) {
          value = arg.substr(prefix.size());
        }
      }
    };

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
    cmd.get_cmd_line_argument("seed", seed, seed);
    cmd.get_cmd_line_argument("mask", use_mask, use_mask);
    cmd.get_cmd_line_argument("reference-check", reference_check, reference_check);
    cmd.get_cmd_line_argument("profile-once", profile_once, profile_once);
    cmd.get_cmd_line_argument("mae-tolerance", mae_tolerance, mae_tolerance);
    cmd.get_cmd_line_argument("host-abs-tolerance", host_abs_tolerance, host_abs_tolerance);
    cmd.get_cmd_line_argument("host-rel-tolerance", host_rel_tolerance, host_rel_tolerance);
    get_string_arg("input-file", input_file);
    get_string_arg("qkv-weight-file", qkv_weight_file);
    get_string_arg("qkv-bias-file", qkv_bias_file);
    get_string_arg("output-weight-file", output_weight_file);
    get_string_arg("output-bias-file", output_bias_file);
    get_string_arg("rel-bias-file", rel_bias_file);

    if (batch_size <= 0 || image_size <= 0 || window_size <= 0 ||
        head_number <= 0 || head_size <= 0 || iterations <= 0 ||
        mae_tolerance <= 0.0f || host_abs_tolerance <= 0.0f ||
        host_rel_tolerance <= 0.0f) {
      error = true;
    }
  }

  SwinAttentionProblem problem() const {
    SwinAttentionProblem p;
    p.batch_size = batch_size;
    p.image_size = image_size;
    p.window_size = window_size;
    p.shift_size = shift_size;
    p.head_number = head_number;
    p.head_size = head_size;
    p.scale = 1.0f / std::sqrt(float(head_size));
    return p;
  }

  double gflops(double runtime_s) const {
    SwinAttentionProblem p = problem();
    int64_t rows = swin_rows(p);
    int64_t c = swin_channels(p);
    int64_t bw = swin_batched_windows(p);
    int64_t h = p.head_number;
    int64_t l = swin_window_len(p);
    int64_t d = p.head_size;
    int64_t qkv = int64_t(2) * rows * c * (3 * c);
    int64_t attention = int64_t(4) * bw * h * l * l * d
                      + int64_t(3) * bw * h * l * l;
    int64_t projection = int64_t(2) * rows * c * c;
    return double(qkv + attention + projection) / 1.0e9 / runtime_s;
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "swin_window\n\n"
        << "Options:\n\n"
        << "  --help                         Display this usage statement.\n"
        << "  --batch_size=<int>             Batch size (default: 2).\n"
        << "  --image_size=<int>             Square image resolution H=W (default: 14).\n"
        << "  --window_size=<int>            Swin local window size (default: 7).\n"
        << "  --shift_size=<int>             Cyclic shift size (default: window_size/2).\n"
        << "  --head_number=<int>            Number of attention heads (default: 3).\n"
        << "  --head_size=<int>              Per-head dimension (default: 32).\n"
        << "  --mask=<bool>                  Apply shifted-window mask (default: true).\n"
        << "  --reference-check=<bool>       Run cuDNN and host references before timing.\n"
        << "  --profile-once=<bool>          Launch one checked Swin path and skip timing.\n"
        << "  --mae-tolerance=<float>        cuDNN attention MAE tolerance (default: 2e-2).\n"
        << "  --iterations=<int>             Number of timed iterations (default: 20).\n"
        << "  --seed=<int>                   Random seed base (default: 2026).\n";
    return out;
  }
};

struct CompareResult {
  double mae = 0.0;
  float max_abs = 0.0f;
  int64_t max_index = 0;
  bool passed = false;
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

bool load_float_file(
    std::string const& path,
    std::vector<Element>& dst,
    size_t expected,
    char const* label) {
  if (path.empty()) {
    return true;
  }
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Could not open " << label << " file: " << path << "\n";
    return false;
  }
  std::streamsize bytes = file.tellg();
  std::streamsize expected_bytes =
      std::streamsize(expected * sizeof(float));
  if (bytes != expected_bytes) {
    std::cerr << label << " file size mismatch: " << path
              << " has " << bytes << " bytes, expected "
              << expected_bytes << "\n";
    return false;
  }
  file.seekg(0, std::ios::beg);
  std::vector<float> tmp(expected);
  if (!file.read(reinterpret_cast<char*>(tmp.data()), bytes)) {
    std::cerr << "Could not read " << label << " file: " << path << "\n";
    return false;
  }
  dst.resize(expected);
  for (size_t i = 0; i < expected; ++i) {
    dst[i] = Element(tmp[i]);
  }
  return true;
}

void fill_weights(SwinAttentionProblem const& p,
                  std::vector<Element>& qkv_weight,
                  std::vector<Element>& qkv_bias,
                  std::vector<Element>& output_weight,
                  std::vector<Element>& output_bias) {
  int c = swin_channels(p);
  qkv_weight.resize(swin_qkv_weight_elements(p));
  qkv_bias.resize(3 * c);
  output_weight.resize(swin_output_weight_elements(p));
  output_bias.resize(c);

  for (int k_col = 0; k_col < c; ++k_col) {
    for (int n = 0; n < 3 * c; ++n) {
      qkv_weight[int64_t(k_col) * (3 * c) + n] =
          Element(0.015f * std::sin(float((k_col + 1) * (n + 3)) * 0.011f));
    }
    for (int n = 0; n < c; ++n) {
      output_weight[int64_t(k_col) * c + n] =
          Element(0.013f * std::cos(float((k_col + 5) * (n + 1)) * 0.009f));
    }
  }
  for (int n = 0; n < 3 * c; ++n) {
    qkv_bias[n] = Element(0.01f * std::sin(float(n + 1) * 0.017f));
  }
  for (int n = 0; n < c; ++n) {
    output_bias[n] = Element(0.01f * std::cos(float(n + 1) * 0.019f));
  }
}

void build_shift_attention_mask(
    SwinAttentionProblem const& p,
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
      int window_y = shifted_y / window;
      int window_x = shifted_x / window;
      int window_idx = window_y * windows_per_row + window_x;
      int idx_in_window = (shifted_y % window) * window + (shifted_x % window);
      window_regions[window_idx * l + idx_in_window] = image_mask[y * image + x];
    }
  }

  for (int w = 0; w < nW; ++w) {
    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < l; ++j) {
        bool same_region = window_regions[w * l + i] == window_regions[w * l + j];
        mask[(int64_t(w) * l + i) * l + j] =
            same_region ? Element(0) : Element(-100);
      }
    }
  }
}

void build_attention_bias(
    SwinAttentionProblem const& p,
    bool use_mask,
    std::vector<Element>& attention_bias,
    std::vector<Element>& rel_bias,
    std::vector<Element>& attention_mask) {
  int bw = swin_batched_windows(p);
  int h = p.head_number;
  int l = swin_window_len(p);
  int lp = swin_window_len_padded(p);
  int nW = swin_num_windows(p);

  rel_bias.resize(int64_t(h) * l * l);
  for (int head = 0; head < h; ++head) {
    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < l; ++j) {
        rel_bias[(int64_t(head) * l + i) * l + j] =
            Element(0.01f * std::sin(float((head + 1) * (i - j))));
      }
    }
  }

  build_shift_attention_mask(p, use_mask, attention_mask);

  attention_bias.assign(swin_attention_bias_elements(p), Element(0));
  for (int batch_window = 0; batch_window < bw; ++batch_window) {
    int window_id = batch_window % nW;
    for (int head = 0; head < h; ++head) {
      for (int i = 0; i < l; ++i) {
        for (int j = 0; j < l; ++j) {
          float v = float(rel_bias[(int64_t(head) * l + i) * l + j]);
          if (use_mask) {
            v += float(attention_mask[(int64_t(window_id) * l + i) * l + j]);
          }
          attention_bias[((int64_t(batch_window) * h + head) * l + i) * lp + j] =
              Element(v);
        }
      }
    }
  }
}

void pack_attention_bias(
    SwinAttentionProblem const& p,
    bool use_mask,
    std::vector<Element>& attention_bias,
    std::vector<Element> const& rel_bias,
    std::vector<Element> const& attention_mask) {
  int bw = swin_batched_windows(p);
  int h = p.head_number;
  int l = swin_window_len(p);
  int lp = swin_window_len_padded(p);
  int nW = swin_num_windows(p);

  attention_bias.assign(swin_attention_bias_elements(p), Element(0));
  for (int batch_window = 0; batch_window < bw; ++batch_window) {
    int window_id = batch_window % nW;
    for (int head = 0; head < h; ++head) {
      for (int i = 0; i < l; ++i) {
        for (int j = 0; j < l; ++j) {
          float v = float(rel_bias[(int64_t(head) * l + i) * l + j]);
          if (use_mask) {
            v += float(attention_mask[(int64_t(window_id) * l + i) * l + j]);
          }
          attention_bias[((int64_t(batch_window) * h + head) * l + i) * lp + j] =
              Element(v);
        }
      }
    }
  }
}

void window_partition_host(
    SwinAttentionProblem const& p,
    std::vector<Element> const& input,
    std::vector<Element>& output) {
  int bsz = p.batch_size;
  int image = p.image_size;
  int window = p.window_size;
  int shift = p.shift_size;
  int c = swin_channels(p);
  int l = swin_window_len(p);
  int windows_per_row = image / window;
  int tokens_per_batch = image * image;

  output.assign(int64_t(bsz) * tokens_per_batch * c, Element(0));
  for (int b = 0; b < bsz; ++b) {
    for (int y = 0; y < image; ++y) {
      for (int x = 0; x < image; ++x) {
        int shifted_y = shift != 0 ? ((y - shift + image) % image) : y;
        int shifted_x = shift != 0 ? ((x - shift + image) % image) : x;
        int window_y = shifted_y / window;
        int window_x = shifted_x / window;
        int window_idx = window_y * windows_per_row + window_x;
        int idx_in_window = (shifted_y % window) * window + (shifted_x % window);
        int64_t input_token = int64_t(b) * image * image + y * image + x;
        int64_t output_token = int64_t(b) * tokens_per_batch
                             + int64_t(window_idx) * l
                             + idx_in_window;
        for (int ch = 0; ch < c; ++ch) {
          output[output_token * c + ch] = input[input_token * c + ch];
        }
      }
    }
  }
}

void window_reverse_host(
    SwinAttentionProblem const& p,
    std::vector<Element> const& input,
    std::vector<Element>& output) {
  int bsz = p.batch_size;
  int image = p.image_size;
  int window = p.window_size;
  int shift = p.shift_size;
  int c = swin_channels(p);
  int l = swin_window_len(p);
  int windows_per_row = image / window;
  int tokens_per_batch = image * image;

  output.assign(swin_input_elements(p), Element(0));
  for (int b = 0; b < bsz; ++b) {
    for (int y = 0; y < image; ++y) {
      for (int x = 0; x < image; ++x) {
        int shifted_y = shift != 0 ? ((y - shift + image) % image) : y;
        int shifted_x = shift != 0 ? ((x - shift + image) % image) : x;
        int window_y = shifted_y / window;
        int window_x = shifted_x / window;
        int window_idx = window_y * windows_per_row + window_x;
        int idx_in_window = (shifted_y % window) * window + (shifted_x % window);
        int64_t input_token = int64_t(b) * tokens_per_batch
                            + int64_t(window_idx) * l
                            + idx_in_window;
        int64_t output_token = int64_t(b) * image * image + y * image + x;
        for (int ch = 0; ch < c; ++ch) {
          output[output_token * c + ch] = input[input_token * c + ch];
        }
      }
    }
  }
}

void qkv_projection_host(
    SwinAttentionProblem const& p,
    std::vector<Element> const& windows,
    std::vector<Element> const& weight,
    std::vector<Element> const& bias,
    std::vector<Element>& q,
    std::vector<Element>& k,
    std::vector<Element>& v) {
  int rows = swin_rows(p);
  int c = swin_channels(p);
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
      Element value = Element(acc);
      if (n < c) {
        q[int64_t(row) * c + n] = value;
      } else if (n < 2 * c) {
        k[int64_t(row) * c + (n - c)] = value;
      } else {
        v[int64_t(row) * c + (n - 2 * c)] = value;
      }
    }
  }
}

void output_projection_host(
    SwinAttentionProblem const& p,
    std::vector<Element> const& input,
    std::vector<Element> const& weight,
    std::vector<Element> const& bias,
    std::vector<Element>& output) {
  int rows = swin_rows(p);
  int c = swin_channels(p);
  output.assign(int64_t(rows) * c, Element(0));
  for (int row = 0; row < rows; ++row) {
    for (int n = 0; n < c; ++n) {
      float acc = float(bias[n]);
      for (int ch = 0; ch < c; ++ch) {
        acc += float(input[int64_t(row) * c + ch]) *
               float(weight[int64_t(ch) * c + n]);
      }
      output[int64_t(row) * c + n] = Element(acc);
    }
  }
}

CompareResult compare_mae(
    cutlass::DeviceAllocation<Element> const& output,
    cutlass::DeviceAllocation<Element> const& reference,
    int64_t elements,
    float mae_tolerance) {
  std::vector<Element> host_output(elements);
  std::vector<Element> host_reference(elements);
  cutlass::device_memory::copy_to_host(host_output.data(), output.get(), elements);
  cutlass::device_memory::copy_to_host(host_reference.data(), reference.get(), elements);

  CompareResult result;
  long double abs_sum = 0.0;
  for (int64_t i = 0; i < elements; ++i) {
    float actual = float(host_output[i]);
    float expected = float(host_reference[i]);
    float diff = std::fabs(actual - expected);
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
      result.mae = std::numeric_limits<double>::infinity();
      result.max_abs = diff;
      result.max_index = i;
      result.passed = false;
      return result;
    }
    abs_sum += diff;
    if (diff > result.max_abs) {
      result.max_abs = diff;
      result.max_index = i;
    }
  }
  result.mae = double(abs_sum / long double(elements));
  result.passed = result.mae <= double(mae_tolerance);
  return result;
}

bool compare_host(
    std::vector<Element> const& computed,
    std::vector<Element> const& reference,
    std::string const& label,
    float abs_tol,
    float rel_tol) {
  if (computed.size() != reference.size()) {
    std::cerr << label << ": size mismatch\n";
    return false;
  }
  for (size_t i = 0; i < computed.size(); ++i) {
    float actual = float(computed[i]);
    float expected = float(reference[i]);
    float diff = std::fabs(actual - expected);
    float rel = diff / (std::fabs(expected) + 1e-5f);
    if (!std::isfinite(actual) || (diff > abs_tol && rel > rel_tol)) {
      std::cerr << label << " mismatch at " << i
                << ": computed=" << actual
                << " reference=" << expected
                << " diff=" << diff
                << " rel=" << rel << "\n";
      return false;
    }
  }
  return true;
}

int run_window(Options const& options) {
  SwinAttentionProblem problem = options.problem();
  cutlass::Status status = Runner::can_implement(problem);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Unsupported Swin attention problem: "
              << cutlassGetStatusString(status) << "\n";
    return -1;
  }

  std::vector<Element> host_input(swin_input_elements(problem));
  std::vector<Element> host_qkv_weight;
  std::vector<Element> host_qkv_bias;
  std::vector<Element> host_output_weight;
  std::vector<Element> host_output_bias;
  std::vector<Element> host_attention_bias;
  std::vector<Element> host_rel_bias;
  std::vector<Element> host_attention_mask;

  fill_random_uniform(host_input, options.seed, -1.0f, 1.0f);
  fill_weights(problem, host_qkv_weight, host_qkv_bias, host_output_weight, host_output_bias);
  build_attention_bias(
      problem,
      options.use_mask,
      host_attention_bias,
      host_rel_bias,
      host_attention_mask);
  if (!load_float_file(
          options.input_file,
          host_input,
          swin_input_elements(problem),
          "input") ||
      !load_float_file(
          options.qkv_weight_file,
          host_qkv_weight,
          swin_qkv_weight_elements(problem),
          "qkv_weight") ||
      !load_float_file(
          options.qkv_bias_file,
          host_qkv_bias,
          3 * swin_channels(problem),
          "qkv_bias") ||
      !load_float_file(
          options.output_weight_file,
          host_output_weight,
          swin_output_weight_elements(problem),
          "output_weight") ||
      !load_float_file(
          options.output_bias_file,
          host_output_bias,
          swin_channels(problem),
          "output_bias") ||
      !load_float_file(
          options.rel_bias_file,
          host_rel_bias,
          int64_t(problem.head_number) * swin_window_len(problem) *
              swin_window_len(problem),
          "relative_position_bias")) {
    return -1;
  }
  if (!options.rel_bias_file.empty()) {
    build_shift_attention_mask(problem, options.use_mask, host_attention_mask);
    pack_attention_bias(
        problem,
        options.use_mask,
        host_attention_bias,
        host_rel_bias,
        host_attention_mask);
  }

  cutlass::DeviceAllocation<Element> input(swin_input_elements(problem));
  cutlass::DeviceAllocation<Element> qkv_weight(swin_qkv_weight_elements(problem));
  cutlass::DeviceAllocation<Element> qkv_bias(3 * swin_channels(problem));
  cutlass::DeviceAllocation<Element> output_weight(swin_output_weight_elements(problem));
  cutlass::DeviceAllocation<Element> output_bias(swin_channels(problem));
  cutlass::DeviceAllocation<Element> attention_bias(swin_attention_bias_elements(problem));
  cutlass::DeviceAllocation<Element> windows(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> qkv(swin_qkv_elements(problem));
  cutlass::DeviceAllocation<Element> query(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> key(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> value(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> attention_output(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> projected(swin_window_elements(problem));
  cutlass::DeviceAllocation<Element> output(swin_input_elements(problem));
  cutlass::DeviceAllocation<Element> cudnn_attention_output(swin_window_elements(problem));

  copy_to_device(input, host_input);
  copy_to_device(qkv_weight, host_qkv_weight);
  copy_to_device(qkv_bias, host_qkv_bias);
  copy_to_device(output_weight, host_output_weight);
  copy_to_device(output_bias, host_output_bias);
  copy_to_device(attention_bias, host_attention_bias);

  cudaMemset(windows.get(), 0, swin_window_elements(problem) * sizeof(Element));
  cudaMemset(qkv.get(), 0, swin_qkv_elements(problem) * sizeof(Element));
  cudaMemset(query.get(), 0, swin_window_elements(problem) * sizeof(Element));
  cudaMemset(key.get(), 0, swin_window_elements(problem) * sizeof(Element));
  cudaMemset(value.get(), 0, swin_window_elements(problem) * sizeof(Element));
  cudaMemset(attention_output.get(), 0, swin_window_elements(problem) * sizeof(Element));
  cudaMemset(projected.get(), 0, swin_window_elements(problem) * sizeof(Element));
  cudaMemset(output.get(), 0, swin_input_elements(problem) * sizeof(Element));
  cudaMemset(cudnn_attention_output.get(), 0, swin_window_elements(problem) * sizeof(Element));

  Tensors tensors;
  tensors.input = input.get();
  tensors.attention.qkv_weight = qkv_weight.get();
  tensors.attention.qkv_bias = qkv_bias.get();
  tensors.attention.output_weight = output_weight.get();
  tensors.attention.output_bias = output_bias.get();
  tensors.attention.attention_bias = attention_bias.get();
  tensors.attention.windows = windows.get();
  tensors.attention.qkv = qkv.get();
  tensors.attention.query = query.get();
  tensors.attention.key = key.get();
  tensors.attention.value = value.get();
  tensors.attention.attention_output = attention_output.get();
  tensors.attention.projected = projected.get();
  tensors.output = output.get();

  status = Runner::run(problem, tensors, nullptr);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Swin attention launch failed: "
              << cutlassGetStatusString(status) << "\n";
    return -1;
  }
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Swin kernel sync failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }

  if (options.reference_check) {
    std::string reference_error;
    err = run_cudnn_swin_attention_reference(
        problem,
        query.get(),
        key.get(),
        value.get(),
        attention_bias.get(),
        cudnn_attention_output.get(),
        nullptr,
        reference_error);
    if (err != cudaSuccess) {
      std::cerr << "cuDNN Swin attention reference failed: "
                << reference_error << "\n";
      return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "cuDNN Swin attention sync failed: "
                << cudaGetErrorString(err) << "\n";
      return -1;
    }

    CompareResult attention_compare = compare_mae(
        attention_output,
        cudnn_attention_output,
        swin_window_elements(problem),
        options.mae_tolerance);
    std::cout << "    Reference: cuDNN SDPA + Swin relative bias/mask\n"
              << "    MAE      : " << attention_compare.mae
              << " (tolerance " << options.mae_tolerance << ")\n"
              << "    Max abs  : " << attention_compare.max_abs
              << " at index " << attention_compare.max_index << "\n";
    if (!attention_compare.passed) {
      std::cout << "\nFailed\n";
      return -1;
    }

    std::vector<Element> host_windows(swin_window_elements(problem));
    std::vector<Element> host_query(swin_window_elements(problem));
    std::vector<Element> host_key(swin_window_elements(problem));
    std::vector<Element> host_value(swin_window_elements(problem));
    std::vector<Element> host_projected(swin_window_elements(problem));
    std::vector<Element> host_output(swin_input_elements(problem));

    cutlass::device_memory::copy_to_host(host_windows.data(), windows.get(), host_windows.size());
    cutlass::device_memory::copy_to_host(host_query.data(), query.get(), host_query.size());
    cutlass::device_memory::copy_to_host(host_key.data(), key.get(), host_key.size());
    cutlass::device_memory::copy_to_host(host_value.data(), value.get(), host_value.size());
    cutlass::device_memory::copy_to_host(host_projected.data(), projected.get(), host_projected.size());
    cutlass::device_memory::copy_to_host(host_output.data(), output.get(), host_output.size());

    std::vector<Element> ref_windows;
    std::vector<Element> ref_query;
    std::vector<Element> ref_key;
    std::vector<Element> ref_value;
    std::vector<Element> ref_projected;
    std::vector<Element> ref_output;

    window_partition_host(problem, host_input, ref_windows);
    qkv_projection_host(
        problem,
        ref_windows,
        host_qkv_weight,
        host_qkv_bias,
        ref_query,
        ref_key,
        ref_value);

    std::vector<Element> host_cudnn_attention(swin_window_elements(problem));
    cutlass::device_memory::copy_to_host(
        host_cudnn_attention.data(),
        cudnn_attention_output.get(),
        host_cudnn_attention.size());
    output_projection_host(
        problem,
        host_cudnn_attention,
        host_output_weight,
        host_output_bias,
        ref_projected);
    window_reverse_host(problem, ref_projected, ref_output);

    bool host_passed =
        compare_host(host_windows, ref_windows, "WindowPartition", 0.0f, 0.0f) &&
        compare_host(host_query, ref_query, "Q", options.host_abs_tolerance, options.host_rel_tolerance) &&
        compare_host(host_key, ref_key, "K", options.host_abs_tolerance, options.host_rel_tolerance) &&
        compare_host(host_value, ref_value, "V", options.host_abs_tolerance, options.host_rel_tolerance) &&
        compare_host(host_projected, ref_projected, "OutputProjection", options.host_abs_tolerance, options.host_rel_tolerance) &&
        compare_host(host_output, ref_output, "WindowReverse", options.host_abs_tolerance, options.host_rel_tolerance);
    if (!host_passed) {
      std::cout << "\nFailed\n";
      return -1;
    }
  }

  if (options.profile_once) {
    std::cout << "\nSwin CUTLASS profile path:\n"
              << "====================================================\n"
              << "    {B, I, window, shift, heads, head_dim} = {"
              << problem.batch_size << ", " << problem.image_size << ", "
              << problem.window_size << ", " << problem.shift_size << ", "
              << problem.head_number << ", " << problem.head_size << "}\n"
              << "    Path: WindowPartition -> QKV -> WindowAttention -> OutputProjection -> WindowReverse\n"
              << "\nPassed\n";
    return 0;
  }

  status = Runner::run(problem, tensors, nullptr);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Warmup failed: " << cutlassGetStatusString(status) << "\n";
    return -1;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Warmup sync failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < options.iterations; ++i) {
    status = Runner::run(problem, tensors, nullptr);
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
  double gflops = options.gflops(runtime_ms / 1000.0);

  std::cout << "\nSwin CUTLASS path:\n"
            << "====================================================\n"
            << "    {B, I, window, shift, heads, head_dim} = {"
            << problem.batch_size << ", " << problem.image_size << ", "
            << problem.window_size << ", " << problem.shift_size << ", "
            << problem.head_number << ", " << problem.head_size << "}\n"
            << "    Path: WindowPartition -> QKV -> WindowAttention -> OutputProjection -> WindowReverse\n"
            << "    num_windows=" << swin_num_windows(problem)
            << ", window_len=" << swin_window_len(problem)
            << ", bias_stride=" << swin_window_len_padded(problem)
            << ", mask=" << (options.use_mask ? "true" : "false") << "\n"
            << "    Runtime: " << runtime_ms << " ms\n"
            << "    GFLOPs : " << gflops << "\n"
            << "\nPassed\n";

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

  return run_window(options);
}
