#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "cutlass/util/command_line.h"

#include "natten/fna_forward.h"

namespace tiny_cutlass {
namespace natten {
namespace {

using FnaPolicy = DefaultFnaForwardPolicy<
    1,
    FnaCausalMask<false>,
    cutlass::half_t,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>>;

static_assert(FnaPolicy::kRank == 1, "FNA smoke policy must be 1D.");
static_assert(!FnaPolicy::CausalMaskTag::kIsCausal,
              "FNA smoke policy should use non-causal mask.");
static_assert(FnaPolicy::kIsAligned, "FNA smoke policy should be aligned.");
static_assert(std::is_same<FnaPolicy::ElementInput, cutlass::half_t>::value,
              "FNA smoke policy input element mismatch.");
static_assert(std::is_same<FnaPolicy::ElementOutput, cutlass::half_t>::value,
              "FNA smoke policy output element mismatch.");
static_assert(std::is_same<FnaPolicy::ElementAccumulator, float>::value,
              "FNA smoke policy accumulator mismatch.");
static_assert(std::is_same<FnaPolicy::Arch, cutlass::arch::Sm80>::value,
              "FNA smoke policy arch mismatch.");
static_assert(FnaPolicy::Threadblock::kM == 64 &&
                  FnaPolicy::Threadblock::kN == 64 &&
                  FnaPolicy::Threadblock::kK == 32,
              "FNA smoke policy threadblock shape mismatch.");

int qkv_fix_dilation(int length, int dilation, int dilation_group) {
  int padding =
      1 - ((dilation_group + (dilation - (length % dilation))) / dilation);
  return (length / dilation) + padding;
}

int window_left(int kernel_size) {
  return kernel_size / 2;
}

int window_right(int kernel_size) {
  return (kernel_size / 2) + ((kernel_size % 2) - 1);
}

int noncausal_window_start(
    int query_coord,
    int kernel_size,
    int stride,
    int length) {
  int leader = std::min(((query_coord / stride) * stride) + (stride / 2),
                        length - 1);
  int right = window_right(kernel_size);
  return std::max(leader - window_left(kernel_size), 0) +
      ((leader + right >= length) * (length - right - leader - 1));
}

bool is_valid_neighbor_1d(FnaForwardProblem const& problem, int query, int key) {
  if (key < 0 || key >= problem.length) {
    return false;
  }

  int query_group = query % problem.dilation;
  if (key % problem.dilation != query_group) {
    return false;
  }

  int query_coord = query / problem.dilation;
  int key_coord = key / problem.dilation;
  int corrected_length =
      qkv_fix_dilation(problem.length, problem.dilation, query_group);
  int start = noncausal_window_start(
      query_coord, problem.kernel_size, problem.stride, corrected_length);
  int end = start + problem.kernel_size;
  return key_coord >= start && key_coord < end;
}

struct HostFnaResult {
  std::vector<float> output;
  std::vector<float> logsumexp;
};

int qkv_offset(
    FnaForwardProblem const& problem,
    int batch,
    int token,
    int head,
    int dim) {
  return ((batch * problem.length + token) * problem.heads + head) *
      problem.head_dim +
      dim;
}

int value_offset(
    FnaForwardProblem const& problem,
    int batch,
    int token,
    int head,
    int dim) {
  return ((batch * problem.length + token) * problem.heads + head) *
      problem.head_dim_value +
      dim;
}

HostFnaResult host_fna_forward_1d_noncausal(
    FnaForwardProblem const& problem,
    std::vector<float> const& query,
    std::vector<float> const& key,
    std::vector<float> const& value) {
  HostFnaResult result;
  result.output.resize(
      problem.batch_size * problem.length * problem.heads *
      problem.head_dim_value);
  result.logsumexp.resize(problem.batch_size * problem.length * problem.heads);

  std::vector<float> scores(problem.length);
  for (int b = 0; b < problem.batch_size; ++b) {
    for (int q = 0; q < problem.length; ++q) {
      for (int h = 0; h < problem.heads; ++h) {
        float max_score = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < problem.length; ++k) {
          if (!is_valid_neighbor_1d(problem, q, k)) {
            scores[k] = -std::numeric_limits<float>::infinity();
            continue;
          }

          float score = 0.0f;
          for (int d = 0; d < problem.head_dim; ++d) {
            score += query[qkv_offset(problem, b, q, h, d)] *
                key[qkv_offset(problem, b, k, h, d)];
          }
          score *= problem.scale;
          scores[k] = score;
          max_score = std::max(max_score, score);
        }

        float sum = 0.0f;
        for (int k = 0; k < problem.length; ++k) {
          if (scores[k] != -std::numeric_limits<float>::infinity()) {
            scores[k] = std::exp(scores[k] - max_score);
            sum += scores[k];
          }
        }

        int lse_offset = (b * problem.length + q) * problem.heads + h;
        result.logsumexp[lse_offset] = std::log(sum) + max_score;

        for (int dv = 0; dv < problem.head_dim_value; ++dv) {
          float acc = 0.0f;
          for (int k = 0; k < problem.length; ++k) {
            if (scores[k] != -std::numeric_limits<float>::infinity()) {
              acc += (scores[k] / sum) * value[value_offset(problem, b, k, h, dv)];
            }
          }
          result.output[value_offset(problem, b, q, h, dv)] = acc;
        }
      }
    }
  }

  return result;
}

bool nearly_equal(float a, float b, float tolerance = 1.0e-5f) {
  return std::abs(a - b) <= tolerance;
}

bool run_uniform_reference_case(
    FnaForwardProblem problem,
    std::vector<float> const& expected_output) {
  problem.batch_size = 1;
  problem.heads = 1;
  problem.head_dim = 2;
  problem.head_dim_value = 1;
  problem.kernel_size = 3;
  problem.stride = 1;
  problem.scale = 1.0f;

  std::vector<float> query(
      problem.batch_size * problem.length * problem.heads * problem.head_dim,
      0.0f);
  std::vector<float> key = query;
  std::vector<float> value(
      problem.batch_size * problem.length * problem.heads *
      problem.head_dim_value);

  for (int token = 0; token < problem.length; ++token) {
    value[value_offset(problem, 0, token, 0, 0)] = static_cast<float>(token);
  }

  HostFnaResult result = host_fna_forward_1d_noncausal(problem, query, key, value);
  if (expected_output.size() != static_cast<size_t>(problem.length)) {
    return false;
  }

  for (int token = 0; token < problem.length; ++token) {
    float actual = result.output[value_offset(problem, 0, token, 0, 0)];
    if (!nearly_equal(actual, expected_output[token])) {
      std::cerr << "Host reference output mismatch at token " << token
                << ": expected " << expected_output[token]
                << ", got " << actual << "\n";
      return false;
    }

    float lse = result.logsumexp[token];
    if (!nearly_equal(lse, std::log(3.0f))) {
      std::cerr << "Host reference LSE mismatch at token " << token
                << ": expected " << std::log(3.0f) << ", got " << lse
                << "\n";
      return false;
    }
  }

  return true;
}

bool run_nonuniform_reference_case() {
  FnaForwardProblem problem;
  problem.batch_size = 1;
  problem.length = 4;
  problem.heads = 1;
  problem.head_dim = 1;
  problem.head_dim_value = 1;
  problem.kernel_size = 3;
  problem.stride = 1;
  problem.dilation = 1;
  problem.scale = 1.0f;

  std::vector<float> query(problem.length, 1.0f);
  std::vector<float> key = {
      0.0f,
      std::log(2.0f),
      std::log(4.0f),
      std::log(8.0f),
  };
  std::vector<float> value = {10.0f, 20.0f, 30.0f, 40.0f};

  HostFnaResult result = host_fna_forward_1d_noncausal(problem, query, key, value);
  std::vector<float> expected_output = {
      170.0f / 7.0f,
      170.0f / 7.0f,
      240.0f / 7.0f,
      240.0f / 7.0f,
  };
  std::vector<float> expected_lse = {
      std::log(7.0f),
      std::log(7.0f),
      std::log(14.0f),
      std::log(14.0f),
  };

  for (int token = 0; token < problem.length; ++token) {
    float actual = result.output[value_offset(problem, 0, token, 0, 0)];
    if (!nearly_equal(actual, expected_output[token])) {
      std::cerr << "Host reference non-uniform output mismatch at token "
                << token << ": expected " << expected_output[token]
                << ", got " << actual << "\n";
      return false;
    }

    float lse = result.logsumexp[token];
    if (!nearly_equal(lse, expected_lse[token])) {
      std::cerr << "Host reference non-uniform LSE mismatch at token "
                << token << ": expected " << expected_lse[token]
                << ", got " << lse << "\n";
      return false;
    }
  }

  return true;
}

bool run_host_reference_smoke() {
  FnaForwardProblem dilation_one;
  dilation_one.length = 8;
  dilation_one.dilation = 1;
  if (!run_uniform_reference_case(
          dilation_one, {1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f})) {
    return false;
  }

  FnaForwardProblem dilation_two;
  dilation_two.length = 8;
  dilation_two.dilation = 2;
  if (!run_uniform_reference_case(
          dilation_two,
          {2.0f, 3.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 5.0f})) {
    return false;
  }

  return run_nonuniform_reference_case();
}

struct Options {
  bool help = false;
  bool error = false;

  int batch_size = 2;
  int length = 128;
  int heads = 2;
  int head_dim = 32;
  int head_dim_value = 32;
  int kernel_size = 33;
  int stride = 1;
  int dilation = 1;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("batch_size", batch_size, batch_size);
    cmd.get_cmd_line_argument("length", length, length);
    cmd.get_cmd_line_argument("heads", heads, heads);
    cmd.get_cmd_line_argument("head_dim", head_dim, head_dim);
    cmd.get_cmd_line_argument("head_dim_value", head_dim_value, head_dim_value);
    cmd.get_cmd_line_argument("kernel_size", kernel_size, kernel_size);
    cmd.get_cmd_line_argument("stride", stride, stride);
    cmd.get_cmd_line_argument("dilation", dilation, dilation);

    if (batch_size <= 0 || length <= 0 || heads <= 0 || head_dim <= 0 ||
        head_dim_value <= 0 || kernel_size <= 0 || stride <= 0 ||
        dilation <= 0) {
      error = true;
    }
  }

  FnaForwardProblem problem() const {
    FnaForwardProblem p;
    p.batch_size = batch_size;
    p.length = length;
    p.heads = heads;
    p.head_dim = head_dim;
    p.head_dim_value = head_dim_value;
    p.kernel_size = kernel_size;
    p.stride = stride;
    p.dilation = dilation;
    p.scale = 1.0f;
    return p;
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "natten_fna_test\n\n"
        << "Options:\n\n"
        << "  --help                         Display this usage statement.\n"
        << "  --batch_size=<int>             Batch size (default: 2).\n"
        << "  --length=<int>                 1D sequence length (default: 128).\n"
        << "  --heads=<int>                  Attention heads (default: 2).\n"
        << "  --head_dim=<int>               Q/K head dim (default: 32).\n"
        << "  --head_dim_value=<int>         V/O head dim (default: 32).\n"
        << "  --kernel_size=<int>            Neighborhood size (default: 33).\n";
    return out;
  }
};

int run(Options const& options) {
  FnaForwardProblem problem = options.problem();

  if (problem.batch_size != options.batch_size || problem.length != options.length ||
      problem.heads != options.heads || problem.head_dim != options.head_dim ||
      problem.head_dim_value != options.head_dim_value ||
      problem.kernel_size != options.kernel_size ||
      problem.stride != options.stride || problem.dilation != options.dilation) {
    std::cerr << "Problem descriptor did not preserve parsed options.\n";
    return -1;
  }

  if (problem.query_tile_size != FnaPolicy::Threadblock::kM ||
      problem.key_tile_size != FnaPolicy::Threadblock::kN ||
      problem.head_dim > FnaPolicy::Threadblock::kK ||
      problem.head_dim_value > FnaPolicy::Threadblock::kK) {
    std::cerr << "Problem descriptor does not match smoke policy tile shape.\n";
    return -1;
  }

  if (!run_host_reference_smoke()) {
    return -1;
  }

  std::cout << "NATTEN FNA forward interface scaffold:\n"
            << "    Policy: DefaultFnaForwardPolicy<Rank, CausalMask, Element, "
               "ArchTag, ThreadblockShape>\n"
            << "    Problem: raw metadata descriptor only\n"
            << "    Host reference: 1D non-causal uniform-softmax smoke\n"
            << "    Launch path: intentionally absent until a CUTLASS-native "
               "TensorOp implementation exists\n"
            << "\nPassed\n";
  return 0;
}

} // namespace
} // namespace natten
} // namespace tiny_cutlass

int main(int argc, char const** args) {
  tiny_cutlass::natten::Options options;
  options.parse(argc, args);
  if (options.help) {
    options.print_usage(std::cout) << "\n";
    return 0;
  }
  if (options.error) {
    std::cerr << "Invalid options.\n";
    return -1;
  }

  return tiny_cutlass::natten::run(options);
}
