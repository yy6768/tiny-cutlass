#include <iostream>
#include <type_traits>

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

  std::cout << "NATTEN FNA forward interface scaffold:\n"
            << "    Policy: DefaultFnaForwardPolicy<Rank, CausalMask, Element, "
               "ArchTag, ThreadblockShape>\n"
            << "    Problem: raw metadata descriptor only\n"
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
