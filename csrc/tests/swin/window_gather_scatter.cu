/*
  Step 2 gate: prove on device that CUTLASS 2.x's stock GatherA / ScatterD
  reproduce Swin's window_partition / window_reverse exactly, so neither needs a
  bespoke kernel or a bespoke iterator.

  ---------------------------------------------------------------------------
  WHAT IS BEING TESTED
  ---------------------------------------------------------------------------
  A Swin token's C channels are contiguous, so one token == one GEMM row. The
  gather in predicated_tile_access_iterator.h:575 is a pure row remap

      if (Gather) coord_strided = indices_[coord_strided];

  and for a row-major A operand the strided rank IS the GEMM M dimension. So
  with B set to the identity matrix:

      GatherA  : D[m] = A[idx[m]]   == window_partition
      ScatterD : D[idx[m]] = A[m]   == window_reverse

  Both run with alpha = 1, beta = 0. A third case runs ScatterD with beta = 1 to
  confirm the epilogue's SOURCE-C path is scattered by the same index array --
  that is what lets residual1 be read back through the projection GEMM's source
  operand with no extra kernel.

  These identity GEMMs are NOT the shipping code path; they isolate the gather
  mechanism from any arithmetic so a mismatch can only be an index-mapping bug.
  Step 3 folds the same index arrays into the real QKV and projection GEMMs.

  ---------------------------------------------------------------------------
  NOTE ON PREDICATION
  ---------------------------------------------------------------------------
  Predication uses the LOGICAL extent (problem_size M), while the gather
  redirects the ADDRESS. So the A buffer may legitimately have a different row
  count than M -- here A has num_tokens rows while M is num_window_rows. That is
  the same arrangement example 36 uses for GatherB.
*/

#include <cstdio>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"

#include "swin/swin_problem.h"
#include "swin/window_index.h"

#include "test_utils.h"

using namespace tiny_cutlass::swin;
using tiny_cutlass::testing::DeviceBuffer;

namespace {

using Element = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

using RowMajor = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

int constexpr kAlignment = 8;
int constexpr kStages = 3;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    Element,
    128 / cutlass::sizeof_bits<Element>::value,
    ElementAccumulator,
    ElementCompute>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Only the three gather/scatter flags differ between these two operators.
template <bool GatherA, bool ScatterD>
using RemapGemm = cutlass::gemm::device::GemmUniversal<
    Element, RowMajor,       // A: token rows
    Element, RowMajor,       // B: identity
    Element, RowMajor,       // C/D
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    Swizzle,
    kStages,
    kAlignment, kAlignment,
    cutlass::arch::OpMultiplyAdd,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    GatherA,
    false,
    ScatterD>;

using GatherGemm = RemapGemm<true, false>;
using ScatterGemm = RemapGemm<false, true>;

int g_failures = 0;

void check(bool condition, char const* what) {
  if (!condition) {
    std::printf("  FAIL  %s\n", what);
    ++g_failures;
  } else {
    std::printf("  ok    %s\n", what);
  }
}

bool check_status(cutlass::Status status, char const* what) {
  if (status != cutlass::Status::kSuccess) {
    std::printf("  FAIL  %s: %s\n", what, cutlass::cutlassGetStatusString(status));
    ++g_failures;
    return false;
  }
  return true;
}

bool check_cuda(char const* what) {
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::printf("  FAIL  %s: %s\n", what, cudaGetErrorString(error));
    ++g_failures;
    return false;
  }
  return true;
}

// Identity matrix, C x C row-major.
std::vector<Element> make_identity(int c) {
  std::vector<Element> identity(size_t(c) * size_t(c), Element(0.0f));
  for (int i = 0; i < c; ++i) {
    identity[size_t(i) * size_t(c) + size_t(i)] = Element(1.0f);
  }
  return identity;
}

// Runs one identity GEMM. `m_rows` is the GEMM M extent; the A and D buffers may
// have different row counts than M when a gather/scatter redirects them.
template <typename Gemm>
bool run_remap(
    int m_rows,
    int channels,
    Element const* device_a,
    Element const* device_identity,
    Element const* device_c,
    Element* device_d,
    int const* gather_a,
    int const* scatter_d,
    float beta) {
  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmCoord{m_rows, channels, channels},
      /*batch_count=*/1,
      {ElementCompute(1.0f), ElementCompute(beta)},
      device_a,
      device_identity,
      device_c,
      device_d,
      /*batch_stride_A=*/int64_t(0),
      /*batch_stride_B=*/int64_t(0),
      /*batch_stride_C=*/int64_t(0),
      /*batch_stride_D=*/int64_t(0),
      /*lda=*/int64_t(channels),
      /*ldb=*/int64_t(channels),
      /*ldc=*/int64_t(channels),
      /*ldd=*/int64_t(channels),
      gather_a,
      /*ptr_gather_B_indices=*/nullptr,
      scatter_d);

  Gemm gemm;
  if (!check_status(Gemm::can_implement(args), "can_implement")) {
    return false;
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  DeviceBuffer<uint8_t> workspace(workspace_size);

  if (!check_status(gemm.initialize(args, workspace.get()), "initialize")) {
    return false;
  }
  if (!check_status(gemm(), "run")) {
    return false;
  }
  return check_cuda("kernel launch");
}

void test_partition_via_gather(SwinStageProblem const& problem, char const* label) {
  std::printf("%s: GatherA == window_partition (H=%d W=%d C=%d window=%d shift=%d)\n",
              label, problem.height, problem.width, problem.channels,
              problem.effective_window(), problem.effective_shift());

  int const C = problem.channels;
  int const rows = problem.num_window_rows();

  std::vector<Element> tokens(size_t(problem.num_tokens()) * size_t(C));
  tiny_cutlass::testing::fill_random_uniform(tokens, /*seed=*/2026);

  std::vector<Element> const identity = make_identity(C);

  DeviceBuffer<Element> d_tokens(tokens.size());
  DeviceBuffer<Element> d_identity(identity.size());
  DeviceBuffer<Element> d_out(size_t(rows) * size_t(C));
  d_tokens.copy_from_host(tokens);
  d_identity.copy_from_host(identity);

  std::vector<int> const index = build_window_row_index(problem);
  DeviceBuffer<int> d_index(index.size());
  d_index.copy_from_host(index);

  if (!run_remap<GatherGemm>(rows, C, d_tokens.get(), d_identity.get(),
                             /*device_c=*/nullptr, d_out.get(), d_index.get(),
                             /*scatter_d=*/nullptr, /*beta=*/0.0f)) {
    return;
  }

  std::vector<Element> got(size_t(rows) * size_t(C));
  d_out.copy_to_host(got);

  std::vector<Element> want;
  window_partition_reference(problem, tokens, want);

  // The identity GEMM is exact in fp16 for these magnitudes (one multiply by
  // 1.0 accumulated in fp32), so this is a bit-for-bit comparison, not a
  // tolerance check -- any difference is a mapping bug, not rounding.
  auto const result = tiny_cutlass::testing::compare_host(got, want);
  std::printf("        MAE %.3e  max_abs %.3e\n", result.mae, result.max_abs);
  check(result.finite && result.max_abs == 0.0f,
        "gathered rows are bit-identical to window_partition");
}

void test_reverse_via_scatter(SwinStageProblem const& problem, char const* label) {
  std::printf("%s: ScatterD == window_reverse\n", label);

  int const C = problem.channels;
  int const rows = problem.num_window_rows();

  std::vector<Element> windows(size_t(rows) * size_t(C));
  tiny_cutlass::testing::fill_random_uniform(windows, /*seed=*/7);

  std::vector<Element> const identity = make_identity(C);

  DeviceBuffer<Element> d_windows(windows.size());
  DeviceBuffer<Element> d_identity(identity.size());
  DeviceBuffer<Element> d_out(size_t(problem.num_tokens()) * size_t(C));
  d_windows.copy_from_host(windows);
  d_identity.copy_from_host(identity);

  std::vector<Element> zeros(size_t(problem.num_tokens()) * size_t(C), Element(0.0f));
  d_out.copy_from_host(zeros);

  std::vector<int> const index = build_window_row_index(problem);
  DeviceBuffer<int> d_index(index.size());
  d_index.copy_from_host(index);

  if (!run_remap<ScatterGemm>(rows, C, d_windows.get(), d_identity.get(),
                              /*device_c=*/nullptr, d_out.get(),
                              /*gather_a=*/nullptr, d_index.get(),
                              /*beta=*/0.0f)) {
    return;
  }

  std::vector<Element> got(size_t(problem.num_tokens()) * size_t(C));
  d_out.copy_to_host(got);

  std::vector<Element> want;
  window_reverse_reference(problem, windows, want);

  auto const result = tiny_cutlass::testing::compare_host(got, want);
  std::printf("        MAE %.3e  max_abs %.3e\n", result.mae, result.max_abs);
  check(result.finite && result.max_abs == 0.0f,
        "scattered rows are bit-identical to window_reverse");
}

// The load path of the epilogue iterator is scattered by the same indices, so a
// residual can be added straight through source C. This is the mechanism
// residual1 will use in step 3, so it is worth pinning down separately.
void test_scatter_with_residual(SwinStageProblem const& problem, char const* label) {
  std::printf("%s: ScatterD source-C path carries residual1\n", label);

  int const C = problem.channels;
  int const rows = problem.num_window_rows();
  size_t const token_elements = size_t(problem.num_tokens()) * size_t(C);

  std::vector<Element> windows(size_t(rows) * size_t(C));
  std::vector<Element> residual(token_elements);
  tiny_cutlass::testing::fill_random_uniform(windows, /*seed=*/11);
  tiny_cutlass::testing::fill_random_uniform(residual, /*seed=*/13);

  std::vector<Element> const identity = make_identity(C);

  DeviceBuffer<Element> d_windows(windows.size());
  DeviceBuffer<Element> d_identity(identity.size());
  DeviceBuffer<Element> d_residual(token_elements);
  DeviceBuffer<Element> d_out(token_elements);
  d_windows.copy_from_host(windows);
  d_identity.copy_from_host(identity);
  d_residual.copy_from_host(residual);

  std::vector<Element> zeros(token_elements, Element(0.0f));
  d_out.copy_from_host(zeros);

  std::vector<int> const index = build_window_row_index(problem);
  DeviceBuffer<int> d_index(index.size());
  d_index.copy_from_host(index);

  if (!run_remap<ScatterGemm>(rows, C, d_windows.get(), d_identity.get(),
                              d_residual.get(), d_out.get(),
                              /*gather_a=*/nullptr, d_index.get(),
                              /*beta=*/1.0f)) {
    return;
  }

  std::vector<Element> got(token_elements);
  d_out.copy_to_host(got);

  // reverse(windows) + residual, computed in fp32 then rounded like the
  // epilogue does.
  std::vector<Element> reversed;
  window_reverse_reference(problem, windows, reversed);
  std::vector<float> want(token_elements);
  for (size_t i = 0; i < token_elements; ++i) {
    want[i] = float(Element(float(reversed[i]) + float(residual[i])));
  }

  auto const result = tiny_cutlass::testing::compare_host(got, want);
  std::printf("        MAE %.3e  max_abs %.3e\n", result.mae, result.max_abs);
  check(result.finite && result.max_abs == 0.0f,
        "scatter + source C == window_reverse(x) + residual");
}

}  // namespace

int main() {
  // Both W-MSA and SW-MSA: the shift lives entirely in the index table, so the
  // kernel side must be identical between them.
  SwinStageProblem stage1;
  stage1.height = 56;
  stage1.width = 56;
  stage1.channels = 96;
  stage1.num_heads = 3;
  stage1.window_size = 4;
  stage1.shift_size = 0;

  SwinStageProblem stage1_shift = stage1;
  stage1_shift.shift_size = 2;

  SwinStageProblem small;
  small.batch = 2;
  small.height = 8;
  small.width = 12;
  small.channels = 32;
  small.num_heads = 2;
  small.window_size = 4;
  small.shift_size = 2;

  test_partition_via_gather(stage1, "stage1 W-MSA");
  test_reverse_via_scatter(stage1, "stage1 W-MSA");
  test_scatter_with_residual(stage1, "stage1 W-MSA");

  test_partition_via_gather(stage1_shift, "stage1 SW-MSA");
  test_reverse_via_scatter(stage1_shift, "stage1 SW-MSA");
  test_scatter_with_residual(stage1_shift, "stage1 SW-MSA");

  test_partition_via_gather(small, "small batched SW-MSA");
  test_reverse_via_scatter(small, "small batched SW-MSA");

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
