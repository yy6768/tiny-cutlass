/*
  Step 5 gate: PatchMerging (2x2 concat gather -> LayerNorm(4C) -> 4C -> 2C).

  The host reference writes the concat out longhand from the official slicing
  expression, independently of build_patch_merging_index(), so the gather table
  is genuinely under test rather than merely reproduced.

  A wrong-convention control pins the concat ORDER: a table using the transposed
  order (0,0) (0,1) (1,0) (1,1) is a permutation of the channel axis, which
  leaves the LayerNorm statistics identical and only shows up after the
  projection -- exactly the kind of bug a shape check cannot catch.
*/

#include <cmath>
#include <cstdio>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "swin/patch_merging/ops/patch_merging.h"
#include "swin/swin_problem.h"
#include "swin/window_index.h"

#include "reference.h"
#include "test_utils.h"

using namespace tiny_cutlass::swin;
using tiny_cutlass::testing::DeviceBuffer;

namespace {

using Element = cutlass::half_t;

int g_failures = 0;

void check(bool condition, char const* what) {
  if (!condition) {
    std::printf("  FAIL  %s\n", what);
    ++g_failures;
  } else {
    std::printf("  ok    %s\n", what);
  }
}

void run_case(PatchMergingProblem const& problem, int seed, char const* label) {
  std::printf("%s (B=%d H=%d W=%d C=%d -> %dx%dx%d)\n", label, problem.batch,
              problem.height, problem.width, problem.channels,
              problem.out_height(), problem.out_width(),
              problem.out_channels());

  int const C = problem.channels;
  int const concat = problem.concat_channels();
  int const out_channels = problem.out_channels();
  int const rows = problem.num_out_tokens();

  std::vector<Element> input(
      size_t(problem.batch) * size_t(problem.height) * size_t(problem.width) *
      size_t(C));
  std::vector<Element> weight(size_t(out_channels) * size_t(concat));
  tiny_cutlass::testing::fill_random_uniform(input, seed, -1.0f, 1.0f);
  tiny_cutlass::testing::fill_random_uniform(weight, seed + 1, -0.1f, 0.1f);

  std::vector<float> gamma(static_cast<size_t>(concat), 0.0f);
  std::vector<float> beta(static_cast<size_t>(concat), 0.0f);
  std::vector<float> bias(static_cast<size_t>(out_channels), 0.0f);
  for (int i = 0; i < concat; ++i) {
    gamma[size_t(i)] = 1.0f + 0.1f * std::cos(0.17f * float(i));
    beta[size_t(i)] = 0.02f * std::sin(0.23f * float(i));
  }
  for (int i = 0; i < out_channels; ++i) {
    bias[size_t(i)] = 0.03f * std::sin(0.11f * float(i));
  }

  std::vector<int> const gather = build_patch_merging_index(problem);

  DeviceBuffer<Element> d_input(input.size());
  DeviceBuffer<Element> d_weight(weight.size());
  DeviceBuffer<int> d_gather(gather.size());
  DeviceBuffer<float> d_gamma(gamma.size());
  DeviceBuffer<float> d_beta(beta.size());
  DeviceBuffer<float> d_bias(bias.size());
  DeviceBuffer<Element> d_output(size_t(rows) * size_t(out_channels));
  DeviceBuffer<Element> d_normalized(size_t(rows) * size_t(concat));

  d_input.copy_from_host(input);
  d_weight.copy_from_host(weight);
  d_gather.copy_from_host(gather);
  d_gamma.copy_from_host(gamma);
  d_beta.copy_from_host(beta);
  d_bias.copy_from_host(bias);

  patch_merging::PatchMergingArguments<Element, float> args;
  args.problem = problem;
  args.input = d_input.get();
  args.gather = d_gather.get();
  args.gamma = d_gamma.get();
  args.beta = d_beta.get();
  args.weight = d_weight.get();
  args.bias = d_bias.get();
  args.output = d_output.get();
  args.normalized = d_normalized.get();

  cutlass::Status status =
      patch_merging::patch_merging_can_implement<Element, float>(args);
  if (status != cutlass::Status::kSuccess) {
    std::printf("  FAIL  can_implement: %s\n",
                cutlass::cutlassGetStatusString(status));
    ++g_failures;
    return;
  }

  status = patch_merging::patch_merging<Element, float>(args);
  if (status != cutlass::Status::kSuccess) {
    std::printf("  FAIL  run: %s\n", cutlass::cutlassGetStatusString(status));
    ++g_failures;
    return;
  }

  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::printf("  FAIL  sync: %s\n", cudaGetErrorString(error));
    ++g_failures;
    return;
  }

  std::vector<Element> got(size_t(rows) * size_t(out_channels));
  d_output.copy_to_host(got);

  std::vector<float> const want = testing::patch_merging_reference<Element>(
      problem, input, gamma, beta, weight, bias);

  auto const result = tiny_cutlass::testing::compare_host(got, want);
  std::printf("        MAE %.3e  max_abs %.3e\n", result.mae, result.max_abs);
  check(result.finite, "output is finite");
  check(result.mae <= 1.0e-2, "MAE within 1e-2 of host reference");

  // Control: the transposed concat order. Same statistics, different result.
  {
    int const di[4] = {0, 0, 1, 1};
    int const dj[4] = {0, 1, 0, 1};
    int const H = problem.height;
    int const W = problem.width;
    int const OH = problem.out_height();
    int const OW = problem.out_width();

    std::vector<float> merged(size_t(rows) * size_t(concat), 0.0f);
    for (int b = 0; b < problem.batch; ++b) {
      for (int i = 0; i < OH; ++i) {
        for (int j = 0; j < OW; ++j) {
          int const out_row = (b * OH + i) * OW + j;
          for (int k = 0; k < 4; ++k) {
            int const src = (b * H + 2 * i + di[k]) * W + 2 * j + dj[k];
            for (int c = 0; c < C; ++c) {
              merged[size_t(out_row) * size_t(concat) + size_t(k * C + c)] =
                  float(input[size_t(src) * size_t(C) + size_t(c)]);
            }
          }
        }
      }
    }
    testing::layernorm_reference(merged, rows, concat, gamma, beta);
    std::vector<Element> merged_rounded(merged.size());
    for (size_t i = 0; i < merged.size(); ++i) {
      merged_rounded[i] = Element(merged[i]);
    }
    std::vector<float> const wrong = testing::linear_reference<Element>(
        merged_rounded, weight, bias, rows, out_channels, concat);

    auto const wrong_result = tiny_cutlass::testing::compare_host(got, wrong);
    std::printf("        control: transposed concat order -> MAE %.3e\n",
                wrong_result.mae);
    check(result.mae < wrong_result.mae,
          "closer to official concat order than the transposed one");
  }
}

}  // namespace

int main() {
  // stage1 -> stage2 transition: 56x56x96 -> 28x28x192.
  PatchMergingProblem stage1;
  stage1.height = 56;
  stage1.width = 56;
  stage1.channels = 96;
  run_case(stage1, 2026, "stage1 -> stage2");

  // stage2 -> stage3: 28x28x192 -> 14x14x384.
  PatchMergingProblem stage2;
  stage2.height = 28;
  stage2.width = 28;
  stage2.channels = 192;
  run_case(stage2, 7, "stage2 -> stage3");

  // Batched and non-square, to catch anything assuming H == W or B == 1.
  PatchMergingProblem small;
  small.batch = 2;
  small.height = 8;
  small.width = 12;
  small.channels = 32;
  run_case(small, 11, "batched non-square");

  // Rejections.
  {
    std::printf("rejects odd spatial extent\n");
    PatchMergingProblem odd;
    odd.height = 7;
    odd.width = 8;
    odd.channels = 96;
    patch_merging::PatchMergingArguments<Element, float> args;
    args.problem = odd;
    args.input = reinterpret_cast<Element const*>(0x1);
    args.gather = reinterpret_cast<int const*>(0x1);
    args.weight = reinterpret_cast<Element const*>(0x1);
    args.output = reinterpret_cast<Element*>(0x1);
    args.normalized = reinterpret_cast<Element*>(0x1);
    check(patch_merging::patch_merging_can_implement<Element, float>(args) ==
              cutlass::Status::kErrorInvalidProblem,
          "can_implement rejects odd H with kErrorInvalidProblem");
  }

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
