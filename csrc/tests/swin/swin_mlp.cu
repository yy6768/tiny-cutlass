/*
  Step 4 gate: Swin MLP (LN2 -> fc1 + GELU -> fc2 + residual2) against a host
  reference.

  Beyond plain parity this test pins down the two orderings that silently give
  wrong answers if they drift:

    * the residual is the UNNORMALIZED block input, not the LayerNorm output
    * GELU is the exact erf form, not the tanh approximation

  Both are checked directly: a "wrong residual" and a "tanh GELU" reference are
  computed alongside the correct one, and the test asserts the device output is
  closer to the correct one. That way the test fails loudly if the kernel ever
  matches the wrong convention, instead of the tolerance quietly absorbing it.
*/

#include <cmath>
#include <cstdio>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "swin/swin_mlp/ops/swin_mlp.h"
#include "swin/swin_problem.h"

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

// tanh approximation of GELU -- used ONLY as a wrong-convention control.
float gelu_tanh(float x) {
  float const inner =
      0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

struct Case {
  int rows;
  int channels;
  int hidden;
  int seed;
};

void run_case(Case const& c, char const* label) {
  std::printf("%s (rows=%d C=%d hidden=%d)\n", label, c.rows, c.channels,
              c.hidden);

  size_t const token_elements = size_t(c.rows) * size_t(c.channels);
  size_t const hidden_elements = size_t(c.rows) * size_t(c.hidden);

  std::vector<Element> input(token_elements);
  std::vector<Element> fc1_weight(size_t(c.hidden) * size_t(c.channels));
  std::vector<Element> fc2_weight(size_t(c.channels) * size_t(c.hidden));
  tiny_cutlass::testing::fill_random_uniform(input, c.seed, -1.0f, 1.0f);
  // Weights scaled down so the two chained GEMMs stay in fp16's comfortable
  // range; otherwise the reference and the kernel diverge on rounding alone.
  tiny_cutlass::testing::fill_random_uniform(fc1_weight, c.seed + 1, -0.1f, 0.1f);
  tiny_cutlass::testing::fill_random_uniform(fc2_weight, c.seed + 2, -0.1f, 0.1f);

  std::vector<float> gamma(static_cast<size_t>(c.channels), 0.0f);
  std::vector<float> beta(static_cast<size_t>(c.channels), 0.0f);
  std::vector<float> fc1_bias(static_cast<size_t>(c.hidden), 0.0f);
  std::vector<float> fc2_bias(static_cast<size_t>(c.channels), 0.0f);
  for (int i = 0; i < c.channels; ++i) {
    gamma[size_t(i)] = 1.0f + 0.1f * std::cos(0.17f * float(i));
    beta[size_t(i)] = 0.02f * std::sin(0.23f * float(i));
    fc2_bias[size_t(i)] = 0.03f * std::sin(0.11f * float(i));
  }
  for (int i = 0; i < c.hidden; ++i) {
    fc1_bias[size_t(i)] = 0.04f * std::cos(0.13f * float(i));
  }

  DeviceBuffer<Element> d_input(token_elements);
  DeviceBuffer<Element> d_fc1_weight(fc1_weight.size());
  DeviceBuffer<Element> d_fc2_weight(fc2_weight.size());
  DeviceBuffer<float> d_gamma(gamma.size());
  DeviceBuffer<float> d_beta(beta.size());
  DeviceBuffer<float> d_fc1_bias(fc1_bias.size());
  DeviceBuffer<float> d_fc2_bias(fc2_bias.size());
  DeviceBuffer<Element> d_output(token_elements);
  DeviceBuffer<Element> d_normalized(token_elements);
  DeviceBuffer<Element> d_hidden(hidden_elements);

  d_input.copy_from_host(input);
  d_fc1_weight.copy_from_host(fc1_weight);
  d_fc2_weight.copy_from_host(fc2_weight);
  d_gamma.copy_from_host(gamma);
  d_beta.copy_from_host(beta);
  d_fc1_bias.copy_from_host(fc1_bias);
  d_fc2_bias.copy_from_host(fc2_bias);

  swin_mlp::SwinMlpArguments<Element, float> args;
  args.rows = c.rows;
  args.channels = c.channels;
  args.hidden = c.hidden;
  args.input = d_input.get();
  args.gamma = d_gamma.get();
  args.beta = d_beta.get();
  args.fc1_weight = d_fc1_weight.get();
  args.fc1_bias = d_fc1_bias.get();
  args.fc2_weight = d_fc2_weight.get();
  args.fc2_bias = d_fc2_bias.get();
  args.output = d_output.get();
  args.normalized = d_normalized.get();
  args.hidden_buffer = d_hidden.get();

  cutlass::Status status = swin_mlp::swin_mlp_can_implement<Element, float>(args);
  if (status != cutlass::Status::kSuccess) {
    std::printf("  FAIL  can_implement: %s\n",
                cutlass::cutlassGetStatusString(status));
    ++g_failures;
    return;
  }

  status = swin_mlp::swin_mlp<Element, float>(args);
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

  std::vector<Element> got(token_elements);
  d_output.copy_to_host(got);

  std::vector<float> const want = testing::swin_mlp_reference<Element>(
      c.rows, c.channels, c.hidden, input, gamma, beta, fc1_weight, fc1_bias,
      fc2_weight, fc2_bias);

  auto const result = tiny_cutlass::testing::compare_host(got, want);
  std::printf("        MAE %.3e  max_abs %.3e\n", result.mae, result.max_abs);
  check(result.finite, "output is finite");
  check(result.mae <= 1.0e-2, "MAE within 1e-2 of host reference");

  //
  // Wrong-convention controls.
  //

  // 1. Residual taken from the NORMALIZED tensor instead of the input.
  {
    std::vector<float> normalized(token_elements);
    for (size_t i = 0; i < token_elements; ++i) {
      normalized[i] = float(input[i]);
    }
    testing::layernorm_reference(normalized, c.rows, c.channels, gamma, beta);
    std::vector<float> wrong = want;
    for (size_t i = 0; i < token_elements; ++i) {
      wrong[i] = wrong[i] - float(input[i]) + normalized[i];
    }
    auto const wrong_result = tiny_cutlass::testing::compare_host(got, wrong);
    std::printf("        control: residual from normalized -> MAE %.3e\n",
                wrong_result.mae);
    check(result.mae < wrong_result.mae,
          "closer to residual-from-input than residual-from-normalized");
  }

  // 2. tanh GELU instead of the exact erf form.
  {
    std::vector<float> normalized(token_elements);
    for (size_t i = 0; i < token_elements; ++i) {
      normalized[i] = float(input[i]);
    }
    testing::layernorm_reference(normalized, c.rows, c.channels, gamma, beta);
    std::vector<Element> normalized_rounded(token_elements);
    for (size_t i = 0; i < token_elements; ++i) {
      normalized_rounded[i] = Element(normalized[i]);
    }
    std::vector<float> h = testing::linear_reference<Element>(
        normalized_rounded, fc1_weight, fc1_bias, c.rows, c.hidden, c.channels);
    std::vector<Element> h_rounded(h.size());
    for (size_t i = 0; i < h.size(); ++i) {
      h_rounded[i] = Element(gelu_tanh(h[i]));
    }
    std::vector<float> wrong = testing::linear_reference<Element>(
        h_rounded, fc2_weight, fc2_bias, c.rows, c.channels, c.hidden);
    for (size_t i = 0; i < token_elements; ++i) {
      wrong[i] += float(input[i]);
    }
    auto const wrong_result = tiny_cutlass::testing::compare_host(got, wrong);
    std::printf("        control: tanh GELU -> MAE %.3e\n", wrong_result.mae);
    check(result.mae < wrong_result.mae, "closer to erf GELU than tanh GELU");
  }
}

}  // namespace

int main() {
  // Stage 1 in window-major order: 196 windows x 16 tokens = 3136 rows, C = 96.
  SwinStageProblem stage1;
  stage1.height = 56;
  stage1.width = 56;
  stage1.channels = 96;
  stage1.num_heads = 3;
  stage1.window_size = 4;
  run_case({stage1.num_window_rows(), stage1.channels, stage1.mlp_hidden(), 2026},
           "stage1");

  // Stage 2 shape, to vary C and hidden together.
  SwinStageProblem stage2;
  stage2.height = 28;
  stage2.width = 28;
  stage2.channels = 192;
  stage2.num_heads = 6;
  stage2.window_size = 4;
  run_case({stage2.num_window_rows(), stage2.channels, stage2.mlp_hidden(), 7},
           "stage2");

  // A row count that is not a multiple of the 128-row tile, to exercise
  // predication on the M edge.
  run_case({200, 96, 384, 11}, "ragged rows");

  // Rejections.
  {
    std::printf("rejects misaligned channels\n");
    swin_mlp::SwinMlpArguments<Element, float> args;
    args.rows = 16;
    args.channels = 12;  // not a multiple of 8
    args.hidden = 48;
    args.input = reinterpret_cast<Element const*>(0x1);
    args.fc1_weight = reinterpret_cast<Element const*>(0x1);
    args.fc2_weight = reinterpret_cast<Element const*>(0x1);
    args.output = reinterpret_cast<Element*>(0x1);
    args.normalized = reinterpret_cast<Element*>(0x1);
    args.hidden_buffer = reinterpret_cast<Element*>(0x1);
    check(swin_mlp::swin_mlp_can_implement<Element, float>(args) ==
              cutlass::Status::kErrorNotSupported,
          "can_implement rejects channels % 8 != 0");
  }

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
