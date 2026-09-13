/*
  Step 1 gate: fused PatchEmbed (conv + bias + channel-axis LayerNorm) against a
  host reference.

  The interesting property being tested is that the LayerNorm is COMPLETE inside
  one kernel. A partial-reduction bug would not blow up -- it would produce
  plausible values normalized by a fraction of the row -- so the test also
  checks the statistical signature of a correct LayerNorm directly: every output
  row must have mean ~0 and variance ~1 once gamma/beta are identity.
*/

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/half.h"

#include "swin/patch_embed/ops/patch_embed.h"
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

// Zero-pads the channel axis of an NHWC activation from `in_channels` to
// `in_channels_padded`. TensorOp fp16 needs an 8-wide C load and this family
// takes no SIMT fallback, so the pad is mandatory rather than an optimization.
std::vector<Element> pad_activation_channels(
    std::vector<Element> const& src, PatchEmbedProblem const& problem) {
  int const H = problem.image_size;
  int const W = problem.image_size;
  int const C = problem.in_channels;
  int const Cp = problem.in_channels_padded;

  std::vector<Element> dst(
      size_t(problem.batch) * size_t(H) * size_t(W) * size_t(Cp), Element(0.0f));
  for (int n = 0; n < problem.batch; ++n) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) {
          size_t const si =
              ((size_t(n) * size_t(H) + size_t(h)) * size_t(W) + size_t(w)) *
                  size_t(C) + size_t(c);
          size_t const di =
              ((size_t(n) * size_t(H) + size_t(h)) * size_t(W) + size_t(w)) *
                  size_t(Cp) + size_t(c);
          dst[di] = src[si];
        }
      }
    }
  }
  return dst;
}

// Same for the filter, KRSC.
std::vector<Element> pad_filter_channels(
    std::vector<Element> const& src, PatchEmbedProblem const& problem) {
  int const K = problem.embed_dim;
  int const R = problem.patch_size;
  int const S = problem.patch_size;
  int const C = problem.in_channels;
  int const Cp = problem.in_channels_padded;

  std::vector<Element> dst(
      size_t(K) * size_t(R) * size_t(S) * size_t(Cp), Element(0.0f));
  for (int k = 0; k < K; ++k) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        for (int c = 0; c < C; ++c) {
          size_t const si =
              ((size_t(k) * size_t(R) + size_t(r)) * size_t(S) + size_t(s)) *
                  size_t(C) + size_t(c);
          size_t const di =
              ((size_t(k) * size_t(R) + size_t(r)) * size_t(S) + size_t(s)) *
                  size_t(Cp) + size_t(c);
          dst[di] = src[si];
        }
      }
    }
  }
  return dst;
}

// A correct channel-axis LayerNorm with gamma=1, beta=0 leaves every row with
// mean 0 and variance 1. Normalizing by a PARTIAL sum would still look
// "reasonable" elementwise but would fail this, so it is checked separately
// from the reference comparison.
void check_row_statistics(
    std::vector<Element> const& output, int rows, int cols, char const* label) {
  double worst_mean = 0.0;
  double worst_var_error = 0.0;
  for (int r = 0; r < rows; ++r) {
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      sum += double(float(output[size_t(r) * size_t(cols) + size_t(c)]));
    }
    double const mean = sum / double(cols);

    double var_sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      double const d =
          double(float(output[size_t(r) * size_t(cols) + size_t(c)])) - mean;
      var_sum += d * d;
    }
    double const variance = var_sum / double(cols);

    worst_mean = std::max(worst_mean, std::fabs(mean));
    worst_var_error = std::max(worst_var_error, std::fabs(variance - 1.0));
  }
  std::printf("        %s: max |mean| %.3e, max |var - 1| %.3e\n", label,
              worst_mean, worst_var_error);
  // fp16 storage of the normalized values sets the floor here.
  check(worst_mean < 2.0e-2, "every row has mean ~ 0");
  check(worst_var_error < 5.0e-2, "every row has variance ~ 1");
}

struct Options {
  PatchEmbedProblem problem;
  bool identity_affine = false;  // gamma = 1, beta = 0 for the stats check
  int seed = 2026;
};

bool run_case(Options const& options, char const* label) {
  PatchEmbedProblem const& problem = options.problem;
  std::printf("%s (B=%d image=%d in_ch=%d->%d embed=%d patch=%d)\n", label,
              problem.batch, problem.image_size, problem.in_channels,
              problem.in_channels_padded, problem.embed_dim, problem.patch_size);

  int const K = problem.embed_dim;
  int const tokens = problem.num_tokens();

  // Unpadded host tensors, then padded copies for both device and reference so
  // the two see byte-identical inputs.
  std::vector<Element> input(
      size_t(problem.batch) * size_t(problem.image_size) *
      size_t(problem.image_size) * size_t(problem.in_channels));
  std::vector<Element> filter(
      size_t(K) * size_t(problem.patch_size) * size_t(problem.patch_size) *
      size_t(problem.in_channels));
  tiny_cutlass::testing::fill_random_uniform(input, options.seed, -1.0f, 1.0f);
  tiny_cutlass::testing::fill_random_uniform(filter, options.seed + 1, -0.5f, 0.5f);

  std::vector<Element> const padded_input = pad_activation_channels(input, problem);
  std::vector<Element> const padded_filter = pad_filter_channels(filter, problem);

  // Two-arg form: `std::vector<float> bias(size_t(K));` is a most-vexing-parse
  // and declares a function instead.
  std::vector<float> bias(static_cast<size_t>(K), 0.0f);
  std::vector<float> gamma(static_cast<size_t>(K), 0.0f);
  std::vector<float> beta(static_cast<size_t>(K), 0.0f);
  for (int k = 0; k < K; ++k) {
    bias[size_t(k)] = 0.05f * std::sin(0.31f * float(k));
    if (options.identity_affine) {
      gamma[size_t(k)] = 1.0f;
      beta[size_t(k)] = 0.0f;
    } else {
      gamma[size_t(k)] = 1.0f + 0.1f * std::cos(0.17f * float(k));
      beta[size_t(k)] = 0.02f * std::sin(0.23f * float(k));
    }
  }

  DeviceBuffer<Element> d_input(padded_input.size());
  DeviceBuffer<Element> d_filter(padded_filter.size());
  DeviceBuffer<float> d_bias(bias.size());
  DeviceBuffer<float> d_gamma(gamma.size());
  DeviceBuffer<float> d_beta(beta.size());
  DeviceBuffer<Element> d_output(size_t(tokens) * size_t(K));
  d_input.copy_from_host(padded_input);
  d_filter.copy_from_host(padded_filter);
  d_bias.copy_from_host(bias);
  d_gamma.copy_from_host(gamma);
  d_beta.copy_from_host(beta);

  patch_embed::PatchEmbedArguments<Element, float> args;
  args.problem = problem;
  args.input = d_input.get();
  args.filter = d_filter.get();
  args.bias = d_bias.get();
  args.gamma = d_gamma.get();
  args.beta = d_beta.get();
  args.output = d_output.get();

  cutlass::Status status =
      patch_embed::patch_embed_can_implement<Element, float>(args);
  if (status != cutlass::Status::kSuccess) {
    std::printf("  FAIL  can_implement: %s\n",
                cutlass::cutlassGetStatusString(status));
    ++g_failures;
    return false;
  }

  status = patch_embed::patch_embed<Element, float>(args);
  if (status != cutlass::Status::kSuccess) {
    std::printf("  FAIL  run: %s\n", cutlass::cutlassGetStatusString(status));
    ++g_failures;
    return false;
  }

  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::printf("  FAIL  sync: %s\n", cudaGetErrorString(error));
    ++g_failures;
    return false;
  }

  std::vector<Element> got(size_t(tokens) * size_t(K));
  d_output.copy_to_host(got);

  std::vector<float> const want = testing::patch_embed_reference<Element>(
      problem, padded_input, padded_filter, bias, gamma, beta);

  auto const result = tiny_cutlass::testing::compare_host(got, want);
  std::printf("        MAE %.3e  max_abs %.3e\n", result.mae, result.max_abs);
  check(result.finite, "output is finite");
  check(result.mae <= 1.0e-2, "MAE within 1e-2 of host reference");

  if (options.identity_affine) {
    check_row_statistics(got, tokens, K, "device");
  }
  return true;
}

}  // namespace

int main() {
  std::printf("fused shared storage: %zu bytes\n",
              patch_embed::patch_embed_shared_storage_size<Element, float>());

  // Shipping stage-1 config.
  Options shipping;
  shipping.problem = PatchEmbedProblem{};
  run_case(shipping, "swin-T stage1");

  // Same shape with identity affine, so the row statistics assert directly that
  // the LayerNorm saw whole rows.
  Options identity = shipping;
  identity.identity_affine = true;
  run_case(identity, "swin-T stage1, identity affine");

  // Smaller image and a batch, to catch anything hardcoded to 224 / B=1.
  Options small;
  small.problem.batch = 2;
  small.problem.image_size = 32;
  small.problem.in_channels = 3;
  small.problem.in_channels_padded = 8;
  small.problem.embed_dim = 96;
  small.problem.patch_size = 4;
  small.identity_affine = true;
  run_case(small, "batched 32x32");

  // embed_dim 64: exercises a channel count below the N-tile with a different
  // lane arrangement.
  Options narrow;
  narrow.problem.batch = 1;
  narrow.problem.image_size = 32;
  narrow.problem.in_channels = 3;
  narrow.problem.in_channels_padded = 8;
  narrow.problem.embed_dim = 64;
  narrow.problem.patch_size = 4;
  narrow.identity_affine = true;
  run_case(narrow, "embed_dim 64");

  // A configuration that must be REFUSED, not silently mis-normalized:
  // embed_dim beyond the N-tile would give partial row statistics.
  {
    std::printf("rejects embed_dim > ThreadblockShape::kN\n");
    Options too_wide;
    too_wide.problem.image_size = 32;
    too_wide.problem.embed_dim = 256;  // > kN = 128
    patch_embed::PatchEmbedArguments<Element, float> args;
    args.problem = too_wide.problem;
    // Non-null pointers so the check under test is the shape check.
    args.input = reinterpret_cast<Element const*>(0x1);
    args.filter = reinterpret_cast<Element const*>(0x1);
    args.output = reinterpret_cast<Element*>(0x1);
    cutlass::Status const status =
        patch_embed::patch_embed_can_implement<Element, float>(args);
    check(status == cutlass::Status::kErrorNotSupported,
          "can_implement rejects embed_dim > kN with kErrorNotSupported");
  }

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
