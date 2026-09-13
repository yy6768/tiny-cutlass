#pragma once

/*
  Host reference for the Swin pipeline. Everything accumulates in fp32 and is
  written straight from the definition -- no tiling, no fast paths -- so it can
  serve as the parity oracle for the fused kernels.

  Numerical conventions that MUST match the kernels (these are the usual sources
  of a false "bug"):

    * LayerNorm uses the BIASED variance (divide by N, not N-1), matching
      PyTorch's LayerNorm.
    * GELU is the EXACT erf form, not the tanh approximation. The tanh version
      differs by ~1e-3, which is large enough to look like a real defect.
    * bias is folded in BEFORE a LayerNorm that follows it, and after the GEMM
      it belongs to.
*/

#include <cmath>
#include <cstddef>
#include <vector>

#include "swin/swin_problem.h"

namespace tiny_cutlass::swin::testing {

// ---------------------------------------------------------------------------
// LayerNorm over the last axis of a [rows, cols] row-major tensor.
//
// gamma / beta may be empty, in which case they are treated as 1 / 0.
// ---------------------------------------------------------------------------
inline void layernorm_reference(
    std::vector<float>& data,
    int rows,
    int cols,
    std::vector<float> const& gamma,
    std::vector<float> const& beta,
    float epsilon = 1e-5f) {
  for (int r = 0; r < rows; ++r) {
    float* row = data.data() + size_t(r) * size_t(cols);

    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      sum += double(row[c]);
    }
    double const mean = sum / double(cols);

    double var_sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      double const d = double(row[c]) - mean;
      var_sum += d * d;
    }
    // Biased variance: divide by N.
    double const variance = var_sum / double(cols);
    double const inv_std = 1.0 / std::sqrt(variance + double(epsilon));

    for (int c = 0; c < cols; ++c) {
      double normalized = (double(row[c]) - mean) * inv_std;
      if (!gamma.empty()) {
        normalized *= double(gamma[size_t(c)]);
      }
      if (!beta.empty()) {
        normalized += double(beta[size_t(c)]);
      }
      row[c] = float(normalized);
    }
  }
}

// Exact GELU. Must match cutlass::epilogue::thread::GELU<float>.
inline float gelu_reference(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

// ---------------------------------------------------------------------------
// PatchEmbed: conv(patch x patch, stride patch) + bias, then LayerNorm along
// the embed_dim axis.
//
// `input` is NHWC with C = problem.in_channels_padded (the padded activation
// the kernel actually reads). `filter` is KRSC with the same padded C. The
// padding channels are expected to be zero in both, so they contribute nothing.
//
// Output is [num_tokens, embed_dim] row-major, token index
// (b * tokens_per_side + p) * tokens_per_side + q.
// ---------------------------------------------------------------------------
template <typename Element>
std::vector<float> patch_embed_reference(
    PatchEmbedProblem const& problem,
    std::vector<Element> const& input,
    std::vector<Element> const& filter,
    std::vector<float> const& bias,
    std::vector<float> const& gamma,
    std::vector<float> const& beta,
    float epsilon = 1e-5f) {
  int const patch = problem.patch_size;
  int const C = problem.in_channels_padded;
  int const K = problem.embed_dim;
  int const H = problem.image_size;
  int const W = problem.image_size;
  int const P = problem.tokens_per_side();
  int const Q = problem.tokens_per_side();

  std::vector<float> out(size_t(problem.num_tokens()) * size_t(K), 0.0f);

  for (int n = 0; n < problem.batch; ++n) {
    for (int p = 0; p < P; ++p) {
      for (int q = 0; q < Q; ++q) {
        int const token = (n * P + p) * Q + q;
        for (int k = 0; k < K; ++k) {
          double acc = 0.0;
          // stride == kernel, so the receptive field is exactly one patch and
          // there is no halo or overlap.
          for (int r = 0; r < patch; ++r) {
            for (int s = 0; s < patch; ++s) {
              int const h = p * patch + r;
              int const w = q * patch + s;
              if (h >= H || w >= W) {
                continue;
              }
              for (int c = 0; c < C; ++c) {
                size_t const ai =
                    ((size_t(n) * size_t(H) + size_t(h)) * size_t(W) + size_t(w)) *
                        size_t(C) +
                    size_t(c);
                size_t const fi =
                    ((size_t(k) * size_t(patch) + size_t(r)) * size_t(patch) +
                     size_t(s)) *
                        size_t(C) +
                    size_t(c);
                acc += double(float(input[ai])) * double(float(filter[fi]));
              }
            }
          }
          if (!bias.empty()) {
            acc += double(bias[size_t(k)]);
          }
          out[size_t(token) * size_t(K) + size_t(k)] = float(acc);
        }
      }
    }
  }

  layernorm_reference(out, problem.num_tokens(), K, gamma, beta, epsilon);
  return out;
}

// ---------------------------------------------------------------------------
// Row-major GEMM with a per-column bias: D[m, n] = sum_k A[m, k] * W[n, k] + bias[n]
//
// W is [N, K] row-major, matching nn.Linear.weight (and read as column-major
// operand B by the kernels).
// ---------------------------------------------------------------------------
template <typename Element>
std::vector<float> linear_reference(
    std::vector<Element> const& a,
    std::vector<Element> const& weight,
    std::vector<float> const& bias,
    int m,
    int n,
    int k) {
  std::vector<float> out(size_t(m) * size_t(n), 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int p = 0; p < k; ++p) {
        acc += double(float(a[size_t(i) * size_t(k) + size_t(p)])) *
               double(float(weight[size_t(j) * size_t(k) + size_t(p)]));
      }
      if (!bias.empty()) {
        acc += double(bias[size_t(j)]);
      }
      out[size_t(i) * size_t(n) + size_t(j)] = float(acc);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Swin MLP: y = x + fc2(GELU(fc1(LayerNorm(x)) + b1)) + b2
//
// Ordering that matters: the residual is the UNNORMALIZED input, b2 is added
// once, and the GELU is the exact erf form.
// ---------------------------------------------------------------------------
template <typename Element>
std::vector<float> swin_mlp_reference(
    int rows,
    int channels,
    int hidden,
    std::vector<Element> const& input,
    std::vector<float> const& gamma,
    std::vector<float> const& beta,
    std::vector<Element> const& fc1_weight,
    std::vector<float> const& fc1_bias,
    std::vector<Element> const& fc2_weight,
    std::vector<float> const& fc2_bias,
    float epsilon = 1e-5f) {
  // LN2 on a float copy of the input.
  std::vector<float> normalized(size_t(rows) * size_t(channels));
  for (size_t i = 0; i < normalized.size(); ++i) {
    normalized[i] = float(input[i]);
  }
  layernorm_reference(normalized, rows, channels, gamma, beta, epsilon);

  // Round to the storage type, matching what the LayerNorm kernel writes out
  // before fc1 reads it. Skipping this shows up as a systematic bias.
  std::vector<Element> normalized_rounded(normalized.size());
  for (size_t i = 0; i < normalized.size(); ++i) {
    normalized_rounded[i] = Element(normalized[i]);
  }

  // fc1 + b1 + GELU.
  std::vector<float> h = linear_reference<Element>(
      normalized_rounded, fc1_weight, fc1_bias, rows, hidden, channels);
  std::vector<Element> h_rounded(h.size());
  for (size_t i = 0; i < h.size(); ++i) {
    h_rounded[i] = Element(gelu_reference(h[i]));
  }

  // fc2 + b2, then the residual on the ORIGINAL input.
  std::vector<float> out = linear_reference<Element>(
      h_rounded, fc2_weight, fc2_bias, rows, channels, hidden);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] += float(input[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// PatchMerging: concat a 2x2 neighbourhood along channels, LayerNorm over 4C,
// then project 4C -> 2C.
//
// The concat order is the official one,
//     cat([x[0::2, 0::2], x[1::2, 0::2], x[0::2, 1::2], x[1::2, 1::2]], -1)
// written out here directly rather than via the gather table, so that the table
// itself is being tested and not merely reproduced.
//
// Note the norm precedes the projection -- the reverse of PatchEmbed.
// ---------------------------------------------------------------------------
template <typename Element>
std::vector<float> patch_merging_reference(
    PatchMergingProblem const& problem,
    std::vector<Element> const& input,
    std::vector<float> const& gamma,
    std::vector<float> const& beta,
    std::vector<Element> const& weight,
    std::vector<float> const& bias,
    float epsilon = 1e-5f) {
  int const H = problem.height;
  int const W = problem.width;
  int const C = problem.channels;
  int const OH = problem.out_height();
  int const OW = problem.out_width();
  int const concat = problem.concat_channels();
  int const out_channels = problem.out_channels();
  int const rows = problem.num_out_tokens();

  int const di[4] = {0, 1, 0, 1};
  int const dj[4] = {0, 0, 1, 1};

  // Concatenate, in fp32.
  std::vector<float> merged(size_t(rows) * size_t(concat), 0.0f);
  for (int b = 0; b < problem.batch; ++b) {
    for (int i = 0; i < OH; ++i) {
      for (int j = 0; j < OW; ++j) {
        int const out_row = (b * OH + i) * OW + j;
        for (int k = 0; k < 4; ++k) {
          int const h = 2 * i + di[k];
          int const w = 2 * j + dj[k];
          int const src = (b * H + h) * W + w;
          for (int c = 0; c < C; ++c) {
            merged[size_t(out_row) * size_t(concat) + size_t(k * C + c)] =
                float(input[size_t(src) * size_t(C) + size_t(c)]);
          }
        }
      }
    }
  }

  layernorm_reference(merged, rows, concat, gamma, beta, epsilon);

  // Round to storage precision, matching what the LayerNorm kernel hands the
  // projection GEMM.
  std::vector<Element> merged_rounded(merged.size());
  for (size_t i = 0; i < merged.size(); ++i) {
    merged_rounded[i] = Element(merged[i]);
  }

  return linear_reference<Element>(
      merged_rounded, weight, bias, rows, out_channels, concat);
}

}  // namespace tiny_cutlass::swin::testing
