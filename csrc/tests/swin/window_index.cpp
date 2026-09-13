/*
  Step 0 gate: the host-built index / bias tables must agree with a naive
  window_partition / window_reverse implementation element by element.

  Nothing here touches the GPU. The point is to nail down the coordinate
  conventions BEFORE any kernel consumes them, because a wrong gather table
  produces plausible-looking garbage that is very hard to localize later.

  Checks:
    1. partition via table == partition via naive six-loop, for shift 0 and w/2
    2. reverse(partition(x)) == x   (the table is a permutation)
    3. relative position index is in range and symmetric under (a,b) -> (b,a)
       mapping to the mirrored table entry
    4. shift mask is symmetric, zero on the diagonal, and all-zero when shift==0
    5. patch merging gather hits every input token exactly once
*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "swin/swin_problem.h"
#include "swin/window_index.h"

using namespace tiny_cutlass::swin;

namespace {

int g_failures = 0;

void check(bool condition, char const* what) {
  if (!condition) {
    std::printf("  FAIL  %s\n", what);
    ++g_failures;
  } else {
    std::printf("  ok    %s\n", what);
  }
}

// Naive window partition written straight from the definition, independent of
// build_window_row_index: roll the grid by -shift, then tile it.
std::vector<float> naive_partition(
    SwinStageProblem const& problem, std::vector<float> const& tokens) {
  int const w = problem.effective_window();
  int const shift = problem.effective_shift();
  int const H = problem.height;
  int const W = problem.width;
  int const C = problem.channels;

  // Step 1: roll. rolled[i][j] = tokens[(i + shift) % H][(j + shift) % W]
  std::vector<float> rolled(tokens.size());
  for (int b = 0; b < problem.batch; ++b) {
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        int const src_i = (i + shift) % H;
        int const src_j = (j + shift) % W;
        for (int c = 0; c < C; ++c) {
          rolled[size_t(((b * H + i) * W + j)) * size_t(C) + size_t(c)] =
              tokens[size_t(((b * H + src_i) * W + src_j)) * size_t(C) + size_t(c)];
        }
      }
    }
  }

  // Step 2: tile into windows, window-major.
  int const wins_col = H / w;
  int const wins_row = W / w;
  std::vector<float> out(size_t(problem.num_window_rows()) * size_t(C));
  for (int b = 0; b < problem.batch; ++b) {
    for (int wi = 0; wi < wins_col; ++wi) {
      for (int wj = 0; wj < wins_row; ++wj) {
        int const window_id = (b * wins_col + wi) * wins_row + wj;
        for (int li = 0; li < w; ++li) {
          for (int lj = 0; lj < w; ++lj) {
            int const row = window_id * (w * w) + li * w + lj;
            int const src = ((b * H + wi * w + li) * W + wj * w + lj);
            for (int c = 0; c < C; ++c) {
              out[size_t(row) * size_t(C) + size_t(c)] =
                  rolled[size_t(src) * size_t(C) + size_t(c)];
            }
          }
        }
      }
    }
  }
  return out;
}

void test_partition(SwinStageProblem problem, char const* label) {
  std::printf("%s (H=%d W=%d C=%d window=%d shift=%d)\n", label, problem.height,
              problem.width, problem.channels, problem.effective_window(),
              problem.effective_shift());

  std::vector<float> tokens(size_t(problem.num_tokens()) * size_t(problem.channels));
  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = float(i);  // unique per element: any mis-mapping shows up
  }

  std::vector<float> via_table;
  window_partition_reference(problem, tokens, via_table);
  std::vector<float> const via_naive = naive_partition(problem, tokens);

  check(via_table.size() == via_naive.size(), "partition sizes match");
  bool equal = via_table.size() == via_naive.size();
  size_t first_bad = 0;
  for (size_t i = 0; equal && i < via_table.size(); ++i) {
    if (via_table[i] != via_naive[i]) {
      equal = false;
      first_bad = i;
    }
  }
  if (!equal && via_table.size() == via_naive.size()) {
    std::printf("        first mismatch at %zu: table=%g naive=%g\n", first_bad,
                double(via_table[first_bad]), double(via_naive[first_bad]));
  }
  check(equal, "partition via table == naive roll+tile");

  // Round trip: the table is a permutation of the token rows.
  std::vector<float> back;
  window_reverse_reference(problem, via_table, back);
  bool round_trip = back.size() == tokens.size();
  for (size_t i = 0; round_trip && i < tokens.size(); ++i) {
    round_trip = back[i] == tokens[i];
  }
  check(round_trip, "reverse(partition(x)) == x");

  // Every token row is hit exactly once.
  std::vector<int> const index = build_window_row_index(problem);
  std::vector<int> hits(size_t(problem.num_tokens()), 0);
  bool in_range = true;
  for (int row : index) {
    if (row < 0 || row >= problem.num_tokens()) {
      in_range = false;
      break;
    }
    ++hits[size_t(row)];
  }
  check(in_range, "gather indices are in range");
  bool once = in_range;
  for (int h : hits) {
    once = once && (h == 1);
  }
  check(once, "each token row gathered exactly once");
}

void test_bias_and_mask(SwinStageProblem problem, char const* label) {
  std::printf("%s bias/mask (window=%d shift=%d heads=%d)\n", label,
              problem.effective_window(), problem.effective_shift(),
              problem.num_heads);

  int const w = problem.effective_window();
  int const L = w * w;
  int const span = 2 * w - 1;

  std::vector<int> const rel = build_relative_position_index(problem);
  bool rel_ok = rel.size() == size_t(L) * size_t(L);
  for (size_t i = 0; rel_ok && i < rel.size(); ++i) {
    rel_ok = rel[i] >= 0 && rel[i] < span * span;
  }
  check(rel_ok, "relative position index in [0, (2w-1)^2)");

  // rel[a][b] and rel[b][a] must be point-reflections through the table centre.
  bool mirrored = true;
  int const centre = (span * span - 1) / 2;
  for (int a = 0; a < L && mirrored; ++a) {
    for (int b = 0; b < L; ++b) {
      int const ab = rel[size_t(a) * size_t(L) + size_t(b)];
      int const ba = rel[size_t(b) * size_t(L) + size_t(a)];
      if (ab + ba != 2 * centre) {
        mirrored = false;
        break;
      }
    }
  }
  check(mirrored, "rel[a][b] + rel[b][a] == 2 * centre");
  check(rel[0] == centre, "self-relation maps to table centre");

  std::vector<float> const mask = build_window_attention_mask(problem);
  int const classes = window_mask_class_count(problem);
  check(mask.size() == size_t(classes) * size_t(L) * size_t(L), "mask size");

  bool diag_zero = true;
  bool symmetric = true;
  for (int cls = 0; cls < classes; ++cls) {
    float const* m = mask.data() + size_t(cls) * size_t(L) * size_t(L);
    for (int a = 0; a < L; ++a) {
      diag_zero = diag_zero && (m[size_t(a) * size_t(L) + size_t(a)] == 0.0f);
      for (int b = 0; b < L; ++b) {
        symmetric = symmetric && (m[size_t(a) * size_t(L) + size_t(b)] ==
                                  m[size_t(b) * size_t(L) + size_t(a)]);
      }
    }
  }
  check(diag_zero, "mask diagonal is zero (a token always sees itself)");
  check(symmetric, "mask is symmetric");

  if (problem.effective_shift() == 0) {
    bool all_zero = true;
    for (float v : mask) {
      all_zero = all_zero && (v == 0.0f);
    }
    check(all_zero, "shift == 0 leaves the mask all zero");
    check(classes == 1, "shift == 0 collapses to one window class");
  } else {
    // The interior class (0) must be unmasked; at least one edge class must
    // carry masked pairs, otherwise the shift is doing nothing.
    bool interior_clear = true;
    for (int i = 0; i < L * L; ++i) {
      interior_clear = interior_clear && (mask[size_t(i)] == 0.0f);
    }
    check(interior_clear, "interior window class is unmasked");

    bool any_masked = false;
    for (float v : mask) {
      any_masked = any_masked || (v != 0.0f);
    }
    check(any_masked, "edge window classes carry masked pairs");
    check(classes == 4, "shift != 0 yields four window classes");
  }

  // Folded table: bias + mask, [class, head, L, L].
  std::vector<float> bias_table(
      size_t(problem.relative_position_entries()) * size_t(problem.num_heads));
  for (size_t i = 0; i < bias_table.size(); ++i) {
    bias_table[i] = 0.001f * float(i);
  }
  std::vector<float> const folded =
      build_attention_bias_table(problem, bias_table);
  check(folded.size() == size_t(classes) * size_t(problem.num_heads) *
                             size_t(L) * size_t(L),
        "folded bias+mask table size");

  // Spot-check the fold against a direct recomputation.
  bool fold_ok = true;
  for (int cls = 0; cls < classes && fold_ok; ++cls) {
    for (int h = 0; h < problem.num_heads && fold_ok; ++h) {
      for (int a = 0; a < L && fold_ok; ++a) {
        for (int b = 0; b < L; ++b) {
          size_t const ab = size_t(a) * size_t(L) + size_t(b);
          float const expect =
              bias_table[size_t(rel[ab]) * size_t(problem.num_heads) + size_t(h)] +
              mask[size_t(cls) * size_t(L) * size_t(L) + ab];
          float const got =
              folded[((size_t(cls) * size_t(problem.num_heads) + size_t(h)) *
                      size_t(L) * size_t(L)) +
                     ab];
          if (expect != got) {
            fold_ok = false;
            break;
          }
        }
      }
    }
  }
  check(fold_ok, "folded table == bias[rel_index] + mask");
}

void test_patch_merging(PatchMergingProblem problem) {
  std::printf("patch merging gather (H=%d W=%d C=%d)\n", problem.height,
              problem.width, problem.channels);

  std::vector<int> const index = build_patch_merging_index(problem);
  int const out_tokens = problem.num_out_tokens();
  check(index.size() == size_t(4) * size_t(out_tokens), "index size == 4 * out tokens");

  std::vector<int> hits(
      size_t(problem.batch) * size_t(problem.height) * size_t(problem.width), 0);
  bool in_range = true;
  for (int row : index) {
    if (row < 0 || row >= int(hits.size())) {
      in_range = false;
      break;
    }
    ++hits[size_t(row)];
  }
  check(in_range, "merging indices in range");

  bool once = in_range;
  for (int h : hits) {
    once = once && (h == 1);
  }
  check(once, "each input token consumed exactly once across the 4 blocks");

  // Block order must be (0,0), (1,0), (0,1), (1,1) -- the official cat order.
  int const OW = problem.out_width();
  int const W = problem.width;
  bool order_ok = true;
  int const expect_di[4] = {0, 1, 0, 1};
  int const expect_dj[4] = {0, 0, 1, 1};
  for (int k = 0; k < 4; ++k) {
    // out token (0, 1) -> input (expect_di[k], 2 + expect_dj[k])
    int const out_row = 1;  // (i=0, j=1)
    int const got = index[size_t(k) * size_t(out_tokens) + size_t(out_row)];
    int const want = expect_di[k] * W + (2 * (out_row % OW) + expect_dj[k]);
    order_ok = order_ok && (got == want);
  }
  check(order_ok, "block order is (0,0) (1,0) (0,1) (1,1)");
}

}  // namespace

int main() {
  // Stage 1 of the window = 4 design line, both W-MSA and SW-MSA.
  SwinStageProblem stage1;
  stage1.height = 56;
  stage1.width = 56;
  stage1.channels = 96;
  stage1.num_heads = 3;
  stage1.window_size = 4;
  stage1.shift_size = 0;
  test_partition(stage1, "stage1 W-MSA");
  test_bias_and_mask(stage1, "stage1 W-MSA");

  SwinStageProblem stage1_shift = stage1;
  stage1_shift.shift_size = 2;
  test_partition(stage1_shift, "stage1 SW-MSA");
  test_bias_and_mask(stage1_shift, "stage1 SW-MSA");

  // A small odd-ish case where windows still tile, to catch hardcoded 56s.
  SwinStageProblem small;
  small.batch = 2;
  small.height = 8;
  small.width = 12;
  small.channels = 16;
  small.num_heads = 2;
  small.window_size = 4;
  small.shift_size = 2;
  test_partition(small, "small batched SW-MSA");
  test_bias_and_mask(small, "small batched SW-MSA");

  // Stage 4 clamp: window 4 does not divide 7, so it degenerates to global
  // attention with no shift.
  SwinStageProblem stage4;
  stage4.height = 7;
  stage4.width = 7;
  stage4.channels = 768;
  stage4.num_heads = 24;
  stage4.window_size = 4;
  stage4.shift_size = 2;
  std::printf("stage4 clamp\n");
  check(stage4.effective_window() == 4, "window 4 < 7 is NOT clamped");
  check(stage4.effective_shift() == 2, "shift survives when window < extent");
  check(!stage4.window_tiles_exactly(), "window 4 does not tile a 7x7 grid");
  SwinStageProblem stage4_global = stage4;
  stage4_global.window_size = 7;
  check(stage4_global.effective_window() == 7, "window == extent stays 7");
  check(stage4_global.effective_shift() == 0, "window == extent forces shift 0");
  check(stage4_global.window_tiles_exactly(), "window 7 tiles a 7x7 grid");

  PatchMergingProblem merge;
  merge.batch = 2;
  merge.height = 8;
  merge.width = 12;
  merge.channels = 16;
  test_patch_merging(merge);

  PatchMergingProblem merge_stage1;
  merge_stage1.height = 56;
  merge_stage1.width = 56;
  merge_stage1.channels = 96;
  test_patch_merging(merge_stage1);

  // PatchEmbed descriptor arithmetic.
  PatchEmbedProblem embed;
  std::printf("patch embed descriptor\n");
  check(embed.tokens_per_side() == 56, "224 / 4 == 56 tokens per side");
  check(embed.num_tokens() == 56 * 56, "token count");
  check(embed.filter_extent() == 4 * 4 * 8, "padded filter extent == 128");

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
