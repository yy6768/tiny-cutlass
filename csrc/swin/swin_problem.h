#pragma once

/*
  Shared problem descriptors for the Swin pipeline.

  These are plain runtime structs of scalars -- no device pointers, no CUTLASS
  types, no ATen/Torch ownership. Every operator in csrc/swin/ takes one of
  these plus raw device pointers and a cudaStream_t, matching the convention set
  by csrc/conv-fused/conv1x1_dual/ops/conv1x1_dual.h.

  Tile shapes, element types, arch tags and the compile-time window size live in
  the kernel-level DefaultXxx<...> factories; the device layer cross-checks the
  compile-time window against SwinStageProblem::effective_window() in
  can_implement().
*/

#include <algorithm>

namespace tiny_cutlass::swin {

// ---------------------------------------------------------------------------
// PatchEmbed: conv(patch x patch, stride = patch) + bias + LayerNorm(channels).
//
// stride == kernel == patch means the patches do not overlap, so the implicit
// GEMM has an im2col expansion factor of exactly 1.0 and needs no halo. The
// activation channel count is padded (3 -> 8) because a TensorOp fp16 mainloop
// requires an 8-wide C-dimension load and this family takes no SIMT fallback.
// ---------------------------------------------------------------------------
struct PatchEmbedProblem {
  int batch = 1;
  int image_size = 224;
  int in_channels = 3;
  int in_channels_padded = 8;
  int embed_dim = 96;
  int patch_size = 4;

  int tokens_per_side() const { return image_size / patch_size; }
  int num_tokens() const { return batch * tokens_per_side() * tokens_per_side(); }
  // Reduction extent of the equivalent GEMM: (patch * patch * padded channels).
  int filter_extent() const {
    return patch_size * patch_size * in_channels_padded;
  }
};

// ---------------------------------------------------------------------------
// One Swin stage's token grid. Shared by window_attention, swin_mlp and
// patch_merging so a stage can be wired end to end from a single descriptor.
//
// window_size / shift_size are the REQUESTED values. Swin's official rule is
// that a window at least as large as the feature map degenerates to global
// attention with no shift, so always read them back through
// effective_window() / effective_shift() rather than the raw fields.
// ---------------------------------------------------------------------------
struct SwinStageProblem {
  int batch = 1;
  int height = 56;       // token rows H
  int width = 56;        // token cols W
  int channels = 96;     // C
  int num_heads = 3;
  int window_size = 4;
  int shift_size = 0;    // 0 for W-MSA, window_size / 2 for SW-MSA
  int mlp_ratio = 4;

  // Official Swin clamp: if the window covers the whole feature map there is
  // nothing left to shift.
  int effective_window() const {
    return std::min(window_size, std::min(height, width));
  }

  int effective_shift() const {
    return (window_size >= std::min(height, width)) ? 0 : shift_size;
  }

  int head_dim() const { return channels / num_heads; }
  int mlp_hidden() const { return channels * mlp_ratio; }

  int tokens_per_window() const {
    int const w = effective_window();
    return w * w;
  }

  int windows_per_row() const { return width / effective_window(); }
  int windows_per_col() const { return height / effective_window(); }

  int windows_per_image() const {
    return windows_per_col() * windows_per_row();
  }

  int num_windows() const { return batch * windows_per_image(); }

  int num_tokens() const { return batch * height * width; }

  // Rows of the window-major layout the attention GEMMs operate on. Equal to
  // num_tokens() whenever the window tiles the grid exactly.
  int num_window_rows() const { return num_windows() * tokens_per_window(); }

  // Relative-position-bias table has (2w - 1)^2 entries per head.
  int relative_position_entries() const {
    int const w = effective_window();
    return (2 * w - 1) * (2 * w - 1);
  }

  bool window_tiles_exactly() const {
    int const w = effective_window();
    return w > 0 && (height % w) == 0 && (width % w) == 0;
  }
};

// ---------------------------------------------------------------------------
// PatchMerging: 2x2 spatial gather -> LayerNorm(4C) -> linear 4C -> 2C.
//
// Note the norm sits BEFORE the projection here, the opposite of PatchEmbed.
// ---------------------------------------------------------------------------
struct PatchMergingProblem {
  int batch = 1;
  int height = 56;
  int width = 56;
  int channels = 96;

  int out_height() const { return height / 2; }
  int out_width() const { return width / 2; }
  int concat_channels() const { return 4 * channels; }
  int out_channels() const { return 2 * channels; }
  int num_out_tokens() const { return batch * out_height() * out_width(); }
};

}  // namespace tiny_cutlass::swin
