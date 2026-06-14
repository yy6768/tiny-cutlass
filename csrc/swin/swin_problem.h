/*
  Shared Swin problem and tensor descriptors.
*/

#pragma once

#include <cstdint>

namespace tiny_cutlass {
namespace swin {

struct SwinProblem {
  int batch_size = 2;
  int image_size = 14;
  int window_size = 7;
  int shift_size = 3;
  int head_number = 3;
  int head_size = 32;
  float scale = 0.0f;
};

struct PatchEmbedProblem {
  int batch_size = 1;
  int image_size = 224;
  int in_channels = 3;
  int input_channels_padded = 8;
  int embed_dim = 96;
  int patch_size = 4;
  float layernorm_eps = 1.0e-5f;
};

template <typename Element_>
struct SwinTensors {
  using Element = Element_;

  Element const* input = nullptr;          // activation [B, H, W, C], NHWC
  Element const* qkv_weight = nullptr;     // [C, 3C]
  Element const* qkv_bias = nullptr;       // [3C]
  Element const* output_weight = nullptr;  // [C, C]
  Element const* output_bias = nullptr;    // [C]
  Element const* attention_bias = nullptr; // [BW, heads, L, L_pad]

  Element* windows = nullptr;              // [BW, L, C]
  Element* qkv = nullptr;                  // [BW * L, 3C]
  Element* query = nullptr;                // [BW, L, heads, D]
  Element* key = nullptr;                  // [BW, L, heads, D]
  Element* value = nullptr;                // [BW, L, heads, D]
  Element* attention_output = nullptr;     // [BW, L, C]
  Element* projected = nullptr;            // [BW, L, C]
  Element* output = nullptr;               // activation [B, H, W, C], NHWC
  Element* patch_merged = nullptr;         // [B, H/2, W/2, 4C], optional
};

template <typename Element_>
struct PatchEmbedTensors {
  using Element = Element_;

  Element const* input = nullptr;      // texture activation [B, H, W, C], NHWC
  Element const* kernel = nullptr;     // [K, C, R, S], PyTorch OIHW
  Element const* bias = nullptr;       // [K]
  Element const* gamma = nullptr;      // [K]
  Element const* beta = nullptr;       // [K]

  // Workspace base. Layout:
  //   [0, output_elements)                  conv output [B, H/4, W/4, K]
  //   next input_padded_elements            padded NHWC activation
  //   next kernel_padded_elements           padded KRSC filter
  Element* conv_output = nullptr;
  Element* output = nullptr;           // activation [B, H/4, W/4, K], NHWC
};

inline int swin_channels(SwinProblem const& p) {
  return p.head_number * p.head_size;
}

inline int swin_window_len(SwinProblem const& p) {
  return p.window_size * p.window_size;
}

inline int swin_window_len_padded(SwinProblem const& p) {
  int l = swin_window_len(p);
  return ((l + 7) / 8) * 8;
}

inline int swin_num_windows(SwinProblem const& p) {
  int windows_per_side = p.image_size / p.window_size;
  return windows_per_side * windows_per_side;
}

inline int swin_batched_windows(SwinProblem const& p) {
  return p.batch_size * swin_num_windows(p);
}

inline int swin_rows(SwinProblem const& p) {
  return swin_batched_windows(p) * swin_window_len(p);
}

inline int64_t swin_input_elements(SwinProblem const& p) {
  return int64_t(p.batch_size) * p.image_size * p.image_size * swin_channels(p);
}

inline int64_t swin_window_elements(SwinProblem const& p) {
  return int64_t(swin_batched_windows(p)) * swin_window_len(p) * swin_channels(p);
}

inline int64_t swin_qkv_elements(SwinProblem const& p) {
  return int64_t(swin_rows(p)) * 3 * swin_channels(p);
}

inline int64_t swin_qkv_weight_elements(SwinProblem const& p) {
  int c = swin_channels(p);
  return int64_t(c) * 3 * c;
}

inline int64_t swin_output_weight_elements(SwinProblem const& p) {
  int c = swin_channels(p);
  return int64_t(c) * c;
}

inline int64_t swin_attention_bias_elements(SwinProblem const& p) {
  return int64_t(swin_batched_windows(p)) * p.head_number
       * swin_window_len(p) * swin_window_len_padded(p);
}

inline int64_t swin_patch_merged_elements(SwinProblem const& p) {
  int c = swin_channels(p);
  return int64_t(p.batch_size) * (p.image_size / 2) * (p.image_size / 2) * (4 * c);
}

inline int patch_embed_output_size(PatchEmbedProblem const& p) {
  return p.image_size / p.patch_size;
}

inline int64_t patch_embed_input_elements(PatchEmbedProblem const& p) {
  return int64_t(p.batch_size) * p.image_size * p.image_size * p.in_channels;
}

inline int64_t patch_embed_kernel_elements(PatchEmbedProblem const& p) {
  return int64_t(p.embed_dim) * p.patch_size * p.patch_size * p.in_channels;
}

inline int64_t patch_embed_input_padded_elements(PatchEmbedProblem const& p) {
  return int64_t(p.batch_size) * p.image_size * p.image_size * p.input_channels_padded;
}

inline int64_t patch_embed_kernel_padded_elements(PatchEmbedProblem const& p) {
  return int64_t(p.embed_dim) * p.patch_size * p.patch_size * p.input_channels_padded;
}

inline int64_t patch_embed_output_elements(PatchEmbedProblem const& p) {
  int out = patch_embed_output_size(p);
  return int64_t(p.batch_size) * out * out * p.embed_dim;
}

} // namespace swin
} // namespace tiny_cutlass
