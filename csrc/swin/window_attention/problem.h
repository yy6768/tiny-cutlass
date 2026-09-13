#pragma once

#include <cstdint>
#include <limits>

namespace tiny_cutlass::swin::window_attention {

#if defined(__CUDACC__)
#define SWIN_WINDOW_HOST_DEVICE __host__ __device__
#else
#define SWIN_WINDOW_HOST_DEVICE
#endif

enum class QkMode { kShared, kSeparate };

// DLSS4SwinWindowAttention inference. The supplied reference does not apply
// SwinStageProblem's small-map window/shift clamp.
struct WindowAttentionProblem {
  int batch = 1, height = 4, width = 4, channels = 32;
  int groups = 1, qk_channels = 32, value_channels = 32;
  int window_size = 4, shift_h = 0, shift_w = 0;
  QkMode qk_mode = QkMode::kShared;
  // nn.RMSNorm(C, eps=None) with half input uses finfo(half).eps.
  float rms_epsilon = 0.0009765625f;
  float attn_drop = 0, proj_drop = 0;
  bool quant_fp8 = false;

  SWIN_WINDOW_HOST_DEVICE int padded_height() const {
    return ((height + shift_h + 3) / 4) * 4;
  }
  SWIN_WINDOW_HOST_DEVICE int padded_width() const {
    return ((width + shift_w + 3) / 4) * 4;
  }
  SWIN_WINDOW_HOST_DEVICE int projected_channels() const {
    return (qk_mode == QkMode::kShared ? 1 : 2) * qk_channels + value_channels;
  }
  SWIN_WINDOW_HOST_DEVICE int hidden_channels() const { return groups * value_channels; }
  int64_t tokens() const { return int64_t(batch) * height * width; }
  int64_t window_rows() const { return int64_t(batch) * padded_height() * padded_width(); }
  int windows() const { return int(window_rows() / 16); }

  bool valid() const {
    // Check bounds before multiplying dimensions or rounding padded extents.
    if (batch <= 0 || batch > 65535 || height <= 0 || width <= 0 ||
        height > 1048576 || width > 1048576 || channels <= 0 || channels > 1024 ||
        groups <= 0 || groups > 64 || qk_channels <= 0 || qk_channels > 64 ||
        value_channels <= 0 || value_channels > 64 || hidden_channels() > 512 ||
        channels % 8 || qk_channels % 8 || value_channels % 8 || window_size != 4 ||
        shift_h < 0 || shift_h >= 4 || shift_w < 0 || shift_w >= 4 ||
        (qk_mode != QkMode::kShared && qk_mode != QkMode::kSeparate) ||
        !(rms_epsilon > 0 && rms_epsilon <= std::numeric_limits<float>::max()) ||
        attn_drop != 0 || proj_drop != 0 || quant_fp8) return false;
    return window_rows() <= std::numeric_limits<int>::max();
  }
};

#undef SWIN_WINDOW_HOST_DEVICE

}  // namespace tiny_cutlass::swin::window_attention
