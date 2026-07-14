/*
  Warp-level Swin window coordinate helpers.
*/

#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"

namespace tiny_cutlass {
namespace swin {
namespace warp {

struct WindowMapping {
  int64_t image_token = 0;
  int64_t window_token = 0;
};

CUTLASS_DEVICE int shifted_coordinate(int coord, int extent, int shift) {
  return shift != 0 ? ((coord - shift + extent) % extent) : coord;
}

CUTLASS_DEVICE WindowMapping window_mapping(
    int64_t pixel,
    int height,
    int width,
    int window_size,
    int shift_size) {
  int x = int(pixel % width);
  int y = int((pixel / width) % height);
  int b = int(pixel / (int64_t(height) * width));

  int shifted_y = shifted_coordinate(y, height, shift_size);
  int shifted_x = shifted_coordinate(x, width, shift_size);

  int windows_per_row = width / window_size;
  int window_y = shifted_y / window_size;
  int window_x = shifted_x / window_size;
  int window_idx = window_y * windows_per_row + window_x;
  int idx_in_window = (shifted_y % window_size) * window_size
                    + (shifted_x % window_size);

  int64_t tokens_per_batch = int64_t(height) * width;
  WindowMapping mapping;
  mapping.image_token = int64_t(b) * tokens_per_batch + y * width + x;
  mapping.window_token = int64_t(b) * tokens_per_batch
                       + int64_t(window_idx) * window_size * window_size
                       + idx_in_window;
  return mapping;
}

} // namespace warp
} // namespace swin
} // namespace tiny_cutlass
