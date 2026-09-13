#pragma once

#include <stdexcept>
#include <vector>
#include "swin/window_attention/problem.h"

namespace tiny_cutlass::swin::window_attention {

struct WindowAttentionIndex {
  std::vector<int> gather;
  // A reflected halo reads valid input but MUST NOT scatter. -1 discards it.
  std::vector<int> scatter;
};

inline int reflect_coordinate(int coordinate, int extent) {
  if (extent == 1) return 0;
  int period = 2 * (extent - 1);
  coordinate = (coordinate % period + period) % period;
  return coordinate < extent ? coordinate : period - coordinate;
}

inline WindowAttentionIndex build_window_attention_index(WindowAttentionProblem const& p) {
  if (!p.valid()) throw std::invalid_argument("unsupported window attention problem");
  WindowAttentionIndex out;
  out.gather.resize(size_t(p.window_rows()));
  out.scatter.resize(size_t(p.window_rows()), -1);
  int hp = p.padded_height(), wp = p.padded_width();
  for (int b = 0; b < p.batch; ++b) {
    for (int y = 0; y < hp; ++y) {
      for (int x = 0; x < wp; ++x) {
        int row = ((b * (hp / 4) + y / 4) * (wp / 4) + x / 4) * 16 + (y % 4) * 4 + x % 4;
        int iy = y - p.shift_h, ix = x - p.shift_w;
        if (iy >= 0 && iy < p.height && ix >= 0 && ix < p.width)
          out.scatter[row] = (b * p.height + iy) * p.width + ix;
        iy = reflect_coordinate(iy, p.height);
        ix = reflect_coordinate(ix, p.width);
        out.gather[row] = (b * p.height + iy) * p.width + ix;
      }
    }
  }
  return out;
}

}  // namespace tiny_cutlass::swin::window_attention
