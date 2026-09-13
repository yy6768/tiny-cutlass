#pragma once

#include <stdexcept>
#include <vector>

namespace tiny_cutlass::swin::window_attention {

// Prepare once per weight update. All windows reuse 2D [groups * 16, 16].
template <typename Element>
std::vector<Element> expand_relative_position_bias(Element const* table, int groups) {
  if (!table || groups <= 0 || groups > 64)
    throw std::invalid_argument("relative bias requires [49, groups], groups in [1,64]");
  std::vector<Element> expanded(size_t(groups) * 256);
  for (int group = 0; group < groups; ++group)
    for (int query = 0; query < 16; ++query)
      for (int key = 0; key < 16; ++key) {
        int relative = (query / 4 - key / 4 + 3) * 7 + query % 4 - key % 4 + 3;
        expanded[(group * 16 + query) * 16 + key] = table[relative * groups + group];
      }
  return expanded;
}

}  // namespace tiny_cutlass::swin::window_attention
