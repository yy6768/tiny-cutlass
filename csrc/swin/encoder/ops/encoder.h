#pragma once

#include <cstdint>

namespace tiny_cutlass::swin::encoder {

// Reconstructed neural slice, starting AFTER the texture feature preparation.
// One tile: 64 interior tokens (8x8 row-major), then top/bottom/left/right
// halo (8 tokens each). All tensors here use ordinary contiguous rows.
struct Problem {
  int tiles = 1;
  int channels = 32;
  int query_tokens = 64;
  int key_tokens = 96;
  int heads = 2;
  int hidden_channels = 128;
};

template <typename Element>
struct Weights {
  // Linear weights are [output_channel, input_channel], not the PTX blob packing.
  Element const* input_weight[2] = {};
  Element const* input_bias[2] = {};
  Element const* norm_weight[2] = {};
  Element const* qk_weight[2] = {};
  Element const* value_weight[2] = {};
  Element const* position_bias[2] = {};  // each [64,96], seeded into QK MMA
  Element const* projection_weight[2] = {};
  Element const* projection_bias = nullptr;  // [32], added once before both heads
  Element const* expand_weight[4] = {};      // each [32,32]
  Element const* expand_bias[4] = {};        // each [32]
  Element const* contract_weight[4] = {};    // each [32,32]
  Element const* contract_bias = nullptr;    // [32], added once before four chunks
  Element const* merge_weight = nullptr;    // [64,128]
  Element const* merge_bias = nullptr;      // [64]
  Element const* head_weight = nullptr;     // [16,32]
  Element const* head_bias = nullptr;       // [16]
};

template <typename Element>
struct Arguments {
  Problem problem;
  Weights<Element> weights;
  Element const* input = nullptr;  // [tiles,96,32], already prepared/rounded
  Element* output = nullptr;      // [tiles,64,32]
  Element* merged = nullptr;      // [tiles,16,64]
  Element* head = nullptr;        // [tiles,64,16], before image postprocessing
};

}  // namespace tiny_cutlass::swin::encoder
