#pragma once

namespace tiny_cutlass { namespace natten {

// Contiguous Q/K [B,L,H,D], V/O [B,L,H,Dv].
struct NeighborhoodProblem {
  int batch_size = 0;
  int length = 0;
  int heads = 0;
  int head_dim = 0;
  int head_dim_value = 0;
  int kernel_size = 0;
  int stride = 1;
  int dilation = 1;
  float scale = 1.0f;
};

}} // namespace tiny_cutlass::natten
