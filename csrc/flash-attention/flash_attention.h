/*
  Shared flash-attention interface.

  Kernel files own only their launch path. The test harness owns input
  generation, reference execution, verification, and timing.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "cutlass/numeric_types.h"

using Element = cutlass::half_t;

struct Problem {
  int head_number = 12;
  int batch_size = 16;
  int head_size = 64;
  int head_size_v = 64;
  int seq_length = 1024;
  int seq_length_kv = 1024;
  float scale = 0.0f;
};

struct Tensors {
  Element const* query = nullptr; // [B, Sq, H, d]
  Element const* key = nullptr;   // [B, Sk, H, d]
  Element const* value = nullptr; // [B, Sk, H, dv]
  Element* output = nullptr;      // [B, Sq, H, dv]
};

struct Workspace {
  void* data = nullptr;
  std::size_t bytes = 0;
};

using WorkspaceBytesFn = std::size_t (*)(Problem const&);
using CanRunFn = bool (*)(Problem const&, std::string&);
using RunFn = cudaError_t (*)(
    Problem const&,
    Tensors const&,
    Workspace,
    cudaStream_t);

struct Kernel {
  char const* id = nullptr;
  char const* name = nullptr;
  WorkspaceBytesFn workspace_bytes = nullptr;
  CanRunFn can_run = nullptr;
  RunFn run = nullptr;
};

inline int64_t total_query_elements(Problem const& p) {
  return int64_t(p.batch_size) * p.seq_length * p.head_number * p.head_size;
}

inline int64_t total_key_elements(Problem const& p) {
  return int64_t(p.batch_size) * p.seq_length_kv * p.head_number * p.head_size;
}

inline int64_t total_value_elements(Problem const& p) {
  return int64_t(p.batch_size) * p.seq_length_kv * p.head_number * p.head_size_v;
}

inline int64_t total_output_elements(Problem const& p) {
  return int64_t(p.batch_size) * p.seq_length * p.head_number * p.head_size_v;
}

inline int64_t total_probability_elements(Problem const& p) {
  return int64_t(p.batch_size) * p.head_number * p.seq_length * p.seq_length_kv;
}

Kernel const& kernel_00_naive();
Kernel const& kernel_01_online_softmax();
Kernel const& kernel_02_tiled_online();
