#pragma once

#include <cmath>
#include <cinttypes>
#include <vector>

#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

struct Params {
    // 输入指针
    scalar_t const* query_ptr;      // [B, Sq, num_heads, head_dim]
    scalar_t const* key_ptr;        // [B, Sk, num_heads, head_dim]
    scalar_t const* value_ptr;      // [B, Sk, num_heads, head_dim]
    scalar_t* output_ptr;           // [B, Sq, num_heads, head_dim]
    float* logsumexp_ptr;           // [B, num_heads, Sq] (可选, 供反向传播)

    // Strides (按行的 stride, 即相邻两行的偏移量)
    int32_t q_strideM;     // = num_heads * head_dim
    int32_t k_strideM;
    int32_t v_strideM;
    int32_t o_strideM;

    // 问题尺寸
    int32_t num_queries;   // Sq
    int32_t num_keys;      // Sk
    int32_t head_dim;      // d (Q/K 的 head dim)
    int32_t head_dim_value;// d_v (V 的 head dim, 通常等于 head_dim)
    int32_t num_heads;
    int32_t num_batches;

    // Attention 参数
    float scale;           // 1.0 / sqrt(head_dim)

    // Causal mask
    int32_t custom_mask_type;       // 0=no mask, 1=causal, 2=causal from top-left
    int32_t causal_diagonal_offset; // 通常为 num_keys - num_queries
};