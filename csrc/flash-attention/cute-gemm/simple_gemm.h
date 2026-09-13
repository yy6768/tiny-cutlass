/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Launch entry for the CuTe warm-up GEMM.

    Only owns problem/tensor structs (re-exported at global scope for the
    test harness) plus the three functions the harness calls. The actual
    kernel lives in gemm_kernel.h.
*/

#pragma once

#include <cuda_runtime.h>
#include <string>

#include "gemm_traits.h"

using GemmProblem = cute_gemm::GemmProblem;
using GemmTensors = cute_gemm::GemmTensors;
using GemmElement = cute_gemm::GemmElement;

bool cute_simple_gemm_can_run(GemmProblem const& problem, std::string& reason);

char const* cute_simple_gemm_name();

cudaError_t cute_simple_gemm(
    GemmProblem const& problem,
    GemmTensors const& tensors,
    cudaStream_t stream);
