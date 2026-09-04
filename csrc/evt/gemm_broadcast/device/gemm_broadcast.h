/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief Device-level operator for the EVT residual-block GEMM.

        D = ((A.B + bias_row) + C1) + C2

    This wraps `GemmUniversalAdapter` and owns the one genuinely error-prone part of using an
    Epilogue Visitor Tree: building the nested `Arguments` initializer. Callers pass flat raw
    device pointers and a `GemmCoord`; this layer assembles the tree-shaped brace initializer.

    ------------------------------------------------------------------------------------------------
    WHY THE ARGUMENTS NESTING IS BACKWARDS
    ------------------------------------------------------------------------------------------------
    A tree is declared node-op-first:

        Sm80EVT<NodeOp, Child0, Child1>

    but its `Params` tuple is stored child-first, because `TreeVisitor2x` inherits
    `VisitorImpl2x<ChildOps..., NodeOp>` -- the parameter pack is rotated so the node op lands in
    the LAST slot. So each brace level reads:

        { child0_args, child1_args, nodeop_args }

    A compute node takes no arguments, hence the trailing `{}` at every level. Getting this order
    wrong still compiles (they are all brace-initialized aggregates) and silently produces wrong
    results, which is exactly why it is centralized here instead of being retyped per call site.
*/

#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/gemm_enumerated_types.h"

#include "cute/int_tuple.hpp"

#include "evt/gemm_broadcast/kernel/default_gemm_broadcast.h"

namespace tiny_cutlass {
namespace evt {
namespace device {

// ---------------------------------------------------------------------------
// GemmBroadcast
//
//   Policy_ : a kernel::DefaultGemmBroadcast<...> instantiation. The swizzle
//             baked into that policy decides data-parallel vs StreamK; this
//             layer is identical for both.
// ---------------------------------------------------------------------------
template <typename Policy_>
class GemmBroadcast {
 public:
  using Policy = Policy_;
  using Element = typename Policy::Element;
  using Gemm = typename Policy::Gemm;
  using EVTStore = typename Policy::EVTStore;

  struct Arguments {
    cutlass::gemm::GemmCoord problem_size{};

    Element const* ptr_a = nullptr;     // [M, K] row-major
    Element const* ptr_b = nullptr;     // [K, N] row-major
    Element const* ptr_bias = nullptr;  // [N]    broadcast along M
    Element const* ptr_c1 = nullptr;    // [M, N] row-major
    Element const* ptr_c2 = nullptr;    // [M, N] row-major
    Element* ptr_d = nullptr;           // [M, N] row-major

    // Split-K slices for the data-parallel kernel / batch count. StreamK
    // ignores this and derives its own work decomposition.
    int split_k_slices = 1;

    // 0 lets StreamK use every SM on the device.
    int avail_sms = -1;
  };

  static cutlass::Status can_implement(Arguments const& args) {
    if (args.ptr_a == nullptr || args.ptr_b == nullptr || args.ptr_d == nullptr ||
        args.ptr_bias == nullptr || args.ptr_c1 == nullptr || args.ptr_c2 == nullptr) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    cutlass::gemm::GemmCoord const& p = args.problem_size;
    if (p.m() <= 0 || p.n() <= 0 || p.k() <= 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Every operand is loaded at full 128-bit width, so the contiguous extent
    // of each tensor must be a multiple of that. Fail loudly instead of quietly
    // dropping to a narrower alignment.
    if (p.k() % Policy::kAlignmentA != 0) {
      return cutlass::Status::kErrorMisalignedOperand;
    }
    if (p.n() % Policy::kAlignmentB != 0 || p.n() % Policy::kAlignmentC != 0) {
      return cutlass::Status::kErrorMisalignedOperand;
    }

    return Gemm::can_implement(to_gemm_arguments(args));
  }

  static size_t get_workspace_size(Arguments const& args) {
    return Gemm::get_workspace_size(to_gemm_arguments(args));
  }

  /// Host-side setup: validates and precomputes Params into the operator.
  ///
  /// Kept separate from run() on purpose. `initialize()` does device queries,
  /// occupancy calculation and (for StreamK) workspace preparation, none of
  /// which belong inside a benchmark loop -- fold it into run() and the loop
  /// measures host overhead instead of the kernel.
  cutlass::Status initialize(
      Arguments const& args,
      void* workspace,
      cudaStream_t stream = nullptr) {
    typename Gemm::Arguments gemm_args = to_gemm_arguments(args);

    cutlass::Status status = gemm_.can_implement(gemm_args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    return gemm_.initialize(gemm_args, workspace, stream);
  }

  /// Launches the kernel using the already-initialized Params.
  cutlass::Status run(cudaStream_t stream = nullptr) { return gemm_.run(stream); }

  cutlass::Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

  /// One-shot convenience path: initialize then launch.
  cutlass::Status operator()(
      Arguments const& args,
      void* workspace,
      cudaStream_t stream = nullptr) {
    cutlass::Status status = initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    return run(stream);
  }

 private:
  Gemm gemm_;

  /// Detects whether the underlying kernel's Arguments carries `avail_sms`.
  ///
  /// This is the ONE place the two swizzles are not interchangeable: the
  /// data-parallel kernel is `GemmUniversal` (Arguments has no `avail_sms`)
  /// while the StreamK kernel is `GemmUniversalStreamk` (Arguments ends with
  /// `avail_sms`). Both ctors are otherwise identical, so detect and dispatch
  /// rather than duplicating the whole argument list per swizzle.
  template <typename T, typename = void>
  struct has_avail_sms : std::false_type {};

  template <typename T>
  struct has_avail_sms<T, decltype(void(std::declval<T&>().avail_sms))> : std::true_type {};

  /// Builds the flat GemmUniversal arguments plus the nested EVT callback tree.
  static typename Gemm::Arguments to_gemm_arguments(Arguments const& args) {
    cutlass::gemm::GemmCoord const& p = args.problem_size;

    int64_t const ldc = static_cast<int64_t>(p.n());
    int64_t const batch_stride_mn = static_cast<int64_t>(p.m()) * static_cast<int64_t>(p.n());

    // Tree-shaped initializer. Each level is {children..., node_op}; see the
    // file header for why the node op comes last.
    typename EVTStore::Arguments callback_args{
        {                                                             // EVTAddC2
            {                                                         //   EVTAddC1
                {                                                     //     EVTAddBias
                    {},                                               //       Accum
                    {const_cast<Element*>(args.ptr_bias),             //       Bias
                     Element(0),
                     {cute::_0{}, cute::_1{}, static_cast<int32_t>(p.n())}},
                    {}                                                //       ComputeAddBias
                },
                {const_cast<Element*>(args.ptr_c1),                   //     ResidualC1
                 Element(0),
                 {ldc, cute::_1{}, batch_stride_mn}},
                {}                                                    //     ComputeAddC1
            },
            {const_cast<Element*>(args.ptr_c2),                       //   ResidualC2
             Element(0),
             {ldc, cute::_1{}, batch_stride_mn}},
            {}                                                        //   ComputeAddC2
        },
        {args.ptr_d, {ldc, cute::_1{}, batch_stride_mn}},             // Store
    };

    // ptr_C / ptr_D / ldc / ldd are all unused on the EVT path: C1 and C2 enter
    // through AuxLoad nodes and D leaves through the tree's AuxStore, all
    // addressed by their own StrideMNL. Only A and B still go through the
    // mainloop's own iterators.
    if constexpr (has_avail_sms<typename Gemm::Arguments>::value) {
      // StreamK kernel.
      return typename Gemm::Arguments(
          cutlass::gemm::GemmUniversalMode::kGemm,
          p,
          args.split_k_slices,
          callback_args,
          const_cast<Element*>(args.ptr_a),
          const_cast<Element*>(args.ptr_b),
          nullptr,                                     // ptr_C
          nullptr,                                     // ptr_D
          static_cast<int64_t>(p.m()) * p.k(),         // batch_stride_A
          static_cast<int64_t>(p.n()) * p.k(),         // batch_stride_B
          0,                                           // batch_stride_C
          0,                                           // batch_stride_D
          static_cast<int64_t>(p.k()),                 // lda (A row-major)
          static_cast<int64_t>(p.n()),                 // ldb (B row-major)
          0,                                           // ldc
          0,                                           // ldd
          args.avail_sms);
    } else {
      // Data-parallel kernel: same list, no avail_sms.
      return typename Gemm::Arguments(
          cutlass::gemm::GemmUniversalMode::kGemm,
          p,
          args.split_k_slices,
          callback_args,
          const_cast<Element*>(args.ptr_a),
          const_cast<Element*>(args.ptr_b),
          nullptr,                                     // ptr_C
          nullptr,                                     // ptr_D
          static_cast<int64_t>(p.m()) * p.k(),         // batch_stride_A
          static_cast<int64_t>(p.n()) * p.k(),         // batch_stride_B
          0,                                           // batch_stride_C
          0,                                           // batch_stride_D
          static_cast<int64_t>(p.k()),                 // lda (A row-major)
          static_cast<int64_t>(p.n()),                 // ldb (B row-major)
          static_cast<int64_t>(0),                     // ldc
          static_cast<int64_t>(0));                    // ldd
    }
  }
};

} // namespace device
} // namespace evt
} // namespace tiny_cutlass
