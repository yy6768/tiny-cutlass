#pragma once

#include <type_traits>
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

namespace tiny_cutlass::swin::window_attention::kernel {

// Group scheduling and the MMA instruction are independent policy choices.
// This primitive consumes already-quantized operands. alpha may contain the
// common dequantization scale sA*sB; per-group scales require a different epilogue.
template <typename ArchTag, typename ElementA, typename ElementB,
          typename ElementOutput = float,
          typename ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 128>,
          typename WarpShape = cutlass::gemm::GemmShape<32, 32, 128>>
struct DefaultGroupedGemm {
  static_assert(std::is_same_v<ArchTag, cutlass::arch::Sm89>, "this grouped policy requires SM89");
  static_assert((std::is_same_v<ElementA, cutlass::float_e4m3_t> || std::is_same_v<ElementA, cutlass::float_e5m2_t>) &&
                (std::is_same_v<ElementB, cutlass::float_e4m3_t> || std::is_same_v<ElementB, cutlass::float_e5m2_t>),
                "FP8 operand policy required");
  static_assert(std::is_same_v<ElementOutput, float>, "this epilogue supports FP32 output");
  // Stock GemmGrouped selects A/B pointers with a conditional expression before
  // casting, which does not compile for distinct operand pointer types.
  static_assert(std::is_same_v<ElementA, ElementB>, "stock grouped kernel requires matching operand types");
  using Epilogue = cutlass::epilogue::thread::LinearCombination<ElementOutput, 4, float, float>;
  using CutlassKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      ElementA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
      ElementB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
      ElementOutput, cutlass::layout::RowMajor, float,
      cutlass::arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
      cutlass::gemm::GemmShape<16, 8, 32>, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
      cutlass::arch::OpMultiplyAdd>::GemmKernel;
  using DeviceOperator = cutlass::gemm::device::GemmGrouped<CutlassKernel>;
};

}  // namespace tiny_cutlass::swin::window_attention::kernel
