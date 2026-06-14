#pragma once

#include <limits>

#include <cuda_runtime_api.h>

#include "cutlass/conv/convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

namespace tiny_cutlass::conv_fused::device {

template <typename ImplicitGemmFusionKernel_>
class ImplicitGemmConvolutionFusion {
 public:
  using ImplicitGemmFusionKernel = ImplicitGemmFusionKernel_;

  using ElementA = typename ImplicitGemmFusionKernel::ElementA;
  using LayoutA = typename ImplicitGemmFusionKernel::LayoutA;
  using ElementB = typename ImplicitGemmFusionKernel::ElementB;
  using LayoutB = typename ImplicitGemmFusionKernel::LayoutB;
  using ElementC = typename ImplicitGemmFusionKernel::ElementC;
  using LayoutC = typename ImplicitGemmFusionKernel::LayoutC;
  using ElementAccumulator =
      typename ImplicitGemmFusionKernel::ElementAccumulator;
  using ElementCompute = typename ImplicitGemmFusionKernel::ElementCompute;
  using ElementScaleBias =
      typename ImplicitGemmFusionKernel::ElementScaleBias;
  using LayoutScaleBias = typename ImplicitGemmFusionKernel::LayoutScaleBias;
  using OperatorClass = typename ImplicitGemmFusionKernel::OperatorClass;
  using ArchTag = typename ImplicitGemmFusionKernel::ArchTag;
  using ThreadblockShape0 =
      typename ImplicitGemmFusionKernel::ThreadblockShape0;
  using ThreadblockShape1 =
      typename ImplicitGemmFusionKernel::ThreadblockShape1;
  using WarpShape0 = typename ImplicitGemmFusionKernel::WarpShape0;
  using WarpShape1 = typename ImplicitGemmFusionKernel::WarpShape1;
  using InstructionShape =
      typename ImplicitGemmFusionKernel::InstructionShape;
  using ThreadblockSwizzle =
      typename ImplicitGemmFusionKernel::ThreadblockSwizzle;
  using EpilogueOutputOp0 =
      typename ImplicitGemmFusionKernel::EpilogueOutputOp0;
  using EpilogueOutputOp1 =
      typename ImplicitGemmFusionKernel::EpilogueOutputOp1;
  using WarpMmaOperator0 =
      typename ImplicitGemmFusionKernel::WarpMmaOperator0;
  using WarpMmaOperator1 =
      typename ImplicitGemmFusionKernel::WarpMmaOperator1;
  using ArchMmaOperator =
      typename ImplicitGemmFusionKernel::ArchMmaOperator;
  using MathOperator = typename ImplicitGemmFusionKernel::MathOperator;
  using Arguments = typename ImplicitGemmFusionKernel::Arguments;

  static int const kStages = ImplicitGemmFusionKernel::kStages;
  static int const kConvDim = ImplicitGemmFusionKernel::kConvDim;
  static cutlass::conv::Operator const kConvolutionalOperator =
      ImplicitGemmFusionKernel::kConvolutionalOperator;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm =
      ImplicitGemmFusionKernel::kIteratorAlgorithm;

  static int const kWarpCount =
      (ThreadblockShape0::kM / WarpShape0::kM) *
      (ThreadblockShape0::kN / WarpShape0::kN);

 private:
  typename ImplicitGemmFusionKernel::Params params_;

 public:
  ImplicitGemmConvolutionFusion() = default;

  static cutlass::Status can_implement(Arguments const& args) {
    cutlass::Status status =
        ImplicitGemmFusionKernel::B2bMma::IteratorA0::can_implement(
            args.problem_size_0);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    status = ImplicitGemmFusionKernel::B2bMma::IteratorB0::can_implement(
        args.problem_size_0);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    status = ImplicitGemmFusionKernel::B2bMma::IteratorB1::can_implement(
        args.problem_size_1);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    ThreadblockSwizzle threadblock_swizzle;
    dim3 grid = threadblock_swizzle.get_grid_shape(
        threadblock_swizzle.get_tiled_shape(
            cutlass::conv::implicit_gemm_problem_size(
                kConvolutionalOperator, args.problem_size_0),
            {ThreadblockShape0::kM,
             ThreadblockShape0::kN,
             ThreadblockShape0::kK},
            args.problem_size_0.split_k_slices));

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max())) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    cutlass::gemm::GemmCoord problem_size_0 =
        cutlass::conv::implicit_gemm_problem_size(
            kConvolutionalOperator, args.problem_size_0);
    cutlass::gemm::GemmCoord problem_size_1 =
        cutlass::conv::implicit_gemm_problem_size(
            kConvolutionalOperator, args.problem_size_1);

    if (problem_size_0.m() != problem_size_1.m()) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    if (problem_size_0.n() != problem_size_1.k()) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    if (args.problem_size_1.R != 1 || args.problem_size_1.S != 1) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    if (problem_size_0.n() > ThreadblockShape0::kN) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    if (problem_size_1.n() > ThreadblockShape1::kN) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    return cutlass::Status::kSuccess;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;

    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord grid_tiled_shape =
        threadblock_swizzle.get_tiled_shape(
            cutlass::conv::implicit_gemm_problem_size(
                kConvolutionalOperator, args.problem_size_0),
            {ThreadblockShape0::kM,
             ThreadblockShape0::kN,
             ThreadblockShape0::kK},
            args.problem_size_0.split_k_slices);

    if (args.split_k_mode == cutlass::conv::SplitKMode::kParallel) {
      workspace_bytes =
          sizeof(ElementAccumulator) *
          size_t(cutlass::conv::implicit_gemm_tensor_c_size(
              kConvolutionalOperator, args.problem_size_0)) *
          size_t(grid_tiled_shape.k());
    } else if (
        args.split_k_mode == cutlass::conv::SplitKMode::kSerial &&
        args.problem_size_0.split_k_slices > 1) {
      workspace_bytes =
          sizeof(int) *
          size_t(grid_tiled_shape.m()) *
          size_t(grid_tiled_shape.n());
    }

    return workspace_bytes;
  }

  cutlass::Status initialize(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr) {
    if (args.problem_size_0.split_k_slices > 1) {
      if (!workspace) {
        return cutlass::Status::kErrorWorkspaceNull;
      }

      cudaError_t status =
          cudaMemsetAsync(workspace, 0, get_workspace_size(args), stream);
      if (status != cudaSuccess) {
        return cutlass::Status::kErrorInternal;
      }
    }

    params_ = typename ImplicitGemmFusionKernel::Params(
        args, static_cast<int*>(workspace));

    int smem_size =
        int(sizeof(typename ImplicitGemmFusionKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(
          cutlass::Kernel<ImplicitGemmFusionKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);

      if (result != cudaSuccess) {
        return cutlass::Status::kErrorInternal;
      }
    }

    return cutlass::Status::kSuccess;
  }

  cutlass::Status update(
      Arguments const& args,
      void* workspace = nullptr) {
    params_.ptr_A0 = args.ref_A0.data();
    params_.ptr_B0 = args.ref_B0.data();
    params_.ptr_C0 = args.ref_C0.data();
    params_.ptr_Scale0 = args.ref_Scale0.data();
    params_.ptr_Bias0 = args.ref_Bias0.data();
    params_.ptr_B1 = args.ref_B1.data();
    params_.ptr_C1 = args.ref_C1.data();
    params_.ptr_D1 = args.ref_D1.data();
    params_.output_op_0 = args.output_op_0;
    params_.output_op_1 = args.output_op_1;
    params_.semaphore = static_cast<int*>(workspace);

    return cutlass::Status::kSuccess;
  }

  cutlass::Status run(cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);

    int smem_size =
        int(sizeof(typename ImplicitGemmFusionKernel::SharedStorage));

    cutlass::arch::synclog_setup();
    cutlass::Kernel<ImplicitGemmFusionKernel>
        <<<grid, block, smem_size, stream>>>(params_);

    cudaError_t result = cudaGetLastError();
    return result == cudaSuccess
        ? cutlass::Status::kSuccess
        : cutlass::Status::kErrorInternal;
  }

  cutlass::Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  cutlass::Status operator()(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr) {
    cutlass::Status status = initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }
    return status;
  }
};

}  // namespace tiny_cutlass::conv_fused::device
