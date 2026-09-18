# Disabling Tensor Core for Specific CUDA Stream

Upstream: https://github.com/NVIDIA/cutlass/issues/289

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

I am trying to run GEMM in double presision (input and output is double ) without using of tensor core and just with CUDA core. For that porpose I am using this line to disable tensor core for the stream that dgemm is going to performe : 

> cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

But after profiling I am seeing: 

    cutlass::kernel<cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1>

Which means that tensor core has benn enabled again. Do you have any idea to keep it disabled?

The GPU that I am using is A100 and cuda 11.0.

Aha! Link: https://nvaiinfa.aha.io/features/CUTLASS-24

## Discussion

### hwu36 · 2021-07-06T19:39:54Z

https://github.com/NVIDIA/cutlass/issues/289#issuecomment-875031347

https://docs.nvidia.com/cuda/cublas/index.html#cublasmath_t says that 



> CUBLAS_DEFAULT_MATH | This is the  default and highest-performance mode that uses compute and intermediate  storage precisions with at least the                                              same number of mantissa and  exponent bits as requested. Tensor Cores will be used whenever  possible.
> -- | --

That said, I don't think you can disable tensor core when using double precision in cublas.  Tensor core and CUDA core provide the same precision and there is no point for cublas to use CUDA cores on Ampere.

### mnicely · 2021-11-23T19:46:02Z

https://github.com/NVIDIA/cutlass/issues/289#issuecomment-977083892

This has been answered here https://forums.developer.nvidia.com/t/disable-tensor-cores-in-cublas-functions-explicity/188795

TL;DR - Can't disable FP64 TCs
