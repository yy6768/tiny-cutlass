# Mixed precision GEMM Performance (A100 & V100)

Upstream: https://github.com/NVIDIA/cutlass/issues/374

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

I want to know about the peak performance of Mixed precision GEMM (Tensor Cores operate on FP16 input data with FP32 accumulation) for Ampere and Volta architecture with cutlass. Do we have any refrence of is it poosible to predeict it without performing an experiment (just by hardware spec)?

I have a code for LU decomposition (mixed fp16 & fp 32), but I am seeing that it is slower with Ampere architecture vs. Volta. So I am looking to find the diference in Ampere whcih make my code to work slower. 

## Discussion

### mnicely · 2021-12-06T14:02:41Z

https://github.com/NVIDIA/cutlass/issues/374#issuecomment-986804063

Hi @nnaron, you should move this question to the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/gpu-accelerated-libraries/12) and provide a reproducer.

### hwu36 · 2021-12-07T16:05:29Z

https://github.com/NVIDIA/cutlass/issues/374#issuecomment-988065434

Are you running the same Volta kernel on Ampere?  Volta tensor core instruction size is 8x8x4 which is no longer natively supported on Ampere.  Ampere uses emulation code to run the old Volta tensor core kernels which is slow.

Ampere is efficient when running its own large tensor core instruction which is 16x8x16.
