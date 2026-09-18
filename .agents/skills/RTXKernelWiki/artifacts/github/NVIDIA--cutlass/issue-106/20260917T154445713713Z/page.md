# Performance on tensor core

Upstream: https://github.com/NVIDIA/cutlass/issues/106

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

I've benchmarked fp16 tensor core performance on Tesla T4 GPU. Compared to cublas, cutlass runs faster in some workloads. But in some cases, cutlass has poor performance which could not reach cublas's 90% speed. Could someone help me to solve that? thx
By the way, which workload does cutlass use in the [gtc presentation](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf)?
m-n-k | cutlas(ms) | cublas(ms)
-- | -- | --
32-1024-2048 | 0.170258 | 0.056032
64-1024-2048 | 0.17238 | 0.0653234
128-1024-2048 | 0.195744 | 0.070236
256-1024-2048 | 0.180191 | 0.09387
512-1024-2048 | 0.194759 | 0.149984
1024-1024-2048 | 0.170392 | 0.274183
128-128-128 | 0.0329984 | 0.01254
256-256-256 | 0.0509088 | 0.017251
512-512-512 | 0.0811725 | 0.04283
1024-1024-1024 | 0.134937 | 0.153455
1024-1024-512 | 0.0880979 | 0.141054
1024-1024-256 | 0.0535661 | 0.063555
1024-512-2048 | 0.163264 | 0.149918
1024-256-2048 | 0.178731 | 0.107762



## Discussion

### kerrmudgeon · 2020-09-23T21:24:10Z

https://github.com/NVIDIA/cutlass/issues/106#issuecomment-697980717

We used 3456-by-4096-by-8192 and 3456-by-4096-by-16384 to estimate peak performance on NVIDIA A100 for the GTC 2020 presentation. This perfectly 'fills' the GPU with CUDA threadblocks and has a sufficiently large inner dimension that runtime is dominated by the mainloop.

Did your study evaluate all tile sizes of CUTLASS GEMM kernels? Consider building as follows:

```bash
cmake .. -DCUTLASS_NVCC_ARCHS=75 -DCUTLASS_LIBRARY_KERNELS=s1688gemm*align8
make cutlass_profiler -j16
```
This results in compiling all kernels targeting Turing Tensor Cores. Running all kernels with the CUTLASS Profiler and picking the fastest for each layout and problem size combination should hopefully depict the best possible performance from CUTLASS.

Closing for now. Feel free to reopen if the above are problematic or yield unexpected results. 

### hwu36 · 2021-04-19T02:46:43Z

https://github.com/NVIDIA/cutlass/issues/106#issuecomment-822129549

CUDA 11.3 significantly improves the performance of Ampere/Turing/Volta Tensor Core kernels. See #241 .

You may want to do the benchmarking again with the latest compiler.
