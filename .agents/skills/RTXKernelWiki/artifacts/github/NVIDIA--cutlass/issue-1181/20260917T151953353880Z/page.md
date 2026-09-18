# [QST] What is the usage of Sm86?

Upstream: https://github.com/NVIDIA/cutlass/issues/1181

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

Hi, currently I am trying to test the performance of different precision on different GPUs.
For RTX 4090, its compute ability is `8.9`. However, the corresponding ArchTag `Sm89`  is not included in CUTLASS. So I choosed `Sm86` as my gemm kernel's `ArchTag`, and I used the compiling command:
 ```nvcc  gemm.cu  -O3 --std=c++17 --gpu-architecture=compute_89 --gpu-code=sm_89,compute_89    -I ~/cutlass-main/include/ -I ~/cutlass-main/tools/util/include/ -I ../common/ -o gemm```
But I got the error: `Incomplete type is not allowed` in class `DefaultGemmConfiguration`, which doesn't have the implemented class whose `ArchTag ` is  `Sm86`.
So I am wondering:
1. What is the usage of Sm86?
2. To test gemm performance on 4090 and A6000, should I use the `ArchTag = Sm80`?

Sincerely thank you for your help!


## Discussion

### hwu36 · 2023-11-11T02:20:41Z

https://github.com/NVIDIA/cutlass/issues/1181#issuecomment-1806632858

just use sm80 in ArchTag.  ArchTag is more like the minimum arch that supports the kernel.  sm86, sm80, sm89 use the same type of kernel.

you need your device arch in your cmake command line.  e.g. sm89 for 4090, so that nvcc will generate the best binary for your gpu.

### Ther-LF · 2023-11-11T02:22:30Z

https://github.com/NVIDIA/cutlass/issues/1181#issuecomment-1806634052

Thank you!
