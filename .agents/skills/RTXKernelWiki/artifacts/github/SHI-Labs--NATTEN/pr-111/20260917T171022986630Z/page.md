# Fused neighborhood attention

Upstream: https://github.com/SHI-Labs/NATTEN/pull/111

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

* Fused neighborhood attention (FNA) kernels (forward pass only for now)
  * 1D, 2D and 3D Neighborhood Attention are supported,
  * Causal neighborhood attention is implemented,
  * Window (kernel) size, dilation, and causality can be defined *per-axis*,
  * All GPU architectures since Maxwell (SM50) are supported,
    * SM50 up to SM70 are SIMT-only, but support both FP16 and FP32,
    * SM70 and SM75 target Tensor Cores in FP16, and SIMT-style in FP32, * SM80 and above target Tensor Cores in FP16, BF16, and FP32.
  * Relative positional biases are implemented (not defined for causal masking yet),
  * Memory layout in FNA is different from existing kernels (`[B, *, heads, dim]` instead of `[B, heads, *, dim]`.)
    * Eventually this layout can skip over the permute/explicit reshape step in the attention module following the QKV projection.
* Naive kernels now implement and allow causal masking,
* Naive kernels (CPU and CUDA) now allow varying parameters (window size, dilation, causal) across axes,
* Major bug fix in Volta GEMM kernels
  * The epilogue was different for Volta, and it slipped through unit tests,
  * Tests are now more aggressive, and the issue has been fixed.
* Minor torch bug fixed
  * Streams were not being selected correctly if users set a tensor to a device other than cuda:0. Thanks to @AdityaKane2001 for discovering it.
* Documentation (finally):
  * Better late than never, but finally added more documentation and reorganized docs under docs/ instead of shoving everything into the readme.
* So much more that I forgot (in part due to lack of documentation).

## Discussion

### Ryu1845 · 2024-03-10T10:49:45Z

https://github.com/SHI-Labs/NATTEN/pull/111#issuecomment-1987182266

Any ETA for the backward pass?

### alihassanijr · 2024-03-10T18:57:13Z

https://github.com/SHI-Labs/NATTEN/pull/111#issuecomment-1987331497

@Ryu1845 we hope to be able to release it very soon, but don't have a specific time frame at the moment.
