# 来源筛选记录

核查日期：2026-09-17。这里的记录不参与 query.py 检索。搜索命中只用于发现；已核查 GitHub API 元数据、PR changed files 和相关讨论。

## 新候选淘汰

- [cutlass-3120](https://github.com/NVIDIA/cutlass/pull/3120)：作者撤回最初的 FP4 转换不支持结论；a/f 编译目标才是关键，不能传播最初 root cause。
- [cutlass-3121](https://github.com/NVIDIA/cutlass/pull/3121)：已关闭未合并的 K=64/zero-stride TMA 候选，作者讨论缩减了适用范围；首批有效集优先采用明确的 upstream 修复。
- [cutlass-3185](https://github.com/NVIDIA/cutlass/pull/3185)：作者表示将采用更干净方案重新提交；已关闭未合并，不作为可用实现。
- [cutlass-3082](https://github.com/NVIDIA/cutlass/pull/3082)：核心修复是让 SM121 通过已有 SM120 guard；SM121-only 扩展超出维护目标。
- [triton-9852](https://github.com/triton-lang/triton/pull/9852)：作者后续测试更正最初 codegen 根因和后缀解释；不能凭标题采用禁用 pipeline 的补丁。
- [vllm-38556](https://github.com/vllm-project/vllm/pull/38556)：后续讨论中出现的交叉引用，但实际 diff 是 async spec decoding host 逻辑，不是本库目标 kernel。

## 旧博客退出

旧 20 篇全部退出有效库，原始文件保留在完整备份。它们围绕 SM100/Hopper 或模型/比赛背景组织，不足以证明新的目标架构实现。并非断言文章内容错误或所有概念不可迁移。

- [amandeep-nvfp4-attempts.md](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)：竞赛/SM100 专用实现，缺少目标架构实现证据。
- [blackwell-microbenchmarking.md](https://arxiv.org/abs/2512.02189)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [colfax-cutlass-blackwell.md](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tmem-for-nvidia-blackwell-gpus/)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [deepgemm.md](https://github.com/deepseek-ai/DeepGEMM)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [flash-attention-4.md](https://tridao.me/blog/2026/flash4/)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [flashmla.md](https://github.com/deepseek-ai/FlashMLA)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [gated-delta-net.md](https://github.com/NVlabs/GatedDeltaNet)：模型/研究背景或目标架构与源码证据不足，不进入首批实现库。
- [gpu-mode-reward-hack.md](https://www.gpumode.com/news/reward-hacking-nvfp4)：竞赛/SM100 专用实现，缺少目标架构实现证据。
- [jax-pallas-blackwell-matmul.md](https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [k-search-kernel-generation.md](https://arxiv.org/abs/2602.19128)：模型/研究背景或目标架构与源码证据不足，不进入首批实现库。
- [modular-blackwell-matmul.md](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [nsa.md](https://arxiv.org/abs/2502.11089)：模型/研究背景或目标架构与源码证据不足，不进入首批实现库。
- [nvfp4-format-details.md](https://haroldbenoit.com/notes/ml/engineering/precision/nvfp4-format)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [qwen3-next-architecture.md](https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/)：模型/研究背景或目标架构与源码证据不足，不进入首批实现库。
- [simon-nvfp4-gemv.md](https://veitner.bearblog.dev/nvfp4-gemv/)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [tcgen05-tutorial.md](https://gau-nernst.github.io/tcgen05/)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [tflops-gap-fp4-moe.md](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [tilus-nvidia.md](https://github.com/NVIDIA/tilus)：原页使用 SM100/Hopper 硬件或专有流水，不重贴 SM120 标签。
- [vllm-deepseek-v3-sparse-attention.md](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)：模型/研究背景或目标架构与源码证据不足，不进入首批实现库。
- [yue-nvfp4-hackathon.md](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)：竞赛/SM100 专用实现，缺少目标架构实现证据。

## 新博客入选

- Ampere async copy：官方实现解释。
- SGEMM worklog：SM86/A100 访存与分块研究，限定 SIMT 学习用途。
- Learn CUTLASS the hard way：SM89 实现，有代码入口，标记硬件参数错误。
- CuTe transpose：只提取可迁移 layout/swizzle 部分，Hopper 数据不转用于目标卡。
- Colfax SM12x NVFP4：直接匹配 SM120，区分 SM121 和 SM100。
