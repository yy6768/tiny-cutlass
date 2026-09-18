# 按仓库

由 `scripts/generate_queries.py` 生成。架构标签表示相关性；状态与本地验证分开记录。

## Dao-AILab/flash-attention

- [issue-flash-attention-190 — Support for NVIDIA GeForce RTX 3090 with Compute Capability 8.6](../sources/issues/flash-attention-190.md) · closed · sm86
  记录 RTX 3090 的早期 FlashAttention backward/head-dimension 限制及后续版本更新。

- [pr-flash-attention-2751 — fix(cute): head-dim-aware SM80 forward tile sizes to fix SMEM overflow on sm_86/sm_89](../sources/prs/flash-attention-2751.md) · open · sm80, sm86, sm89
  为 SM80 系列 attention forward 按 head dimension 选 tile，处理 SM86/89 的 SMEM 溢出。

## NVIDIA/FasterTransformer

- [pr-fastertransformer-333 — Fix/swin qk scale](../sources/prs/fastertransformer-333.md) · merged · sm89
  Swin WindowAttention 的 qk_scale 修正，用来检查接口 scale 和 fused runner 内部 scale 是否重复。

## NVIDIA/cutlass

- [issue-cutlass-1181 — [QST] What is the usage of Sm86?](../sources/issues/cutlass-1181.md) · closed · sm80, sm86, sm89
  维护者解释 ArchTag 表示 kernel 的最低需求；某些 SM86/89 kernel 应保留 Sm80 policy，再选择实际编译目标。

- [issue-cutlass-1937 — [QST] FP8 with row-wise scaling on Ada-Lovelace](../sources/issues/cutlass-1937.md) · open · sm89
  Ada FP8 row-wise scaling 讨论提供 EVT 生成代码的思路。

- [issue-cutlass-2053 — [QST] Grouped GEMM on A10 GPUs](../sources/issues/cutlass-2053.md) · closed · sm80, sm86
  A100 上工作的 grouped GEMM 在 A10 上失败；讨论通过减小 ThreadblockShape 处理 SMEM 预算。

- [issue-cutlass-2158 — [DOC] CUTLASS INT4 GEMM: Missing SM89 Dispatch Configuration for L40S/4090](../sources/issues/cutlass-2158.md) · open · sm80, sm89
  维护者建议为 SM89 INT4 使用对应 SM80 kernel policy 并编译到 SM89。

- [issue-cutlass-2189 — [QST] Fusion paths on Ada/SM89](../sources/issues/cutlass-2189.md) · closed · sm89
  围绕 Ada implicit-GEMM convolution 讨论 mainloop、epilogue、back-to-back fusion 与 2.x EVT。

- [issue-cutlass-2212 — [QST]Why cp.async.ca will influence bank conflcit of Shared Store From Global Load ?](../sources/issues/cutlass-2212.md) · open · sm86, sm89
  提供 cp.async.ca/cg 与 swizzle、bank conflict 之间关系的复现讨论，涉及 RTX 3060/4060 Ti。

- [issue-cutlass-2766 — [QST] Does sm120 gemm kernels support fp16/tf32 inputs?](../sources/issues/cutlass-2766.md) · closed · sm120
  解释 SM100 GEMM 不能直接用于 SM120，并讨论 FP16/TF32 kernel 选择。

- [issue-cutlass-2906 — [BUG] SM120 NVF4 GEMM (example 79a): misaligned address crash](../sources/issues/cutlass-2906.md) · open · sm120
  记录 Windows 11、CUDA 13.1、CUTLASS 4.3.4 下 example 79a 的对齐故障。

- [issue-cutlass-3044 — CuTe DSL: FP8 MMA segfaults on SM120 — MmaAtomSM80Type missing kind::f8f6f4 lowering](../sources/issues/cutlass-3044.md) · closed · sm120
  记录把 SM80/SM89 FP8 atom 用于 SM120 lowering 时的编译崩溃与不同指令格式。

- [issue-cutlass-3096 — SM120 (Bug) (With FIx)(RTX Blackwell) NVFP4 MoE: CUTLASS Grouped GEMM Produces Garbage Output; Fixed via FlashInfer SM120 Patches + compute_120f (CUDA 13.0) — 39 tok/s Native FP4](../sources/issues/cutlass-3096.md) · closed · sm120
  记录 RTX PRO 6000 上 NVFP4 grouped GEMM 的错误输出、初始化失败与工具链排查。

- [issue-cutlass-3281 — [BUG] CuTeDSL 4.5.2: MmaFP8Op.make_fragment_A segfaults on SM120 (Blackwell)](../sources/issues/cutlass-3281.md) · open · sm120
  CuTe DSL 4.5.2 在 RTX 5090 上使用 MmaFP8Op.make_fragment 时的 crash reproducer。

- [pr-cutlass-1187 — fix cp.async L2 prefetch typo](../sources/prs/cutlass-1187.md) · merged · sm80, sm86, sm89
  修正 CuTe SM80 cp.async 的 L2 prefetch 拼写；从 copy atom 跟踪真正发出的 PTX。

- [pr-cutlass-1604 — Add Ampere GEMM example using Cute and CUTLASS 3.x](../sources/prs/cutlass-1604.md) · open · sm80
  提供 Ampere CuTe + CUTLASS 3.x Collective MMA/Epilogue 示例候选。

- [pr-cutlass-1753 — Fix EVT for cutlass::gemm::kernel::DefaultGemmWithVisitor's behavior when constructing GemmUniversalAdapter](../sources/prs/cutlass-1753.md) · merged · sm80, sm89
  修复 DefaultGemmWithVisitor 配合 GemmUniversalAdapter 时的继承与类型可见性。

- [pr-cutlass-2177 — add support for sm89 in cute and the unit tests](../sources/prs/cutlass-2177.md) · merged · sm89
  引入 Ada CuTe FP8 MMA atom、traits 和对应单元测试。

- [pr-cutlass-2328 — Add SM80/89 blockwise scaling kernel, support FP8 block/groupwise on Ada, INT8 on Ampere](../sources/prs/cutlass-2328.md) · open · sm80, sm86, sm89
  提出 Ampere INT8 与 Ada FP8 的 block/groupwise scaling CuTe 路径；正文包含 A100、A10 和 4090。

- [pr-cutlass-2351 — Fix sgemm_sm80 example bug](../sources/prs/cutlass-2351.md) · merged · sm80, sm86, sm89
  修正 sgemm_sm80 示例中 copy fragment 的 K 维与 MMA fragment 不匹配问题。

- [pr-cutlass-2378 — support fp16 accmulator for sm89 fp8 mma](../sources/prs/cutlass-2378.md) · merged · sm89
  补充 SM89 FP8 MMA 的 FP16 accumulator 路径及测试。

- [pr-cutlass-2582 — Fix Copy_Atom type mismatch in sgemm_sm80.cu](../sources/prs/cutlass-2582.md) · merged · sm80, sm86, sm89
  修正 sgemm_sm80 泛型 gemm_nt/gemm_tn 的 Copy_Atom 类型不匹配。

- [pr-cutlass-3030 — [CuTeDSL] Flash Attention v2 for SM120 (Blackwell GeForce)](../sources/prs/cutlass-3030.md) · open · sm120
  SM120 FA2 forward 候选同时提供 cp.async 和 TMA pipeline，使用 warp MMA。

- [pr-cutlass-3055 — Replace std::min with cute::min in sm120 blockwise scaling device functions](../sources/prs/cutlass-3055.md) · merged · sm120
  把 SM120 blockwise scaling device 函数里的 std::min 替换成可用于 device 的 cute::min。

- [pr-cutlass-3196 — Small Tile M BlockScaled GEMM + Grouped GEMM on SM12x](../sources/prs/cutlass-3196.md) · closed-unmerged · sm120
  提出 SM12x blockscaled 小 M/K tile、scale padding 与 grouped GEMM 扩展。

- [pr-cutlass-3273 — [CuTeDSL] Add SM120 MXF4/NVFP4 native-TMA path](../sources/prs/cutlass-3273.md) · closed-unmerged · sm120
  提出 SM120 warp MMA + native TMA 的 CuTe DSL 实现，覆盖 scale fragment、copy layout 和 microtile 示例。

- [pr-cutlass-3278 — Fix the ScatterD issue in predicated_tile_iterator](../sources/prs/cutlass-3278.md) · merged · sm89
  修正 2.x PredicatedTileIterator 的 ScatterD 指针推进，供 partition/reverse 组件审查。

- [pr-cutlass-3394 — Fix SM89 FP8 blockwise scale indexing under threadblock swizzling](../sources/prs/cutlass-3394.md) · merged · sm89
  将 blockwise scale 的索引从原始 blockIdx 改为 swizzle 后的逻辑 M/N tile 坐标。

- [pr-cutlass-828 — fMHA: Sync FW with xFormers](../sources/prs/cutlass-828.md) · merged · sm89
  CUTLASS example 41 的 online softmax、attention bias 和 shared-memory PV 组件。

- [pr-cutlass-992 — Update fMHA kernels](../sources/prs/cutlass-992.md) · merged · sm89
  example 41 更新：迭代 softmax、同步、寄存器使用和从 shared memory 执行第二次 MMA。

## SHI-Labs/NATTEN

- [pr-natten-111 — Fused neighborhood attention](../sources/prs/natten-111.md) · merged · sm89
  FNA forward：1D/2D/3D 邻域 mask、dilation、相对位置偏置与 QK→softmax→PV 融合。

- [pr-natten-114 — Disable 64x128x128 GEMM config for SM86 and 89](../sources/prs/natten-114.md) · merged · sm89
  为 SM86/SM89 排除 64×128×128 GEMM 配置，避免照搬 A100 的 shared-memory 用量。

- [pr-natten-337 — cutlass-fna: 64b strides](../sources/prs/natten-337.md) · merged · sm89
  FNA token strides 改为 64 位；固定 head 保存较新的 forward kernel 和邻域索引实现。

- [pr-natten-341 — Fix/cutlass fna dv nan issue](../sources/prs/natten-341.md) · merged · sm89
  修复 FNA backward 在 SM80+ partial tile 上遗漏 mask 边界检查而出现 dV NaN。

## flashinfer-ai/flashinfer

- [pr-flashinfer-2786 — feat: K=64 block-scaled MoE GEMM for SM120 (RTX PRO 6000)](../sources/prs/flashinfer-2786.md) · closed-unmerged · sm120
  K=64 blockscaled MoE 候选展示 scale layout、SMEM 与 TMA 的交互。

- [pr-flashinfer-2798 — Upgrade cutlass 4.2.1 -> 4.4.2](../sources/prs/flashinfer-2798.md) · merged · sm120
  升级 CUTLASS 依赖并调整 kernel 构建配置，关联 SM120/SM121 NVFP4 与 TMA descriptor 故障。

## thu-ml/SageAttention

- [pr-sageattention-196 — Update to SageAttention v2.2.0 (sage2++)](../sources/prs/sageattention-196.md) · merged · sm89
  SM89 的 INT8 QK + FP8 PV，以及 FP16 局部累加/FP32 缓冲的混合精度参考。

## triton-lang/triton

- [issue-triton-11733 — 3.8.0 on sm_120: MN-packed mxfp4 dot_scaled returns wrong results and batched scaled dots fail an assert; both fixed on main (#10726, #11262)](../sources/issues/triton-11733.md) · open · sm120
  在 Triton 3.8.0/RTX PRO 6000 报告 MN-packed FP4 错误输出和 batched scale layout 失败。

- [pr-triton-10726 — [SM120] Fallback MN-packed FP4 MMA to the decomposition path](../sources/prs/triton-10726.md) · merged · sm120
  为 SM120 MN-packed FP4 scaled dot 增加分解路径选择，避免送入要求 K packing 的 native MMA。

- [pr-triton-11262 — [NVIDIA] Support batched SM120 scaled-dot scale layouts](../sources/prs/triton-11262.md) · merged · sm120
  把 SM120 scaled-dot scale layout 转换扩展到 rank-3，保留 batch 维。

- [pr-triton-11386 — [NVIDIA] Use block-scaled MMA with unit scales for fp8 dot on sm120](../sources/prs/triton-11386.md) · closed-unmerged · sm120
  提出用单位 scale 的 blockscaled MMA 替换部分 SM120 FP8 dot。

## vllm-project/vllm

- [pr-vllm-38423 — [NVIDIA] Bugfix NVFP4 DGX Spark and RTX50](../sources/prs/vllm-38423.md) · merged · sm120
  修正 SM12x NVFP4 的编译能力判断、runtime guard 和关联 CUTLASS/FlashInfer 依赖。

- [pr-vllm-5275 — [Kernel] Update Cutlass int8 kernel configs for SM80](../sources/prs/vllm-5275.md) · merged · sm80
  通过 A100 benchmark sweep 为 INT8 CUTLASS GEMM 选择 shape-dependent tile/dispatch。

- [pr-vllm-5560 — [Kernel] Adding bias epilogue support for `cutlass_scaled_mm`](../sources/prs/vllm-5560.md) · merged · sm80, sm89
  在 cutlass_scaled_mm 的 2.x/3.x epilogue 体系中添加 bias。

- [pr-vllm-6677 — [Kernel] Tuned FP8 Kernels for Ada Lovelace](../sources/prs/vllm-6677.md) · merged · sm89
  Ada FP8 CUTLASS 2.x GEMM 的 M/N 分派、tile/stages 和 scale epilogue；用于 FP8 primitive 选型。
