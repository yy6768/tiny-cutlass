# 架构与实现入口

核查日期：2026-09-17。下面是实现选择的起点；具体 dtype/shape/API 支持仍需查对应版本的源码和 PTX target notes。

| 目标 | 设备例子 | 首选研究路线 | 需要单独检查 |
|---|---|---|---|
| SM80 | A100、A30 | CUTLASS 2.x TensorOp、CuTe warp MMA、cp.async 多级流水 | A100 的 tile/stages、FP64 能力不能泛化到所有 Ampere |
| SM86 | RTX 3090、A10、RTX A6000 | 复用合适的 SM80 policy，编译到实际 SM86 | SMEM、occupancy、FP32 与 TensorOp 路线的吞吐差异 |
| SM89 | RTX 4090、L4、L40/L40S | warp MMA；FP8 使用 Ada atom；2.x epilogue/EVT | FP8 累加精度、scale 维度、2.x/3.x 支持按具体算子判断 |
| SM120 | RTX 5090、RTX PRO 6000 Blackwell | warp MMA；SM120 FP8/FP4 blockscale；cp.async 或 TMA | a/f 编译后缀、PTX 版本、scale fragment、TMA 对齐、设备资源 |

这些设备映射以 NVIDIA [CUDA GPU 列表](https://developer.nvidia.com/cuda-gpus)为准。

## SM80、SM86、SM89 的共同点与边界

`ArchTag=Sm80` 可以表示某条 CUTLASS kernel policy 的最低指令需求，不要求目标 GPU 恰好为 SM80。nvcc 编译目标仍应按实际 GPU 设置。不能机械地把所有模板参数替换成 `Sm86` 或 `Sm89`，因为对应 family 可能没有该 specialization。参见 [CUTLASS #1181](../sources/issues/cutlass-1181.md) 和 [#2158](../sources/issues/cutlass-2158.md)。

SM80 与 SM86 的 shared-memory 和 occupancy 上限不同；SM89 也应独立计算。查询 `cudaDeviceGetAttribute`/`cudaGetDeviceProperties`、kernel 的 `SharedStorage` 大小与寄存器用量，再决定 tile 和 stages。官方 [Ampere guide](../sources/docs/ampere-tuning.md) 与 [Ada guide](../sources/docs/ada-tuning.md) 是硬件边界依据。A10 grouped GEMM 的实际迁移问题见 [#2053](../sources/issues/cutlass-2053.md)。

SM80/86 没有 Ada FP8 Tensor Core 路径；FP8 存储后转换成 FP16/BF16 再计算，应明确称为转换方案。SM89 的 FP8 atom、累加类型和 blockwise scaling 分别查 [#2177](../sources/prs/cutlass-2177.md)、[#2378](../sources/prs/cutlass-2378.md)、[#3394](../sources/prs/cutlass-3394.md)。

## SM120 不能按 SM100 编写

PTX 的支持目标区分 `mma`、`wgmma`、`tcgen05`。SM120 的 Tensor Core 实现以 warp-level `mma.sync` 和寄存器 fragment 为依据，不使用 SM100 的 TMEM accumulator / tcgen05 主循环。SM120 支持 TMA；不能因其 MMA 路线接近 Ampere 就排除 TMA、CLC、PDL 或部分目标上的 `setmaxnreg`。每项高级指令分别核对 [PTX ISA](../sources/docs/ptx-isa.md) 的 Target ISA Notes 与工具链版本。

`sm_120`、`sm_120a`、`sm_120f` 不是任意互换的字符串；不要通过删除 a/f 后缀“修复”构建。不要从 `__CUDA_ARCH__ >= 1000` 推断 TMEM 支持。SM121 不在本 skill 的目标集，SM121-only 的修复不能直接当作 SM120 实测。详细实现入口见 [Colfax SM12x](../sources/blogs/colfax-sm12x-nvfp4.md) 和 [CUTLASS 固定版本源码](../sources/docs/cutlass-implementation-map.md)。

Blackwell 官方 tuning guide 同时讨论多个 compute capability。资源上限还应对照实际 SKU 的 runtime 属性；不抄 B200 的配置到 RTX，也不从不同 RTX 型号的博客数据推导统一预算。
