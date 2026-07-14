# Swin workspace 记忆

这个文件只记录会影响后续修改的稳定边界和已知约束。

## Public operator

- public operator 只保留 `PatchEmbed<ArchTag, Element>`、
  `SwinAttention<ArchTag, Element>`、`SwinBlock<ArchTag, Element>`。
- 不要恢复 `device::Xxx<Policy>` 入口，也不要为旧 `device::Swin` 增加兼容 alias。
- `DefaultPatchEmbed`、`DefaultSwinAttention`、`DefaultSwinBlock` 是 CUDA 实现侧的
  CUTLASS factory；不要从 public facade header include CUDA-only factory 依赖。
- 当前显式实例化是 `cutlass::arch::Sm80` + `cutlass::half_t`，构建目标是 SM89。
- 三个 public operator 的 `can_implement` 和 `run` 统一返回 `cutlass::Status`。

## Descriptor

- PatchEmbed、SwinAttention、SwinBlock 使用各自的 Problem/Tensors。
- attention 与 block 只通过内部 `WindowAttentionTensors` 共享 projection/attention
  权重和 workspace；不要重新合并成一个同时包含 MLP/LayerNorm 的大 `SwinTensors`。
- core runtime 使用 raw device pointer 和 `cudaStream_t`，不依赖 Torch/ATen ownership。

## Operator 边界

- `SwinAttention` 只负责 partition -> attention -> reverse。
- `SwinBlock` 负责完整 v1 pre-norm block，包括两处 residual 和 MLP。
- PatchMerging 是 BasicLayer stage transition，不允许通过 nullable output pointer 作为
  SwinAttention 的可选副作用。如果恢复，必须是独立 problem/factory/operator/test。
- TensorRT plugin 当前封装 SwinAttention，不要把它描述成完整 SwinBlock 或整网。

## 验证与文档

- 唯一入口是 `scripts/kernels/swin/run.bat`，顺序固定 build -> verify -> bench。
- verify 覆盖三个 public operator；失败后不得 benchmark/profile。
- benchmark、Nsys、NCU 采集和 NCU CSV 解析统一放在 `bench.py`。
- workflow 只消费 `checkpoint/manifest.json` 已描述的 artifact，不在 verify/bench 下载模型。
- cuDNN correctness reference 固定为 `csrc/tests/swin/reference.h/.cpp`；cuDNN 不进入
  runtime fallback。
- 学习和设计文档放在 `docs/`，按 `00-...`、`01-...` 连续编号；删除过期路径，
  不保留与当前实现冲突的历史说明。
