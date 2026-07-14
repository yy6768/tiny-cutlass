# 00 Swin CUTLASS 总览

## 前言

这个 workspace 不是完整 Swin 网络实现。当前目标是把三个边界明确、能独立验证的
operator 做扎实，再决定怎样向 BasicLayer 和整网组合。

## 三个 public operator

```text
device::PatchEmbed<ArchTag, Element>
device::SwinAttention<ArchTag, Element>
device::SwinBlock<ArchTag, Element>
```

调用方直接选择 arch 和 element type，不再构造或传入一个含义模糊的 `Policy`。
`DefaultPatchEmbed`、`DefaultSwinAttention`、`DefaultSwinBlock` 只在 CUDA 实现侧
组装 CUTLASS kernel 配置。

## 当前代码分层

```text
device/       public facade 和内部 launch helper
kernel/       DefaultXxx factory、CUDA kernel wrapper
threadblock/  单个 glue stage 的计算
warp/         window 坐标映射
tests/swin/   host/cuDNN reference 和 executable
```

`SwinAttentionProblem/SwinAttentionTensors` 与
`SwinBlockProblem/SwinBlockTensors` 已经分开。两条路径只共享内部的
`WindowAttentionTensors`，完整 block 不再把 MLP、LayerNorm workspace 塞进
attention descriptor。

## 当前结论

- PatchEmbed、SwinAttention、SwinBlock 都有独立 reference。
- PatchMerging 不属于 attention 的可选副作用，当前没有 public operator。
- TensorRT plugin 当前封装的是 `SwinAttention`，不是完整 `SwinBlock`。
- 统一入口是 `scripts/kernels/swin/run.bat`，顺序固定为 build -> verify -> bench。
