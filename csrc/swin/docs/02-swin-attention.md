# 02 SwinAttention

## 前言

`SwinAttention` 表示一个明确的 attention 子路径，不冒充完整 Swin block：

```text
WindowPartition
  -> QKV GEMM
  -> QKV bias/split
  -> tiled-online attention
  -> output GEMM + bias
  -> WindowReverse
```

## Operator 边界

public 入口是：

```cpp
device::SwinAttention<ArchTag, Element>
```

`SwinAttentionProblem` 保存 window/head/shift 参数，`SwinAttentionTensors` 保存输入输出，
并组合一个内部 `WindowAttentionTensors`。QKV、Q/K/V、attention output 等 workspace
不再与完整 block 的 MLP workspace 混在同一个大 descriptor 中。

`kernel::DefaultSwinAttention` 负责 projection GEMM tile 和 attention tile；它在
`swin.cu` 中选择，不是 public API 的模板实参。

## Reference

`csrc/tests/swin/reference.h/.cpp` 使用 cuDNN frontend SDPA 写 attention reference。
`window.cpp` 另外检查 partition、QKV、projection 和 reverse 的 host reference。
cuDNN 只属于 correctness gate，不是 runtime fallback。

## 当前结论

PatchMerging 已从这条路径移除。stage transition 需要独立 operator 和独立验证，不能
通过一个 nullable output pointer 改变 attention 的语义。
