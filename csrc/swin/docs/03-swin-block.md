# 03 SwinBlock

## 前言

`device::SwinBlock<ArchTag, Element>` 实现 Swin v1 pre-norm block 的 inference 路径：

```text
norm1 -> shifted window attention -> residual1
      -> norm2 -> fc1 -> GELU -> fc2 -> residual2
```

DropPath/Dropout 在 inference 下是 identity，因此不生成额外 kernel。

## 十次 launch

1. `LayerNormShiftPartition`
2. QKV GEMM
3. `AddQkvBiasSplit`
4. tiled-online attention
5. output GEMM + bias
6. `ReverseAddResidualLayerNorm`
7. MLP fc1 GEMM
8. `AddBiasGelu`
9. MLP fc2 GEMM
10. `AddBiasResidual`

`SwinBlockProblem` 在 attention problem 上增加 `mlp_ratio` 和 `layernorm_eps`。
`SwinBlockTensors` 组合内部 attention tensors，再增加 norm/MLP 权重和 workspace。

## 验证

`csrc/tests/swin/block.cpp` 有自包含 host reference，覆盖 shifted、non-shifted 和较大
feature map 三组 case。它已经进入 `verify.py`，因此 block parity 失败时不会继续 bench。

## 当前结论

完整 block 和 attention 子路径可以共享内部 attention 实现，但 public descriptor 和
operator 名必须分开。这样 plugin 或上层 runtime 能明确选择自己需要的语义。
