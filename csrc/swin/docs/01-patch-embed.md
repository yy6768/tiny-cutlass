# 01 PatchEmbed

## 前言

PatchEmbed 的输入是 texture-like NHWC 图像，输出是 NHWC patch token。RGB 的
`C=3` 不满足 TensorOp optimized iterator 对齐，所以实现显式 pad 到 8 通道。

## 数据流

```text
ImagePadChannels
  -> FilterOihwToKrscPadded
  -> CUTLASS implicit-GEMM Conv2d
  -> AddBiasLayerNorm
```

`device::PatchEmbed<ArchTag, Element>` 是 public facade。
`kernel::DefaultPatchEmbed<ArchTag, Element, ...>` 在 `patch_embed.cu` 内部产出
`Conv2dFpropKernel`；factory 不进入 public header。

`PatchEmbedProblem` 只保存 shape 和 LayerNorm epsilon，`PatchEmbedTensors` 保存输入、
权重、输出和显式 workspace。core runtime 不依赖 Torch 或 ATen。

## 验证

`csrc/tests/swin/patch_embed.cu` 同时支持随机 host reference 和 checkpoint artifact。
reference parity 通过后，`bench.py` 才会记录 runtime；profiling 模式下同一脚本采集
Nsys 和 NCU。

## 当前结论

这一条路径已经是独立 operator。后续优化重点是提前 pack filter，避免 inference
期间重复做 OIHW -> KRSC reorder。
