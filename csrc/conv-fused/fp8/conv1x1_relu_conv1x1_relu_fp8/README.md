# conv1x1_relu_conv1x1_relu_fp8

这个子目录只放 FP8 family 的稳定设计，不记录临时构建状态。

## 目标

固定算子语义：

```text
input_e4m3
  -> conv1x1(input, weight0, bias0_fp32)
  -> relu
  -> quantize to e4m3 with stage0_scale
  -> conv1x1(stage0_e4m3, weight1, bias1_fp32)
  -> relu
  -> quantize to e4m3 with output_scale
```

公开函数名保留 `conv1x1_relu_conv1x1_relu_fp8`。实际 dtype 只支持 CUTLASS `cutlass::float_e4m3_t`；本地 CUTLASS 没有标准 `e5m3` 类型。

## 分层

- `ops/conv1x1_relu_conv1x1_relu_fp8.{h,cu}`: raw-pointer core API、problem/scale/pointer 检查、CUTLASS device 调用。
- `device/conv1x1_relu_conv1x1_relu_fp8.{h,cu}`: CUTLASS device wrapper 和 `Arguments` 组装。
- `kernel/conv1x1_relu_conv1x1_relu_fp8.h`: `DefaultConv1x1ReluConv1x1ReluFp8<ArchTag, ...>` type factory，直接持有 CTA/warp shape 默认模板参数。
- `threads/conv1x1_relu_conv1x1_relu_fp8_epilogue_ops.h`: stage0/stage1 的 thread-level epilogue 别名。

## 量化位置

- stage0 的 `relu + bias + stage0 quantize` 由中间 accumulator loader / smem-staged path 通过 `EpilogueOutputOp0` 完成，第二段卷积直接消费 e4m3 fragment。
- stage1 的 `relu + output quantize` 由最终 epilogue 完成。
- ModelOpt `FP8_DEFAULT_CFG` 的第一版验证按 per-tensor scale 语义处理，host 侧只把 scalar stage0 scale 展开成 hidden-channel 等值向量以适配 CUTLASS example 13 的 loader 接口。
- amax/动态 scale 更新不是第一版目标；若要与 ModelOpt 动态 amax 训练/校准闭环对齐，需要改成支持 absmax 的 CUTLASS epilogue 或扩展 final epilogue。

## CUTLASS policy

- 当前 kernel 走 CUTLASS example 13 的 back-to-back implicit-GEMM convolution。
- stage0 accumulator 通过 CUTLASS 的 smem-accumulator specialization 暂存，再作为 stage1 的 A operand。
- `kernel/` 只提供模板化 policy factory；当前 device 默认实例是 `ArchTag = cutlass::arch::Sm89` 与 `cutlass::float_e4m3_t`，但入口保留 arch/type 模板参数。
- CTA/warp shape 不再单独建 traits 文件；当前 shape 直接作为 `DefaultConv1x1ReluConv1x1ReluFp8` 的默认模板参数，真实 pipeline 仍由 CUTLASS `DefaultB2bConv2dFprop` 组装。

## 约束

- 不允许 raw CUDA kernel。
- 主计算只能走 CUTLASS implicit-GEMM。
- 构建只允许写入 `build/`。
- 不允许引入 PyTorch/ATen/pybind binding；ModelOpt 只能作为外部量化语义 reference。
