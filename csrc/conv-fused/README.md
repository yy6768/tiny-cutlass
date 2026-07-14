# conv-fused

此目录只保留已验证的 CUTLASS implicit-GEMM convolution family。运行时边界是
raw device pointer、problem descriptor 和 `cudaStream_t`；core 不依赖 PyTorch、ATen
或 TensorRT。

## 已验证 family

- `conv1x1_dual`：`conv1x1 -> relu -> conv1x1`，使用 CUTLASS example 13 的
  B2B implicit-GEMM device operator。
- `conv1x1_upsample`：`conv1x1 -> nearest-neighbor upsample`。卷积在低分辨率完成，
  输出迭代器将结果写入高分辨率块。

每个 family 都按 `ops/`、`device/`、`kernel/` 分层：`ops` 暴露 raw-pointer API，
`device` 组装 CUTLASS device operator，`kernel` 只装配模板类型。测试位于
`csrc/tests/conv-fused/<family>/`，对应脚本位于 `scripts/kernels/<family>/run.bat`。

目前只显式实例化 fp16 TensorOp 路径，并要求 `CMAKE_CUDA_ARCHITECTURES=89`。不支持的
shape 或对齐必须由 CUTLASS 明确拒绝，不能退回 SIMT 或 raw CUDA fallback。

FP8、TensorRT 和 pool 实验已移除。`conv3x3_pool` 只是后续重写的预留方向，尚未注册为
构建目标或实现。
