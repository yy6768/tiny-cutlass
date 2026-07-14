# conv-fused Agent 指南

## 当前范围

- 只维护已验证的 `conv1x1_dual` 和 `conv1x1_upsample` family。
- `conv3x3_pool` 是未来方向；在真实设计、reference parity 和验证完成前，不新增
  placeholder、CMake target 或正确性声明。
- FP8、TensorRT 和旧 pool pipeline 已移除，不恢复它们的绑定、脚本或文档。

## 设计约束

- 主干必须是 CUTLASS implicit-GEMM；不写 raw CUDA kernel，也不提供 SIMT/raw CUDA
  fallback。
- core 只接受 raw device pointer、problem descriptor 和 `cudaStream_t`，不依赖
  PyTorch、ATen、pybind 或 TensorRT。
- `ops/` 暴露 API，`device/` 负责 CUTLASS device operator 与显式支持性检查，
  `kernel/` 只装配 `CutlassKernel`。
- policy primary type 保持 `DefaultXxx<ArchTag, Element..., ThreadblockShape..., WarpShape...>`
  模板工厂风格；arch、dtype、layout 和 tile 选择留在模板参数、实例化、测试或 CMake。
- 不支持的 arch、dtype、layout 或 shape 必须通过 CMake、`can_implement` 或明确状态失败。

## 验证与组织

- 每个 family 的验证位于 `csrc/tests/conv-fused/<family>/`，目标使用短名。
- 脚本位于 `scripts/kernels/<family>/run.bat`，保持 build 后执行正确性验证；未通过
  reference parity 前不做 benchmark。
- 构建产物只能放在 `build/`。
