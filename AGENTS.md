# tiny-cutlass Agent Guide

这个文件记录整个仓库都会影响后续 agent 修改的规则。子工作区可以有自己的
`AGENTS.md`，但不能覆盖这里的全局约束。

## 子工作区索引

- `csrc/flash-attention/AGENTS.md`
- `csrc/conv-fused/AGENTS.md`
- `csrc/conv-fused/fp8/conv1x1_relu_conv1x1_relu_fp8/AGENTS.md`
- `csrc/natten/AGENTS.md`

## 全局规则

- 做 CUDA/CUTLASS kernel 编程时，必须先使用 `cutlass-kernel` skill。
- 每个 kernel family 优先提供一个脚本入口：
  `scripts/kernels/<family>/<family>.bat`。
- `.bat` 工作流顺序必须是 build -> verify -> bench；verify 失败时不要 benchmark。
- 构建输出只能放在 `build/`。
- 验证和 benchmark harness 放在 `csrc/tests/`。
- 先对齐可信 reference，再讨论性能；未通过 reference parity 的数据不能当作性能结论。
- CUTLASS kernel policy primary type 必须保持模板工厂风格，例如
  `DefaultXxx<ArchTag, Element..., ThreadblockShape..., WarpShape...>`。
- 不要新增 primary class、struct、function、target 或公共 API 名字把架构写死进去，
  例如 `Sm80Fna...`、`Sm89...`、`SM90...`。
- 架构、dtype、tile shape 应该作为模板参数或显式 policy 选择出现；架构特化 alias
  只能是薄 convenience，主实现仍必须模板化。
- device/launcher 层负责把模板 policy 映射到底层 CUTLASS TensorOp 或生成 kernel
  实例，并在该层显式检查 arch、dtype、layout、shape 是否受支持。
- TensorOp 实验不允许悄悄退回 SIMT 或 raw CUDA fallback；不支持的配置必须通过
  CMake、assert、`cutlass::Status` 或 CUDA error 明确失败。
- core runtime 入口应朝 raw device pointer、problem descriptor、`cudaStream_t`
  收敛，不要把 ATen/Torch tensor ownership 带进核心 kernel 设计。
