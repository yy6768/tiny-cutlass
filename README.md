# tiny-cutlass

这是一个 CUDA/CUTLASS 学习仓库，用来拆解 attention、卷积融合、Swin、NATTEN
等 kernel family 的实现、验证和 profiling 流程。

## 工作区

- `csrc/flash-attention`: FlashAttention 学习实现和 cuDNN reference 验证。
- `csrc/conv-fused`: CUTLASS implicit GEMM 风格的卷积融合实验。
- `csrc/swin`: Swin WindowAttention 主路径和 cuDNN SDPA 对照。
- `csrc/natten`: NATTEN/FNA 的接口骨架、论文导读和后续实现路线。
- `csrc/tests`: correctness、benchmark 和 reference harness。
- `scripts/kernels`: 每个 kernel family 的 `.bat` 工作流入口。
- `profiling`: Nsight Compute/Systems profiling 脚本。

## 全局约束

修改 CUDA/CUTLASS kernel 时先阅读根目录 `AGENTS.md`。核心原则是：

- 每个 kernel family 优先维护一个 `.bat` 入口。
- 工作流顺序是 build -> verify -> bench；verify 失败时不能 benchmark。
- 构建和 profiling 产物只放在 `build/`。
- 性能结论必须建立在 reference parity 之上。
- CUTLASS policy 保持模板工厂风格，不把架构写进 primary API 名字。
