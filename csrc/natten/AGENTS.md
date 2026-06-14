# NATTEN 记忆

这个文件记录 NATTEN 工作区后续修改必须保留的约束和当前事实。

## 当前范围

- 当前只保留 Fused Neighborhood Attention (FNA) forward 的仅头文件接口骨架。
- 这是独立 CUDA/CUTLASS 工作区，不引入 PyTorch、ATen 或 pybind 扩展。
- 不保留上游的 ATen generic binding headers、本地 CUDA/CUTLASS helper shim、
  生成 kernel 符号、自定义 MMA、自定义 iterator 或自定义 epilogue。
- 当前没有 device launcher 或公开运行时入口。不要伪造一个“可调用但未实现”的
  运行时路径。
- 只有在真实 CUTLASS 原生 TensorOp kernel 存在后，才添加启动/支持性检查。
- 测试入口是 `csrc/tests/natten/natten_fna_test.cu`。
- 脚本入口是 `scripts/kernels/natten/natten.bat`，当前只做 build + 接口冒烟测试；
  真实 kernel 完成并通过 reference parity 前不要 benchmark。

## 命名和分层

- tiny-cutlass 公共 API 必须保持模板化：
  `DefaultFnaForwardPolicy<...>`、`FnaForwardProblem`。
- 不要新增把架构写进主 API、类、target、generated symbol 或文件名的实现。
- 架构只能作为 CUTLASS `ArchTag` 模板参数或测试选择出现，不进入 primary name
  或 generated symbol name。
- 增加真实 NATTEN 实例时，先扩展模板 policy，再添加模板化的 device/launcher；
  不要新增具体架构 wrapper function 或把架构写进名字的 wrapper。
- device wrapper 应调用底层 CUTLASS TensorOp kernel，并通过 `cutlass::Status`
  显式拒绝不支持的 arch、dtype、layout、shape。
- 启动/支持性检查优先返回和传播 `cutlass::Status`；不要新增本地
  CUDA/CUTLASS check macro 或 `DeviceKernel` clone。

## 对齐原版 NATTEN 接口

- 原版 general FNA forward 的概念是 `fna_generic_forward` 统一校验 Tensor、
  组装 problem metadata，再分发到 `fna_forward_generic` / autogen dispatch。
- tiny-cutlass 不复制 PyTorch Tensor ownership；对应的核心契约必须是
  原始设备指针 + `FnaForwardProblem` + `cudaStream_t`。
- 原版 `na1d_forward/na2d_forward/na3d_forward` 是 PyTorch binding 层；tiny-cutlass
  只保留 kernel/device 层 scaffold。
- 如果未来需要 2D/3D，先扩展 problem descriptor 和 reference，再扩 kernel policy。
