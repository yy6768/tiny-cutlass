# NATTEN 工作区

这里是 tiny-cutlass 中用于学习和重写 NATTEN Fused Neighborhood Attention
(FNA) 前向的独立工作区。当前目标不是把上游 NATTEN 原样搬进来，而是
保留一个干净、模板化、CUTLASS 原生的实现入口。

## 当前范围

- 当前只有接口骨架：`fna_forward.h`、一个接口冒烟测试、中文说明文档。
- 当前没有真实 CUDA kernel、设备端启动器或公开运行时入口。
- 不引入 PyTorch、ATen 或 pybind 扩展层；核心设计后续应收敛到
  原始设备指针、`FnaForwardProblem`、`cudaStream_t`。
- 不保留上游的生成 kernel 符号、自定义 MMA、自定义 iterator、
  自定义 epilogue 或本地 CUDA/CUTLASS helper 封装。
- 不允许用伪造状态返回路径表示“未实现但可调用”。没有真实 TensorOp kernel 前，
  不应该暴露 `run`/`launch` 之类的运行入口。

## 文件结构

- `include/natten/fna_forward.h`: FNA 前向的模板 policy 和问题描述符。
- `../../tests/natten/natten_fna_test.cu`: 验证接口形态、问题字段传播，以及最小
  1D non-causal host reference 冒烟 case。
- `CMakeLists.txt`: 只构建 `natten_fna_test`。
- `AGENTS.md`: NATTEN 子工作区后续修改约束。
- `PAPERS.md`: NATTEN/NAT/FNA 相关论文导读，以及和 Swin 的对照。
- `ROADMAP.md`: 从接口骨架推进到 correctness、benchmark、profiling 的实现路线图。

## 当前接口

`fna_forward.h` 只保留三个概念：

| 概念 | 当前名字 | 说明 |
| --- | --- | --- |
| 模板 policy 工厂 | `DefaultFnaForwardPolicy<Rank, CausalMask, Element, ArchTag, ThreadblockShape>` | rank、mask、dtype、arch、tile shape 都是模板参数，不进入主类名字。 |
| causal mask tag | `FnaCausalMask<IsCausal>` | 只表达 mask 语义，不做分发。 |
| 问题描述符 | `FnaForwardProblem` | 保存 batch、length、heads、head_dim、kernel_size、stride、dilation、tile size、scale 等归一化元数据。 |

## 和原版 General NATTEN 接口对比

| 概念 | 原版 NATTEN general FNA forward | tiny-cutlass 当前状态 |
| --- | --- | --- |
| PyTorch binding | `na1d_forward`、`na2d_forward`、`na3d_forward` 接收 ATen Tensor 并校验 PyTorch 元数据。 | 不复制 binding 层；当前没有 ATen/Torch tensor ownership。 |
| 统一入口 | `fna_generic_forward<StdNADim, StdCausal>` 统一推导 shape、dtype、device、stream、workspace allocator 和分发元数据。 | 暂不提供统一入口；只保留已经归一化后的 `FnaForwardProblem` 和模板 policy。 |
| 设备端分发 | `fna_forward_generic` 通过自动生成分发和生成实例符号进入具体 kernel。 | 暂不提供设备端分发；未来必须是模板化启动器，调用 CUTLASS 原生 TensorOp/FMHA 能力。 |
| kernel policy | 原版生成实例把 rank、causal mask、dtype、arch、alignment、tile shape 绑定到具体 kernel。 | `DefaultFnaForwardPolicy` 保持模板工厂风格，禁止主 API 名字写死架构。 |
| MMA/iterator/epilogue | 原版旧路径带有 NATTEN 自定义 MMA、iterator、epilogue，且部分来自对 CUTLASS 内部的修改。 | 不复制这些文件；未来实现必须优先使用 `3rdparty/cutlass` 已提供的接口。 |
| 当前正确性状态 | 原版在完整 autogen/CUTLASS 栈存在时可以分发真实 kernel。 | 当前只是接口骨架，没有可运行 kernel，也没有伪造的未支持运行时路径。 |

## 验证和性能

脚本入口：

```bat
scripts\kernels\natten\natten.bat
```

当前脚本仍遵守 `cutlass-kernel` 的顺序约束：

1. build：配置并构建 `natten_fna_test`。
2. verify：运行接口冒烟测试和 1D non-causal host reference 冒烟测试，确认
   policy/problem 形态与基础窗口语义没有漂移。
3. bench：当前没有真实 kernel，因此不 benchmark、不 profile、不产生性能结论。

当前 host reference 只覆盖小型 uniform-softmax case，作用是固定窗口、dilation 和
`logsumexp` 的基础语义。后续添加真实 FNA kernel 时，必须先补完整 reference parity：

- reference 可以来自 PyTorch/NATTEN 官方实现或清晰可审计的 host reference。
- 先比较输出和 `logsumexp` 等中间量，再讨论性能。
- verify 未通过时，`.bat` 必须停止，不能继续 benchmark。
- profiling artifact 只能落在 `build/` 下，例如 `build/reports/natten`。

## 和 Swin 工作区的对齐方向

NATTEN 后续要和 `csrc/swin` 保持相近风格：

- 一个 kernel family 一个脚本入口：`scripts/kernels/natten/natten.bat`。
- 测试放在 `csrc/tests/natten`，不要散在 kernel 目录里。
- 文档先说明主路径、接口对照、reference gate、当前约束。
- 真实 device 层必须调用 CUTLASS TensorOp/FMHA 级别能力，不新增 raw CUDA fallback。
- Swin 现在是完整路径 + cuDNN reference；NATTEN 当前只是接口骨架，这个差异需要在后续
  实现路线图中逐步收敛。
