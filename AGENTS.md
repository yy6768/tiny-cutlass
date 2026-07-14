# tiny-cutlass Agent 指南

这个文件记录整个仓库都会影响后续 agent 修改的规则。子工作区可以有自己的
`AGENTS.md`，但不能覆盖这里的全局约束。

## 子工作区索引

- `csrc/flash-attention/AGENTS.md`
- `csrc/conv-fused/AGENTS.md`
- `csrc/conv-fused/fp8/conv1x1_relu_conv1x1_relu_fp8/AGENTS.md`
- `csrc/natten/AGENTS.md`
- `csrc/swin/AGENTS.md`

## 全局规则

- 做 CUDA/CUTLASS kernel 编程时，必须先使用 `cutlass-kernel` skill。
- 每个 kernel family 优先提供一个脚本入口：
  `scripts/kernels/<family>/<family>.bat`。
- `.bat` 工作流顺序必须是 build -> verify -> bench；verify 失败时不要 benchmark。
- 构建输出只能放在 `build/`。
- 验证和 benchmark harness 放在 `csrc/tests/`。
- `csrc/tests/<family>/` 下的 C++ 测试文件和 CMake target 使用短名，不再额外添加
  `_test` 后缀；目录已经表达 test 语义。
- 验证脚本统一命名为 `verify.py`，性能脚本统一命名为 `bench.py`；不要新增
  `compare_*.py`、`run_*_reference.py` 或 `*_perf.py` 这类临时名字。
- 先对齐可信 reference，再讨论性能；未通过 reference parity 的数据不能当作性能结论。
- CUTLASS kernel policy primary type 必须保持模板工厂风格，例如
  `DefaultXxx<ArchTag, Element..., ThreadblockShape..., WarpShape...>`。
- 算子元素类型、架构、layout、tile shape 必须通过模板参数、policy 参数、CMake 配置
  或显式实例化表达；dtype/layout/arch 不进入 primary operator、target、测试文件或
  公共 API 名字，除非它是不可模板化的真实 family 边界。
- 不要新增 primary class、struct、function、target 或公共 API 名字把架构写死进去，
  例如 `Sm80Fna...`、`Sm89...`、`SM90...`。
- 不要新增把 dtype、layout 或架构写进名字的 concrete alias、launcher、target、
  测试名或公共 API，例如 `XxxFp16`、`XxxBF16`、`XxxNHWC`、`XxxRowMajor`、
  `XxxSm80`。
- 不要用 `using OldXxx = NewTemplatedXxx<...>` 这类旧名 alias 来支持 fallback
  兼容；需要把调用点改成模板/policy 形式。
- 架构、dtype、tile shape 应该作为模板参数或显式 policy 选择出现；具体选择只应
  留在 launcher、显式实例化、benchmark/test 或 CMake 配置层。
- device/launcher 层负责把模板 policy 映射到底层 CUTLASS TensorOp 或生成 kernel
  实例，并在该层显式检查 arch、dtype、layout、shape 是否受支持。
- TensorOp 实验不允许悄悄退回 SIMT 或 raw CUDA fallback；不支持的配置必须通过
  CMake、assert、`cutlass::Status` 或 CUDA error 明确失败。
- core runtime 入口应朝 raw device pointer、problem descriptor、`cudaStream_t`
  收敛，不要把 ATen/Torch tensor ownership 带进核心 kernel 设计。
