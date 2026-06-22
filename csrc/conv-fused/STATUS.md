# conv-fused STATUS

这个文件记录本轮重构的实际修改流水。稳定设计放在 `README.md`，约束和坑放在
`AGENTS.md`，这里记录“我具体改了什么”。

## 2026-06-18

- 建立 `docs/00-overview.md` 到 `docs/05-porting-plan.md` 的学习文档骨架。
- 文档重点改为读取 CUTLASS example 13 的 SM80 RF fused convolution，不再只描述
  tiny-cutlass 当前代码。
- 明确 RF fusion 的含义：中间 accumulator 留在 register file；A/B operand 仍然
  可以使用 shared memory pipeline。
- 明确 `conv_relu_pool` 当前 staged baseline 不是 monolithic RF fusion，后续必须
  删除、隔离或重写为真实 threadblock-level fusion。
- 开始把 `conv1x1_relu_conv1x1` device 层重构为 CUTLASS-style device operator。
- 删除 staged `conv_relu_pool` 的 core source、kernel policy、TensorRT binding 和
  对应 C++ tests；它不是 example 13 风格的 RF/threadblock fusion。
- `conv-fused` build workflow 不再构建或运行 `conv_relu_pool*` targets。
- FP8 `conv1x1_relu_conv1x1_relu` device 层也改成 CUTLASS-style class，不再以
  standalone `run_*` free function 作为 device 入口。
- `conv-fused.bat` 移除 staged pool workflow，并调整 FP8 顺序为 build -> 生成
  reference/candidate -> verify -> bench。
- 已验证 `cmake --build build\conv-fused --target conv_fused_core --config Release`
  通过；测试 executable 链接仍受当前 shell 缺少 Windows SDK `kernel32.lib` 路径影响，
  需要在完整 VS/SDK 环境中运行。
- 删除本地复刻的 `device/implicit_gemm_convolution_fusion.h`，family device class
  直接包 CUTLASS example 13 的 `cutlass::conv::device::B2bImplicitGemmConvolution`。
- 将非 FP8 C++ 测试从误导性的 `conv_pool` 改名为 `conv1x1_relu_conv1x1`，同步
  CMake target、`.bat` workflow、README 和 AGENTS 记忆。
- 新增 `docs/06-source-walk.md`，按源码顺序走读 SM80 RF fused conv：入口 policy
  选择、官方 device operator、RF kernel policy、threadblock fragment loader。
- 重新构建 `conv_fused_core conv1x1_relu_conv1x1 conv1x1_relu_conv1x1_relu`：CUDA
  编译和 `conv_fused_core.lib` 生成通过；两个 executable 仍在链接阶段因当前 shell
  缺少 Windows SDK `kernel32.lib` 的 LIB 路径失败，不是 CUTLASS 模板编译错误。
- 清理 FP8 family 的 dtype 后缀公共 API：删除 `E4m3Arguments` alias 和
  `conv1x1_relu_conv1x1_relu_fp8(...)` 兼容 wrapper；kernel policy factory 改为
  `DefaultConv1x1ReluConv1x1Relu<ArchTag, ...>`。
- 重新构建 `conv_fused_core` 通过，确认删除 wrapper 和改名后的 FP8 显式实例化能编译。
- 通过 `scripts\kernels\conv-fused\conv-fused.bat`（设置
  `CONV_FUSED_SKIP_BENCH=1`、`CONV_FUSED_SKIP_FP8=1`）在 VS DevCmd 环境中完成
  build 和 C++ smoke：`conv1x1_relu_conv1x1.exe`、`conv1x1_relu_conv1x1_relu.exe`
  均通过；`conv_fused_plugin.dll` 和 `conv1x1_relu_conv1x1_relu_trt.exe` 也成功链接。
- 复审 example 13 分层后确认当前非 FP8 主线已经改成 `ops -> device -> kernel ->
  threads`：`kernel/` 只产出 `DefaultB2bConv2dFprop<...>::Kernel`，`device/`
  直接包官方 `cutlass::conv::device::B2bImplicitGemmConvolution<Kernel>` 并提供
  `can_implement/get_workspace_size/initialize/run/operator()`。
- 复审 FP8 TensorRT plugin：当前 `getWorkspaceSize()` 返回 0，enqueue 只把 TensorRT
  tensor 指针转成 core raw-pointer `Arguments`；修正 `AGENTS.md` 中旧的 stage0
  workspace 记忆。
- 修正 FP8 README 的 runtime contract 描述：`bias1` 当前按 e4m3 final epilogue
  source 输入，`bias0/stage0_scale` 是 float。
- 重新运行 `python -m py_compile scripts\kernels\conv-fused\bench.py
  scripts\kernels\conv-fused\verify.py scripts\kernels\conv-fused\export_fp8_reference.py`
  通过。
- 重新运行 `scripts\kernels\conv-fused\conv-fused.bat`（设置
  `CONV_FUSED_SKIP_BENCH=1`、`CONV_FUSED_SKIP_FP8=1`）通过：CMake/Ninja 无需增量构建，
  `conv1x1_relu_conv1x1.exe` 的 unaligned reject 与 4 个 aligned correctness case
  全部通过，`conv1x1_relu_conv1x1_relu.exe` 的 3 个 FP8 smoke case 全部通过。
