# conv-fused 记忆

这个文件只记会影响后续修改的事实、约束和已知坑，不记临时构建状态。

## 约束

- `conv-fused` 依赖仓库内的 `3rdparty/cutlass`。
- 不允许 raw CUDA kernel。
- 卷积主干必须是 CUTLASS implicit-GEMM。
- 默认 runtime 不绑定 PyTorch：core header/source/CMake 不允许依赖 `torch`、ATen、pybind，也不再保留 `conv_fused_torch` adapter。
- 命名保持简洁：文件、target、函数只描述模块组成；默认 layout 和 dtype 是 contract 或模板参数，不进入新 core 名字。
- kernel policy primary type 必须是 CUTLASS-style 模板工厂，例如 `DefaultXxx<ArchTag, Element..., ThreadblockShape..., WarpShape...>`；不要新增 `SomethingSm89` 这种把架构写进 primary struct 名的实现。
- 不要为单个 `cutlass::gemm::GemmShape` 新增 threadblock/warp traits 文件；shape 简单时直接作为 `Default...` 模板参数默认值。
- 不要为 CUTLASS 已经存在的 arch、storage、swizzle、layout、TensorRef packing 能力写本地函数或长名字 wrapper；直接使用 CUTLASS 类型和函数。
- 即使当前默认固定 fp16/Sm80 与 fp8/Sm89，也必须通过模板参数控制 arch 和 element type，默认实例只做薄实例化。
- 不允许 SIMT 或 raw-kernel fallback；SM、dtype、layout 或 problem shape 不支持时必须通过 CMake、assert 或 `cutlass::Status` 明确报错。
- 长期 runtime 入口必须朝 raw device pointer、problem descriptor、`cudaStream_t` 收敛，不能把 ATen tensor ownership 带进 core。
- 构建输出只允许在 `build/`。
- 参考验证放在 `csrc/tests/`。
- 测试可以用 PyTorch/ModelOpt/TensorRT 做外部 reference，但 C++ 执行段必须脱离 Torch。

## 已知设计

- `ops/conv1x1_relu_conv1x1.{h,cu}` 是默认 core API，名字只描述 `conv1x1 -> relu -> conv1x1`。
- `conv_fused_core` 是默认构建目标；不要重新引入 `conv_fused_torch`。
- `conv_pool_test` 是 C++ raw-pointer 测试入口，不加载 Python extension。
- `device/conv1x1_relu_conv1x1.{h,cu}` 负责 device wrapper。
- `kernel/conv1x1_relu_conv1x1.h` 负责 kernel 类型组合和 CTA/warp shape 默认值。
- `fp8/conv1x1_relu_conv1x1_relu_fp8/` 使用同样分层，不能把外部框架 binding、layout 准备和 CUTLASS 参数组装混在一个 `.cu` 文件里。

## 已知问题

- 当前 kernel 只支持 `float16`。
- `build/conv-fused` 之前曾被 Debug 缓存污染，必须用 `CMAKE_BUILD_TYPE=Release` 重新配置。
- 非 FP8 path 当前只接受 CUTLASS optimized iterator 能实现的 channel 规模；小 channel/unaligned case 必须由 CUTLASS `can_implement` 返回错误，不要用本地 helper 或 SIMT fallback 覆盖。
- fused path 仍有 `hidden_channels <= ThreadblockShape0::kN`、`output_channels <= ThreadblockShape1::kN` 的 policy 约束。
- 未来测试组织优先参考 `3rdparty/cutlass/examples/41_fused_multi_head_attention`：C++ executable/testbed 管理 host/device allocation 和 raw-pointer invocation，Python 只做 reference/driver。
- 用户目标运行环境可能是和 DX12 interop 的普通 CUDA 环境；不要把 PyTorch 作为核心设计假设。

## 约定

- 新增 family 相关文件时，先写清楚职责再补实现。
- 如果需要记录新坑，只追加到这里，不回填 README。
