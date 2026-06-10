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
- 即使当前默认实例按 family 分成 legacy fp16/Sm80、`conv_relu_pool` fp16/Sm89、FP8 e4m3/Sm89，也必须通过模板参数控制 arch 和 element type，默认实例只做薄实例化。
- 不允许 SIMT 或 raw-kernel fallback；SM、dtype、layout 或 problem shape 不支持时必须通过 CMake、assert 或 `cutlass::Status` 明确报错。
- 长期 runtime 入口必须朝 raw device pointer、problem descriptor、`cudaStream_t` 收敛，不能把 ATen tensor ownership 带进 core。
- 构建输出只允许在 `build/`。
- 参考验证放在 `csrc/tests/`。
- 测试可以用 PyTorch/ModelOpt/TensorRT 做外部 reference，但 C++ 执行段必须脱离 Torch。
- `csrc/tests/<family>/` 下的测试文件和 CMake target 使用短名，不再添加 `_test` 后缀；目录已经表达 test 语义。
- 验证脚本统一命名为 `verify.py`，性能脚本统一命名为 `bench.py`，不要新增 `compare_*.py` 或 `run_*_reference.py` 这类临时名字。

## 已知设计

- `ops/conv1x1_relu_conv1x1.{h,cu}` 是默认 core API，名字只描述 `conv1x1 -> relu -> conv1x1`。
- `ops/conv_relu_pool.{h,cu}` 是 `conv3x3 -> relu -> pool -> conv1x1 -> relu -> pool` raw-pointer contract；`device/conv_relu_pool.{h,cu}` 是 staged CUTLASS device wrapper，串联两次 implicit-GEMM conv 和两次 CUTLASS reduction pool。
- `kernel/conv_relu_pool.h` 持有 `DefaultConvRelu<ArchTag, Element, ...>` 与 `DefaultPool<Element, ...>` 模板工厂；不要把它拆成只有一个 `GemmShape` 的 traits 文件。
- `conv_fused_core` 是默认构建目标；不要重新引入 `conv_fused_torch`。
- `conv_pool` 是 C++ raw-pointer 测试入口，不加载 Python extension。
- `conv_relu_pool` 覆盖 staged conv/pool core small correctness、missing workspace、bad shape 和 null pointer。
- `conv_relu_pool_plugin` 覆盖小规模 TensorRT IPluginV3 构图、序列化/反序列化和 plugin vs direct core 一致性。
- `conv_relu_pool_trt` 是 full-shape ours runner：读取 `build/conv-fused/reference` 下的 PyTorch artifacts，构建 TensorRT IPluginV3 network，导出 `ours_output.bin` 和 `ours_times.json`。
- `device/conv1x1_relu_conv1x1.{h,cu}` 负责 device wrapper。
- `kernel/conv1x1_relu_conv1x1.h` 负责 kernel 类型组合和 CTA/warp shape 默认值。
- `fp8/conv1x1_relu_conv1x1_relu_fp8/` 使用同样分层，不能把外部框架 binding、layout 准备和 CUTLASS 参数组装混在一个 `.cu` 文件里。
- `binding/tensorrt/conv_relu_pool.{h,cpp}` 是 TensorRT IPluginV3 绑定层，不属于 core kernel；`conv_fused_core` 不能依赖 TensorRT。

## 已知问题

- 当前 kernel 只支持 `float16`。
- `build/conv-fused` 之前曾被 Debug 缓存污染，必须用 `CMAKE_BUILD_TYPE=Release` 重新配置。
- 非 FP8 path 当前只接受 CUTLASS optimized iterator 能实现的 channel 规模；小 channel/unaligned case 必须由 CUTLASS `can_implement` 返回错误，不要用本地 helper 或 SIMT fallback 覆盖。
- fused path 仍有 `hidden_channels <= ThreadblockShape0::kN`、`output_channels <= ThreadblockShape1::kN` 的 policy 约束。
- 未来测试组织优先参考 `3rdparty/cutlass/examples/41_fused_multi_head_attention`：C++ executable/testbed 管理 host/device allocation 和 raw-pointer invocation，Python 只做 reference/driver。
- 用户目标运行环境可能是和 DX12 interop 的普通 CUDA 环境；不要把 PyTorch 作为核心设计假设。
- 本机 TensorRT 10/cu12 安装在 `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT`；headers 在 `include`，import libs 与 runtime DLL 都在 `lib`，`trtexec.exe` 在 `bin`。运行 TensorRT 测试时需要把 `...\TensorRT\lib` 加到 PATH。
- 当前 Python 是 3.11，但没有安装 `tensorrt` package；本机 wheel 在 `...\TensorRT\python\tensorrt-10.13.3.9-cp311-none-win_amd64.whl`。C++ plugin/test 不依赖 Python TensorRT，Python reference 先走 PyTorch export + `trtexec`。
- TensorRT 10.13 的 plugin registry C API 是全局 `getPluginRegistry()`，不是 `nvinfer1::getPluginRegistry()`。
- `binding/tensorrt/conv_relu_pool.cpp` 的 `enqueue` 已接入 staged CUTLASS core；后续不要再把它改回 unsupported stub，也不要用 TensorRT layer、raw CUDA 或 SIMT fallback 假装 ours。
- `conv_relu_pool` staged baseline 已经能经过 C++ plugin test；它不是最终 monolithic fusion，后续性能结论必须明确区分 staged CUTLASS baseline 和真正单 kernel fusion。
- CUTLASS example 13 的 B2B conv `can_implement` 要求两段 implicit GEMM 的 `M` 相同；`2x2 stride2 maxpool` 会把第二段输入空间降为四分之一，所以不能直接把现有 `DefaultB2bConv2dFprop` 当成 conv+pool fusion。
- example 41 的 `downsample_load_with_byte_offset` 只提供空间地址映射，不提供 `2x2 max` reduction；不能把它当作 maxpool 实现。

## 约定

- 新增 family 相关文件时，先写清楚职责再补实现。
- 如果需要记录新坑，只追加到这里，不回填 README。
