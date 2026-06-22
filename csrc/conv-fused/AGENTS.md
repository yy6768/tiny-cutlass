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
- `kernel/` 层只产出 `CutlassKernel`，不 include CUTLASS device adapter、TensorRT、Torch 或 reduction device adapter；B2B fusion 的 device operator 由 family-specific `device/` class 直接包 `cutlass::conv::device::B2bImplicitGemmConvolution<Kernel>`。
- 即使当前默认实例按 family 分成 legacy fp16/Sm80、FP8 e4m3/Sm89，也必须通过模板参数控制 arch 和 element type，默认实例只做薄实例化。
- 不允许 SIMT 或 raw-kernel fallback；SM、dtype、layout 或 problem shape 不支持时必须通过 CMake、assert 或 `cutlass::Status` 明确报错。
- 长期 runtime 入口必须朝 raw device pointer、problem descriptor、`cudaStream_t` 收敛，不能把 ATen tensor ownership 带进 core。
- 构建输出只允许在 `build/`。
- 参考验证放在 `csrc/tests/`。
- 测试可以用 PyTorch/ModelOpt/TensorRT 做外部 reference，但 C++ 执行段必须脱离 Torch。
- `csrc/tests/<family>/` 下的测试文件和 CMake target 使用短名，不再添加 `_test` 后缀；目录已经表达 test 语义。
- 验证脚本统一命名为 `verify.py`，性能脚本统一命名为 `bench.py`，不要新增 `compare_*.py` 或 `run_*_reference.py` 这类临时名字。

## 已知设计

- `ops/conv1x1_relu_conv1x1.{h,cu}` 是默认 core API，名字只描述 `conv1x1 -> relu -> conv1x1`。
- `conv_fused_core` 是默认构建目标；不要重新引入 `conv_fused_torch`。
- `conv1x1_relu_conv1x1` 是非 FP8 C++ raw-pointer 测试入口，不加载 Python extension。
- `conv1x1_relu_conv1x1_relu_trt` 是 FP8 ours runner：读取 `build/conv-fused/fp8-reference` 下的 ModelOpt artifacts，构建 TensorRT IPluginV3 network，plugin 输出 kFP8 后接 TensorRT `IDequantizeLayer`，最终导出 half `ours_output.bin`。
- `device/conv1x1_relu_conv1x1.{h,cu}` 负责 device wrapper。
- `device/conv1x1_relu_conv1x1.{h,cu}` 持有 family-specific CUTLASS-style device operator；它必须直接包 example 13 的 `B2bImplicitGemmConvolution<Kernel>`，不要恢复本地复刻的 generic device wrapper。
- `kernel/conv1x1_relu_conv1x1.h` 负责 kernel 类型组合和 CTA/warp shape 默认值。
- `fp8/conv1x1_relu_conv1x1_relu_fp8/` 使用同样分层，不能把外部框架 binding、layout 准备和 CUTLASS 参数组装混在一个 `.cu` 文件里。
- FP8 back-to-back conv 也必须直接包 example 13 的 `B2bImplicitGemmConvolution<Kernel>`；不要恢复本地复刻的 generic device wrapper，也不要新增只定义 alias 的 device 文件。
- `binding/tensorrt/conv1x1_relu_conv1x1_relu.{h,cpp}` 是 FP8 TensorRT IPluginV3 绑定层；它的 runtime 输入 contract 是 `input/weight0/weight1/bias1` 为 E4M3 raw bytes，`stage0_scale/bias0` 为 FP32，`output_alpha` 通过 plugin field 固定；当前插件不为 stage0 暴露或申请 TensorRT workspace，`getWorkspaceSize()` 返回 0。

## 已知问题

- 非 FP8 path 当前只显式实例化 `cutlass::half_t`；FP8 family 独立显式实例化 `cutlass::float_e4m3_t`。
- `build/conv-fused` 之前曾被 Debug 缓存污染，必须用 `CMAKE_BUILD_TYPE=Release` 重新配置。
- 非 FP8 path 当前只接受 CUTLASS optimized iterator 能实现的 channel 规模；小 channel/unaligned case 必须由 CUTLASS `can_implement` 返回错误，不要用本地 helper 或 SIMT fallback 覆盖。
- fused path 仍有 `hidden_channels <= ThreadblockShape0::kN`、`output_channels <= ThreadblockShape1::kN` 的 policy 约束。
- 未来测试组织优先参考 `3rdparty/cutlass/examples/41_fused_multi_head_attention`：C++ executable/testbed 管理 host/device allocation 和 raw-pointer invocation，Python 只做 reference/driver。
- 用户目标运行环境可能是和 DX12 interop 的普通 CUDA 环境；不要把 PyTorch 作为核心设计假设。
- 本机 TensorRT 10/cu12 安装在 `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT`；headers 在 `include`，import libs 与 runtime DLL 都在 `lib`，`trtexec.exe` 在 `bin`。运行 TensorRT 测试时需要把 `...\TensorRT\lib` 加到 PATH。
- 当前 Python 是 3.11，但没有安装 `tensorrt` package；本机 wheel 在 `...\TensorRT\python\tensorrt-10.13.3.9-cp311-none-win_amd64.whl`。C++ plugin/test 不依赖 Python TensorRT，Python reference 先走 PyTorch export + `trtexec`。
- ModelOpt 正确 pip 包名是 `nvidia-modelopt`，不是 `modelopt`。当前可用组合是 `nvidia-modelopt==0.43.0`、`torch==2.7.1+cu126`、`numpy==1.24.3`、`scipy==1.11.3`；不要直接升级到 `nvidia-modelopt==0.44.0`，它会拉取 PyPI `torch==2.12.0`，在当前 Windows/conda 环境下触发 `c10.dll` 初始化失败。
- 当前 FP8 ONNX reference 只需要 `modelopt.torch.quantization` 提供校准 scale，并用基础 `onnx` package 手工建图；不要为了这条链路默认安装完整 `nvidia-modelopt[onnx]`，它会额外拉 ONNXRuntime/CuPy 等依赖。只有需要 `modelopt.onnx` post-process native QDQ 时，再单独补缺失依赖并记录原因。
- 当前 `conv1x1_relu_conv1x1_relu` FP8 reference 没有继续依赖 `torch.onnx.export`：ModelOpt 的 FP8 custom QDQ 让 Torch exporter 在 Conv symbolic 阶段把 shape 变成 unknown，报 `Unsupported: ONNX export of convolution for kernel of unknown shape`。现阶段脚本用 ModelOpt 校准出的 amax/scale 手工生成 TensorRT/ModelOpt 风格 `trt::TRT_FP8QuantizeLinear/DequantizeLinear` ONNX。
- TensorRT FP8 QDQ ONNX reference 必须同时传 `trtexec --fp8 --fp16`；只传 `--fp8` 会因为 graph 内 high-precision tensor 是 FP16 而报 builder 没启用 fp16。
- TensorRT 10.13 构建 FP8 QDQ reference 时可能打印 `Skipping tactic` / `Unsupported data type FP8` 的 tactic warning；只要 `trtexec` 最终 `PASSED` 且 `verify.py --mode fp8` 通过，不要把这些 skipped tactic 当成 workflow 失败。
- 如果 ModelOpt import 报 `numpy.dtype size changed`，优先检查 NumPy 是否被升级到 2.x；本环境需要回到 NumPy 1.24.x/SciPy 1.11.x。
- ModelOpt import 可能警告 triton/HF plugin 不可用；当前 conv-fused PTQ/export 路线只依赖 `modelopt.torch.quantization`，不依赖这些 plugin。
- TensorRT 10.13 的 plugin registry C API 是全局 `getPluginRegistry()`，不是 `nvinfer1::getPluginRegistry()`。
- `conv_relu_pool` staged baseline 已从正式 core/build/test/plugin 路径删除。后续不要用 TensorRT layer、raw CUDA、SIMT fallback 或 staged global-memory pipeline 假装 ours。
- CUTLASS example 13 的 B2B conv `can_implement` 要求两段 implicit GEMM 的 `M` 相同；`2x2 stride2 maxpool` 会把第二段输入空间降为四分之一，所以不能直接把现有 `DefaultB2bConv2dFprop` 当成 conv+pool fusion。
- example 41 的 `downsample_load_with_byte_offset` 只提供空间地址映射，不提供 `2x2 max` reduction；不能把它当作 maxpool 实现。

## 约定

- 新增 family 相关文件时，先写清楚职责再补实现。
- 如果需要记录新坑，只追加到这里，不回填 README。
- 学习文档放在 `docs/`，从 `00-overview.md` 开始编号；实际修改流水放在
  `STATUS.md`。README 只放稳定设计。
