# conv-fused

这个目录的目标是实现 `tiny-cutlass` 的 CUTLASS-native conv-fused family。最终运行环境不是 PyTorch extension，而是可以在普通 CUDA runtime 中被调用的 NHWC raw-pointer 算子入口。

## 设计

稳定目标只有一条主线：

```text
NHWC input / KRSC weights / NHWC output
  -> raw device pointers + problem descriptors + cudaStream_t
  -> CUTLASS implicit-GEMM conv1x1 -> relu -> conv1x1
```

默认 runtime 不依赖 PyTorch，也不提供 PyTorch binding。验证入口由 C++ executable 自己持有 host/device allocation，直接调用 core API。参考实现可以来自 PyTorch、TensorRT 或 ModelOpt 导出的外部流程，但不能把 ATen tensor ownership 带进本目录的可运行入口。

### CUTLASS 思路

这不是“自己写一个卷积 kernel”，而是“用 CUTLASS 的积木拼一个固定 family”。

```text
Kernel
  -> CTA / threadblock
    -> Warp
      -> Threads
```

- `Kernel` 层决定用哪一个 CUTLASS kernel 组合模板。
- `CTA` 层决定一个 threadblock 怎么做两段 implicit-GEMM、怎么走 shared memory、怎么串起两次卷积。
- `Warp` 层只固定 warp 的 tile 形状。
- `Threads` 层只固定 epilogue 的逐线程算子。

本目录不为单个 `GemmShape` 包一层 traits。真实 kernel policy 应该是 `DefaultXxx<ArchTag, Element..., ThreadblockShape..., WarpShape...>` 这样的模板工厂，直接把 CUTLASS shape 类型作为模板参数；只有多架构或多 dtype 映射真的复杂时才新增 traits。不能把 primary implementation 写成 `SomethingSm89` 这类 concrete struct。真正的通用能力来自 CUTLASS 自己，比如 `GemmShape`、`DefaultMmaCore`、`MmaTensorOpFragmentIterator`、`GemmIdentityThreadblockSwizzle`、`TensorRef`、`LinearCombination` 这一套。

### 当前 family 约束

1. 卷积主干必须走 CUTLASS implicit-GEMM。
2. 不写 raw CUDA kernel。
3. 默认 runtime 不绑定 PyTorch；本目录不再构建 pybind/ATen adapter。
4. 结构分层必须清楚，职责不能交叉。
5. 当前构建目标固定 SM89；CMake 配置不是 `CMAKE_CUDA_ARCHITECTURES=89` 时直接失败。kernel policy 仍用模板控制 `ArchTag` 和 element type；当前默认实例是 fp16/Sm80 policy 与 fp8/Sm89 policy。
6. 不提供 SIMT 或 raw-kernel fallback；SM 或 problem shape 不支持时只允许返回明确错误状态。
7. 不维护自定义 arch、storage、swizzle helper；如果 CUTLASS 已经有对应类型或函数，直接使用 CUTLASS 的实现。

### 文件职责

- `ops/conv1x1_relu_conv1x1.{h,cu}`: 默认 core API，接收 raw device pointer、problem descriptor、`cudaStream_t`，调用 device wrapper。
- `device/conv1x1_relu_conv1x1.{h,cu}`: device 层，负责把 `Conv2dProblemSize` 和裸指针喂给 CUTLASS device wrapper。
- `kernel/conv1x1_relu_conv1x1.h`: kernel 层，负责组装 `DefaultB2bConv2dFprop` 的模板参数，包括 CTA/warp shape 默认值。
- `threads/epilogue_ops.h`: threads 层的 epilogue 别名，只固定逐线程输出算子。
- `fp8/conv1x1_relu_conv1x1_relu_fp8/ops/conv1x1_relu_conv1x1_relu_fp8.{h,cu}`: FP8 core API，接收 e4m3 raw device pointer、scale/bias 指针和 `cudaStream_t`。
### 核心结构

- `ops` 只放无 ATen 的 core API。
- `device` 只管 CUTLASS device wrapper。
- `kernel` 只负责模板装配。
- CTA 和 warp shape 当前直接作为 `Default...` factory 的模板参数；没有额外逻辑时不单独建 traits 文件。
- `threads` 对应 CUTLASS 的逐线程 epilogue 层。

### 复用点

这里优先复用 CUTLASS 自带的能力，而不是自己造同义工具：

- `cutlass::gemm::GemmShape`
- `cutlass::gemm::threadblock::DefaultMmaCore`
- `cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle`
- `cutlass::conv::kernel::DefaultB2bConv2dFprop`
- `cutlass::conv::device::B2bImplicitGemmConvolution`
- `cutlass::epilogue::thread::LinearCombination`
- `cutlass::epilogue::thread::LinearCombinationRelu`

### 现状

- 当前 `float16` 主路径只走 CUTLASS back-to-back TensorOp implicit-GEMM。
- 非 FP8 path 要求 `channels`、`hidden_channels`、`output_channels` 满足 TensorOp vector access 对齐；不满足时返回 `kErrorInvalidProblem`，不降级。
- `fp8/conv1x1_relu_conv1x1_relu_fp8/` 是独立 FP8 family，保持 `ops/device/kernel/threads` 分层；只有出现真实 CTA 或 warp 逻辑时才新增 `threadblock/` 或 `warp/`。
- FP8 family 当前只支持 CUTLASS `cutlass::float_e4m3_t`，目标硬件是 SM89。

### 测试方向

当前测试入口参考 CUTLASS example 41 的方向：C++ executable/testbed 自己持有 host/device allocation、构造 problem descriptors、调用 raw-pointer kernel。Python、PyTorch、TensorRT 或 ModelOpt 可以作为外部 reference/driver，但不是主运行时，也不通过本目录的 binding 进入 kernel。

- `conv_pool_test`: 覆盖非 FP8 aligned fused conv 正确性，以及 unaligned problem 被明确拒绝。
- `conv_fused_relu_fp8_test`: 覆盖 FP8 e4m3 raw-pointer smoke case。
