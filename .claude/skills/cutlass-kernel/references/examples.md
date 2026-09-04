# Anchor examples for the SM80/89 operator pattern

Two upstream CUTLASS examples define the shape every operator in this repository
follows. Paths are relative to the CUTLASS repository root.

## `examples/14_ampere_tf32_tensorop_gemm` — GEMM baseline

The minimal `device`-layer GEMM instantiation:

```cpp
using MMAOp   = cutlass::arch::OpClassTensorOp;
using SmArch  = cutlass::arch::Sm80;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>;
using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 16>;
using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 8>;   // mma.sync tile for tf32

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, ElementComputeEpilogue>;
constexpr int NumStages = 4;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput, ElementAccumulator,
    MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
    EpilogueOp, SwizzleThreadBlock, NumStages>;
```

The host-side call sequence — this is the contract every `device`-layer operator
here must expose:

```cpp
typename Gemm::Arguments arguments{
    problem_size, tensor_a.device_ref(), tensor_b.device_ref(),
    tensor_c.device_ref(), tensor_d.device_ref(), {alpha, beta}, split_k_slices};

size_t workspace_size = Gemm::get_workspace_size(arguments);
cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

Gemm gemm_op;
cutlass::Status status = gemm_op.can_implement(arguments);
status = gemm_op.initialize(arguments, workspace.get());
status = gemm_op();  // == run(); launches the kernel
```

Correctness is checked against `cutlass::reference::device::Gemm<...>` — the same
role a PyTorch or cuDNN reference plays for other families.

## `examples/16_ampere_tensorop_conv2dfprop` — implicit-GEMM conv baseline

Same five-parameter structure, plus conv-specific knobs:

```cpp
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;      // half_t mma.sync tile
constexpr int NumStages = 3;

static auto const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;
static auto const OutputStride      = cutlass::conv::StrideSupport::kUnity;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput, ElementAccumulator,
    MMAOp, SmArch, ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp, SwizzleThreadBlock, NumStages,
    cutlass::arch::OpMultiplyAdd, IteratorAlgorithm, OutputStride>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
```

The conv problem is described independently of the GEMM shape and converted by
CUTLASS, never by hand:

```cpp
cutlass::conv::Conv2dProblemSize problem_size(
    input_size, filter_size, padding, conv_stride, dilation,
    output_size, cutlass::conv::Mode::kCrossCorrelation, split_k_slices);

typename ImplicitGemm::Arguments arguments{
    problem_size, tensor_a.device_ref(), tensor_b.device_ref(),
    tensor_c.device_ref(), tensor_d.device_ref(), {alpha, beta}};
```

`can_implement` / `initialize` / `operator()` follow the identical sequence.
Reference: `cutlass::reference::host::Conv2dFprop<...>`.

`IteratorAlgorithm::kOptimized` requires channel counts aligned to the vectorized
access width (128 bits — 8 elements for `half_t`). Unaligned problems must be
rejected by `can_implement`, not routed to a slower iterator or a SIMT fallback.

## Secondary examples worth knowing

| Example | What it demonstrates |
|---|---|
| `examples/13_two_tensor_op_fusion` | B2B GEMM: second GEMM's operand A stays in registers/smem |
| `examples/36_gather_scatter_fusion` | `GatherA` / `GatherB` / `ScatterD` row remapping |
| `examples/37_gemm_layernorm_gemm_fusion` | Epilogue visitor doing a full reduction inside one N-tile |
| `examples/41_fused_multi_head_attention` | Multi-head attention with from-smem chaining and online softmax |
| `examples/59_ampere_gather_scatter_conv` | The only CUTLASS 3.x Sm80 precedent — cited to explain why 3.x is off the table here, not as a template |
