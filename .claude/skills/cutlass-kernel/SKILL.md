---
name: cutlass-kernel
description: Generic workflow and guardrails for developing, verifying, benchmarking, and profiling CUTLASS-based CUDA kernels across any operator family, including GEMM, convolution, attention, fused epilogues, and related experiments. Use when creating or modifying CUTLASS experiments, writing .bat-driven build/verify/bench flows, adding reference tests under csrc/tests, or capturing Nsight Compute CSV and .ncu-rep artifacts.
---

# CUTLASS Kernel Workflow

## CUTLASS pipeline overview

CUTLASS decomposes one kernel into a fixed hierarchy that mirrors the GPU
execution hierarchy. Every operator family (GEMM, convolution, attention,
fused epilogues) is an instance of the same five layers:

```text
device      host-side operator class: Arguments, can_implement, initialize, run, operator()
  -> kernel     device-side __global__ entry: Params, SharedStorage, the Mma+Epilogue composition
    -> threadblock  CTA-level mainloop: global -> shared memory staging, multistage cp.async pipeline
      -> warp       warp-level MMA: mma.sync / ldmatrix wrappers, operand and accumulator fragment iterators
        -> thread     per-thread math and per-element epilogue ops
```

A `DefaultXxx<ArchTag, Element..., ThreadblockShape, WarpShape,
InstructionShape, EpilogueOp, ...>` factory in the `kernel` layer picks and
assembles the `threadblock`/`warp`/`thread` types for one architecture and
element combination, and exposes the resulting `::Kernel` type. The `device`
layer never contains algorithm code; it only builds `Params` from `Arguments`
and launches `cutlass::Kernel<Kernel><<<grid, block, smem>>>`.

On SM80/SM89, the threadblock mainloop is the multistage `cp.async` pipeline:

```text
tensor in global memory
  --cp_async-->
tile in shared memory
  --smem loads-->
registers
  --mma-->
registers
  --global stores-->
output tensor in global memory
```

`NumStages` (a kernel template parameter, typically 3-5) controls how many
tiles of the mainloop are in flight at once, trading shared memory usage for
latency hiding. This is the deciding architectural difference from SM70/SM75,
which use a 2-stage software-pipelined (`mma_pipelined`) mainloop without
`cp.async`.

### `include/cutlass/gemm/`

- `device/`: host-side GEMM operator classes (`cutlass::gemm::device::Gemm`,
  `GemmUniversal`, ...). Owns `Arguments`, `can_implement`,
  `get_workspace_size`, `initialize`, `run`/`operator()`.
- `kernel/`: `DefaultGemm<...>` and friends assemble the device-side `Kernel`
  type from a threadblock `Mma` and an `Epilogue`.
- `threadblock/`: CTA-level mainloop. `DefaultMmaCore<ArchTag, ...>` picks
  operand iterators and shared-memory layouts; `mma_multistage.h` implements
  the `cp.async` pipeline for SM80+; `mma_pipelined.h` is the 2-stage
  predecessor for SM70/SM75; `threadblock_swizzle.h` maps CTA index to
  problem tile.
- `warp/`: warp-level MMA. `DefaultMmaTensorOp` wraps `mma.sync`/`ldmatrix`
  instructions; the accumulator and operand tile iterators here
  (`mma_tensor_op_tile_iterator.h`, `mma_tensor_op_fragment_iterator.h`)
  define how a warp's 32 lanes partition an M x N accumulator tile. This
  partitioning is fixed by the instruction shape and is not aligned to any
  problem-space adjacency (e.g. spatial neighbors in a convolution) — any
  fusion that needs cross-lane communication over the accumulator has to
  reason about this layout explicitly.
- `thread/`: thread-level scalar/SIMT MMA (`mma_sm50.h`, `mma_sm60.h`,
  `mma_sm61.h`) for `OpClassSimt`; not the SM80/89 TensorOp path.
- `collective/`: CUTLASS 3.x `CollectiveMma` builders, primarily targeting
  SM90+ (`cute`-based warp-specialized pipelines). Out of scope for an
  SM80/89-only kernel; do not pull these in unless the experiment explicitly
  targets SM90.

### `include/cutlass/conv/`

Convolution is implemented as an implicit GEMM: the conv problem
(`Conv2dProblemSize`: `N,H,W,C,K,R,S,pad,stride,dilation`) is mapped to a
GEMM problem (`M = N*P*Q`, `K = R*S*C_in`, `N = C_out` in matmul terms) without
materializing the unrolled matrix. The layers reuse most of `gemm/`:

- `device/`: `cutlass::conv::device::ImplicitGemmConvolution<Kernel>` is the
  host-side operator; same `Arguments`/`can_implement`/`initialize`/`run`
  shape as `gemm::device::Gemm`, but `Arguments` carries a `Conv2dProblemSize`
  instead of a `GemmCoord`.
- `kernel/`: `DefaultConv2dFprop<...>` (also `Dgrad`, `Wgrad`, `Conv3d*`
  variants) assembles the `Kernel` type. It reuses `gemm::threadblock`'s
  `DefaultMmaCore` for the actual MMA, and pairs it with conv-specific
  activation/filter iterators.
- `threadblock/`: conv-specific tile access iterators
  (`conv2d_fprop_activation_tile_access_iterator_optimized.h`,
  `..._filter_tile_access_iterator_optimized.h`) that convert a GEMM-M/K
  offset into `(n, p, q)` / `(k, r, s)` conv coordinates and apply the
  padding/stride predicate. `implicit_gemm_multistage.h` is the conv analog of
  `gemm::threadblock::mma_multistage.h`, built on the same operand iterators
  but driving the conv-specific activation/filter iterators.
- `warp/`, `thread/`: depthwise-convolution-specific warp/thread helpers; the
  regular (non-depthwise) TensorOp math itself is the same type reused from
  `gemm/warp`.
- `collective/`: SM90+ implicit-GEMM collectives; same out-of-scope note as
  `gemm/collective/`.

## Reference examples

Two upstream CUTLASS examples anchor the SM80/89 "CUTLASS-style operator"
pattern this skill follows. Paths below are relative to the CUTLASS repository
root (`https://github.com/NVIDIA/cutlass`), not to any specific checkout
location.

### `examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu` — GEMM baseline

The minimal shape of a `device`-layer GEMM instantiation:

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

Host-side call sequence — this is the contract every `device`-layer operator
in this skill must expose:

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

Correctness is checked against `cutlass::reference::device::Gemm<...>`, a
CUTLASS-provided (unoptimized) reference kernel — the same role a PyTorch or
cuDNN reference plays for other operator families in this skill.

### `examples/16_ampere_tensorop_conv2dfprop/ampere_tensorop_conv2dfprop.cu` — implicit-GEMM conv baseline

Same five-parameter shape structure as example 14, plus conv-specific knobs:

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

The conv problem is described independently of the GEMM shape and converted
by CUTLASS, not by hand:

```cpp
cutlass::conv::Conv2dProblemSize problem_size(
    input_size, filter_size, padding, conv_stride, dilation,
    output_size, cutlass::conv::Mode::kCrossCorrelation, split_k_slices);

typename ImplicitGemm::Arguments arguments{
    problem_size, tensor_a.device_ref(), tensor_b.device_ref(),
    tensor_c.device_ref(), tensor_d.device_ref(), {alpha, beta}};
```

`can_implement`/`initialize`/`operator()` follow the identical sequence as
example 14. Correctness is checked against
`cutlass::reference::host::Conv2dFprop<...>`.

`IteratorAlgorithm::kOptimized` requires channel counts aligned to the
vectorized access width (128 bits, e.g. 8 elements for `half_t`); unaligned
problems must be rejected by `can_implement`, not silently routed to a slower
iterator or a SIMT fallback.

## Goal
1. Optimize and program CUDA kernels for NVIDIA SM80 and SM89 targets — not SM90 — implementing CUTLASS-style operators: `DefaultXxx<ArchTag, Element..., ThreadblockShape..., WarpShape..., EpilogueOp...>` kernel policy factories composed with a CUTLASS-style `device` operator class (`Arguments` / `can_implement` / `initialize` / `run` / `operator()`), following the pipeline and reference examples above.
2. Treat every kernel experiment as a learning loop around the current project needs: study the operator, implement it, measure it, explain the optimization choices, and preserve the lessons for future kernels.
3. For each kernel, first match a trusted reference implementation, usually PyTorch, cuDNN, TensorRT, a CUTLASS reference kernel (`cutlass::reference::device`/`host`), or another explicitly chosen reference. Only pursue performance once the implementation stays within the required numerical tolerance, usually MAE <= 1e-3 unless the user gives a different tolerance.
4. Make real profiling artifacts the center of performance work. Use Nsight Compute, Nsight Systems, and available CUDA profiling APIs to produce human-readable reports such as `.ncu-rep` and `.nsys-rep`, plus `.csv` outputs intended for agent-side analysis.
5. Preserve new learning by collaborating with the human on Chinese technical blog material suitable for Zhihu, X, GitHub, or similar public engineering notes.

## Scope
- Keep this skill generic for CUTLASS kernel work on SM80/SM89.
- Treat experiments as kernel families with any number of numbered variants, normally `00`, `01`, `02`, and so on.
- Do not hardcode attention, a fixed `00`/`01` relationship, or any single demo as the default structure.
- Keep `blogs/` as notes and learning material only. Do not make blogs part of the build, verification, or benchmark path.

## Workflow
1. Create or maintain one `.bat` entrypoint per kernel family, for example `scripts\kernels\<family>\<family>.bat`.
2. Make the `.bat` execute the workflow in this order: `build`, then `verify`, then `bench`.
3. Stop after `verify` if correctness fails. Do not benchmark or profile kernels that do not match the reference within the required tolerance.
4. Put all build outputs under `build/`.
5. Put verification and benchmark harnesses under `csrc/tests/`.
6. Compare against the chosen reference implementation before trusting any performance data.
7. Capture Nsight Compute and Nsight Systems artifacts in both human-readable report formats and machine-readable CSV formats.
8. Keep variants numbered and comparable within the same kernel family. Each new variant should have a clear reason to exist, such as a tiling change, memory-layout change, pipeline change, epilogue fusion, or architecture-specific path.

## DO
- Keep build artifacts inside `build/`.
- Keep generated reports, build debris, and `__pycache__` out of the worktree.
- Treat reference parity as a required gate before benchmark conclusions.
- Prefer small diffs that make the optimization hypothesis easy to inspect.
- Emit both machine-readable and human-readable profiling artifacts.
- Express CUTLASS kernel policies as template type factories, normally `DefaultXxx<ArchTag, Element..., ThreadblockShape..., WarpShape..., EpilogueOp...>`.
- Keep concrete policy choices in launcher code, explicit instantiation code, CMake configuration, or local tests. Public operator, policy, target, and runtime names should remain template-driven.
- Pass CUTLASS shape types such as `cutlass::gemm::GemmShape` directly as template parameters when the shape choice is local and simple.
- Use CUTLASS-provided arch tags, storage/layout types, swizzles, and tensor reference helpers directly instead of wrapping them in local one-off functions or long concrete names.
- Use traits only when they remove real duplication across multiple architecture or dtype specializations; do not add one-off traits that merely wrap a single `GemmShape`.
- Report unsupported architectures or problem shapes through CMake checks, assertions, or explicit unsupported status paths.
- If the optimized TensorOp or architecture-specific path cannot support a case, reject it explicitly instead of silently selecting a weaker implementation.

## DO NOT
- Do not target SM90; this skill is scoped to SM80/SM89 only.
- Do not write build outputs outside `build/`.
- Do not leave performance reports or temporary build files in source directories.
- Do not treat numbers from an unverified kernel as valid performance analysis.
- Do not accept a performance report that lacks reference parity.
- Do not make a kernel-family workflow depend on attention-specific assumptions unless the current user request is explicitly about attention.
- Do not introduce concrete policy implementation structs, aliases, launchers, targets, public APIs, or test names whose primary name bakes in architecture, dtype, or layout, such as `SomethingSm89`, `SomethingSm80`, `SomethingFp16`, `SomethingBF16`, `SomethingNHWC`, or `SomethingRowMajor`.
- Do not expose non-template specialization aliases such as `using SomethingFp16 = DefaultSomething<cutlass::arch::Sm80, cutlass::half_t>`. Select concrete policies through template arguments at the launcher, explicit instantiation, benchmark, or test site.
- Do not preserve fallback compatibility with alias shims such as `using OldSomething = NewTemplatedSomething<...>`. Rename call sites to the template form instead.
- Do not make threadblock or warp shape files expose only one concrete architecture struct or one-off traits wrapper; keep shape defaults in the `DefaultXxx` factory until a separate layer earns its keep.
- Do not add lower-quality fallback kernels, such as SIMT fallbacks for a TensorOp experiment. Broader support must be a separate explicit variant with its own policy, tests, and name.
