# CUTLASS source directory index

Paths are relative to the CUTLASS repository root
(`https://github.com/NVIDIA/cutlass`), vendored here under `3rdparty/cutlass`.

## `include/cutlass/gemm/`

- **`device/`** — host-side GEMM operator classes (`cutlass::gemm::device::Gemm`,
  `GemmUniversal`, `GemmUniversalWithBroadcast`, ...). Owns `Arguments`,
  `can_implement`, `get_workspace_size`, `initialize`, `run` / `operator()`.
- **`kernel/`** — `DefaultGemm<...>` and friends assemble the device-side `Kernel`
  type from a threadblock `Mma` and an `Epilogue`.
- **`threadblock/`** — CTA-level mainloop. `DefaultMmaCore<ArchTag, ...>` picks
  operand iterators and shared-memory layouts; `mma_multistage.h` implements the
  `cp.async` pipeline for SM80+; `mma_pipelined.h` is the 2-stage predecessor for
  SM70/SM75; `threadblock_swizzle.h` maps CTA index to problem tile.
- **`warp/`** — warp-level MMA. `DefaultMmaTensorOp` wraps `mma.sync` / `ldmatrix`.
  The tile iterators here fix the lane partitioning of the accumulator — see
  [layer-map.md](layer-map.md) on why that matters for fusion.
- **`thread/`** — thread-level scalar/SIMT MMA (`mma_sm50.h`, `mma_sm60.h`,
  `mma_sm61.h`) for `OpClassSimt`. Not the SM80/89 TensorOp path.
- **`collective/`** — CUTLASS 3.x `CollectiveMma` builders, primarily SM90+
  (`cute`-based warp-specialized pipelines). **Out of scope.** Do not pull these
  in; `collective_builder.hpp` has no Sm80/89 specialization.

## `include/cutlass/conv/`

Convolution is an implicit GEMM: the conv problem (`Conv2dProblemSize`:
`N,H,W,C,K,R,S,pad,stride,dilation`) maps to a GEMM problem
(`M = N*P*Q`, `K = R*S*C_in`, `N = C_out`) without materializing the unrolled
matrix. The layers reuse most of `gemm/`:

- **`device/`** — `cutlass::conv::device::ImplicitGemmConvolution<Kernel>`. Same
  `Arguments` / `can_implement` / `initialize` / `run` shape as
  `gemm::device::Gemm`, but `Arguments` carries a `Conv2dProblemSize` instead of a
  `GemmCoord`.
- **`kernel/`** — `DefaultConv2dFprop<...>` (also `Dgrad`, `Wgrad`, `Conv3d*`).
  Reuses `gemm::threadblock`'s `DefaultMmaCore` for the MMA and pairs it with
  conv-specific activation/filter iterators.
- **`threadblock/`** — conv tile access iterators
  (`conv2d_fprop_activation_tile_access_iterator_optimized.h`,
  `..._filter_tile_access_iterator_optimized.h`) convert a GEMM-M/K offset into
  `(n, p, q)` / `(k, r, s)` conv coordinates and apply the padding/stride
  predicate. `implicit_gemm_multistage.h` is the conv analog of
  `gemm::threadblock::mma_multistage.h`.
- **`warp/`, `thread/`** — depthwise-convolution helpers. Regular (non-depthwise)
  TensorOp math is the same type reused from `gemm/warp`.
- **`collective/`** — SM90+ implicit-GEMM collectives. Out of scope.

## `include/cutlass/epilogue/`

- **`thread/`** — per-element output ops: `LinearCombination`,
  `LinearCombinationBiasElementwise`, `LinearCombinationResidualBlock`,
  `LinearCombinationGELU` (erf-exact, matches host `std::erf`). These are pure
  elementwise functors and cannot reduce.
- **`threadblock/`** — output tile iterators and epilogue drivers.
  `predicated_tile_iterator.h` is where `ScatterD` is honoured, on **both** the
  load and store paths — which is what lets a residual arrive through the
  source-C tensor under the same scatter indices. `epilogue_with_visitor.h`
  supplies the `begin_row` / `end_row` reduction hooks;
  `epilogue_with_broadcast.h` supplies per-column bias broadcast;
  `epilogue_with_reduction.h` supplies a reduction alongside the store.
- **`warp/`** — accumulator fragment iterators used by the above.

## Gather / scatter, without a bespoke iterator

`GemmUniversal` carries `GatherA` / `GatherB` / `ScatterD` template flags
(`gemm/device/gemm_universal.h`), demonstrated in
`examples/36_gather_scatter_fusion`. The semantics are **row-number remapping**:

```cpp
// gemm/threadblock/predicated_tile_access_iterator.h
if (Gather) coord_strided = indices_[coord_strided];
```

When A is row-major, the strided rank is GEMM M. So any permutation, selection, or
concat that is expressible as "row `m` of the operand comes from row `idx[m]` of
the buffer" needs zero new iterator code — build the index array on the host.

Two constraints found the hard way:

- Predication uses the **logical extent** (`problem_size`'s M) while gather
  rewrites the **address**. The buffer's row count may therefore differ from M.
- `GemmUniversalWithBroadcast` does **not** carry the gather/scatter flags, and
  `EpilogueWithBroadcast` has no scatter path. One GEMM cannot both broadcast a
  bias vector and gather its A operand. Pick one, or move the other into a pass
  that has to exist anyway.

## Attention / from-smem chaining

Reusable pieces for back-to-back GEMMs where the second operand A is already in
shared memory (local to this repo, under `csrc/flash-attention/gemm/`):

- `B2bGemm<...>::accumToSmem(smem, accum, lane, tile_coords)` dumps a warp-MMA
  register accumulator to smem as the next GEMM's operand A.
- `DefaultMmaFromSharedMemory<DefaultGemm::Mma, kMaxK, WarpIteratorA, false>`
  builds a GEMM whose operand A already lives in smem; the warp iterator comes
  from `DefaultWarpIteratorAFromSharedMemory`.
- `MemoryEfficientAttentionNormalize` (`epilogue/epilogue_rescale_output.h`) is the
  template pattern to mimic for a custom output op that needs per-row state.

Upstream `examples/13_two_tensor_op_fusion` is the B2B precedent for the fully
CUTLASS-side version.
