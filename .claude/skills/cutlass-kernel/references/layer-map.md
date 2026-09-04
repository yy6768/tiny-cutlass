# CUTLASS 2.x layer map (SM80 / SM89)

CUTLASS decomposes one kernel into a fixed hierarchy that mirrors the GPU
execution hierarchy. Every operator family — GEMM, convolution, attention, fused
epilogues — is an instance of the same five layers:

```text
device      host-side operator class: Arguments, can_implement, initialize, run, operator()
  -> kernel     device-side __global__ entry: Params, SharedStorage, the Mma+Epilogue composition
    -> threadblock  CTA-level mainloop: global -> shared memory staging, multistage cp.async pipeline
      -> warp       warp-level MMA: mma.sync / ldmatrix wrappers, operand and accumulator fragment iterators
        -> thread     per-thread math and per-element epilogue ops
```

A `DefaultXxx<ArchTag, Element..., ThreadblockShape, WarpShape, InstructionShape,
EpilogueOp, ...>` factory in the `kernel` layer picks and assembles the
`threadblock` / `warp` / `thread` types for one architecture and element
combination, and exposes the resulting `::Kernel` type.

The `device` layer never contains algorithm code. It builds `Params` from
`Arguments`, checks support, and launches
`cutlass::Kernel<Kernel><<<grid, block, smem>>>`.

## The multistage mainloop

On SM80/SM89 the threadblock mainloop is the multistage `cp.async` pipeline:

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

`NumStages` (a kernel template parameter, typically 3–5) controls how many tiles
of the mainloop are in flight at once, trading shared memory for latency hiding.
This is the deciding architectural difference from SM70/SM75, which use a 2-stage
software-pipelined (`mma_pipelined`) mainloop without `cp.async`.

Shared memory per CTA on SM89 caps at ~99 KB by default and ~163 KB with
`cudaFuncAttributeMaxDynamicSharedMemorySize` opt-in. A fusion design that stages
more than one intermediate tile needs its smem budget written down before any
code is written — that number, not the algorithm, is usually what decides whether
a fused variant is possible at all.

## Where fusion hooks exist — and where they do not

Two facts worth knowing before designing a fused epilogue, both verified in this
repository:

- **Epilogue visitors are a GEMM-layer mechanism.** `EpilogueWithVisitor`'s
  `begin_row` / `end_row` hooks (`epilogue/threadblock/epilogue_with_visitor.h`)
  are driven from GEMM kernels only. `conv::device::ImplicitGemmConvolution`
  calls a plain threadblock epilogue directly, so a conv that needs a
  cross-column reduction (e.g. LayerNorm along channels) must fork its own conv
  kernel to drive the visitor. Precedent:
  `csrc/conv-fused/conv1x1_dual/kernel/b2b_implicit_gemm_convolution.h`.
- **A plain `EpilogueOutputOp` cannot reduce.** It is a per-element thread
  functor: no cross-lane hook, no second pass. Any reduction along the epilogue's
  N axis needs either a visitor or a separate pass.

The lever that makes a single-kernel full reduction possible is tile containment:
choose `ThreadblockShape::kN >= reduction_extent` so the whole reduced axis lands
in one N-tile, and a `__shfl_xor_sync` butterfly inside `end_row` produces the
complete statistic with no cross-CTA finalize. Pattern: upstream `examples/37`.

## The accumulator layout is not problem-space adjacency

In `gemm/warp/`, the accumulator and operand tile iterators
(`mma_tensor_op_tile_iterator.h`, `mma_tensor_op_fragment_iterator.h`) define how
a warp's 32 lanes partition an M×N accumulator tile. That partitioning is fixed
by the instruction shape and bears no relation to adjacency in the problem space
— spatial neighbors in a convolution, or adjacent tokens in an attention window,
generally do not land in the same lane or even the same fragment.

Any fusion that needs cross-lane communication over the accumulator has to reason
about this layout explicitly. `DefaultMmaAccumLambdaIterator` is the supported way
to walk a register accumulator fragment with knowledge of its lane mapping.
