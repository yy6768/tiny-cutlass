# Blackwell RTX CUDA programming: SM120 / RTX 50x0

## Architecture boundary

RTX 50x0 uses GB20x, CC 12.0. GB202 is an RTX chip, not B200/GB100.
Do not transplant the datacenter SM100 `tcgen05`/TMEM/2-CTA MMA design.

SM120 TensorOp kernels use supported warp-level `mma.sync` forms and register
operands/accumulators, including architecture-specific low-precision forms.
NVIDIA CUTLASS explicitly excludes SM120 from its `tcgen05` path:
[CUTLASS implementation](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py).
Check each instruction's [PTX target notes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html);
“fifth-generation Tensor Core” does not establish ISA compatibility.

## Hardware context

The RTX Blackwell whitepaper describes 256 KiB registers and a 128 KiB unified
L1/shared pool per SM, GDDR7, FP4/FP6 Tensor capabilities, and unified FP32/INT32
execution resources. Integer throughput gains depend on the instruction and
compete for shared execution resources. Full GB202 differs from enabled RTX 5090.
Source: the user's [RTX Blackwell whitepaper, SM/memory/Tensor sections](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

The CUDA capability tables specify 48 resident warps, 24 resident blocks,
65,536 32-bit registers and 100 KiB allocatable shared memory per SM, with
99 KiB maximum opt-in shared memory per block. Query actual attributes and occupancy.
The [CUDA capability tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
and [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
have differed in SM120 shared-pool/block-limit descriptions (checked 2026-09-17).
The values above follow the capability tables. Record document
version and runtime attributes rather than treating all 128 KiB as allocatable
shared memory. Use dynamic shared-memory opt-in above the default block limit.

## Implementation and profiling

- Check `nvcc --list-gpu-code` and library support. Ordinary kernels may use
  `sm_120`; use a feature target such as `sm_120a` only where the chosen
  instruction requires it. `sm_100a` is not a substitute.
- Accumulators remain in registers: measure pressure, epilogue live ranges,
  compiler spills and local accesses.
- Async/bulk/tensor copies need their own target, layout and synchronization
  checks. Copy support does not imply datacenter MMA, multicast or scheduling support.
- Include scales, packing, conversion and epilogue in FP4/FP6/FP8 end-to-end
  comparisons. Verify the intended numerical reference before timing.
- Match dense/sparse mode and accumulator precision in rooflines.
  GDDR7 bandwidth does not eliminate latency or access inefficiency.

Follow [the NCU pipeline](reference/01-workflow.md). Discover real device metrics;
do not copy SM100 substitutions. Pair Tensor activity with SASS, eligibility,
staging traffic and resource limits. Test one hypothesis and rerun parity,
unprofiled timing and focused counters. PM is optional evidence with sample
count, timestamps, context and pass-alignment limits.
