# Ada CUDA programming: SM89 / RTX 40x0

## Resources and execution

Ada AD10x is CC 8.9: 48 resident warps, 24 blocks and 65,536 32-bit registers per
SM; 100 KiB shared memory per SM and 99 KiB opt-in per block. The 128 KiB unified
L1/shared pool is not a 128 KiB shared allocation. Above 48 KiB per block, use
dynamic allocation with explicit opt-in.
Source: [Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).

Fourth-generation Tensor Cores add FP8 capabilities. Verify the exact supported
instruction/library path, accumulator and numerical tolerance. FP16/BF16 paths
use warp-level MMA and register accumulators. `cp.async` staging is relevant;
Hopper WGMMA/TMA and SM100 TMEM are not Ada implementation choices.

AD10x retains FP32/INT32 resource sharing and increases L2 substantially.
Full AD102 has 96 MiB L2; RTX 4090 has 72 MiB. Neither value applies to every
RTX 40x0, especially laptop configurations.
See the user's [Ada whitepaper, SM/memory sections and SKU table](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).

## NCU-guided experiments

- Record replay/cache policy before comparing cold and warm behavior. A working
  set fitting L2 changes the apparent bottleneck; low DRAM traffic alone is not
  evidence of efficient execution.
- Pair occupancy with eligible warps, issue activity and resource limits.
  Reducing register limits can introduce spills; larger tiles can reduce residency.
- Inspect Tensor counters and SASS together. Missing counters mean unknown.
  Match dtype, accumulation and sparsity when choosing a roofline.
- Investigate shared layout/bank conflicts before adding pipeline stages.
  For ragged work, inspect CTA work sizes and wave geometry.
- On a display/laptop GPU, record graphics contention, clocks, power and thermal
  state. Establish timing variability outside NCU.

Compile explicitly for supported `sm_89` through policy/build configuration.
Check actual device limits before reusing A100 tiles. Follow
[collection](reference/03-collection.md) and
[analysis](reference/05-analysis-dimensions.md).
