# Ampere CUDA programming: SM80 / SM86

## GA100 and GA10x are separate targets

SM80 is GA100/A100. RTX 30x0 uses GA10x/SM86. Do not transfer an A100 shared-memory
budget or HBM roofline into a GeForce experiment.

| Architecture limit | SM80 | SM86 |
|---|---:|---:|
| Resident warps / SM | 64 | 48 |
| Resident blocks / SM | 32 | 16 |
| 32-bit registers / SM | 65,536 | 65,536 |
| Shared memory / SM | 164 KiB | 100 KiB |
| Opt-in shared memory / block | 163 KiB | 99 KiB |

Use device/kernel attributes and occupancy APIs for actual allocations. Above
48 KiB per block, use dynamic shared memory with explicit opt-in.
Source: [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).

## Tensor and copy pipeline

Use supported warp-level `mma.sync`/CUTLASS TensorOp policies and a dtype,
accumulator and sparsity mode matching the numerical contract. GA100 and GA10x
have different Tensor throughput and FP64 capabilities; native FP8/FP4 is not
an RTX 30x0 assumption. GA10x combines dedicated FP32 with shared FP32/INT32
execution resources, so indexing can compete with math.
See the user's [GA102 whitepaper, SM and Tensor Core sections](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf).

`cp.async` overlaps global-to-shared staging and avoids intermediate copy
registers. Wait for copy completion and synchronize consumers correctly.
Stage count trades latency hiding against shared memory, registers and residency.

## NCU-guided experiments

| Evidence | Experiment | Recheck |
|---|---|---|
| Poor global efficiency | Change lane layout/legal vector width | Requests, sectors, alignment, parity |
| Few eligible warps, long-scoreboard samples | More reuse or independent load overlap | Issue, duration, register growth |
| Shared-memory occupancy limit | Smaller tile/fewer stages | Tensor activity and latency |
| Local traffic plus compiler spills | Smaller live accumulator footprint | Spill diagnostics/SASS; arrays also use local memory |
| Partial wave or ragged CTA work | Justified split/repartition | Whole-operator time including reduction |

Compile for the actual supported `sm_80` or `sm_86` target. Keep architecture
in build/policy parameters. Query SKU-specific SM count, memory/cache size and
bandwidth; full-chip whitepaper parameters are not every shipping card.
Use [the pipeline](reference/01-workflow.md) and
[metric discovery](reference/08-metric-discovery.md), not a fixed architecture counter list.
