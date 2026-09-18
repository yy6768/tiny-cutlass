# Six analysis dimensions

Names are [discovery candidates](08-metric-discovery.md), not guaranteed counters.

| Dimension | Evidence | Main question |
|---|---|---|
| Launch/occupancy | Grid/block, registers/shared memory, waves, occupancy | Enough work? Which resource limits residency? |
| Work distribution | CTA work sizes, SM active-cycle distribution when collected | Partial wave or ragged-work imbalance? |
| Scheduler/stalls | Eligible warps, issue, WarpStateStats, per-PC samples | Why cannot warps issue, and where? |
| Compute/Tensor | ComputeWorkloadAnalysis, Tensor activity, SASS | Is the intended instruction path busy doing useful work? |
| Time variation | Supported PM, timestamps, sample count/context | Is a tail/phase actually resolved? |
| Memory | DRAM/L2/L1 traffic, sectors/requests, conflicts/local accesses | Bandwidth, latency, inefficiency or redundant traffic? |

## Launch and scheduler

For conventional non-cluster launches with occupancy-derived B resident
blocks/SM and S SMs, wave capacity is B*S. G blocks imply full waves and a
remainder, not an exact runtime model. G < S leaves SMs unused; G < B*S may
still use every SM. Do not infer idle SMs from waves < 1 alone or take the
minimum of differently dimensioned occupancy counters.

Long scoreboard means an L1TEX dependency, not necessarily a DRAM miss.
Short scoreboard involves MIO dependencies, often shared memory; it is not a
generic local-memory/math-chain label. Wait denotes fixed-latency dependencies.
Inspect SASS and producer instructions.

Per-issued-warp ratios are not percentages and do not have an assumed bound
of 15. PC samples are statistical counts, not elapsed cycles. Use compatible
documented denominators for percentages. Selected is productive; not_selected
means the scheduler chose another eligible warp. Exclude both from dependency
stall totals. An occupancy gap can arise from launch/tail behavior as well as
other effects; high occupancy does not guarantee issue progress.

## Compute and memory

Absent Tensor counters mean unknown. A measured zero needs dispatch/SASS
context; elementwise kernels need not use Tensor Cores. SM120 follows its
[own warp-level MMA path](../blackwell-rtx-cuda-programming.md).

Sectors/request depends on active lanes, width, vectorization and alignment.
Four 32-byte sectors is an example for 32 lanes each loading one contiguous,
aligned 4-byte value, not a universal target. Local traffic can be arrays or
spills; inspect compiler diagnostics and SASS.

Low DRAM throughput plus long-scoreboard sampling motivates a latency
hypothesis, but check small grid, L1/L2 pressure and dependencies too.
High hit rate does not imply low traffic. Normalize to useful work and record
cache policy.

## Timeline

Retain leading/trailing zeros and unknowns. A falling curve could be partial
waves, ragged work, computation phases, contention or replay alignment.
Combine with geometry/work distribution. Short captures may resolve no tail.
Without PM, use launch/work data and repeated timings, leaving time-resolved
claims unproven.
Source: [NCU metric/sampling semantics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
