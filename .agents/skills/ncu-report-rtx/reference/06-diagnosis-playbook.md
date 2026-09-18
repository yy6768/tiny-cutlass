# Observation → hypothesis → experiment

Use after [analysis](05-analysis-dimensions.md), without universal cutoffs.

| Observation | Hypotheses | Experiment | Validation/tradeoff |
|---|---|---|---|
| Grid smaller than SM count | Insufficient work | Smaller tiles or justified split | Whole operator incl. reduction |
| Partial waves/variable CTA work | Geometry vs ragged imbalance | Repartition/bucket work | Scheduling/reordering cost |
| Low issue, few eligible, long scoreboard | Dependencies, locality, low parallelism | Reuse/layout or independent loads | Registers and issue progress |
| Shared conflicts plus short scoreboard | Bank layout or shared dependency | Padding/layout | Operand constraints/parity |
| Barrier samples and uneven warps | Divergent arrival or excess sync | Equalize preceding work | Preserve synchronization correctness |
| Local traffic plus compiler spills | Live state pressure | Smaller live ranges/tile | Reuse/instruction cost |
| High DRAM traffic near sustainable BW | Redundant bytes | Fuse/reuse materialization | End-to-end timing |
| Low Tensor activity, heavy staging/epilogue | Delivery or surrounding work | One stage/tile/epilogue change | Residency and instruction cost |
| Math-pipe pressure | Useful or redundant compute | Reuse/simplify indexing | Numerical contract |
| PM decline alone | Tail, phase, contention or pass skew | Repeated capture + work audit | No invented speedup |

Read the matching [Ampere](../ampere-cuda-programming.md),
[Ada](../ada-cuda-programming.md) or
[Blackwell RTX](../blackwell-rtx-cuda-programming.md) guide first.

Persistent kernels cannot create independent work. Register caps may increase
spills. SM100 TMEM/2-CTA MMA is not an SM120 fix. Quantization needs supported
instructions and a new, explicit numerical validation.

Each recommendation records exact report/action/metric/source evidence,
mechanism and alternative, one distinguishing experiment, expected direction
and tradeoff, and rebuild → verify → bench → focused-counter acceptance criteria.
NCU rule estimates overlap and are not measured speedups. Success means
repeatable workload improvement with parity, not merely higher utilization.
