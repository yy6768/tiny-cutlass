# Report template

Save `profile/<run>/REPORT.md`; mark unavailable data explicitly.

```markdown
# <operator/workload> NCU report

<Main finding with measured values, confidence and unresolved alternative.>

## Provenance and correctness
- GPU SKU/UUID/CC/SM count/memory/OS; driver/CUDA/compiler/NCU versions.
- Target, build command, source revision/relevant diff, binary path/hash.
- Shapes/strides/layout/dtype/accumulator/seed or dataset identity.
- Exact dispatch, grid/block/shared memory.
- Reference, tolerance, command and result; unknown if absent.

## Measurements
- Unprofiled warmup/repetitions, median/variability, whole-operation scope.
- NCU filter/skip/count, range/action, replay/cache/clock policies.
- Discovery artifacts and missing/failed collections.

| Exact measurement and validated unit | Baseline | Candidate | Evidence |
|---|---:|---:|---|
| Unprofiled operator latency | ... | ... | benchmark artifact |
| Profiled duration | ... | ... | report/range/action |
| SM/DRAM throughput | ... | ... | exact names |
| Resources/waves/eligible/issue/Tensor | ... | ... | metric records |

## Diagnosis
Cover relevant launch, work balance, scheduler/source, compute, timeline and
memory findings; mark unavailable dimensions. Cite full paths/PCs, sample
counts and units. Separate observations from hypotheses. Preserve timeline
zero/unknown regions and state sampling scope.

## Experiments
| Priority | Evidence | One change | Tradeoff | Acceptance test |
|---|---|---|---|---|
| 1 | ... | ... | ... | parity + repeated timing + focused counters |

## Artifacts and limits
Link metadata, commands, verification/bench logs, .ncu-rep, CSV/details and
analysis outputs. State replay/sampling limits, unknown provenance and
comparison mismatches. Distinguish measured from estimated speedup.
```

Historical reports without parity logs permit structural analysis, with
performance conclusions conditional on correctness. Identify tested hardware;
single-GPU evidence does not establish four-architecture hardware validation.
