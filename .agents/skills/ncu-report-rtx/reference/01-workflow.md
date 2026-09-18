# Capability-driven NCU pipeline

## Entry and preflight

For new collection, establish workload and question. For existing reports,
inventory actions/provenance first; recollect only for missing evidence.
Map SM80 → GA100/A100, SM86 → RTX 30x0, SM89 → RTX 40x0, SM120 → RTX 50x0.
Query actual SKU/CC/SM count/memory and device ordinal/visibility.

Save versions, `ncu --help`, `--list-sections`, `--list-sets` and device
metric queries. Installed options govern collection. Offline `--chips`
queries do not prove driver/device collection support.
See [metric discovery](08-metric-discovery.md).

## Build → verify → bench

Reuse the family entrypoint, optimized `-lineinfo` build, and supported target.
Harness source stays in `csrc/tests/<family>/`; outputs stay in `build/`.
Verification failure stops timing and profiling. Preserve trusted reference,
tolerance and results. Benchmark outside NCU with representative inputs,
warmup, repetitions and synchronization. Store variability and whole-operation
scope. A standalone harness must preserve dispatch and relevant cache/dependencies.
See [harness guide](02-harness-guide.md).

## Progressive collection

Create a [fresh run](00-directory-layout.md). Start with listed LaunchStats,
Occupancy and SpeedOfLight sections on the intended post-warmup launch.
Check exit status, nonempty report, selected name/shape/range/action.
Export raw CSV and details. NCU replay/serialization/cache/clock controls may
change timing; retain the unprofiled baseline.

| Open question | Next evidence |
|---|---|
| Insufficient or uneven work | Launch/Occupancy/WorkloadDistribution; input work sizes |
| Issue/dependencies | SchedulerStats, WarpStateStats, ComputeWorkloadAnalysis |
| Traffic/reuse/access efficiency | MemoryWorkloadAnalysis and listed chart/table sections |
| Waiting instruction | SourceCounters with line tables and SASS |
| Time variation | Supported PM sections, timestamps and sufficient samples |

Only use installed sections. Full capture is an optional escalation. Collect
PM/source separately so unavailable sampling does not invalidate the overview.
See [recipes](03-collection.md).

## Analysis and optimization

Enumerate ranges/actions, retain report lifetime, select explicitly and preserve
missing values. Cover the [six dimensions](05-analysis-dimensions.md), use
[the playbook](06-diagnosis-playbook.md), and write
[the report](07-report-template.md) with artifact links and limits.

Change one justified variable; rebuild → verify → bench. Compare matching
workload, timing scope and policies. Recollect only necessary counters.
Measured speedup and noise are separate from NCU estimated speedup.
Identify tested hardware; never imply all four architectures were hardware-tested
from a single report. Stop collecting when the question is answered.
