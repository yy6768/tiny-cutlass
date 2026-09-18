---
name: ncu-report-skill
description: Profile and diagnose CUDA kernels on SM80/SM86 (Ampere), SM89 (Ada), and SM120 (Blackwell RTX), including RTX 30/40/50 series. Use for Nsight Compute collection, report analysis, source stalls, and evidence-backed optimization plans (profile 一下、为什么慢、ncu 报告). Supports Windows and Linux with device-specific metric discovery.
---

# CUDA profiling: SM80 / SM86 / SM89 / SM120

Diagnose verified kernels from measured evidence. For existing reports, start at
report inspection; a new collection or local GPU is not required.

| Target | Hardware | Programming reference |
|---|---|---|
| SM80 / CC 8.0 | GA100 / A100, separate from GeForce | [Ampere](ampere-cuda-programming.md) |
| SM86 / CC 8.6 | GA10x / RTX 30x0 | [Ampere](ampere-cuda-programming.md) |
| SM89 / CC 8.9 | AD10x / RTX 40x0 | [Ada](ada-cuda-programming.md) |
| SM120 / CC 12.0 | GB20x / RTX 50x0 | [Blackwell RTX](blackwell-rtx-cuda-programming.md) |

Read the matching guide. Query the actual SKU, CC, SM count, memory and toolchain;
desktop/laptop configurations differ. SM120 does not use the SM100/B200
`tcgen05`/TMEM accumulator path.

## Workflow

1. **Scope and discover.** Identify shapes, strides, dtype, numerical tolerance,
   dispatch and performance question. Save device attributes, tool versions,
   `ncu --help`, `--list-sections`, `--list-sets` and metric queries.
   See [workflow](reference/01-workflow.md).
2. **Preserve provenance.** Create `profile/<unique-run>/` for metadata,
   commands, reports and analysis. In tiny-cutlass, harness source belongs in
   `csrc/tests/<family>/`; binaries/intermediates belong in `build/`.
   See [layout](reference/00-directory-layout.md).
3. **Build → verify → bench.** Use optimized `-lineinfo` builds and a supported
   compiler target. Pass trusted reference parity before benchmark or profiling.
   Keep unprofiled timing. Reuse the application when isolation changes its
   dispatch/cache/dependencies. See [harness guide](reference/02-harness-guide.md).
4. **Collect progressively.** Start with launch/occupancy/SOL. Add scheduler,
   compute or memory sections for open questions. Source counters and PM sampling
   are conditional separate captures; neither PM nor `--set full` is mandatory.
   Record launch selection, replay/cache/clock policies; export `.ncu-rep`,
   raw CSV and details. See [recipes](reference/03-collection.md).
5. **Select the action.** Enumerate ranges/actions; do not silently select the
   first kernel. Preserve names, raw values, units and missing-data status.
   See [Python API](reference/04-python-api.md) and
   [metric discovery](reference/08-metric-discovery.md).
6. **Diagnose and test.** Use [six dimensions](reference/05-analysis-dimensions.md)
   and [playbook](reference/06-diagnosis-playbook.md). Separate observations,
   hypotheses and experiments. Change one variable, rerun parity and unprofiled
   timing, then recollect only necessary counters.
7. **Report.** Write `REPORT.md` with the
   [template](reference/07-report-template.md), artifacts and limits.
   Use [troubleshooting](reference/09-common-issues.md) for failures.

Historical reports with unknown correctness/build provenance still permit
structural analysis; performance conclusions remain conditional on parity.

## Helpers

See [usage](helpers/README.md).

- [harness_template.cu](helpers/harness_template.cu): fail-closed scaffold,
  actual device launcher, device metadata and error checks.
- [safetensors_loader.h](helpers/safetensors_loader.h) and
  [list_flashinfer_workloads.py](helpers/list_flashinfer_workloads.py):
  optional local workload loading/browsing.
- [ncu_utils.py](helpers/ncu_utils.py): lazy Windows/Linux API discovery,
  explicit action selection, neutral metric candidates and typed access.
- [analyze_reports.py](helpers/analyze_reports.py): action inventory, archives,
  availability and comparisons.
- [extract_stall_hotspots.py](helpers/extract_stall_hotspots.py): PC/source
  attribution; productive scheduler states kept separate from stalls.
- [plot_timeline.py](helpers/plot_timeline.py): sampled series retaining zeros,
  unknown values and the entire tail, with timestamps when available.

## Interpretation constraints

Missing is not zero. Low DRAM traffic alone does not establish memory latency.
Occupancy is not the optimization objective. A sampled PC identifies a waiting
instruction; inspect its producer. Do not mix active/elapsed denominators.
Preserve raw API units and verify timing scale through NCU CSV.
NCU rule speedups are estimates, not measurements.

PM support and resolution depend on device/tool/platform. Without PM, use launch
geometry, input work distribution and repeated timings; mark time-resolved
claims unproven. For kernel changes, apply `cutlass-kernel` and repository
template/policy and no-silent-fallback rules.
