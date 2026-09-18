# Run layout and provenance

Repository rules take precedence: live harness source in `csrc/tests/`;
binaries and compiler intermediates in `build/`.

```text
csrc/tests/<family>/          harness, verify.py, bench.py
build/<family>/              binaries, objects, compiler logs
profile/<unique-run>/
  metadata.json              device/build/workload/parity/capture policies
  commands.ps1 or commands.sh exact commands and exit-status checks
  environment/               versions, properties, sections, metrics
  reports/                   overview_<tag>.ncu-rep; source_/pm_ as needed
  analysis/                  raw CSV, details, metrics, stalls, timelines
  REPORT.md                  evidence, findings and next experiment
```

Choose a fresh name for changed builds, workloads or capture policies, e.g.
`20260917-attention-rtx4070-baseline`. Additional source/PM evidence for the
same build/workload may join that run. Do not overwrite old reports.
Imported reports retain their original provenance.

Record revision/relevant dirty diff, binary path/hash, build flags, reference
command/result/tolerance, seed or dataset identity, shapes/strides, dtype,
dispatch/launch geometry, benchmark distribution, GPU UUID/SKU/CC/SM count,
driver/CUDA/NCU/OS, replay/cache/clock policies, filter/skip/count and failures.
Use unknown for unavailable fields.

An optional `harness/` snapshot under a run contains source only. Record
dependency hashes or archive relevant source for reproducibility. Do not copy
large datasets or build debris into runs. Keep generated data out of source
control according to repository policy.

A comparison may reference reports from two runs; preserve both paths/hashes
and their metadata. Helper `--run-dir` is required for analysis outputs.
