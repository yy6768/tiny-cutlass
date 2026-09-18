# NCU collection recipes

Use installed help/discovery. Replace example executable, arguments, kernel
and warmup skip with the verified workload. Check command exit codes.

## PowerShell discovery

```powershell
$Run = 'profile/20260917-example-baseline'
if (Test-Path -LiteralPath $Run) { throw 'Choose a new run directory' }
New-Item -ItemType Directory -Path "$Run/environment","$Run/reports","$Run/analysis" | Out-Null
ncu --version | Set-Content -Encoding utf8 "$Run/environment/ncu-version.txt"
ncu --help | Set-Content -Encoding utf8 "$Run/environment/ncu-help.txt"
ncu --list-sections | Set-Content -Encoding utf8 "$Run/environment/sections.txt"
ncu --list-sets | Set-Content -Encoding utf8 "$Run/environment/sets.txt"
ncu --query-metrics --devices 0 --query-metrics-mode all | Set-Content -Encoding utf8 "$Run/environment/metrics.txt"
if ($LASTEXITCODE -ne 0) { throw 'Discovery failed' }
```

Also save compiler/GPU/driver versions and CUDA device properties.
Use the intended visible device and record CUDA_VISIBLE_DEVICES mapping.

## Overview, after build/parity/benchmark

```powershell
$Exe = (Resolve-Path 'build/example/harness.exe').Path
$WorkloadArgs = @('--shape','1024') # replace with actual harness CLI
$Kernel = 'regex:my_kernel'
$Skip = 5 # verified number of matching warmup launches
$Common = @('--devices','0','--kernel-name-base','demangled','-k',$Kernel,
            '--launch-skip',"$Skip",'--launch-count','1',
            '--replay-mode','kernel','--cache-control','all','--clock-control','base')
ncu @Common --section LaunchStats --section Occupancy --section SpeedOfLight -o "$Run/reports/overview_main" $Exe @WorkloadArgs
if ($LASTEXITCODE -ne 0) { throw 'Overview failed' }
ncu --import "$Run/reports/overview_main.ncu-rep" --page raw --csv | Set-Content -Encoding utf8 "$Run/analysis/raw_main.csv"
if ($LASTEXITCODE -ne 0) { throw 'CSV export failed' }
ncu --import "$Run/reports/overview_main.ncu-rep" --page details | Set-Content -Encoding utf8 "$Run/analysis/details_main.txt"
if ($LASTEXITCODE -ne 0) { throw 'Details export failed' }
```

Confirm section availability first. Verify the selected action/name/shape.
`--launch-skip` counts matching launches with filters; it differs from
`--launch-skip-before-match`. `-c 1` does not establish the intended dispatch.

Kernel replay suits isolated replayable launches. Application/range replay may
be needed for host-device or inter-kernel dependencies; use installed supported
modes and deterministic inputs/launch order. Serialization changes concurrency.
The explicit cache/clock policies above describe controlled collection, not
production-warm timing. If clock control is unsupported, record that and use
`none` consistently for both comparison captures.

## Targeted and source follow-ups

Use the same selection and workload, only with listed section identifiers:

```powershell
ncu @Common --section SchedulerStats --section WarpStateStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis -o "$Run/reports/targeted_main" $Exe @WorkloadArgs
if ($LASTEXITCODE -ne 0) { throw 'Targeted capture failed' }
ncu @Common --section SourceCounters --import-source yes -o "$Run/reports/source_main" $Exe @WorkloadArgs
if ($LASTEXITCODE -ne 0) { throw 'Source capture failed' }
```

SourceCounters needs supported collection; CUDA line mapping additionally needs
`-lineinfo`. SASS remains useful without line tables. A `source` set is not
assumed. `--set full` is optional, with no fixed pass count.

## Optional PM

If installed help supports it, query:
`ncu --query-metrics --devices 0 --query-metrics-collection pmsampling --query-metrics-mode all`.
Otherwise use listed PM sections and a small capability probe. Availability does
not guarantee single-pass collectability.

```powershell
ncu @Common --section PmSampling -o "$Run/reports/pm_main" $Exe @WorkloadArgs
if ($LASTEXITCODE -ne 0) { throw 'PM failed; preserve overview and record limitation' }
```

Add PmSampling_WarpStates only if listed and relevant. Short kernels can yield
too few samples. Device activity, context filtering and cross-pass alignment
affect interpretation. Preserve timestamps, zeros and unknown values.

## Bash equivalent

In saved scripts use `set -euo pipefail` to stop after failures. After setup:

```bash
ncu --devices 0 --kernel-name-base demangled -k 'regex:my_kernel' \
  --launch-skip 5 --launch-count 1 --replay-mode kernel \
  --cache-control all --clock-control base \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  -o "$PROFILE_RUN_DIR/reports/overview_main" ./build/example/harness "$@"
ncu --import "$PROFILE_RUN_DIR/reports/overview_main.ncu-rep" --page raw --csv \
  > "$PROFILE_RUN_DIR/analysis/raw_main.csv"
ncu --import "$PROFILE_RUN_DIR/reports/overview_main.ncu-rep" --page details \
  > "$PROFILE_RUN_DIR/analysis/details_main.txt"
```

Export every report used as evidence. Avoid force-overwrite on existing runs.
For explicit metrics, query exact full names first.
Sources: [NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html),
[profiling/replay/sampling](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
