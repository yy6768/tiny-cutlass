# Profiling helpers

GPU-independent analysis for SM80/SM86/SM89/SM120. Candidate counters are checked
against each report; missing values remain N/A. No B200-specific alias list.

## Python usage (PowerShell)

```powershell
$Helpers = '.agents/skills/ncu-report-rtx/helpers'
$Run = 'profile/myrun'
$Report = "$Run/reports/overview_main.ncu-rep"
python "$Helpers/analyze_reports.py" --report $Report --list-actions
python "$Helpers/analyze_reports.py" --run-dir $Run --report $Report --tag main --range-index 0 --action-index 0
python "$Helpers/extract_stall_hotspots.py" --run-dir $Run --report "$Run/reports/source_main.ncu-rep" --tag main --range-index 0 --action-index 0
python "$Helpers/plot_timeline.py" --run-dir $Run --report "$Run/reports/pm_main.ncu-rep" --tag main --range-index 0 --action-index 0
```

Use the inventory's actual indices. Selectors are optional only for single-action
reports. `analyze_reports.py --all-actions` extracts distinct `tag_rN_aM`
files without assuming different kernels are comparable.
Repeated `--report` / `--tag` pairs produce raw comparisons in selected-action
mode; use separate commands when each report needs different action indices.

| Helper | Output / role |
|---|---|
| ncu_utils.py | Lazy API discovery, report lifetime/selection, typed access, candidate metrics |
| analyze_reports.py | metrics_all/key_<tag>.json, key text and optional comparison |
| extract_stall_hotspots.py | stall_hotspots_<tag>.txt; selected/not_selected kept separate |
| plot_timeline.py | pm_timeline_<tag>.txt and raw pm_samples_<tag>.json; all samples retained |
| list_flashinfer_workloads.py | Local dataset inspection; no report API dependency |
| harness_template.cu | Fail-closed scaffold to adapt under csrc/tests/<family>/ |
| safetensors_loader.h | Minimal architecture-independent host reader |

All extraction outputs go to the required `--run-dir/analysis`. Missing reports,
ambiguous selection or invalid tags fail instead of silently skipping.
Raw API units/values must be checked against CLI exports before conversions;
see [API notes](../reference/04-python-api.md).

Hotspots default to all warp-state samples; use --sample-scope not-issued for
the scheduler-not-issuing subset. These are overlapping populations and are
never summed together. Use distinct tags when saving both analyses.

API discovery probes common Windows Program Files/CUDA_PATH/PATH and Linux paths.
Set `NCU_PYTHON_PATH` to a compatible NCU `extras/python` directory when needed.
`--help` works without loading the native API.

## Harness and workloads

Copy the CUDA scaffold and reader into the actual test family, configure the
architecture through the build, implement the real dispatch and verification.
Compile outputs under `build/`; no binary goes into `profile/`.
See [harness guide](../reference/02-harness-guide.md).
The template intentionally fails before adaptation; it does not benchmark an
empty launcher.

Dataset browser example:
```bash
python helpers/list_flashinfer_workloads.py --dataset /path/to/trace --definition NAME --show-definition
```

Check dtype/shape/byte count before using the minimal safetensors reader.
Packed FP4/FP6/FP8 data needs its own decoding/scale contract.
