# NCU Python report API

Use a report-compatible NCU `extras/python` installation. The helper imports
lazily and probes Windows/Linux paths. Set `NCU_PYTHON_PATH` to a specific
extras/python directory or configure PYTHONPATH when needed.

## Select a launch

```powershell
python .agents/skills/ncu-report-rtx/helpers/analyze_reports.py --report profile/myrun/reports/overview_main.ncu-rep --list-actions
```

All analysis helpers accept `--range-index` / `--action-index`. Without them,
exactly one action must exist. Multi-report invocations apply the same selectors
to each report; use separate invocations if indices differ.

```python
from ncu_utils import load_report, metric_record
report, action = load_report("profile/myrun/reports/overview_main.ncu-rep",
                             range_index=0, action_index=1)
print(action.name())
print(metric_record(action, "gpu__time_duration.sum"))
# Keep report alive while using actions, metrics and source information.
```

## Values, availability and units

Records preserve name/raw value/API unit/status. Status distinguishes available,
not_collected, unavailable and error. A name missing from the report does not
prove hardware support; compare with device queries.

Do not assume API unit strings include numeric scaling. Some reports return
duration numerically in nanoseconds while `unit()` says `s`. Preserve raw
data and cross-check the same report's CLI raw CSV with `--units base` and
the metric definition before converting timing or deriving bandwidth/speedup.

Use `metric.value(i)` for typed instances; “try integer first” can silently
truncate floats or zero-fill missing values.

## Correlation

PC metrics have instruction-address correlations; map through
`action.source_info(pc)` and keep unknown mappings as PC evidence.
PM correlations are timestamps (documented as ns); preserve ordering and values.
Never align separate passes by array index.
Other instance arrays can represent different hardware units; length alone
does not make them per-SM.

Stalls identify waiting instructions, not necessarily producers.
Hotspot output separates productive selected / eligible-not-selected states.
It also keeps all-state samples and the overlapping not-issued subset separate
(--sample-scope all or not-issued); adding both would double-count stalls.

## Fallback

Archive scalar metrics, availability, report path/range/action and rule results.
Compare only matching names and reported units; cross-device peak denominators
differ. The analyzer does not establish workload equivalence.

Without Python API support, export raw CSV/details/source using NCU CLI and
inspect headers/units. Do not install a similarly named third-party module.
Source: [NVIDIA Python Report Interface](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html).
