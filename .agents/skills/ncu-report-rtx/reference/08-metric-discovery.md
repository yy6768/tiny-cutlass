# Metric discovery: SM80 / SM86 / SM89 / SM120

Device, driver, NCU and sections determine availability. `KEY_METRICS` is an
extraction candidate catalog, not guaranteed hardware support or a collection request.

## Device versus report inventory

Save `ncu --query-metrics --devices 0 --query-metrics-mode all`,
versions/help/sections/sets. To query a base:
`ncu --query-metrics --devices 0 --query-metrics-mode suffix --metrics sm__throughput`.

Then enumerate `action.metric_names()` for the selected report action.
A supported metric may simply not have been collected.
For offline queries first `--list-chips`, then plural `--chips` with a listed
identifier. GB202 is RTX, not a datacenter chip.
Check installed help before `--query-metrics-collection pmsampling`.

| Question | Candidates |
|---|---|
| Identity | device__attribute_display_name, compute_capability_major/minor, multiprocessor_count |
| Launch | launch__grid_size, block_size, registers_per_thread, shared_mem_per_block, waves_per_multiprocessor |
| Occupancy | sm__warps_active.avg.pct_of_peak_sustained_active; Occupancy section |
| Issue | smsp__warps_eligible.avg.per_cycle_active, smsp__issue_active.avg.pct_of_peak_sustained_active |
| SOL | sm__throughput.avg.pct_of_peak_sustained_elapsed, dram__throughput.avg.pct_of_peak_sustained_elapsed |
| Traffic | dram__bytes_read.sum, dram__bytes_write.sum, lts__t_sector_hit_rate.pct |
| Coalescing | l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / matching requests |
| Tensor | sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed; available HMMA/IMMA counters |
| Source | smsp__pcsamp_warps_issue_stalled_*; SourceCounters |
| Timeline | Actual pmsampling:* names and timestamp correlations |

Abbreviations/wildcards in the table are descriptive, not literal CLI names.

## Semantics

Sum/avg/max differ. Active-cycle and elapsed-cycle peak percentages cannot be
substituted. Per-issued-warp ratios are not percentages. Missing/unavailable/zero
are distinct. Preserve API unit and raw value; verify scale through CSV.

Sum read/write bytes only when both exist with matching scope/units; label the
derived formula. Divide sectors by requests only with a nonzero denominator,
then account for access width/lanes. Cross-device peak percentages have different
denominators. Do not invent architecture-specific aliases for unlike counters.

Source: [NCU CLI/metrics](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
