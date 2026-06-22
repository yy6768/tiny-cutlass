# SM89 (RTX 4090 / Ada Lovelace) Metric 名称参考

本文档列出在 SM89 上验证可用的 NCU metric 名称。
大部分与 SM80 (A100) 通用，少数 SM100 (B200) 专属名称不适用。

---

## 硬件参数

| 参数 | RTX 4090 |
|---|---|
| SM 数 | 128 |
| Compute Capability | 8.9 |
| Max warps / SM | 48 |
| Max threads / SM | 1536 |
| Max blocks / SM | 24 |
| Registers / SM | 65536 |
| Shared Memory / SM | 100 KB (configurable) |
| L2 Cache | 72 MB |
| DRAM Bandwidth | ~1 TB/s |
| Tensor Core Gen | 4th gen (FP8, FP16, BF16, TF32, INT8) |

---

## Launch Geometry

```
launch__grid_size
launch__block_size
launch__grid_dim_x
launch__grid_dim_y
launch__grid_dim_z
launch__block_dim_x
launch__block_dim_y
launch__block_dim_z
launch__waves_per_multiprocessor
launch__registers_per_thread
launch__shared_mem_per_block
launch__shared_mem_per_block_dynamic
launch__thread_count
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
device__attribute_multiprocessor_count
device__attribute_max_warps_per_multiprocessor
```

---

## Timing

```
gpu__time_duration.sum                              (ns)
gpu__time_duration.avg
smsp__cycles_active.avg
smsp__cycles_active.sum
smsp__cycles_elapsed.avg
```

---

## Speed of Light (SOL)

```
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_active
lts__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
```

---

## Occupancy

```
sm__maximum_warps_per_active_cycle_pct
sm__warps_active.avg.pct_of_peak_sustained_active
sm__warps_active.avg.per_cycle_active
sm__warps_active.max.per_cycle_active
sm__warps_active.min.per_cycle_active
smsp__warps_active.avg.per_cycle_active
smsp__warps_eligible.avg.per_cycle_active
smsp__warps_eligible.max.per_cycle_active
```

---

## IPC (Instructions Per Cycle)

```
sm__inst_executed.avg.per_cycle_active
smsp__issue_active.avg.per_cycle_active
smsp__issue_active.avg.pct_of_peak_sustained_active
smsp__inst_executed.avg
smsp__inst_executed.sum
```

---

## Compute Pipes

```
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active
```

---

## Tensor Core

```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active
```

注意：SM89 的 TC 支持 FP8 (e4m3/e5m2)。相关 ops metric 名因 ncu 版本而异，
建议用 `action.metric_names()` 过滤 `tensor` 关键字确认。

---

## DRAM

```
dram__bytes_read.sum
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_read.sum.per_second
dram__bytes_write.sum
dram__bytes_write.sum.pct_of_peak_sustained_elapsed
dram__bytes_write.sum.per_second
dram__sectors_read.sum
dram__sectors_write.sum
```

---

## Cache

```
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct
```

---

## Memory Instructions & Coalescing

```
smsp__sass_inst_executed_op_global_ld.sum
smsp__sass_inst_executed_op_global_st.sum
smsp__sass_inst_executed_op_local_ld.sum            ← > 0 = register spill!
smsp__sass_inst_executed_op_local_st.sum
smsp__sass_inst_executed_op_shared_ld.sum
smsp__sass_inst_executed_op_shared_st.sum

l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum      ← sectors
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum     ← requests
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum

# Coalescing quality = sectors / requests
# 理想: 4 (128B aligned warp-wide load)
# > 8: 严重 uncoalesced

smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio
```

---

## Stall Reasons — Aggregate

```
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio
smsp__average_warps_issue_stalled_misc_per_issue_active.ratio
```

---

## Stall Reasons — Per-PC (requires `--set source`)

```
smsp__pcsamp_sample_count
smsp__pcsamp_warps_issue_stalled_long_scoreboard
smsp__pcsamp_warps_issue_stalled_short_scoreboard
smsp__pcsamp_warps_issue_stalled_wait
smsp__pcsamp_warps_issue_stalled_barrier
smsp__pcsamp_warps_issue_stalled_math_pipe_throttle
smsp__pcsamp_warps_issue_stalled_mio_throttle
smsp__pcsamp_warps_issue_stalled_lg_throttle
smsp__pcsamp_warps_issue_stalled_tex_throttle
smsp__pcsamp_warps_issue_stalled_not_selected
smsp__pcsamp_warps_issue_stalled_dispatch_stall
smsp__pcsamp_warps_issue_stalled_drain
smsp__pcsamp_warps_issue_stalled_no_instructions
smsp__pcsamp_warps_issue_stalled_selected
smsp__pcsamp_warps_issue_stalled_branch_resolving
smsp__pcsamp_warps_issue_stalled_membar
smsp__pcsamp_warps_issue_stalled_sleeping
smsp__pcsamp_warps_issue_stalled_misc
```

---

## PM Sampling (时序)

```
pmsampling:sm__throughput.avg.pct_of_peak_sustained_elapsed
pmsampling:sm__warps_active.avg.pct_of_peak_sustained_active
pmsampling:dram__throughput.avg.pct_of_peak_sustained_elapsed
pmsampling:l1tex__throughput.avg.pct_of_peak_sustained_active
pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_wait.avg
pmsampling:smsp__warps_issue_stalled_dispatch_stall.avg
pmsampling:smsp__warps_issue_stalled_math_pipe_throttle.avg
pmsampling:smsp__warps_issue_stalled_mio_throttle.avg
```

---

## SM89 vs SM100 (B200) 差异

以下名称在 SM100 上可用但 SM89 上**可能不存在**：
- `gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed`（SM89 用 `gpu__compute_memory_throughput...` 替代）
- `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg`（SM100 特有 sparsity 标记）

建议：遇到 `None` 时用 `ncu_utils.metric_or_none()` 提供 fallback 候选名。

---

## 验证 metric 是否可用

```python
from ncu_utils import load_action

action = load_action("path/to/report.ncu-rep")
all_names = action.metric_names()

# 搜索含特定关键字的 metric
tensor_metrics = [n for n in all_names if "tensor" in n]
```
