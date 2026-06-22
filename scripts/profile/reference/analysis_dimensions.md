# 六维分析法

每个 kernel profile 报告都要按以下六个维度逐一检查。不要在第一个发现处停下。

---

## 维度 1 — SM Occupancy & Launch Geometry

**问题：** grid 够大吗？occupancy 被什么限制？

**关键 Metrics：**
```
launch__grid_size
launch__block_size
launch__waves_per_multiprocessor
launch__registers_per_thread
launch__shared_mem_per_block
launch__occupancy_limit_blocks / _registers / _shared_mem / _warps
device__attribute_multiprocessor_count          (RTX 4090 = 128 SM)
sm__maximum_warps_per_active_cycle_pct          (理论 occupancy %)
sm__warps_active.avg.pct_of_peak_sustained_active  (实际 occupancy %)
```

**如何判断：**
- `waves_per_multiprocessor < 1`：grid 太小，有 SM 全程空闲
- `waves_per_multiprocessor ∈ [1, 2)`：有 tail wave，尾部效应
- `waves_per_multiprocessor > 4`：grid 足够大
- 理论 occupancy 100% 但实际远低于：瓶颈在 stall，看维度 3
- `occupancy_limit_registers` 最紧：考虑 `__launch_bounds__` 或减少活跃变量

**Wave 数学：**
```
blocks_per_sm = min(occ_limit_blocks, occ_limit_registers, occ_limit_shared_mem, occ_limit_warps)
wave_size = blocks_per_sm × num_sms
num_waves = ceil(total_blocks / wave_size)
last_wave_utilization = (total_blocks - (num_waves-1) × wave_size) / wave_size × 100%
```

---

## 维度 2 — Thread-block Balance (Tail Effect)

**问题：** 所有 block 是否差不多同时完成？还是有几个拖后腿的？

**信号：**
- NCU details 页："One or more SMs have a much lower number of active cycles"
- PM timeline 尾部渐降（用 `plot_timeline.py` 查看）
- 时间线形状分类：
  - **Flat high → clean drop**：理想
  - **Flat high → gradual tail**：tail effect，几个慢 block 拖延
  - **Flat low**：grid 太小（回看维度 1）
  - **Sawtooth**：compute/memory 交替，没有 overlap

**常见原因：**
1. 变长内循环（如 seq_len 不一致的 attention）
2. 条件提前 return
3. 不均匀的 work-stealing

---

## 维度 3 — Stall Reason Breakdown + Per-line Hotspots

**问题：** warp 不 issue 时在等什么？哪行源码最热？

**Aggregate Metrics：**
```
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
...
```

**Per-line Metrics（需要 --set source）：**
```
smsp__pcsamp_warps_issue_stalled_long_scoreboard
smsp__pcsamp_warps_issue_stalled_short_scoreboard
...
```

**Stall 原因速查：**

| 原因 | 含义 | 修复方向 |
|---|---|---|
| `long_scoreboard` | 等 global memory load | coalesce, 加 ILP, prefetch |
| `short_scoreboard` | 等 shared/local/计算链 | 加 ILP, 缩短依赖链 |
| `wait` | 等 SFU/tensor core | 更多独立操作 |
| `barrier` | `__syncthreads` | 减少 sync, 修 divergence |
| `math_pipe_throttle` | FMA 管线饱和 | 已是 compute-bound，找别处 |
| `mio/lg_throttle` | LSU 管线饱和 | 向量化, 用 shared mem |
| `not_selected` | 有资格但没被选中 | **好信号**，有并行度 |
| `selected` | 正在 issue | **生产周期** |

**规则：**
- `long_scoreboard > 40%` samples：memory-latency-bound → 看维度 6
- `barrier > 20%`：同步开销过大
- `selected < 10%`：几乎没有有效 issue

---

## 维度 4 — Tensor Core Utilization

**问题：** 矩阵运算用上 TC 了吗？效率如何？

**Metrics：**
```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
```

**判断：**
- `= 0%`：没用 TC。若是 GEMM/attention/conv，这是重大优化机会
- `< 50%`：TC 被用但 underutilized，通常数据供应不上
- `> 50%`：TC 利用良好

---

## 维度 5 — SM Utilization Timeline

**问题：** SM 利用率随时间如何变化？

用 `plot_timeline.py` 观察 PM sampling 时序：
```
pmsampling:sm__throughput.avg.pct_of_peak_sustained_elapsed
pmsampling:dram__throughput.avg.pct_of_peak_sustained_elapsed
pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg
```

**形状分类：**
- Flat high, clean drop → 理想
- Flat high, long tail → tail effect（维度 2）
- Flat low → grid 太小 or stall-bound
- Sawtooth → compute/memory 没 overlap，需要 double-buffer / pipeline

---

## 维度 6 — Memory Access Pattern & Cache Efficiency

**问题：** global load coalesced 吗？cache 命中？DRAM 带宽实际在用吗？

**Metrics：**
```
dram__bytes_read.sum.pct_of_peak_sustained_elapsed    ← DRAM BW 利用率
l1tex__t_sector_hit_rate.pct                         ← L1 命中率
lts__t_sector_hit_rate.pct                           ← L2 命中率
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       ← 总 sectors
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum      ← 总 requests
    sectors/request: 理想 = 4 (128B aligned)，> 8 说明严重 uncoalesced
smsp__sass_inst_executed_op_local_ld.sum             ← > 0 表示 register spill！
```

**判断：**
- DRAM BW > 80%：真正 bandwidth-bound → 减少 bytes 或用 shared mem
- DRAM BW < 10% 且 kernel 慢：不是 BW-bound → latency-bound（看维度 3）
- sectors/request > 8：严重 uncoalesced，重排线程-数据映射
- local_ld > 0：register spill → `__launch_bounds__` 或拆 kernel

---

## 综合诊断

六维走完后，写一行总结：

> "Kernel 运行在 X% peak SM throughput / Y% peak DRAM BW（维度 1,6）。
> Stall 以 `<reason>` 为主（Z% samples，维度 3），集中在 N 行源码，
> 访存模式为 coalesced/uncoalesced（维度 6）。
> PM timeline 呈 flat/tail/sawtooth（维度 2/5）。
> Tensor core 使用率 W%（维度 4）。"

填入实际数字，这就是交付物。
