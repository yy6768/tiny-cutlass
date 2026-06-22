# 诊断 Playbook — 信号 → 诊断 → 修复

按 NCU 观察到的信号，查找对应的 pattern，获得诊断和修复方向。

大多数 kernel 会同时命中 2-4 个 pattern。**按影响排序**（用 NCU `Est. Speedup: X%`），先修最大的。

---

## Pattern A — 小 Grid / SM 空闲

**信号：**
- `launch__waves_per_multiprocessor < 0.5`
- `launch__grid_size < 128`（RTX 4090 有 128 SM）
- NCU 规则："The grid is configured to execute only N blocks, which is less than M multiprocessors"

**诊断：** CTA 数少于 SM 数，部分 SM 全程空闲。

**修复：**
- 增加 grid 维度（split-K、按 head/channel 拆分）
- Persistent kernel（每 SM 一个 block，循环 dequeue）
- 与相邻 kernel 融合

---

## Pattern B — Tail Effect（变长输入）

**信号：**
- Per-SM active cycles 分布不均（max 远高于 avg）
- PM timeline 尾部渐降
- `max_seq_len / avg_seq_len > 3`

**诊断：** 每个 CTA 工作量不一致，几个慢的拖延整体。

**修复：**
- 按长度排序 batching
- 拆长序列到多个 CTA + 后置 reduce
- Chunkwise kernel（固定 chunk 大小）

---

## Pattern C — Uncoalesced Global Loads

**信号：**
- `sectors / requests > 5`（理想 = 4）
- NCU："uncoalesced global accesses resulting in N excessive sectors"
- 热点行 stall 类型为 `long_scoreboard`

**诊断：** warp 内线程访问不连续地址，多余 sector 传输。

**修复：**
- 重排 thread↔data 映射（stride → contiguous）
- AoS → SoA
- 用 shared mem 做转置
- 向量化（float4 / ushort2）

---

## Pattern D — Store 效率低

**信号：**
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio < 16`

**诊断：** 只有部分 lane 写出，sector 半空刷出。

**修复：** 先在 warp 内聚合，再由连续 lane 写出。

---

## Pattern E — Latency-bound (long_scoreboard 主导)

**信号：**
- `long_scoreboard > 40%` of stall samples
- `dram BW < 10%`（不是带宽瓶颈）
- 热点行是 global load

**诊断：** load 发出后 stall 等返回，ILP 不够或 occupancy 低。

**修复：**
- 展开 load 循环（4-8 个 load 连发再用）
- 提高 occupancy（更多 warp 掩盖延迟）
- 软件流水线 / double-buffer
- 异步 copy（cp.async）

---

## Pattern F — Compute-bound 但没用 Tensor Core

**信号：**
- `sm__inst_executed_pipe_fma > 50%` of peak
- `sm__pipe_tensor_cycles_active = 0%`
- Workload 是 GEMM / attention / conv

**诊断：** 用标量 FMA 做矩阵运算，没用 TC。RTX 4090 的 TC 是标量的 ~8×。

**修复：** 用 WMMA / mma.sync / CUTLASS。

---

## Pattern G — Atomic 竞争

**信号：** `long_scoreboard` 集中在 ATOM/RED 指令

**修复：** 层级规约（warp → block → grid），减少 atomic 粒度。

---

## Pattern H — Shared Memory Bank Conflict

**信号：** `short_scoreboard` stall 集中在 shared mem load 行

**修复：** padding（`[32][33]`）或 swizzle。

---

## Pattern I — 同步开销过大

**信号：** `barrier > 20%` of stall samples，热点在 `BAR.SYNC`

**修复：**
- 用 warp-level 原语替代 block-level sync
- 减少 `__syncthreads` 次数
- Warp-specialization + mbarrier

---

## Pattern J — 实际 Occupancy 远低于理论

**信号：** 理论 occupancy 高但实际低，gap > 20%

**诊断：** 理论可以放这么多 warp，但实际没有 — 通常由 stall 或 imbalance 导致。

**修复：** 先找主导 stall 原因（Pattern E/H/I），解决它就能关闭 gap。

---

## Pattern K — Register Spill

**信号：**
- `smsp__sass_inst_executed_op_local_ld.sum > 0`
- `launch__registers_per_thread > 128`

**诊断：** 编译器无法把所有活跃变量放进 register，溢出到 local memory（DRAM 速度）。

**修复：**
- `__launch_bounds__(maxThreads, minBlocks)`
- 减少活跃变量（重计算代替缓存、拆 kernel）
- 移到 shared memory

---

## Pattern L — 意外使用 FP64

**信号：** `sm__pipe_fp64_cycles_active > 0` 但 kernel 应该是 FP32

**修复：** 所有浮点字面量加 `f` 后缀：`1.0f`。用 `__expf` 等快速数学函数。

---

## Pattern M — Pipeline Bubble（无 compute/memory overlap）

**信号：** PM timeline sawtooth（SM 吞吐 ↕ DRAM 吞吐交替）

**诊断：** 单 buffer — load tile, compute tile, 再 load。

**修复：** Double-buffer（compute tile A 同时 load tile B）。多 stage pipeline。

---

## Pattern N — Warp Divergence

**信号：** `smsp__thread_inst_executed_per_inst_executed.ratio < 32`

**修复：** 重排数据让同 warp 走同分支；branchless（mask × a + (1-mask) × b）。

---

## 报告排序模板

```
Priority 1: <pattern> — <具体修复>
  证据: <metric 值>
  NCU Est. Speedup: X%
  工程量: low / medium / high

Priority 2: ...
```

最多 3-5 条。超过 5 条会稀释信号。
