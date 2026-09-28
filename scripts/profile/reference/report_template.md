# REPORT.md 模板

Profile 报告是交付物。所有其他产物（`.ncu-rep`、Python 分析、CSV）是证据。

保存到 `profile/<run_name>/REPORT.md`。

---

## 模板

```markdown
# `<kernel_name>` Profiling Report

**Kernel:** `<exact kernel name or template instantiation>`
**Target GPU:** NVIDIA RTX 4090/4070 (128 SM, CC 8.9)
**Nsight Compute:** 2025.2.0
**编译 flags:** `nvcc -O2 -std=c++17 --generate-line-info -gencode=arch=compute_89,code=sm_89`
**Profile date:** YYYY-MM-DD
**Run directory:** `profile/<run_name>/`

---

## 0. Profiling Setup

- **可执行文件：** `build/csrc/.../Release/<target>.exe` 或 `profile/<run>/harness/<harness>.exe`
- **Workload shape：** (具体参数)
- **Dispatch path：** (template 参数、grid/block config)
- **收集命令：**

    python scripts/profile/run_ncu.py ^
        --run-name <run_name> ^
        --exe <path> ^
        --kernel-regex "<regex>" ^
        --args "<args>" ^
        --set both --count 1

### 产出物

    profile/<run_name>/
    ├── REPORT.md                       ← 本文件
    ├── reports/full_default.ncu-rep
    ├── reports/source_default.ncu-rep
    └── analysis/
        ├── metrics_key_default.txt
        ├── details_default.txt
        ├── stall_hotspots_default.txt
        └── pm_timeline_plots.txt

---

## 1. Headline Numbers

| Metric | 值 | 来源 |
|---|---:|---|
| **Duration** | X µs | `gpu__time_duration.sum` |
| SM throughput (% peak) | X% | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM throughput (% peak) | X% | `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` |
| L1 hit rate | X% | `l1tex__t_sector_hit_rate.pct` |
| L2 hit rate | X% | `lts__t_sector_hit_rate.pct` |
| Tensor Core usage | X% | `sm__pipe_tensor_cycles_active.*` |
| Registers / thread | X | `launch__registers_per_thread` |
| 理论 / 实际 Occupancy | X% / Y% | |
| Waves / SM | X | `launch__waves_per_multiprocessor` |

**一句话：** <"Kernel 运行在 X% peak SM throughput — 它是 latency-bound on Y，不是 DRAM-BW-bound。">

---

## 2. Per-dimension 分析

### 2.1 SM Occupancy & Launch Geometry
<grid size, block size, waves/SM, occupancy, 限制因素>

### 2.2 Thread-block Balance (Tail Effect)
<per-SM 分布, PM timeline 形状>

### 2.3 Stall Breakdown + Hotspots
<stall 百分比, 热点源码行 file:line + 类型 + 样本数>

### 2.4 Tensor Core Utilization
<TC % 或 "0%, N/A">

### 2.5 SM Utilization Timeline
<flat-high / tail / sawtooth — 引用 pm_timeline_plots.txt>

### 2.6 Memory Access Pattern
<sectors/request, L1/L2, DRAM BW, store 效率, register spill>

### 2.7 NCU 规则引擎建议
<从 details_default.txt 提取, 每条带 Est. Speedup: X%>

---

## 3. Summary Diagnosis

| 因素 | 状态 | 影响 |
|---|---|---|
| <因素 1> | <状态> | <排序> |
| <因素 2> | ... | ... |

---

## 4. Optimization Directions（按影响排序）

### Priority 1 — <一行描述>

<具体做什么，引用源码行/函数名>

**证据：**
- <metric + 值>
- <NCU rule + Est. Speedup>

**预期收益：** <X%>
**工程量：** <low/medium/high>

### Priority 2 — ...

### Priority 3 — ...

（最多 3-5 条）

---

## 5. Confidence & Caveats

- 确信的：<列表>
- 不确定的：<列表 + 如何验证>
- Profile 无法回答的：<列表>
```

---

## 风格规则

- **每个结论必须引用具体 metric 值。** "SM 利用率 37.2%" > "SM 利用率低"
- **引用文件和行号。** "harness.cu:42 的 LDG 指令" > "主循环中的 load"
- **用 NCU 的估计。** `Est. Speedup: X%` 通常量级正确
- **按影响排序。** 先修 50% 的问题，再修 5% 的
- **头部摘要要密。** 读者应该 10 秒内看到 #1 发现
- **链接到 analysis/ 下的文件。** 不要把大表格粘到 prose 里

## 反模式

- ❌ 没有证据的泛泛建议（"可以考虑用 shared memory"）
- ❌ 超过 5 条 "priority" — 在凑数
- ❌ 不引用 metric 的结论
- ❌ 直接贴 CLI 输出 — 应该提取、解读、写结论
- ❌ 省略 setup section — 没有它没人能复现
