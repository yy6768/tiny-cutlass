# `ncu_report` Python API 参考

使用 Python 模块（而非 CLI 输出）做结构化提取、对比和归档。

---

## 导入

```python
# 设置 PYTHONPATH 指向 Nsight Compute 的 extras/python
# Windows:
#   set PYTHONPATH=%PYTHONPATH%;C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\extras\python
# 或让 ncu_utils.py 自动定位

from ncu_utils import load_action, safe, safe_many, per_instance_values, dump_all_metrics
```

本项目的 `scripts/profile/ncu_utils.py` 会自动尝试常见路径。如果失败：

```bat
set PYTHONPATH=%PYTHONPATH%;C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\extras\python
python -c "import ncu_report; print('OK')"
```

---

## 基本加载

```python
import ncu_report

report = ncu_report.load_report("path/to/full_default.ncu-rep")

# 一个 report 包含多个 "range"，每个 range 含多个 "action"（= 一次 kernel launch）
# 用 -c 1 收集时，只有一个 range 一个 action
rng = report.range_by_idx(0)
action = rng.action_by_idx(0)

print(f"Kernel name: {action.name()}")
print(f"Total metrics: {len(action.metric_names())}")
```

---

## 读取单个 metric

```python
from ncu_utils import safe

sm_util = safe(action, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
duration_ns = safe(action, "gpu__time_duration.sum")
dram_bw = safe(action, "dram__bytes_read.sum.per_second")

print(f"SM throughput: {sm_util}%")
print(f"Duration: {duration_ns/1e3:.2f} µs")
print(f"DRAM read BW: {dram_bw/1e9:.2f} GB/s")
```

**总是用 `safe()` 包裹。** Metric 名在不同 GPU 代际可能不同 — `safe()` 出错时返回 `None` 而非崩溃。

---

## 枚举可用 metrics

```python
all_names = action.metric_names()  # 2000+ metrics for --set full

# 按关键字过滤
for name in sorted(all_names):
    if "tensor" in name:
        print(name, "=", safe(action, name))
```

当不确定 metric 是否存在时，先用这个方法确认。

---

## Per-instance 值（PM Sampling 时序、Per-SM 等）

```python
from ncu_utils import per_instance_values

# PM sampling — 时间有序的采样序列
vals = per_instance_values(action, "pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg")
if vals:
    print(f"PM samples: {len(vals)}")
    print(f"Max stall: {max(vals):.3f}")
```

PM sampling 的 index i 是时间顺序 — 可以用来画时序图（`plot_timeline.py` 做的就是这个）。

---

## Per-PC → 源码行映射

Source-level 报告（`--set source --section SourceCounters`）有 per-PC stall 采样数据：

```python
from ncu_utils import per_pc_values, pc_to_source_line

stalls = per_pc_values(action, "smsp__pcsamp_warps_issue_stalled_long_scoreboard")
# stalls = [(pc, count), (pc, count), ...]

# 聚合到 (file, line)
from collections import defaultdict
per_line = defaultdict(int)
for pc, count in stalls:
    file, line = pc_to_source_line(action, pc)
    per_line[(file, line)] += count

# 排序输出
for (f, l), total in sorted(per_line.items(), key=lambda x: -x[1])[:10]:
    print(f"  {total:>8}  {f}:{l}")
```

需要编译时使用 `--generate-line-info`（本项目 CMake 已默认开启）。

---

## ValueKind 处理

每个 metric 有特定的值类型：

```python
from ncu_utils import metric_value_at

m = action["gpu__time_duration.sum"]
print(f"Kind: {m.kind()}")      # UINT64, DOUBLE, FLOAT, STRING
print(f"Value: {m.value()}")    # 聚合值
print(f"Unit: {m.unit()}")      # "ns", "%", "cycle" 等

# Per-instance 时需要按 kind 选择访问方法 — metric_value_at() 已封装
n = m.num_instances()
if n > 0:
    for i in range(min(n, 5)):
        print(f"  [{i}] = {metric_value_at(m, i)}")
```

---

## action / metric 常用方法

```python
# Action（= 一次 kernel launch 的 profile 数据）
action.name()                    # demangled kernel name
action.metric_names()            # 所有 metric 名列表
action.metric_by_name(name)      # 同 action[name]
action.source_info(pc)           # PC → SourceInfo (file, line) — 需要 -lineinfo
action.rule_results_as_dicts()   # NCU 规则引擎结果（结构化）

# Metric
m.value()                        # 聚合值
m.unit()                         # 单位字符串
m.kind()                         # ValueKind (UINT64 / DOUBLE / ...)
m.num_instances()                # per-instance 数量（0 = 只有聚合值）
m.has_correlation_ids()          # True = per-PC metric
m.correlation_ids()              # 与 num_instances 并行的 PC 数组
m.description()                  # metric 人类描述
```

---

## NCU 规则引擎结果提取

`--page details` 中的 "OPT Est. Speedup: X%" 规则可以程序化提取：

```python
results = action.rule_results_as_dicts()
for r in sorted(results, key=lambda x: -(x.get("estimated_speedup_pct") or 0)):
    severity = r.get("type", "?")           # "OPT" / "INF" / "WRN"
    rule_name = r.get("rule_name", "?")
    est = r.get("estimated_speedup_pct", 0)
    msg = r.get("message_for_display", "")
    print(f"[{severity}] {rule_name}: Est. Speedup {est}%")
    print(f"    {msg[:200]}")
```

按 `estimated_speedup_pct` 降序排列 = 最高影响优先。

---

## 对比两个报告

```python
from ncu_utils import load_action, safe, SM89_KEY_METRICS

a1 = load_action("profile/v1/reports/full_default.ncu-rep")
a2 = load_action("profile/v2/reports/full_default.ncu-rep")

t1 = safe(a1, "gpu__time_duration.sum")
t2 = safe(a2, "gpu__time_duration.sum")
print(f"Speedup: {t1/t2:.2f}x")

# 全量对比
for m in SM89_KEY_METRICS[:20]:
    v1, v2 = safe(a1, m), safe(a2, m)
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1:
        chg = (v2 - v1) / v1 * 100
        print(f"{m:70s} {v1:>12.4g} → {v2:>12.4g} ({chg:+.1f}%)")
```

或用 `analyze_report.py --report ... --tag v1 --report ... --tag v2` 自动生成对比文件。

---

## 全量归档

```python
from ncu_utils import dump_all_metrics

n = dump_all_metrics(action, "analysis/metrics_all_default.json")
print(f"Archived {n} metrics")
```

保存为 JSON 后，后续分析不需要重新打开 `.ncu-rep`。

---

## 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `KeyError` on metric name | SM89 上名字不同 | 用 `action.metric_names()` 搜索，或用 `ncu_utils.metric_or_none()` |
| `num_instances() == 0` | 没收集 instanced 数据 | 重跑 ncu 加对应 `--section` |
| `source_info(pc)` 返回 None | 编译时没加 `-lineinfo` | 重编加 `--generate-line-info` |
| Metric 值是字符串 | 枚举类型 | 用 `m.as_string()` 或 `m.value()` |
| `import ncu_report` 失败 | PYTHONPATH 没设 | 见上方导入说明 |
