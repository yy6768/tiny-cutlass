---
name: ncu-report
description: Profile CUDA kernels with Nsight Compute on SM89 (RTX 4090/4070). Use when the user asks to profile a kernel, analyze its performance, diagnose bottlenecks, read an ncu report, or write an optimization plan — including variants like "profile 一下", "为什么慢", "ncu 报告", "性能分析".
---

# Skill: CUDA Kernel Profiling (SM89 / Nsight Compute)

**When to use:** user asks to profile a CUDA kernel, analyze its performance, find bottlenecks, or write an optimization plan based on Nsight Compute data. Triggers include: "profile X", "为什么这个 kernel 慢", "ncu report", "下一步怎么优化", "帮我看一下这份 ncu 报告", "性能分析".

**Target hardware:** NVIDIA RTX 4090/4070 (SM89, CC 8.9, 128 SM, ~1 TB/s DRAM BW, 4th gen Tensor Core with FP8/FP16/BF16/TF32/INT8).

---

## Golden Rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing CUDA kernels are under-performing for exactly one reason that ncu can tell you in 10 seconds. Don't invent hypotheses before you have the report. Don't start coding a fix before you've matched the observed pattern to a known diagnosis.

---

## Quick Start — "Profile this kernel"

### Step 1: Build (确保有 line-info)

本项目的 CMake target 已默认带 `--generate-line-info`。直接编译：

```bat
cmake -S . -B build -DTINY_CUTLASS_BUILD_<FAMILY>=ON
cmake --build build --config Release --target <target_name>
```

### Step 2: Verify Correctness (必须先过 correctness gate)

```bat
build\csrc\<family>\Release\<target>.exe --reference-check=true [small shape args]
```

### Step 3: Profile (一条命令搞定)

```bat
python scripts/profile/run_ncu.py ^
    --run-name <descriptive_name> ^
    --exe build\csrc\<family>\Release\<target>.exe ^
    --kernel-regex "<kernel_name_regex>" ^
    --args "<exe_args>" ^
    --set both --count 1
```

这会自动：
1. 创建 `profile/<run_name>/` 目录结构
2. 执行 `ncu --set full + PmSampling`
3. 执行 `ncu --set source + SourceCounters`
4. 导出 details page + CSV
5. 调用 `analyze_report.py` / `extract_stalls.py` / `plot_timeline.py`

### Step 4: Diagnose

按六维分析法检查 `profile/<run_name>/analysis/` 下的产出：
1. `metrics_key_default.txt` — headline numbers + occupancy + stalls
2. `details_default.txt` — NCU 规则引擎建议（**先看这个**，每条带 `Est. Speedup: X%`）
3. `stall_hotspots_default.txt` — per-line stall 热点
4. `pm_timeline_plots.txt` — 时序图（揭示 tail effect / pipeline bubble）

### Step 5: Write REPORT.md

在 `profile/<run_name>/REPORT.md` 中写诊断报告。用 `reference/report_template.md` 的模板。

---

## File Index

### 工具脚本 (`scripts/profile/`)

| File | Purpose |
|---|---|
| `run_ncu.py` | **统一入口** — 创建目录 → ncu 收集 → 导出 → 分析，一条命令搞定 |
| `ncu_utils.py` | 核心工具：加载 .ncu-rep、安全 metric 读取、SM89 key metrics 列表 |
| `analyze_report.py` | 提取 key metrics + 多报告对比 |
| `extract_stalls.py` | Per-line stall hotspots（需要 source-level 报告） |
| `plot_timeline.py` | ASCII PM sampling 时序图 |

### Helpers (`scripts/profile/helpers/`)

| File | Purpose |
|---|---|
| `harness_template.cu` | 独立 profiling harness 模板（当不想走 CMake 时使用） |
| `safetensors_loader.h` | Header-only safetensors 读取器（加载真实 workload tensor） |
| `README.md` | 使用说明 |

### Reference Docs (`scripts/profile/reference/`)

| File | Purpose |
|---|---|
| `directory_layout.md` | **先读** — 目录命名规则 |
| `workflow.md` | 端到端 profiling 流程（Phase 0-6） |
| `harness_guide.md` | 何时/如何构建独立 harness |
| `collection.md` | ncu 命令 recipes（SM89 + Windows） |
| `python_api.md` | `ncu_report` Python API 详细参考 |
| `analysis_dimensions.md` | 六维分析法 |
| `diagnosis_playbook.md` | Signal → Diagnosis → Fix（14 patterns A-N） |
| `report_template.md` | REPORT.md 模板与风格指南 |
| `sm89_metrics.md` | SM89 metric name 完整参考 |
| `common_issues.md` | 权限、PM Sampling、匹配失败等排查 |

---

## Critical Lessons

1. **先看 `details_<tag>.txt`。** NCU 规则引擎的 `Est. Speedup: X%` 通常直接指向答案。

2. **编译必须带 `--generate-line-info`。** 本项目 CMake 已默认开启。如果 source view 空白 = 没有 line info。

3. **PM sampling 是唯一能看到 tail effect 的方式。** 静态 metric 做全 kernel 平均；只有时序图能揭示利用率随时间的形状。

4. **不要委托理解。** 自己跑 profile，打开报告，引用具体 metric 值。不要写"profile 显示是 memory-bound" — 而是说明哪两三个 metric 值支撑了这个结论。

5. **Sectors/request = 4 是理想值。** > 8 = 严重 uncoalesced。用 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` 计算。

6. **`local_ld > 0` = register spill。** 这是 DRAM 速度的溢出。用 `__launch_bounds__` 或拆 kernel。

---

## Driver Script (Smoke Test)

验证整个 profiling 工具链工作正常：

```bat
rem 如果有已编译的 target 可用：
python scripts/profile/run_ncu.py ^
    --run-name smoke_test ^
    --exe build\csrc\flash-attention\Release\naive_attention.exe ^
    --kernel-regex ".*" ^
    --args "--batch_size=1 --head_number=1 --head_size=32 --head_size_v=32 --seq_length=32 --seq_length_kv=32 --iterations=1 --reference-check=false" ^
    --set full --count 1

rem 检查产出
dir profile\smoke_test\analysis\
type profile\smoke_test\analysis\metrics_key_default.txt
```

如果 `ncu_report` import 失败：
```bat
set PYTHONPATH=%PYTHONPATH%;C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\extras\python
```

---

## Integration with cutlass-kernel Skill

本 skill 与 `cutlass-kernel` skill 配合使用：
- `cutlass-kernel` 定义 kernel 开发的 **build → verify → bench** 工作流
- `ncu-report` 在 bench 之后提供 **深度性能分析** 能力

典型流程：
1. `cutlass-kernel` workflow: build → verify → bench → 发现 kernel 比 cuBLAS 慢
2. `ncu-report` workflow: profile → diagnose → 写 REPORT.md → 得到优化方向
3. 回到 `cutlass-kernel`: 按优化方向实现新 variant → verify → bench → 确认提升
