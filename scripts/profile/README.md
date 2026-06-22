# Profiling 工具链

本目录包含 tiny-cutlass 项目的 NCU (Nsight Compute) profiling 工具集，
借鉴 [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) 的设计理念。

## 核心原则

1. **Profile → Diagnose → Plan**：先有数据再下结论，不猜测
2. **One run = one directory**：每次 profiling 独立目录，不互相污染
3. **`-lineinfo` 必须**：没有 source mapping 就没有 per-line stall 分析
4. **结构化输出**：.ncu-rep + 分析产出物在固定位置

## 目标硬件

SM89 (RTX 4090 / Ada Lovelace) 为主。Metric 名称以 SM89 验证。

## 快速开始

```bat
rem 一键收集 + 分析
python scripts/profile/run_ncu.py ^
    --run-name naive_baseline ^
    --exe build/csrc/flash-attention/Release/naive_attention.exe ^
    --kernel-regex "attention_.*" ^
    --args "--batch_size=1 --head_number=32 --head_size=256 --head_size_v=256 --seq_length=1024 --seq_length_kv=1024 --iterations=1 --reference-check=false" ^
    --set both --count 1
```

产出物在 `profile/naive_baseline/`：
```
profile/naive_baseline/
├── reports/
│   ├── full_default.ncu-rep       ← ncu --set full 输出
│   └── source_default.ncu-rep     ← ncu --set source 输出
└── analysis/
    ├── metrics_key_default.json   ← 精选 key metrics
    ├── metrics_key_default.txt    ← 同上，可读文本
    ├── metrics_all_default.json   ← 全量 2000+ metrics 归档
    ├── details_default.txt        ← NCU 内置规则建议（含 Est. Speedup）
    ├── raw_default.csv            ← CSV 原始导出
    ├── stall_hotspots_default.txt ← Per-line stall 热点排名
    └── pm_timeline_plots.txt      ← ASCII 时序图
```

## 脚本说明

| 文件 | 用途 |
|---|---|
| `run_ncu.py` | 统一入口：创建目录 → ncu 收集 → 导出 → 分析 |
| `ncu_utils.py` | 核心工具函数（加载报告、安全读取 metric、SM89 metric 列表） |
| `analyze_report.py` | 从 .ncu-rep 提取关键 metrics，支持多报告对比 |
| `extract_stalls.py` | 聚合 per-PC stall samples 到源码行级热点 |
| `plot_timeline.py` | ASCII PM-sampling 时序图（揭示 tail effect） |

## 单独使用分析脚本

已有 .ncu-rep 文件时，可跳过收集直接分析：

```bat
rem 重新分析已有报告
python scripts/profile/run_ncu.py --run-name naive_baseline --exe dummy --analyze-only

rem 或单独调用
python scripts/profile/analyze_report.py ^
    --run-dir profile/naive_baseline ^
    --report profile/naive_baseline/reports/full_default.ncu-rep --tag default

python scripts/profile/extract_stalls.py ^
    --run-dir profile/naive_baseline ^
    --report profile/naive_baseline/reports/source_default.ncu-rep --tag default
```

## A/B 对比

优化前后对比两个 run 的 key metrics：

```bat
python scripts/profile/analyze_report.py ^
    --run-dir profile/naive_compare ^
    --report profile/naive_v1/reports/full_default.ncu-rep --tag v1 ^
    --report profile/naive_v2/reports/full_default.ncu-rep --tag v2
```

生成 `compare_v1_vs_v2.txt`，逐项对照。

## 分析维度（六维诊断法）

每次 profile 后按以下六个维度逐一检查：

1. **SM Occupancy & Launch Geometry** — grid 够大吗？occupancy 受限于什么？
2. **Thread-block Balance (Tail Effect)** — block 执行时间一致吗？PM timeline 尾巴长吗？
3. **Stall Reason Breakdown** — warp 不 issue 时在等什么？哪行代码最热？
4. **Tensor Core Utilization** — 矩阵运算用上 TC 了吗？
5. **SM Utilization Timeline** — flat-high / tail / sawtooth？
6. **Memory Access Pattern** — coalescing 好吗？cache 命中？spill 了吗？

详见 `reference/` 目录。

## 参考文档

- `reference/workflow.md` — 端到端 profiling 流程
- `reference/collection.md` — ncu 命令 recipes
- `reference/analysis_dimensions.md` — 六维分析详解
- `reference/diagnosis_playbook.md` — 信号 → 诊断 → 修复
- `reference/sm89_metrics.md` — SM89 metric 名称参考

## 前置要求

- CUDA Toolkit + nvcc（用于 `-lineinfo` 编译）
- Nsight Compute CLI `ncu`（推荐 2025.x+）
- `ncu_report` Python 模块（Nsight Compute 自带，在 `extras/python/` 下）
- Python 3.10+

设置 PYTHONPATH：
```bat
set PYTHONPATH=%PYTHONPATH%;C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\extras\python
```
