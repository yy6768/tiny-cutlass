# Profile 目录布局

**首先读这个。** 目录混乱是最常见的问题来源。

---

## 顶层规则

所有 profiling 产物在 `profile/` 下。不要把 `.ncu-rep` 散落在 `build/` 或 `csrc/` 中。

```
<repo_root>/
├── profile/                        ← 所有 profiling 产出
│   ├── <run_1>/
│   ├── <run_2>/
│   └── ...
├── scripts/profile/                ← 工具脚本（不变）
├── csrc/                           ← 源码（不动）
└── build/                          ← 编译产物
```

---

## 一次 run = 一个子目录

每次 profile（新 kernel、新版本、新 workload）**创建新目录**。永远不要往已有 run 里写。

好名字（kebab-case，描述**什么**被 profile 了）：
```
profile/naive_attention_baseline/
profile/tiled_attention_v2_optimized/
profile/swin_window_attn_128sm/
profile/gemm_fp16_m128n128k64/
```

坏名字：
```
profile/test/                   # 太模糊
profile/run1/                   # 无意义
profile/20260628/               # 日期无上下文
profile/final/                  # 从来没有"最终"
```

---

## 标准 run 目录结构

```
profile/<run_name>/
├── REPORT.md                       ← 人类可读报告
├── harness/                        ← (可选) 独立 harness
│   ├── my_kernel_harness.cu
│   └── my_kernel_harness.exe
├── reports/
│   ├── full_<tag>.ncu-rep          ← ncu --set full 输出
│   └── source_<tag>.ncu-rep        ← ncu --set source 输出
└── analysis/
    ├── metrics_all_<tag>.json      ← 全量 2000+ metrics 归档
    ├── metrics_key_<tag>.json      ← 精选 key metrics
    ├── metrics_key_<tag>.txt       ← 同上，文本格式
    ├── details_<tag>.txt           ← ncu --page details 规则引擎输出
    ├── raw_<tag>.csv               ← ncu CSV 导出
    ├── stall_hotspots_<tag>.txt    ← per-line stall 聚合
    └── pm_timeline_plots.txt       ← ASCII PM 时序图
```

`<tag>` = workload/dispatch-path 的短标签，如 `default`、`path_a_shape1024`。
单 tag 时可以省略后缀（`full_default.ncu-rep`），但有多个时必须区分。

---

## 对比两个 run

```
profile/<kernel>_v2_vs_v1/
├── REPORT.md                       ← 对比结论
└── analysis/
    └── compare_v1_vs_v2.txt        ← analyze_report.py 输出的对比表
    (不含 .ncu-rep — 它们在各自的 run 中)
```

---

## 不该进 run 目录的东西

- `.ncu-rep.old` 备份 — 应该是单独的 run
- 临时文件 — 用系统 temp 目录
- 数据集/workload 文件 — 通过路径引用，不要复制进来
- 编译中间文件（*.obj）— 放 `harness/build/` 下或重编即可
- `__pycache__` — .gitignore 已忽略

---

## .gitignore（已配置）

本项目的 `.gitignore` 已忽略：
- `profile/` 下的 `.ncu-rep`、`.csv`、`.json`（大文件）
- 只有 `REPORT.md` 适合提交到 git

---

## 使用 `run_ncu.py` 时

`run_ncu.py --run-name <name>` 会自动：
1. 创建 `profile/<name>/reports/` 和 `profile/<name>/analysis/`
2. 收集 full + source reports
3. 导出 details + CSV
4. 调用三个分析脚本

你只需要最后写 `REPORT.md`。
