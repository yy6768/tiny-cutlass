# Profiling 工作流 — 端到端

从"要 profile 一个 kernel"到"写出诊断报告"的完整流程。

---

## Phase 0 — 创建 run 目录

```bat
set PROFILE_RUN_DIR=profile\<descriptive_run_name>
mkdir %PROFILE_RUN_DIR%\reports
mkdir %PROFILE_RUN_DIR%\analysis
```

命名规则：kebab-case，描述**什么**被 profile 了。
- ✅ `naive_attention_baseline`、`tiled_attention_v2_optimized`
- ❌ `test`、`run1`、`final`

使用 `run_ncu.py` 时目录会自动创建。

---

## Phase 0.5 — 明确问题

开始前回答：
1. **Profile 哪个 kernel？** 精确的 kernel 名或 regex
2. **什么输入 shape？** 必须是实际 workload 的代表性 shape
3. **什么 dispatch path？** 不同 shape 可能走不同 template 实例
4. **要回答什么问题？** "为什么慢"太笼统 → "是 latency-bound 还是 bandwidth-bound？"

---

## Phase 1 — 环境检查

```bat
ncu --version
nvidia-smi
nvcc --version
python -c "import ncu_report; print('OK')"
```

如果 `ncu_report` 导入失败：
```bat
set PYTHONPATH=%PYTHONPATH%;C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\extras\python
```

---

## Phase 2 — 构建 profile 目标

确保编译带 `-lineinfo`（本项目 CMake 中 flash-attention 和 swin 已默认开启 `--generate-line-info`）。

```bat
cmake -S . -B build -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON
cmake --build build --config Release --target naive_attention
```

先验证 correctness：
```bat
build\csrc\flash-attention\Release\naive_attention.exe ^
    --batch_size=1 --head_number=1 --head_size=32 --head_size_v=32 ^
    --seq_length=32 --seq_length_kv=32 --iterations=1 --reference-check=true
```

---

## Phase 3 — 收集 profiles

使用统一入口：
```bat
python scripts/profile/run_ncu.py ^
    --run-name naive_baseline ^
    --exe build\csrc\flash-attention\Release\naive_attention.exe ^
    --kernel-regex "attention_.*" ^
    --args "--batch_size=1 --head_number=32 --head_size=256 --head_size_v=256 --seq_length=1024 --seq_length_kv=1024 --iterations=1 --reference-check=false" ^
    --set both --count 1
```

或手动分两步：
```bat
rem (1) Full overview + PM sampling
ncu --set full --section PmSampling ^
    -k "regex:attention_.*" -c 1 ^
    -o profile\naive_baseline\reports\full_default ^
    build\...\naive_attention.exe [args]

rem (2) Source-level stall sampling
ncu --set source --section SourceCounters ^
    -k "regex:attention_.*" -c 1 ^
    -o profile\naive_baseline\reports\source_default ^
    build\...\naive_attention.exe [args]
```

---

## Phase 4 — 提取结构化数据

`run_ncu.py` 自动完成。手动时：

```bat
python scripts/profile/analyze_report.py ^
    --run-dir profile\naive_baseline ^
    --report profile\naive_baseline\reports\full_default.ncu-rep --tag default

python scripts/profile/extract_stalls.py ^
    --run-dir profile\naive_baseline ^
    --report profile\naive_baseline\reports\source_default.ncu-rep --tag default

python scripts/profile/plot_timeline.py ^
    --run-dir profile\naive_baseline ^
    --report profile\naive_baseline\reports\full_default.ncu-rep --tag default
```

---

## Phase 5 — 诊断

按六维分析法逐一检查（见 `analysis_dimensions.md`）：
1. SM occupancy & launch geometry
2. Thread-block balance (tail effect)
3. Stall reason breakdown + per-line hotspots
4. Tensor Core utilization
5. SM utilization timeline
6. Memory access pattern

对照 `diagnosis_playbook.md` 中的 pattern。

---

## Phase 6 — 写报告

在 `profile/<run_name>/REPORT.md` 中写：
1. Setup section（怎么 profile 的，可复现）
2. Headline numbers 表格
3. Per-dimension 分析 + 引用具体 metric 值
4. Optimization directions（按影响排序，最多 3-5 条）
5. Confidence & caveats

---

## 反面模式

- ❌ 不引用具体 metric 值的结论（"SM 利用率低" → 应给出百分比）
- ❌ 用合成 shape 代替真实 workload
- ❌ 不排序就列举优化建议
- ❌ 不先验证 correctness 就 profile
