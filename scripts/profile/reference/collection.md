# NCU 命令 Recipes

适配 SM89 (RTX 4090) + Windows 环境。

---

## Recipe 1: Full overview（首次 profile 必做）

```bat
ncu --set full ^
    --section PmSampling ^
    -k "regex:KERNEL_REGEX" ^
    -c 1 ^
    -o profile\<run_name>\reports\full_<tag> ^
    your_exe.exe [args]
```

| Flag | 含义 |
|---|---|
| `--set full` | 所有内置 section（SOL、Occupancy、Memory、Compute、Scheduler 等） |
| `--section PmSampling` | PM 时序采样（不包含在 full 中），揭示 tail effect |
| `-k "regex:..."` | 只 profile 匹配的 kernel（减少 replay 次数） |
| `-c 1` | 只 profile 第一次匹配的 launch |
| `-o ...` | 输出路径（.ncu-rep 自动追加） |

耗时：约 45 个 replay pass，30-60 秒。

---

## Recipe 2: Source-level（per-line stall）

```bat
ncu --set source --section SourceCounters ^
    -k "regex:KERNEL_REGEX" ^
    -c 1 ^
    -o profile\<run_name>\reports\source_<tag> ^
    your_exe.exe [args]
```

需要 `-lineinfo` 编译。约 5 个 pass，5-10 秒。

---

## Recipe 3: Details page（规则引擎建议）

不需要重新收集，从已有 .ncu-rep 导出：

```bat
ncu --import profile\<run_name>\reports\full_<tag>.ncu-rep --page details > profile\<run_name>\analysis\details_<tag>.txt
```

NCU 规则引擎会给出形如 `OPT  Est. Speedup: 10.8%` 的建议。**总是先看这个**。

---

## Recipe 4: CSV 导出

```bat
ncu --import profile\<run_name>\reports\full_<tag>.ncu-rep --page raw --csv > profile\<run_name>\analysis\raw_<tag>.csv
```

---

## Recipe 5: 只收集特定 metrics（快速验证）

```bat
ncu --metrics ^
    sm__throughput.avg.pct_of_peak_sustained_elapsed,^
    sm__warps_active.avg.pct_of_peak_sustained_active,^
    dram__bytes_read.sum.pct_of_peak_sustained_elapsed,^
    l1tex__t_sector_hit_rate.pct,^
    gpu__time_duration.sum ^
  -k "regex:KERNEL_REGEX" -c 1 ^
  your_exe.exe [args]
```

1-2 个 replay pass，直接打印到 stdout。

---

## Recipe 6: A/B 对比

```bat
rem Before
ncu --set full -k "regex:my_kernel" -c 1 ^
    -o profile\<run>\reports\full_v1 exe_v1.exe [args]

rem After
ncu --set full -k "regex:my_kernel" -c 1 ^
    -o profile\<run>\reports\full_v2 exe_v2.exe [args]

rem 用分析脚本对比
python scripts/profile/analyze_report.py ^
    --run-dir profile\<run> ^
    --report profile\<run>\reports\full_v1.ncu-rep --tag v1 ^
    --report profile\<run>\reports\full_v2.ncu-rep --tag v2
```

---

## 跳过 warmup launches

```bat
rem 跳过前 5 次匹配，然后收集 3 次
ncu -k "regex:my_kernel" --launch-skip-before-match 5 --launch-count 3 -o report exe [args]
```

---

## 查找 kernel 名

如果不确定 kernel 的 demangled name：

```bat
cuobjdump --dump-function-names your_exe.exe
```

或用 `ncu --set basic` 快速跑一遍看 kernel name。

---

## 常见问题

- **`regex:...` 没匹配到**：检查 demangled name，模板化 kernel 名很长
- **报告文件 0 KB**：kernel 没 launch 或 regex 没匹配
- **ERR_NVGPUCTRPERM**：Windows 上通常不会遇到；Linux 需要 root 或配置 modprobe
- **收集很慢（> 2 分钟）**：正常，`--set full` 需要 45+ replay passes
