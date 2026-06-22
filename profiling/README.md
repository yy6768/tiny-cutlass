# Profiling 入口

本目录是 profiling 的快速入口点。核心工具链位于 `scripts/profile/`。

## 使用方式

### 推荐：统一 Python 入口

```bat
python scripts/profile/run_ncu.py ^
    --run-name naive_baseline ^
    --exe build\csrc\flash-attention\Release\naive_attention.exe ^
    --kernel-regex "attention_.*" ^
    --args "--batch_size=1 --head_number=32 --head_size=256 --head_size_v=256 --seq_length=1024 --seq_length_kv=1024 --iterations=1 --reference-check=false" ^
    --set both --count 1
```

### 旧式 .bat 入口

`profiling/scripts/` 下的 `.bat` 脚本面向 `naive_attention` 变体，
快速完成 build → verify → profile 流程：

```bat
profiling\scripts\run_ncu.bat
profiling\scripts\run_nsys.bat
```

环境变量：
- `TINY_CUTLASS_PROFILE_SKIP_BUILD=1`: 复用已有 executable，不重新构建。
- `TINY_CUTLASS_PROFILE_SKIP_VERIFY=1`: 只应在 correctness 已验证通过后使用。

## 产出物

profiling 数据统一输出到 `profile/` 目录（已在 .gitignore 中）：

```
profile/<run_name>/
├── reports/       ← .ncu-rep 文件
└── analysis/      ← 分析脚本产出物
```

## 详细文档

- [scripts/profile/README.md](../scripts/profile/README.md) — 完整工具链文档
- [scripts/profile/reference/](../scripts/profile/reference/) — 参考手册（workflow、六维分析法等）

## 注意

profiling 数据不能替代 correctness。没有 reference parity 的 kernel 不能
进入 benchmark 或性能结论。
