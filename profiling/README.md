# Profiling 说明

这个目录保存独立 profiling 工作流脚本，主要用于复用上游 CUTLASS
FlashAttention 学习脚本的 Nsight Compute / Nsight Systems 路线。

当前脚本面向 `00-naive-attention` 变体，生成的报告统一放在：

```text
build/reports/profiling/flash-attention/00-naive-attention/
```

## 命令

```bat
profiling\scripts\run_ncu.bat
profiling\scripts\run_nsys.bat
```

也可以用 PowerShell 直接运行 Nsight Compute 脚本：

```powershell
powershell -ExecutionPolicy Bypass -File profiling\scripts\run_ncu.ps1
```

环境变量：

- `TINY_CUTLASS_PROFILE_SKIP_BUILD=1`: 复用已有 executable，不重新构建。
- `TINY_CUTLASS_PROFILE_SKIP_VERIFY=1`: 只应在 correctness 已经验证通过后使用。

注意：profiling 数据不能替代 correctness。没有 reference parity 的 kernel 不能进入
benchmark 或性能结论。
