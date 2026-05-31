# Profiling

This folder mirrors the standalone profiling workflow used by the upstream
CUTLASS flash-attention study scripts.

The scripts target the current `00-naive-attention` variant and keep generated
reports under:

```text
build/reports/flash-attention/00-naive-attention/
```

## Commands

```bat
profiling\scripts\run_ncu.bat
profiling\scripts\run_nsys.bat
```

PowerShell can also run Nsight Compute:

```powershell
powershell -ExecutionPolicy Bypass -File profiling\scripts\run_ncu.ps1
```

Set `TINY_CUTLASS_PROFILE_SKIP_BUILD=1` to reuse an existing executable.
Set `TINY_CUTLASS_PROFILE_SKIP_VERIFY=1` only after correctness has already
been checked.
