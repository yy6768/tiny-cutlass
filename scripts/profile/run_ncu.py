#!/usr/bin/env python3
"""统一 NCU profiling 收集入口。

完成：创建 run 目录 → 执行 ncu 收集 → 导出 details → 调用分析脚本。

用法：
    python scripts/profile/run_ncu.py \\
        --run-name naive_baseline \\
        --exe build/csrc/flash-attention/Release/naive_attention.exe \\
        --kernel-regex "attention_.*" \\
        --args "--batch_size=1 --head_number=32 --head_size=256 --head_size_v=256 --seq_length=1024 --seq_length_kv=1024 --iterations=1 --reference-check=false" \\
        --set both \\
        --count 1

    # 只收集 full report（跳过 source-level）
    python scripts/profile/run_ncu.py ... --set full

    # 跳过 ncu 收集，只重新跑分析脚本（已有 .ncu-rep）
    python scripts/profile/run_ncu.py ... --analyze-only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # scripts/profile/ -> repo root


def find_ncu() -> str:
    """定位 ncu 可执行文件。"""
    found = shutil.which("ncu")
    if found:
        return found
    if sys.platform.startswith("win"):
        found = shutil.which("ncu.bat")
        if found:
            return found
        # 常见 Windows 安装路径
        nv_corp = Path(r"C:\Program Files\NVIDIA Corporation")
        if nv_corp.is_dir():
            for sub in sorted(nv_corp.glob("Nsight Compute */ncu.bat"), reverse=True):
                return str(sub)
    raise FileNotFoundError(
        "找不到 ncu。请确认 Nsight Compute 已安装且 ncu 在 PATH 中。"
    )


def run_cmd(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    """执行命令并打印。"""
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture and result.stdout:
        # 只打印关键行
        for line in result.stdout.splitlines():
            if any(k in line for k in ["==ERROR==", "==WARNING==", "==PROF==", "Report:"]):
                print(line, flush=True)
    if result.returncode != 0:
        print(f"[错误] 命令退出码: {result.returncode}", file=sys.stderr)
        if capture and result.stdout:
            print(result.stdout[-2000:], file=sys.stderr)
    return result


def collect_full(
    ncu: str, run_dir: Path, tag: str, exe: str, kernel_regex: str,
    exe_args: list[str], skip: int, count: int,
) -> Path:
    """收集 --set full + PmSampling。"""
    report_path = run_dir / "reports" / f"full_{tag}"
    cmd = [
        ncu,
        "--force-overwrite",
        "--target-processes", "all",
        "--set", "full",
        "--section", "PmSampling",
        "-k", f"regex:{kernel_regex}",
        "--launch-skip-before-match", str(skip),
        "--launch-count", str(count),
        "--import-source", "1",
        "-o", str(report_path),
        "--", exe, *exe_args,
    ]
    result = run_cmd(cmd, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"ncu --set full 失败 (exit {result.returncode})")
    return report_path.with_suffix(".ncu-rep")


def collect_source(
    ncu: str, run_dir: Path, tag: str, exe: str, kernel_regex: str,
    exe_args: list[str], skip: int, count: int,
) -> Path:
    """收集 --set source --section SourceCounters。"""
    report_path = run_dir / "reports" / f"source_{tag}"
    cmd = [
        ncu,
        "--force-overwrite",
        "--target-processes", "all",
        "--set", "source",
        "--section", "SourceCounters",
        "-k", f"regex:{kernel_regex}",
        "--launch-skip-before-match", str(skip),
        "--launch-count", str(count),
        "--import-source", "1",
        "-o", str(report_path),
        "--", exe, *exe_args,
    ]
    result = run_cmd(cmd, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"ncu --set source 失败 (exit {result.returncode})")
    return report_path.with_suffix(".ncu-rep")


def export_details(ncu: str, full_rep: Path, run_dir: Path, tag: str) -> Path:
    """从 full report 导出 --page details 文本。"""
    out_path = run_dir / "analysis" / f"details_{tag}.txt"
    cmd = [ncu, "--import", str(full_rep), "--page", "details"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out_path.write_text(result.stdout or "", encoding="utf-8")
    print(f"  -> {out_path}")
    return out_path


def export_csv(ncu: str, full_rep: Path, run_dir: Path, tag: str) -> Path:
    """从 full report 导出 CSV。"""
    out_path = run_dir / "analysis" / f"raw_{tag}.csv"
    cmd = [ncu, "--import", str(full_rep), "--page", "raw", "--csv"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out_path.write_text(result.stdout or "", encoding="utf-8")
    print(f"  -> {out_path}")
    return out_path


def run_analyze(run_dir: Path, report: Path, tag: str):
    """调用 analyze_report.py。"""
    script = HERE / "analyze_report.py"
    cmd = [sys.executable, str(script),
           "--run-dir", str(run_dir),
           "--report", str(report), "--tag", tag]
    run_cmd(cmd)


def run_extract_stalls(run_dir: Path, report: Path, tag: str):
    """调用 extract_stalls.py。"""
    script = HERE / "extract_stalls.py"
    cmd = [sys.executable, str(script),
           "--run-dir", str(run_dir),
           "--report", str(report), "--tag", tag]
    run_cmd(cmd)


def run_plot_timeline(run_dir: Path, report: Path, tag: str):
    """调用 plot_timeline.py。"""
    script = HERE / "plot_timeline.py"
    cmd = [sys.executable, str(script),
           "--run-dir", str(run_dir),
           "--report", str(report), "--tag", tag]
    run_cmd(cmd)


def main():
    ap = argparse.ArgumentParser(
        description="统一 NCU profiling 收集与分析入口。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-name", required=True,
                    help="run 名称（在 profile/ 下创建同名目录）")
    ap.add_argument("--exe", required=True,
                    help="要 profile 的可执行文件路径")
    ap.add_argument("--kernel-regex", default=".*",
                    help="ncu -k regex（默认匹配所有 kernel）")
    ap.add_argument("--args", default="",
                    help="传给可执行文件的参数（引号包裹的字符串）")
    ap.add_argument("--set", choices=["full", "source", "both"], default="both",
                    help="收集模式：full / source / both（默认 both）")
    ap.add_argument("--skip", type=int, default=0,
                    help="--launch-skip-before-match（跳过前 N 次匹配）")
    ap.add_argument("--count", type=int, default=1,
                    help="--launch-count（收集几次 kernel launch）")
    ap.add_argument("--tag", default="default",
                    help="报告标签（默认 'default'）")
    ap.add_argument("--profile-dir", type=Path, default=None,
                    help="profile 根目录（默认 <repo_root>/profile）")
    ap.add_argument("--analyze-only", action="store_true",
                    help="跳过 ncu 收集，只重新跑分析脚本")
    args = ap.parse_args()

    profile_root = args.profile_dir or (REPO_ROOT / "profile")
    run_dir = profile_root / args.run_name
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)

    exe_args = args.args.split() if args.args else []
    tag = args.tag

    full_rep = run_dir / "reports" / f"full_{tag}.ncu-rep"
    source_rep = run_dir / "reports" / f"source_{tag}.ncu-rep"

    if not args.analyze_only:
        ncu = find_ncu()
        print(f"[ncu] 使用: {ncu}")
        print(f"[run] 目录: {run_dir}")
        print(f"[exe] {args.exe}")
        print(f"[kernel] regex: {args.kernel_regex}")
        print(f"[args] {' '.join(exe_args)}")
        print()

        if args.set in ("full", "both"):
            print("=" * 60)
            print("Phase 1: --set full + PmSampling")
            print("=" * 60)
            full_rep = collect_full(
                ncu, run_dir, tag, args.exe, args.kernel_regex,
                exe_args, args.skip, args.count,
            )

        if args.set in ("source", "both"):
            print()
            print("=" * 60)
            print("Phase 2: --set source + SourceCounters")
            print("=" * 60)
            source_rep = collect_source(
                ncu, run_dir, tag, args.exe, args.kernel_regex,
                exe_args, args.skip, args.count,
            )

        # 导出 details + CSV
        if full_rep.exists():
            print()
            print("=" * 60)
            print("Phase 3: 导出 details & CSV")
            print("=" * 60)
            export_details(ncu, full_rep, run_dir, tag)
            export_csv(ncu, full_rep, run_dir, tag)

    # 分析阶段
    print()
    print("=" * 60)
    print("Phase 4: 分析")
    print("=" * 60)

    if full_rep.exists():
        run_analyze(run_dir, full_rep, tag)
        run_plot_timeline(run_dir, full_rep, tag)
    else:
        print(f"[skip] {full_rep} 不存在，跳过 analyze + timeline")

    if source_rep.exists():
        run_extract_stalls(run_dir, source_rep, tag)
    else:
        print(f"[skip] {source_rep} 不存在，跳过 stall 分析")

    print()
    print("=" * 60)
    print(f"完成！产出物在: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
