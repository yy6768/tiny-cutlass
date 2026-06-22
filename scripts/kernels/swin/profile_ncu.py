from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfileCase:
    name: str
    target: str
    args: tuple[str, ...]
    launch_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--config", default="Release")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--profile-dir", type=Path, default=Path("profile/swin"))
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--set", default="detailed")
    parser.add_argument("--case", action="append", default=[])
    return parser.parse_args()


def executable(build_dir: Path, config: str, name: str) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return build_dir / "csrc" / "swin" / config / f"{name}{suffix}"


def run(command: list[str], echo_output: bool = True) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if echo_output:
        print(completed.stdout, end="", flush=True)
    else:
        for line in completed.stdout.splitlines():
            if line.startswith("==ERROR==") or line.startswith("==WARNING=="):
                print(line, flush=True)
            elif line.startswith("==PROF== Report:"):
                print(line, flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout
        )
    return completed.stdout


def ncu_path() -> str:
    found = shutil.which("ncu")
    if found:
        return found
    found = shutil.which("ncu.bat")
    if found:
        return found
    default = Path(
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\ncu.bat"
    )
    if default.exists():
        return str(default)
    raise FileNotFoundError("could not find ncu or ncu.bat")


def prepare(checkpoint_dir: Path) -> None:
    run(
        [
            sys.executable,
            "scripts/kernels/swin/prepare_official.py",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
    )


def path_arg(name: str, base: Path, file_name: str) -> str:
    return f"--{name}={base / file_name}"


def cases_from_manifest(manifest: dict[str, object], checkpoint_dir: Path) -> list[ProfileCase]:
    patch = manifest["patch_embed"]
    cases = [
        ProfileCase(
            "patch_embed",
            "swin_patch_embed",
            (
                f"--batch_size={patch['batch_size']}",
                f"--image_size={patch['image_size']}",
                f"--in_channels={patch['in_channels']}",
                f"--input_channels_padded={patch['input_channels_padded']}",
                f"--embed_dim={patch['embed_dim']}",
                f"--patch_size={patch['patch_size']}",
                "--iterations=1",
                "--reference-check=true",
                "--profile-once=true",
                path_arg("input-file", checkpoint_dir, patch["input"]),
                path_arg("kernel-file", checkpoint_dir, patch["kernel"]),
                path_arg("bias-file", checkpoint_dir, patch["bias"]),
                path_arg("gamma-file", checkpoint_dir, patch["gamma"]),
                path_arg("beta-file", checkpoint_dir, patch["beta"]),
            ),
            4,
        )
    ]

    for stage in manifest["stages"]:
        launch_count = 8 if int(stage["image_size"]) % 2 == 0 else 7
        for shift in stage["shifts"]:
            cases.append(
                ProfileCase(
                    f"stage{stage['stage']}_shift{shift['shift_size']}",
                    "swin_window",
                    (
                        f"--batch_size={manifest['batch_size']}",
                        f"--image_size={stage['image_size']}",
                        f"--window_size={stage['window_size']}",
                        f"--shift_size={shift['shift_size']}",
                        f"--head_number={stage['head_number']}",
                        f"--head_size={stage['head_size']}",
                        "--iterations=1",
                        "--reference-check=true",
                        "--profile-once=true",
                        f"--mask={str(bool(shift['use_mask'])).lower()}",
                        path_arg("input-file", checkpoint_dir, stage["input"]),
                        path_arg("qkv-weight-file", checkpoint_dir, shift["qkv_weight"]),
                        path_arg("qkv-bias-file", checkpoint_dir, shift["qkv_bias"]),
                        path_arg("output-weight-file", checkpoint_dir, shift["output_weight"]),
                        path_arg("output-bias-file", checkpoint_dir, shift["output_bias"]),
                        path_arg("rel-bias-file", checkpoint_dir, shift["rel_bias"]),
                    ),
                    launch_count,
                )
            )
    return cases


def profile_case(
    case: ProfileCase,
    exe: Path,
    profile_dir: Path,
    ncu: str,
    ncu_set: str,
) -> dict[str, str | int]:
    base = profile_dir / case.name / case.name
    csv_path = base.with_suffix(".csv")
    report_path = base.with_suffix(".ncu-rep")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ncu,
        "--force-overwrite",
        "--target-processes",
        "all",
        "--set",
        ncu_set,
        "--launch-count",
        str(case.launch_count),
        "--page",
        "raw",
        "--csv",
        "--import-source",
        "1",
        "-o",
        str(base),
        "--",
        str(exe),
        *case.args,
    ]
    try:
        output = run(command, echo_output=False)
    except subprocess.CalledProcessError as exc:
        output = exc.output or ""
        csv_path.write_text(output, encoding="utf-8")
        raise
    csv_path.write_text(output, encoding="utf-8")
    return {
        "case": case.name,
        "target": case.target,
        "launch_count": case.launch_count,
        "csv": str(csv_path),
        "ncu_rep": str(report_path),
    }


def summarize_kernel_times(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not csv_path.exists():
        return rows
    raw_lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = next(i for i, line in enumerate(raw_lines) if line.startswith('"ID"'))
    except StopIteration:
        return rows
    lines = raw_lines[start:]
    if not lines:
        return rows
    reader = csv.DictReader(lines)
    for row in reader:
        if row.get("ID", "") == "":
            continue
        rows.append(
            {
                "id": row.get("ID", ""),
                "kernel": row.get("Kernel Name", ""),
                "duration_ns": row.get("gpu__time_duration.sum", ""),
                "dram_read_bytes": row.get("dram__bytes_read.sum", ""),
                "dram_write_bytes": row.get("dram__bytes_write.sum", ""),
                "sm_throughput_pct": row.get(
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed", ""
                ),
                "dram_throughput_pct": row.get(
                    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", ""
                ),
                "grid_size": row.get("launch__grid_size", ""),
                "block_size": row.get("launch__block_size", ""),
                "registers_per_thread": row.get("launch__registers_per_thread", ""),
                "shared_mem_bytes": row.get("launch__shared_mem_per_block", ""),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if not args.skip_prepare:
        prepare(args.checkpoint_dir)
    manifest_path = args.checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = set(args.case)
    cases = cases_from_manifest(manifest, args.checkpoint_dir)
    if selected:
        cases = [case for case in cases if case.name in selected]
    if not cases:
        raise RuntimeError("no profile cases selected")

    args.profile_dir.mkdir(parents=True, exist_ok=True)
    ncu = ncu_path()
    rows = []
    summaries = {}
    for case in cases:
        exe = executable(args.build_dir, args.config, case.target)
        row = profile_case(case, exe, args.profile_dir, ncu, args.set)
        rows.append(row)
        summaries[case.name] = summarize_kernel_times(Path(row["csv"]))

    manifest_out = args.profile_dir / "ncu_manifest.json"
    manifest_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary_out = args.profile_dir / "ncu_kernel_times.json"
    summary_out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {manifest_out}")
    print(f"wrote {summary_out}")


if __name__ == "__main__":
    main()
