from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RUNTIME_RE = re.compile(r"Runtime:\s*([0-9.eE+-]+)\s*ms")
GFLOPS_RE = re.compile(r"GFLOPs\s*:\s*([0-9.eE+-]+)")


@dataclass(frozen=True)
class Case:
    name: str
    target: str
    args: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--config", default="Release")
    parser.add_argument("--report-dir", type=Path, default=Path("build/reports/swin"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--patch-iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--nsys-iterations", type=int, default=20)
    return parser.parse_args()


def executable(build_dir: Path, config: str, name: str) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return build_dir / "csrc" / "swin" / config / f"{name}{suffix}"


def cases(batch_size: int, iterations: int, patch_iterations: int) -> list[Case]:
    window_common = ("--window_size=7", "--head_size=32", "--reference-check=false")
    return [
        Case(
            "patch_embed",
            "swin_patch_embed",
            (
                f"--batch_size={batch_size}",
                "--image_size=224",
                "--in_channels=3",
                "--input_channels_padded=8",
                "--embed_dim=96",
                "--patch_size=4",
                f"--iterations={patch_iterations}",
            ),
        ),
        Case(
            "stage0_shift0",
            "swin_window",
            (
                f"--batch_size={batch_size}",
                "--image_size=56",
                "--shift_size=0",
                "--head_number=3",
                *window_common,
                "--mask=false",
                f"--iterations={iterations}",
            ),
        ),
        Case(
            "stage0_shift3",
            "swin_window",
            (
                f"--batch_size={batch_size}",
                "--image_size=56",
                "--shift_size=3",
                "--head_number=3",
                *window_common,
                "--mask=true",
                f"--iterations={iterations}",
            ),
        ),
        Case(
            "stage1_shift0",
            "swin_window",
            (
                f"--batch_size={batch_size}",
                "--image_size=28",
                "--shift_size=0",
                "--head_number=6",
                *window_common,
                "--mask=false",
                f"--iterations={iterations}",
            ),
        ),
        Case(
            "stage1_shift3",
            "swin_window",
            (
                f"--batch_size={batch_size}",
                "--image_size=28",
                "--shift_size=3",
                "--head_number=6",
                *window_common,
                "--mask=true",
                f"--iterations={iterations}",
            ),
        ),
        Case(
            "stage2_shift0",
            "swin_window",
            (
                f"--batch_size={batch_size}",
                "--image_size=14",
                "--shift_size=0",
                "--head_number=12",
                *window_common,
                "--mask=false",
                f"--iterations={iterations}",
            ),
        ),
        Case(
            "stage2_shift3",
            "swin_window",
            (
                f"--batch_size={batch_size}",
                "--image_size=14",
                "--shift_size=3",
                "--head_number=12",
                *window_common,
                "--mask=true",
                f"--iterations={iterations}",
            ),
        ),
        Case(
            "stage3_tail",
            "swin_window",
            (
                f"--batch_size={batch_size}",
                "--image_size=7",
                "--shift_size=0",
                "--head_number=24",
                *window_common,
                "--mask=false",
                f"--iterations={iterations}",
            ),
        ),
    ]


def run_capture(command: list[str]) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="", flush=True)
    return completed.stdout


def parse_metric(output: str, pattern: re.Pattern[str], label: str) -> float:
    match = pattern.search(output)
    if match is None:
        raise RuntimeError(f"could not parse {label} from output")
    return float(match.group(1))


def nsys_profile(
    case: Case,
    exe: Path,
    report_dir: Path,
    nsys_iterations: int,
) -> dict[str, str]:
    nsys_args = []
    for arg in case.args:
        if arg.startswith("--iterations="):
            nsys_args.append(f"--iterations={nsys_iterations}")
        else:
            nsys_args.append(arg)

    base = report_dir / f"{case.name}_b{case_arg_value(case, '--batch_size')}"
    rep = base.with_suffix(".nsys-rep")
    command = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--stats=false",
        "-o",
        str(base),
        str(exe),
        *nsys_args,
    ]
    run_capture(command)

    artifacts = {"nsys_rep": str(rep)}
    sqlite = base.with_suffix(".sqlite")
    if sqlite.exists():
        artifacts["nsys_sqlite"] = str(sqlite)
    for report in ("cuda_gpu_kern_sum", "cuda_api_sum"):
        output_base = report_dir / f"{base.name}_{report}"
        stats_cmd = [
            "nsys",
            "stats",
            "--force-export=true",
            "--force-overwrite=true",
            "--report",
            report,
            "--format",
            "csv",
            "--output",
            str(output_base),
            str(rep),
        ]
        run_capture(stats_cmd)
        artifacts[report] = str(
            output_base.with_name(f"{output_base.name}_{report}.csv")
        )
    return artifacts


def case_arg_value(case: Case, option: str) -> str:
    prefix = f"{option}="
    for arg in case.args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return "unknown"


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in cases(args.batch_size, args.iterations, args.patch_iterations):
        exe = executable(args.build_dir, args.config, case.target)
        output = run_capture([str(exe), *case.args])
        row = {
            "case": case.name,
            "target": case.target,
            "batch_size": case_arg_value(case, "--batch_size"),
            "runtime_ms": parse_metric(output, RUNTIME_RE, "runtime"),
            "gflops": parse_metric(output, GFLOPS_RE, "gflops"),
        }
        if args.nsys:
            row.update(nsys_profile(case, exe, args.report_dir, args.nsys_iterations))
        rows.append(row)

    csv_path = args.report_dir / f"bench_b{args.batch_size}.csv"
    json_path = args.report_dir / f"bench_b{args.batch_size}.json"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
