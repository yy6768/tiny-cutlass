from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RUNTIME_RE = re.compile(r"Runtime\s*:\s*([0-9.eE+-]+)\s*ms")
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
    parser.add_argument("--report-dir", type=Path, default=Path("profile/swin/runtime"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--patch-iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--nsys-iterations", type=int, default=20)
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--ncu-set", default="detailed")
    parser.add_argument("--case", action="append", default=[])
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
            "block_shift3",
            "swin_block",
            (
                f"--batch_size={batch_size}",
                "--image_size=14",
                "--window_size=7",
                "--shift_size=3",
                "--head_number=3",
                "--head_size=32",
                f"--iterations={iterations}",
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


def path_arg(name: str, base: Path, file_name: str) -> str:
    return f"--{name}={base / file_name}"


def official_cases(
    checkpoint_dir: Path,
    iterations: int,
    patch_iterations: int,
) -> list[Case]:
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing official manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patch = manifest["patch_embed"]
    result = [
        Case(
            "patch_embed",
            "swin_patch_embed",
            (
                f"--batch_size={patch['batch_size']}",
                f"--image_size={patch['image_size']}",
                f"--in_channels={patch['in_channels']}",
                f"--input_channels_padded={patch['input_channels_padded']}",
                f"--embed_dim={patch['embed_dim']}",
                f"--patch_size={patch['patch_size']}",
                f"--iterations={patch_iterations}",
                "--reference-check=false",
                path_arg("input-file", checkpoint_dir, patch["input"]),
                path_arg("kernel-file", checkpoint_dir, patch["kernel"]),
                path_arg("bias-file", checkpoint_dir, patch["bias"]),
                path_arg("gamma-file", checkpoint_dir, patch["gamma"]),
                path_arg("beta-file", checkpoint_dir, patch["beta"]),
            ),
        )
    ]
    for stage in manifest["stages"]:
        for shift in stage["shifts"]:
            result.append(
                Case(
                    f"stage{stage['stage']}_shift{shift['shift_size']}",
                    "swin_window",
                    (
                        f"--batch_size={manifest['batch_size']}",
                        f"--image_size={stage['image_size']}",
                        f"--window_size={stage['window_size']}",
                        f"--shift_size={shift['shift_size']}",
                        f"--head_number={stage['head_number']}",
                        f"--head_size={stage['head_size']}",
                        "--reference-check=false",
                        f"--mask={str(bool(shift['use_mask'])).lower()}",
                        f"--iterations={iterations}",
                        path_arg("input-file", checkpoint_dir, stage["input"]),
                        path_arg("qkv-weight-file", checkpoint_dir, shift["qkv_weight"]),
                        path_arg("qkv-bias-file", checkpoint_dir, shift["qkv_bias"]),
                        path_arg(
                            "output-weight-file",
                            checkpoint_dir,
                            shift["output_weight"],
                        ),
                        path_arg(
                            "output-bias-file",
                            checkpoint_dir,
                            shift["output_bias"],
                        ),
                        path_arg("rel-bias-file", checkpoint_dir, shift["rel_bias"]),
                    ),
                )
            )
    result.append(
        Case(
            "block_shift3",
            "swin_block",
            (
                f"--batch_size={manifest['batch_size']}",
                "--image_size=14",
                "--window_size=7",
                "--shift_size=3",
                "--head_number=3",
                "--head_size=32",
                f"--iterations={iterations}",
            ),
        )
    )
    return result


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


def parse_metric(
    output: str,
    pattern: re.Pattern[str],
    label: str,
    required: bool = True,
) -> float | None:
    match = pattern.search(output)
    if match is None:
        if required:
            raise RuntimeError(f"could not parse {label} from output")
        return None
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


def ncu_path() -> str:
    for name in ("ncu", "ncu.bat"):
        found = shutil.which(name)
        if found:
            return found
    default = Path(
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\ncu.bat"
    )
    if default.exists():
        return str(default)
    raise FileNotFoundError("could not find ncu or ncu.bat")


def ncu_launch_count(case: Case) -> int:
    if case.target == "swin_patch_embed":
        return 4
    if case.target == "swin_block":
        return 10
    return 7


def ncu_args(case: Case) -> list[str]:
    result = []
    for arg in case.args:
        if arg.startswith("--iterations="):
            result.append("--iterations=1")
        else:
            result.append(arg)
    if case.target in {"swin_patch_embed", "swin_window"}:
        result.append("--profile-once=true")
    return result


def summarize_ncu_csv(csv_path: Path) -> list[dict[str, str]]:
    raw_lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = next(i for i, line in enumerate(raw_lines) if line.startswith('"ID"'))
    except StopIteration:
        return []
    rows = []
    for row in csv.DictReader(raw_lines[start:]):
        if not row.get("ID", ""):
            continue
        rows.append(
            {
                "id": row.get("ID", ""),
                "kernel": row.get("Kernel Name", ""),
                "duration_ns": row.get("gpu__time_duration.sum", ""),
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


def ncu_profile(
    case: Case,
    exe: Path,
    report_dir: Path,
    ncu: str,
    ncu_set: str,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    case_dir = report_dir / "ncu" / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    base = case_dir / case.name
    csv_path = base.with_suffix(".csv")
    rep_path = base.with_suffix(".ncu-rep")
    command = [
        ncu,
        "--force-overwrite",
        "--target-processes",
        "all",
        "--set",
        ncu_set,
        "--launch-count",
        str(ncu_launch_count(case)),
        "--page",
        "raw",
        "--csv",
        "--import-source",
        "1",
        "-o",
        str(base),
        "--",
        str(exe),
        *ncu_args(case),
    ]
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    csv_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout, end="", flush=True)
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout
        )
    return (
        {"ncu_rep": str(rep_path), "ncu_csv": str(csv_path)},
        summarize_ncu_csv(csv_path),
    )


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
    ncu_rows: dict[str, list[dict[str, str]]] = {}
    ncu = ncu_path() if args.ncu else ""
    benchmark_cases = (
        official_cases(args.checkpoint_dir, args.iterations, args.patch_iterations)
        if args.official
        else cases(args.batch_size, args.iterations, args.patch_iterations)
    )
    if args.case:
        requested = set(args.case)
        benchmark_cases = [case for case in benchmark_cases if case.name in requested]
        found = {case.name for case in benchmark_cases}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"unknown benchmark case(s): {', '.join(missing)}")
    for case in benchmark_cases:
        exe = executable(args.build_dir, args.config, case.target)
        output = run_capture([str(exe), *case.args])
        row = {
            "case": case.name,
            "target": case.target,
            "batch_size": case_arg_value(case, "--batch_size"),
            "runtime_ms": parse_metric(output, RUNTIME_RE, "runtime"),
            "gflops": parse_metric(output, GFLOPS_RE, "gflops", required=False),
        }
        if args.nsys:
            row.update(nsys_profile(case, exe, args.report_dir, args.nsys_iterations))
        if args.ncu:
            artifacts, kernels = ncu_profile(
                case, exe, args.report_dir, ncu, args.ncu_set
            )
            row.update(artifacts)
            ncu_rows[case.name] = kernels
        rows.append(row)

    csv_path = args.report_dir / f"bench_b{args.batch_size}.csv"
    json_path = args.report_dir / f"bench_b{args.batch_size}.json"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.ncu:
        ncu_path_json = args.report_dir / "ncu_kernel_times.json"
        ncu_path_json.write_text(json.dumps(ncu_rows, indent=2), encoding="utf-8")
        print(f"wrote {ncu_path_json}")

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
