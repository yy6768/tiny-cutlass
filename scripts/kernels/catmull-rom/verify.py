from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TESTS = (
    "catmull_forward",
    "catmull_backward",
    "catmull_dilation",
    "catmull_autodiff",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Catmull-Rom correctness and reference checks."
    )
    parser.add_argument("--build-dir", type=Path, default=Path("build/catmull-rom"))
    parser.add_argument("--config", default="Release")
    parser.add_argument(
        "--report-dir", type=Path, default=Path("build/reports/catmull-rom")
    )
    return parser.parse_args()


def find_executable(build_dir: Path, config: str, name: str) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    filename = f"{name}{suffix}"
    candidates = (
        build_dir / "tests" / "catmull-rom" / filename,
        build_dir / "tests" / "catmull-rom" / config / filename,
        build_dir / "csrc" / "catmull-rom" / config / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing executable {name}; searched:\n  {searched}")


def write_report(report_dir: Path, status: str, cases: list[dict[str, object]]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "VERIFY.json"
    payload = {
        "schema_version": 1,
        "status": status,
        "reference": "host analytic checks plus checked-in Slang autodiff CUDA",
        "cases": cases,
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    build_dir = args.build_dir.resolve()
    report_dir = args.report_dir.resolve()
    cases: list[dict[str, object]] = []

    for name in TESTS:
        try:
            executable = find_executable(build_dir, args.config, name)
        except FileNotFoundError as error:
            cases.append({"name": name, "status": "failed", "error": str(error)})
            report_path = write_report(report_dir, "failed", cases)
            print(error, file=sys.stderr)
            print(f"verify_report={report_path}", file=sys.stderr)
            return 1

        command = [str(executable)]
        print(f"[verify:{name}]", flush=True)
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout, end="", flush=True)
        case = {
            "name": name,
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "command": command,
        }
        cases.append(case)
        if completed.returncode != 0:
            report_path = write_report(report_dir, "failed", cases)
            print(f"verify_report={report_path}", file=sys.stderr)
            return completed.returncode or 1

    report_path = write_report(report_dir, "passed", cases)
    print(f"verify_report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
