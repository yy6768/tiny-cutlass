from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate Catmull-Rom benchmarking on reference parity."
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("build/reports/catmull-rom")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_report = args.report_dir.resolve() / "VERIFY.json"
    if not verify_report.is_file():
        raise RuntimeError(f"verification report not found: {verify_report}")

    payload = json.loads(verify_report.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError(f"verification did not pass: {verify_report}")

    print("[catmull-rom] benchmark skipped")


if __name__ == "__main__":
    main()
