"""Build-independent correctness gate for the standalone FNA executable."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = subprocess.run([str(args.exe.resolve()), "--suite"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(result.stdout, encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
