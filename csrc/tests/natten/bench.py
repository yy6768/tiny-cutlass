"""Each timed case performs its own output/LSE reference gate first."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    outputs = []
    code = 0
    for length, window in [(1024, 33), (1024, 129), (4096, 33)]:
        result = subprocess.run([str(args.exe.resolve()), "--bench", "--iterations=50",
                                 f"--length={length}", f"--kernel_size={window}"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        outputs.append(result.stdout)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            code = result.returncode
            break
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(outputs), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
