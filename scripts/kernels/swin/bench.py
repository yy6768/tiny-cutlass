"""Forward to the benchmark harness under csrc/tests."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    script = Path(__file__).resolve().parents[3] / "csrc/tests/swin/bench.py"
    raise SystemExit(subprocess.call([sys.executable, "-B", str(script), *sys.argv[1:]]))
