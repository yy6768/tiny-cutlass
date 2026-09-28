"""Forward to the verification harness under csrc/tests."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    script = Path(__file__).resolve().parents[3] / "csrc/tests/swin/verify.py"
    raise SystemExit(subprocess.call([sys.executable, "-B", str(script), "--all-families", *sys.argv[1:]]))
