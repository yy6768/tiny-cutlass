"""Swin's source gate, shared by verification and benchmark entrypoints."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'csrc/tests/common'))
from kernel_policy import enforce_policy


def check_sources() -> bool:
    return enforce_policy(ROOT, ROOT / 'csrc/swin/kernel-policy.json')
