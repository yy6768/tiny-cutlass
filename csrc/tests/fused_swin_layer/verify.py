"""Full independent PyTorch parity, contracts, and single-kernel graph replay."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from source_policy import ROOT, check_source


def executable(build):
    candidates = [build/'tests/fused_swin_layer/Release/fused_swin_layer.exe',
                  build/'tests/fused_swin_layer/fused_swin_layer.exe',
                  build/'tests/fused_swin_layer/fused_swin_layer']
    return next((p for p in candidates if p.is_file()), candidates[0])


def run_case(binary, directory, iterations=0):
    command = [str(binary), '--case-dir', str(directory), '--graph']
    if iterations:
        command += ['--iterations', str(iterations)]
    completed = subprocess.run(command, text=True, encoding='utf-8', errors='replace',
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (directory/('bench.log' if iterations else 'verify.log')).write_text(completed.stdout, encoding='utf-8')
    print(completed.stdout, end='', flush=True)
    completed.check_returncode()
    return completed.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build/fused_swin_layer')
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--quick', action='store_true', help='omit 720p cases for debugging')
    args = parser.parse_args()
    files = check_source()
    if args.source_only:
        return
    sys.path.insert(0, str(ROOT/'csrc/tests/swin'))
    from block_reference import cases, write_fixture
    import torch
    suite = cases(full=not args.quick)
    suite += [replace(suite[0], name='block_no_output_bias', output_bias=False)]
    build = args.build_dir.resolve()
    binary = executable(build)
    evidence = {'status': 'running', 'scope': 'quick' if args.quick else 'full',
                'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
                'torch': torch.__version__, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'cases': []}
    report = build/'verification.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    try:
        for case in suite:
            directory, tensors, expected = write_fixture(case, build/'fixtures')
            del tensors, expected
            run_case(binary, directory)
            evidence['cases'].append(case.name)
        evidence['status'] = 'pass'
    except Exception:
        evidence['status'] = 'fail'
        raise
    finally:
        report.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(f'PARITY PASS: {len(suite)} full-block cases; graph replay and unsupported contracts passed')


if __name__ == '__main__':
    main()
