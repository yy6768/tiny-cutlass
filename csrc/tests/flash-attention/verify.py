"""cuDNN parity gate for the split-KV learning kernel and shared registration."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

# B, H, Sq, Sk, D; these same shapes are used in bench.py.
BENCHMARK_CASES = [(1, 1, 256, 256, 64), (1, 4, 1024, 1024, 64),
                   (1, 4, 1024, 1024, 128), (16, 12, 1024, 1024, 64)]


def arguments(case):
    b, h, sq, sk, d = case
    return [f'--batch_size={b}', f'--head_number={h}', f'--seq_length={sq}',
            f'--seq_length_kv={sk}', f'--head_size={d}', f'--head_size_v={d}']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    exe = args.exe.resolve()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = directory / 'verification.json'
    record = {'status': 'running', 'executable': str(exe),
              'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(), 'cases': []}
    stamp.write_text(json.dumps(record, indent=2), encoding='utf-8')
    cases = [(1, 1, 1, 1, 32), (1, 1, 17, 33, 32), (2, 3, 63, 65, 32),
             (1, 1, 64, 64, 64), (2, 3, 65, 129, 64), (1, 2, 129, 17, 64),
             (1, 3, 65, 257, 96), (1, 2, 64, 64, 96),
             (2, 3, 17, 65, 128), (1, 1, 129, 257, 128)]
    jobs = [('02-split-kv', case, 1.0, 3080 + i) for i, case in enumerate(cases)]
    jobs += [('02-split-kv', (1, 2, 65, 129, d), scale, 941)
             for d in (32, 64, 96, 128) for scale in (0.0, 4.0)]
    # All registered variants must pass each benchmark shape before any timing.
    jobs += [('all', case, 1.0, 3080) for case in BENCHMARK_CASES]
    try:
        with (directory / 'verify.log').open('w', encoding='utf-8') as log:
            for kernel, case, scale, seed in jobs:
                command = [str(exe), f'--kernel={kernel}', *arguments(case),
                           '--reference-check=true', '--verify-only=true',
                           f'--input-scale={scale}', f'--seed={seed}']
                result = subprocess.run(command, capture_output=True, text=True, errors='replace')
                log.write(json.dumps(command) + '\n' + result.stdout + result.stderr + '\n')
                log.flush()
                expected = 3 if kernel == 'all' else 1
                passed = result.returncode == 0 and result.stdout.count('Reference passed:') == expected
                record['cases'].append({'kernel': kernel, 'shape': case, 'scale': scale,
                                        'seed': seed, 'passed': passed, 'command': command})
                print(f'{kernel} {case} scale={scale}: {"PASS" if passed else "FAIL"}', flush=True)
                if not passed:
                    raise RuntimeError(f'Parity failed: see {directory / "verify.log"}')
            for options in [('--head_size=48', '--head_size_v=48'),
                            ('--head_size=64', '--head_size_v=32')]:
                command = [str(exe), '--kernel=02-split-kv', '--verify-only=true', *options]
                result = subprocess.run(command, capture_output=True, text=True, errors='replace')
                log.write(json.dumps(command) + '\n' + result.stdout + result.stderr + '\n')
                if result.returncode == 0 or 'requires D=Dv' not in result.stderr:
                    raise RuntimeError('Expected explicit rejection of unsupported head shape')
        record['status'] = 'passed'
    except Exception:
        record['status'] = 'failed'
        raise
    finally:
        stamp.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Verified {len(jobs)} cases plus 2 rejection cases. Stamp: {stamp}')


if __name__ == '__main__':
    main()
