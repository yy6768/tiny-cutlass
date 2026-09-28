"""cuDNN parity gate for attention; split-Q requires MAE strictly below 1e-6."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

# B, H, Sq, Sk, D; these same shapes are used in bench.py.
BENCHMARK_CASES = [(1, 1, 256, 256, 64), (1, 4, 1024, 1024, 64),
                   (1, 4, 1024, 1024, 128), (16, 12, 1024, 1024, 64)]
REGISTERED_KERNELS = ('00-naive', '01-online-softmax', '02-split-kv', '03-split-q')
BASELINE_MAE_TOLERANCE = 1e-3
SPLIT_Q_MAE_TOLERANCE = 1e-6
MAX_ABS_TOLERANCE = 1e-2


def arguments(case):
    b, h, sq, sk, d = case
    return [f'--batch_size={b}', f'--head_number={h}', f'--seq_length={sq}',
            f'--seq_length_kv={sk}', f'--head_size={d}', f'--head_size_v={d}']


def validation_jobs(selected_kernel):
    cases = [(1, 1, 1, 1, 32), (1, 1, 17, 33, 32), (2, 3, 63, 65, 32),
             (1, 1, 64, 64, 64), (2, 3, 65, 129, 64), (1, 2, 129, 17, 64),
             (1, 3, 65, 257, 96), (1, 2, 64, 64, 96),
             (2, 3, 17, 65, 128), (1, 1, 129, 257, 128),
             # FA1 capacity tiles: cross both Q and KV boundaries, including
             # Bc=192/96/64/48 and the physical padding of logical Bc=48.
             (2, 2, 65, 385, 32), (1, 2, 129, 193, 64),
             (1, 2, 129, 129, 96), (2, 2, 97, 97, 128)]
    # Tail, exactly full tiles, and repeated logical-Bc advances.
    # Sq spans three Q tiles, including a partial tile. D=128 pads Bc=48 to 64.
    for d, br, bc in [(32, 32, 192), (64, 64, 96), (96, 64, 64), (128, 48, 48)]:
        cases += [(1, 2, 2 * br + 1, sk, d) for sk in (bc - 1, bc, bc + 1, 3 * bc)]
    jobs = [(selected_kernel, case, 1.0, 3080 + i) for i, case in enumerate(cases)]
    jobs += [(selected_kernel, (1, 2, 65, 129, d), scale, 941)
             for d in (32, 64, 96, 128) for scale in (0.0, 4.0)]
    if selected_kernel == '03-split-q':
        # Inherit every 02 input above unchanged, then cover the split-Q tile.
        jobs += [(selected_kernel, (1, 2, sq, sk, d), 1.0, 1203)
                 for d in (32, 64, 96, 128)
                 for sq, sk in ((63, 63), (64, 64), (65, 65), (129, 192))]
        # Candidate precision and baseline parity are separate requirements.
        jobs += [(kernel, case, 1.0, 3080) for case in BENCHMARK_CASES
                 for kernel in ('02-split-kv', '03-split-q')]
    # All registered variants must pass each benchmark shape before any timing.
    jobs += [('all', case, 1.0, 3080) for case in BENCHMARK_CASES]
    return jobs


def output_metrics(output):
    number = r'([\d.eE+-]+|inf|nan)'
    maes = re.findall(r'MAE\s*:\s*' + number, output)
    maxima = re.findall(r'Max abs\s*:\s*' + number, output)
    metrics = [{'mae': float(mae), 'max_abs': float(maximum)}
               for mae, maximum in zip(maes, maxima)]
    reused = re.search(r'Repeated invocation passed: workspace_bytes=(\d+), MAE='
                       + number + r', max abs=' + number, output)
    reuse = None if reused is None else {
        'workspace_bytes': int(reused[1]), 'mae': float(reused[2]),
        'max_abs': float(reused[3])}
    return metrics, reuse


def backward_metrics(output):
    number = r'([\d.eE+-]+|inf|nan)'
    pattern = (r'(Training|Gradient|Zero-dO) (O|LSE|dQ|dK|dV): MAE=' + number
               + r' max_abs=' + number + r' finite=([01]) passed=([01])'
               + r' max_index=(\d+) actual=' + number + r' reference=' + number)
    metrics = {}
    for match in re.finditer(pattern, output):
        phase, name = match[1], match[2]
        metrics.setdefault(phase, {})[name] = {
            'mae': float(match[3]), 'max_abs': float(match[4]),
            'finite': match[5] == '1', 'passed': match[6] == '1',
            'max_index': int(match[7]), 'actual_at_max': float(match[8]),
            'reference_at_max': float(match[9])}
    return metrics


def verify_backward(args, exe, directory):
    if args.kernel != '03-split-q':
        raise ValueError('Backward requires --kernel=03-split-q')
    stamp = args.stamp_artifact.resolve() if args.stamp_artifact else directory / 'backward-verification.json'
    stamp.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'running', 'phase': 'backward', 'executable': str(exe),
              'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
              'selected_kernel': args.kernel, 'reference': 'cudnn',
              'training_state': 'independent candidate and cuDNN O/LSE',
              'reference_deterministic': True, 'mae_tolerance': SPLIT_Q_MAE_TOLERANCE,
              'mae_comparison': '<', 'max_abs_tolerance': MAX_ABS_TOLERANCE,
              'dO_distribution': 'uniform[-2,2], HALF, seed+4',
              'cases': [], 'rejections': []}
    stamp.write_text(json.dumps(record, indent=2), encoding='utf-8')
    jobs = [job for job in validation_jobs('03-split-q') if job[0] == '03-split-q']
    try:
        with (directory / 'backward-verify.log').open('w', encoding='utf-8') as log:
            for kernel, case, scale, seed in jobs:
                command = [str(exe), f'--kernel={kernel}', *arguments(case),
                           '--reference-check=true', '--verify-only=true', '--verify-reuse=true',
                           f'--input-scale={scale}', f'--seed={seed}',
                           f'--mae-tolerance={SPLIT_Q_MAE_TOLERANCE}',
                           f'--max-abs-tolerance={MAX_ABS_TOLERANCE}']
                result = subprocess.run(command, capture_output=True, text=True, errors='replace')
                log.write(json.dumps(command) + '\n' + result.stdout + result.stderr + '\n')
                log.flush()
                metrics = backward_metrics(result.stdout)
                gradients = metrics.get('Gradient', {})
                reused = metrics.get('Zero-dO', {})
                passed = result.returncode == 0 and 'Backward reference passed: 03-split-q' in result.stdout
                passed = passed and set(gradients) == {'dQ', 'dK', 'dV'} and set(reused) == {'dQ', 'dK', 'dV'}
                passed = passed and all(row['finite'] and row['passed'] and row['mae'] < SPLIT_Q_MAE_TOLERANCE
                                        and row['max_abs'] <= MAX_ABS_TOLERANCE for row in gradients.values())
                passed = passed and all(row['finite'] and row['passed'] and row['max_abs'] == 0.
                                        for row in reused.values())
                passed = passed and 'Workspace rejection passed: missing and undersized' in result.stdout
                passed = passed and 'Backward reuse passed:' in result.stdout
                singleton = case[2] == 1 and case[3] == 1
                reference = 'analytic-singleton' if singleton else 'cudnn'
                expected_reference = 'Backward reference: analytic singleton' if singleton else 'Backward reference: cuDNN SDPA backward'
                passed = passed and expected_reference in result.stdout
                record['cases'].append({'kernel': kernel, 'shape': case, 'scale': scale, 'seed': seed,
                                        'dO_seed': seed + 4, 'reference': reference, 'metrics': metrics,
                                        'passed': passed, 'command': command, 'returncode': result.returncode})
                print(f'backward {case} scale={scale}: {"PASS" if passed else "FAIL"}', flush=True)
                if not passed:
                    raise RuntimeError(f'Backward parity failed: see {directory / "backward-verify.log"}')
            for options in [('--head_size=48', '--head_size_v=48'),
                            ('--head_size=64', '--head_size_v=32')]:
                command = [str(exe), '--kernel=03-split-q', '--verify-only=true', *options]
                result = subprocess.run(command, capture_output=True, text=True, errors='replace')
                log.write(json.dumps(command) + '\n' + result.stdout + result.stderr + '\n')
                passed = result.returncode != 0 and 'requires D=Dv' in result.stderr
                record['rejections'].append({'command': command, 'passed': passed,
                                              'returncode': result.returncode})
                if not passed:
                    raise RuntimeError('Expected explicit rejection of unsupported backward head shape')
        record['status'] = 'passed'
    except Exception:
        record['status'] = 'failed'
        raise
    finally:
        stamp.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Verified {len(jobs)} backward cases plus 2 rejection cases. Stamp: {stamp}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--kernel', choices=('02-split-kv', '03-split-q'), default='02-split-kv')
    parser.add_argument('--phase', choices=('forward', 'backward'), default='forward')
    parser.add_argument('--stamp-artifact', type=Path,
                        help='Optional verification JSON path; phase-specific default is inside output-dir')
    args = parser.parse_args()
    exe = args.exe.resolve()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.phase == 'backward':
        verify_backward(args, exe, directory)
        return
    stamp = args.stamp_artifact.resolve() if args.stamp_artifact else directory / 'verification.json'
    stamp.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'running', 'executable': str(exe),
              'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
              'selected_kernel': args.kernel, 'registered_kernels': list(REGISTERED_KERNELS),
              'reference': 'cudnn', 'split_q_mae_tolerance': SPLIT_Q_MAE_TOLERANCE,
              'baseline_mae_tolerance': BASELINE_MAE_TOLERANCE,
              'max_abs_tolerance': MAX_ABS_TOLERANCE, 'cases': [], 'rejections': []}
    stamp.write_text(json.dumps(record, indent=2), encoding='utf-8')
    jobs = validation_jobs(args.kernel)
    try:
        with (directory / 'verify.log').open('w', encoding='utf-8') as log:
            for kernel, case, scale, seed in jobs:
                strict = kernel == '03-split-q'
                tolerance = SPLIT_Q_MAE_TOLERANCE if strict else BASELINE_MAE_TOLERANCE
                reuse_required = kernel in ('02-split-kv', '03-split-q')
                command = [str(exe), f'--kernel={kernel}', *arguments(case),
                           '--reference-check=true', '--verify-only=true',
                           f'--input-scale={scale}', f'--seed={seed}',
                           f'--mae-tolerance={tolerance}', f'--max-abs-tolerance={MAX_ABS_TOLERANCE}']
                if reuse_required:
                    command.append('--verify-reuse=true')
                result = subprocess.run(command, capture_output=True, text=True, errors='replace')
                log.write(json.dumps(command) + '\n' + result.stdout + result.stderr + '\n')
                log.flush()
                expected = len(REGISTERED_KERNELS) if kernel == 'all' else 1
                metrics, reuse = output_metrics(result.stdout)
                passed = result.returncode == 0 and result.stdout.count('Reference passed:') == expected
                passed = passed and len(metrics) == expected
                passed = passed and all((row['mae'] < tolerance if strict else row['mae'] <= tolerance)
                                        and row['max_abs'] <= MAX_ABS_TOLERANCE for row in metrics)
                if reuse_required:
                    b, h, sq, _, d = case
                    br = min((24576 + 4 * d - 1) // (4 * d), d)
                    padded_sq = ((sq + br - 1) // br) * br
                    expected_workspace = b * h * padded_sq * (d + 2) * 4 if kernel == '02-split-kv' else 0
                    passed = passed and reuse is not None and reuse['workspace_bytes'] == expected_workspace
                    if expected_workspace:
                        passed = passed and 'Workspace rejection passed: missing, undersized and misaligned' in result.stdout
                    passed = passed and (reuse['mae'] < tolerance if strict else reuse['mae'] <= tolerance)
                    passed = passed and reuse['max_abs'] <= MAX_ABS_TOLERANCE
                record['cases'].append({'kernel': kernel, 'shape': case, 'scale': scale,
                                        'seed': seed, 'passed': passed, 'command': command,
                                        'returncode': result.returncode, 'mae_tolerance': tolerance,
                                        'mae_comparison': '<' if strict else '<=',
                                        'metrics': metrics, 'reuse': reuse})
                print(f'{kernel} {case} scale={scale}: {"PASS" if passed else "FAIL"}', flush=True)
                if not passed:
                    raise RuntimeError(f'Parity failed: see {directory / "verify.log"}')
            for options in [('--head_size=48', '--head_size_v=48'),
                            ('--head_size=64', '--head_size_v=32')]:
                command = [str(exe), f'--kernel={args.kernel}', '--verify-only=true', *options]
                result = subprocess.run(command, capture_output=True, text=True, errors='replace')
                log.write(json.dumps(command) + '\n' + result.stdout + result.stderr + '\n')
                passed = result.returncode != 0 and 'requires D=Dv' in result.stderr
                record['rejections'].append({'command': command, 'passed': passed,
                                              'returncode': result.returncode})
                if not passed:
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
