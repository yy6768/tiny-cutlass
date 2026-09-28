"""Compare verified attention variants on identical shapes with CUDA events."""
import argparse
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

from verify import (BASELINE_MAE_TOLERANCE, BENCHMARK_CASES, MAX_ABS_TOLERANCE,
                    SPLIT_Q_MAE_TOLERANCE, arguments, backward_metrics, validation_jobs)


def require_parity(stamp, selected_kernel):
    if selected_kernel == '03-split-q' and stamp.get('selected_kernel') != selected_kernel:
        raise RuntimeError('Run verify.py --kernel=03-split-q before benchmarking split-Q')
    for case in BENCHMARK_CASES:
        if selected_kernel != '03-split-q':
            if not any(row['kernel'] == 'all' and tuple(row['shape']) == case and row['passed']
                       for row in stamp['cases']):
                raise RuntimeError(f'Missing all-kernel parity for {case}')
            continue
        for kernel in ('02-split-kv', '03-split-q'):
            strict = kernel == '03-split-q'
            tolerance = SPLIT_Q_MAE_TOLERANCE if strict else BASELINE_MAE_TOLERANCE
            eligible = [row for row in stamp['cases'] if row['kernel'] == kernel
                        and tuple(row['shape']) == case and row['passed']
                        and row.get('seed') == 3080 and row.get('scale') == 1.0
                        and row.get('mae_tolerance', float('inf')) <= tolerance
                        and row.get('mae_comparison') == ('<' if strict else '<=')]
            if not any(len(row.get('metrics', [])) == 1
                       and (row['metrics'][0]['mae'] < tolerance if strict
                            else row['metrics'][0]['mae'] <= tolerance)
                       and row['metrics'][0]['max_abs'] <= MAX_ABS_TOLERANCE
                       for row in eligible):
                raise RuntimeError(f'Missing {kernel} parity at MAE {"<" if strict else "<="} '
                                   f'{tolerance} for {case}')


def valid_backward_metrics(metrics):
    training = metrics.get('Training', {})
    gradients = metrics.get('Gradient', {})
    reuse = metrics.get('Zero-dO', {})
    if set(training) != {'O', 'LSE'} or set(gradients) != {'dQ', 'dK', 'dV'} or set(reuse) != {'dQ', 'dK', 'dV'}:
        return False
    for row in [*training.values(), *gradients.values(), *reuse.values()]:
        if not row.get('finite') or not row.get('passed'):
            return False
        if not all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                   and row[key] >= 0 for key in ('mae', 'max_abs')):
            return False
    for row in [training['O'], *gradients.values()]:
        if row['mae'] >= SPLIT_Q_MAE_TOLERANCE or row['max_abs'] > MAX_ABS_TOLERANCE:
            return False
    # LSE has an auxiliary finite-state check; it has no invented error bound.
    return all(row['mae'] == 0 and row['max_abs'] == 0 for row in reuse.values())


def require_backward_parity(stamp, binary_hash):
    if (stamp.get('status') != 'passed' or stamp.get('binary_sha256') != binary_hash
            or stamp.get('phase') != 'backward' or stamp.get('selected_kernel') != '03-split-q'
            or stamp.get('reference') != 'cudnn' or stamp.get('reference_deterministic') is not True
            or stamp.get('training_state') != 'independent candidate and cuDNN O/LSE'
            or stamp.get('mae_comparison') != '<'
            or not 0 < stamp.get('mae_tolerance', float('inf')) <= SPLIT_Q_MAE_TOLERANCE
            or not 0 < stamp.get('max_abs_tolerance', float('inf')) <= MAX_ABS_TOLERANCE):
        raise RuntimeError('Run strict end-to-end backward verification on this exact executable first')
    expected = Counter((kernel, tuple(case), scale, seed)
                       for kernel, case, scale, seed in validation_jobs('03-split-q')
                       if kernel == '03-split-q')
    observed = Counter()
    for row in stamp.get('cases', []):
        signature = (row.get('kernel'), tuple(row.get('shape', [])), row.get('scale'), row.get('seed'))
        observed[signature] += 1
        case = row.get('shape', [])
        reference = 'analytic-singleton' if len(case) == 5 and case[2:4] == [1, 1] else 'cudnn'
        command = row.get('command', [])
        if (not row.get('passed') or row.get('returncode') != 0 or signature not in expected
                or row.get('dO_seed') != signature[3] + 4 or row.get('reference') != reference
                or not valid_backward_metrics(row.get('metrics', {}))
                or '--verify-only=true' not in command or '--verify-reuse=true' not in command
                or '--reference-check=true' not in command
                or any(arg.startswith('--diagnostic-state') for arg in command)):
            raise RuntimeError(f'Invalid strict backward verification evidence for {signature}')
    if observed != expected:
        raise RuntimeError('Backward benchmark requires every one of the 58 strict verification jobs')
    rejections = stamp.get('rejections', [])
    expected_rejections = {('--head_size=48', '--head_size_v=48'), ('--head_size=64', '--head_size_v=32')}
    actual_rejections = set()
    for row in rejections:
        options = tuple(arg for arg in row.get('command', []) if arg.startswith(('--head_size=', '--head_size_v=')))
        if not row.get('passed') or row.get('returncode', 0) == 0 or options not in expected_rejections:
            raise RuntimeError('Missing successful backward unsupported-shape rejection evidence')
        actual_rejections.add(options)
    if len(rejections) != 2 or actual_rejections != expected_rejections:
        raise RuntimeError('Backward benchmark requires both unsupported-shape rejection checks')


def benchmark_backward(args, exe, directory):
    if args.kernel != '03-split-q':
        raise RuntimeError('Backward benchmark requires --kernel=03-split-q')
    binary_hash = hashlib.sha256(exe.read_bytes()).hexdigest()
    stamp = json.loads((directory / 'backward-verification.json').read_text(encoding='utf-8'))
    # Validate the full suite before creating benchmark output or starting a process.
    require_backward_parity(stamp, binary_hash)
    record = {'status': 'running', 'phase': 'backward', 'executable': str(exe),
              'binary_sha256': binary_hash, 'kernel': args.kernel,
              'reference': 'cudnn', 'training_state': 'independent candidate and cuDNN O/LSE',
              'repeats': args.repeats, 'iterations': args.iterations, 'seed': 3080, 'dO_seed': 3084,
              'input_scale': 1.0, 'samples': [], 'timings': []}
    artifact = directory / 'backward-benchmark.json'
    artifact.write_text(json.dumps(record, indent=2), encoding='utf-8')
    try:
        with (directory / 'backward-bench.log').open('w', encoding='utf-8') as log:
            for case in BENCHMARK_CASES:
                samples = []
                for repeat in range(args.repeats):
                    if hashlib.sha256(exe.read_bytes()).hexdigest() != binary_hash:
                        raise RuntimeError('Backward executable changed during benchmark')
                    command = [str(exe), '--kernel=03-split-q', *arguments(case),
                               '--reference-check=true', '--verify-only=false', '--verify-reuse=true',
                               '--seed=3080', '--input-scale=1.0', f'--iterations={args.iterations}',
                               f'--mae-tolerance={SPLIT_Q_MAE_TOLERANCE}',
                               f'--max-abs-tolerance={MAX_ABS_TOLERANCE}']
                    result = subprocess.run(command, capture_output=True, text=True, errors='replace')
                    log.write(json.dumps(command) + '\n' + result.stdout + result.stderr + '\n')
                    log.flush()
                    metrics = backward_metrics(result.stdout)
                    match = re.search(r'Backward runtime:\s*([\d.eE+-]+) ms', result.stdout)
                    accepted = result.stdout.find('Backward reference passed: 03-split-q (end-to-end; non-default stream)')
                    runtime = float(match[1]) if match else None
                    passed = (result.returncode == 0 and 'DIAGNOSTIC' not in result.stdout
                              and valid_backward_metrics(metrics) and accepted >= 0 and match is not None
                              and accepted < match.start() and math.isfinite(runtime) and runtime > 0
                              and 'Backward reference: cuDNN SDPA backward' in result.stdout
                              and 'Backward reuse passed:' in result.stdout
                              and 'Workspace rejection passed: missing and undersized' in result.stdout)
                    record['samples'].append({'shape': case, 'repeat': repeat, 'runtime_ms': runtime,
                                               'passed': passed, 'returncode': result.returncode,
                                               'metrics': metrics, 'command': command,
                                               'command_sha256': hashlib.sha256(json.dumps(command).encode('utf-8')).hexdigest()})
                    if not passed:
                        raise RuntimeError(f'Backward benchmark parity/timing failed: {directory / "backward-bench.log"}')
                    samples.append(runtime)
                row = dict(zip(('B', 'H', 'Sq', 'Sk', 'D'), case))
                row.update(kernel=args.kernel, median_ms=statistics.median(samples), min_ms=min(samples),
                           max_ms=max(samples), samples_ms=json.dumps(samples), iterations=args.iterations)
                record['timings'].append(row)
                print(f'{case} backward 03-split-q: {row["median_ms"]:.6f} ms', flush=True)
        with (directory / 'backward-benchmark.csv').open('w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=list(record['timings'][0]))
            writer.writeheader()
            writer.writerows(record['timings'])
        record['status'] = 'passed'
    except Exception:
        record['status'] = 'failed'
        raise
    finally:
        artifact.write_text(json.dumps(record, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--kernel', choices=('02-split-kv', '03-split-q'), default='02-split-kv')
    parser.add_argument('--phase', choices=('forward', 'backward'), default='forward')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 1 or args.iterations < 1:
        parser.error('Repeat and iteration counts must be positive')
    exe = args.exe.resolve()
    directory = args.output_dir.resolve()
    if args.phase == 'backward':
        benchmark_backward(args, exe, directory)
        return
    stamp = json.loads((directory / 'verification.json').read_text(encoding='utf-8'))
    binary_hash = hashlib.sha256(exe.read_bytes()).hexdigest()
    if stamp['status'] != 'passed' or stamp['binary_sha256'] != binary_hash:
        raise RuntimeError('Run verify.py successfully on this exact executable first')
    require_parity(stamp, args.kernel)
    kernels = ('02-split-kv', '03-split-q') if args.kernel == '03-split-q' else (
        '00-naive', '01-online-softmax', '02-split-kv')
    record = {'status': 'running', 'executable': str(exe), 'binary_sha256': binary_hash,
              'selected_kernel': args.kernel, 'kernels': list(kernels),
              'reference': stamp.get('reference', 'cudnn'),
              'repeats': args.repeats, 'iterations': args.iterations, 'seed': 3080,
              'input_scale': 1.0, 'samples': [], 'ratios': [],
              'ratio_requirement': '< 0.95' if args.kernel == '03-split-q' else None}
    rows = []
    with (directory / 'bench.log').open('w', encoding='utf-8') as log:
        for case in BENCHMARK_CASES:
            timings = {key: [] for key in kernels}
            # Interleave variants to reduce bias from gradual clock/thermal drift.
            for repeat in range(args.repeats):
                order = kernels if repeat % 2 == 0 else tuple(reversed(kernels))
                for key in order:
                    command = [str(exe), f'--kernel={key}', *arguments(case),
                               '--reference-check=false', f'--iterations={args.iterations}',
                               '--seed=3080', '--input-scale=1.0']
                    result = subprocess.run(command, capture_output=True, text=True, errors='replace')
                    log.write(json.dumps(command) + '\n' + result.stdout + result.stderr + '\n')
                    match = re.search(r'Runtime:\s*([\d.eE+-]+) ms', result.stdout)
                    if result.returncode or not match:
                        raise RuntimeError(f'Benchmark failed: {directory / "bench.log"}')
                    timings[key].append(float(match[1]))
                    record['samples'].append({'shape': case, 'kernel': key, 'repeat': repeat,
                                               'runtime_ms': float(match[1]), 'command': command})
            for key, samples in timings.items():
                row = dict(zip(('B', 'H', 'Sq', 'Sk', 'D'), case))
                row.update(kernel=key, median_ms=statistics.median(samples), min_ms=min(samples),
                           max_ms=max(samples), samples_ms=json.dumps(samples), iterations=args.iterations)
                rows.append(row)
                print(f'{case} {key}: {row["median_ms"]:.6f} ms', flush=True)
            if args.kernel == '03-split-q':
                ratio = statistics.median(timings['03-split-q']) / statistics.median(timings['02-split-kv'])
                passed = ratio < 0.95
                record['ratios'].append({'shape': case, 't03_over_t02': ratio, 'passed': passed})
                print(f'{case} t03/t02={ratio:.6f} (< 0.95): {"PASS" if passed else "FAIL"}', flush=True)
    with (directory / 'benchmark.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = all(row['passed'] for row in record['ratios'])
    record['status'] = 'passed' if passed else 'failed-performance-target'
    record['timings'] = rows
    (directory / 'benchmark.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    if not passed:
        raise RuntimeError('Split-Q performance contract failed; see benchmark.json')


if __name__ == '__main__':
    main()
