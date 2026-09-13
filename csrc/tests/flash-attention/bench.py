"""Compare verified attention variants on identical shapes with CUDA events."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

from verify import BENCHMARK_CASES, arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 1 or args.iterations < 1:
        parser.error('Repeat and iteration counts must be positive')
    exe = args.exe.resolve()
    directory = args.output_dir.resolve()
    stamp = json.loads((directory / 'verification.json').read_text(encoding='utf-8'))
    if stamp['status'] != 'passed' or stamp['binary_sha256'] != hashlib.sha256(exe.read_bytes()).hexdigest():
        raise RuntimeError('Run verify.py successfully on this exact executable first')
    for case in BENCHMARK_CASES:
        if not any(row['kernel'] == 'all' and tuple(row['shape']) == case and row['passed']
                   for row in stamp['cases']):
            raise RuntimeError(f'Missing all-kernel parity for {case}')
    rows = []
    with (directory / 'bench.log').open('w', encoding='utf-8') as log:
        for case in BENCHMARK_CASES:
            timings = {key: [] for key in ('00-naive', '01-online-softmax', '02-split-kv')}
            # Interleave variants to reduce bias from gradual clock/thermal drift.
            for repeat in range(args.repeats):
                for key in timings:
                    command = [str(exe), f'--kernel={key}', *arguments(case),
                               '--reference-check=false', f'--iterations={args.iterations}']
                    result = subprocess.run(command, capture_output=True, text=True, errors='replace')
                    log.write(json.dumps(command) + '\n' + result.stdout + result.stderr + '\n')
                    match = re.search(r'Runtime:\s*([\d.eE+-]+) ms', result.stdout)
                    if result.returncode or not match:
                        raise RuntimeError(f'Benchmark failed: {directory / "bench.log"}')
                    timings[key].append(float(match[1]))
            for key, samples in timings.items():
                row = dict(zip(('B', 'H', 'Sq', 'Sk', 'D'), case))
                row.update(kernel=key, median_ms=statistics.median(samples), min_ms=min(samples),
                           max_ms=max(samples), samples_ms=json.dumps(samples), iterations=args.iterations)
                rows.append(row)
                print(f'{case} {key}: {row["median_ms"]:.6f} ms', flush=True)
    with (directory / 'benchmark.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
