"""Benchmark the full 720p layer only after same-input parity in each process."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from source_policy import ROOT, check_source
from verify import executable, run_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build/fused_swin_layer')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error('positive iterations and repeats required')
    check_source()
    sys.path.insert(0, str(ROOT/'csrc/tests/swin'))
    from block_reference import benchmark_cases, write_fixture
    import torch
    build = args.build_dir.resolve()
    run_dir = build/'benchmark-runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'run_dir': str(run_dir),
              'capability': list(torch.cuda.get_device_capability()),
              'binaries': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [executable(build)] + ([args.baseline] if args.baseline else [])},
              'timing': 'full_block_cuda_graph', 'iterations': args.iterations,
              'repeats': args.repeats, 'cases': []}
    for case in benchmark_cases():
        directory, tensors, expected = write_fixture(case, run_dir/'fixtures')
        del tensors, expected
        times = {'candidate': [], 'baseline': []}
        binaries = [('candidate', executable(build))]
        if args.baseline:
            binaries.append(('baseline', args.baseline.resolve()))
        for repeat in range(args.repeats):
            for name, binary in (binaries if repeat % 2 == 0 else list(reversed(binaries))):
                output = run_case(binary, directory, args.iterations)
                (directory/f'{name}-{repeat}.log').write_text(output, encoding='utf-8')
                times[name].append(float(re.search(r'Runtime: ([\d.eE+-]+) ms', output)[1]))
        item = {'name': case.name, 'milliseconds': times, 'candidate_median_ms': statistics.median(times['candidate'])}
        if times['baseline']:
            item['baseline_median_ms'] = statistics.median(times['baseline'])
            item['speedup'] = item['baseline_median_ms']/item['candidate_median_ms']
        report['cases'].append(item)
    report_name = 'benchmark-comparison.json' if args.baseline else 'benchmark.json'
    (run_dir/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (build/report_name).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
