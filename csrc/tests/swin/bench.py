"""Benchmark verified fused attention; optional NCU/NSYS artifacts under build/."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import statistics
import hashlib

from source_policy import check_sources


def run(command: list[str], log: Path | None = None) -> str:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout, encoding='utf-8')
    if result.returncode:
        print(result.stdout)
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.stdout


def pytorch_time(case, tensors, iterations: int) -> float:
    import torch
    from window_attention_reference import expand_position_bias, reference

    # Eager CUDA-event latency for both paths includes gaps caused by host
    # dispatch. NCU provides the separate, isolated fused kernel duration.
    bias = expand_position_bias(case, tensors['position_bias'])
    with torch.inference_mode():
        for _ in range(10): reference(case, tensors, bias)
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations): reference(case, tensors, bias)
        stop.record()
        stop.synchronize()
    return start.elapsed_time(stop) / iterations


def benchmark_block(args):
    import torch
    from window_attention_reference import artifact_root, configure_reference, executable
    from block_reference import benchmark_cases as block_cases, write_fixture as block_fixture, prepare, reference as block_reference
    configure_reference()
    exe = executable(args.build_dir, args.config, 'swin_block')
    report = artifact_root(args.report_dir or args.build_dir / 'reports/block')
    report.mkdir(parents=True, exist_ok=True)
    cases = block_cases()
    if args.case:
        unknown = set(args.case) - {c.name for c in cases}
        if unknown: raise ValueError(f'unknown block cases: {unknown}')
        cases = [c for c in cases if c.name in args.case]
    from dataclasses import replace
    paths = {'candidate': exe}
    if args.baseline_executable: paths['baseline'] = args.baseline_executable.resolve()
    results = []
    for case in cases:
        case = replace(case, batch=args.batch_size)
        directory, tensors, expected = block_fixture(case, artifact_root(args.build_dir / 'fixtures/block'))
        # Every process validates the full output before entering its timed graph loop.
        samples = {name: [] for name in paths}
        order = list(paths)
        for repeat in range(args.repeats):
            for name in order[::1 if repeat % 2 == 0 else -1]:
                output = run([str(paths[name]), '--case-dir', str(directory), '--graph',
                              '--iterations', str(args.iterations)], report / f'{case.name}_{name}_{repeat}.log')
                match = re.search(r'Runtime:\s*([0-9.eE+-]+)\s*ms', output)
                if not match: raise RuntimeError('block timing missing')
                samples[name].append(float(match[1]))
        bias, axes = prepare(case, tensors)
        with torch.inference_mode():
            for _ in range(3): block_reference(case, tensors, bias, axes)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = block_reference(case, tensors, bias, axes)
            graph.replay(); torch.cuda.synchronize()
            error = (graph_output.float() - expected.float()).abs()
            if not bool(torch.isfinite(error).all()) or error.mean().item() > 1e-3 or error.max().item() > 2e-2:
                raise RuntimeError('PyTorch graph reference parity failed')
            torch_samples = []
            for _ in range(args.repeats):
                for _ in range(20): graph.replay()
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations): graph.replay()
                stop.record(); stop.synchronize()
                torch_samples.append(start.elapsed_time(stop) / args.iterations)
        del graph, graph_output, error
        samples['pytorch'] = torch_samples
        row = {'case': case.name, 'shape': [case.batch,case.height,case.width,case.channels], 'mlp_ratio':4,
               'timing':'full_block_cuda_graph', 'samples_ms':samples,
               'median_ms': {k: statistics.median(v) for k,v in samples.items()},
               'min_ms':{k:min(v) for k,v in samples.items()}, 'max_ms':{k:max(v) for k,v in samples.items()}}
        if 'baseline' in samples: row['speedup'] = row['median_ms']['baseline'] / row['median_ms']['candidate']
        results.append(row)
        print(case.name, row['median_ms'], flush=True)
        del tensors, expected, bias, axes
        torch.cuda.empty_cache()
        command = [str(exe), '--case-dir', str(directory)]
        if args.ncu:
            ncu = shutil.which('ncu') or shutil.which('ncu.bat')
            if not ncu: raise FileNotFoundError('ncu')
            base = report / 'ncu' / case.name
            base.parent.mkdir(parents=True, exist_ok=True)
            for mode in ('full', 'source'):
                target = base.with_name(base.name + '_' + mode)
                run([ncu, '--force-overwrite', '--set', mode, '--import-source', 'yes',
                     '--source-folders', str(Path(__file__).resolve().parents[3]), '--launch-count', '1',
                     '--export', str(target), '--page', 'raw', '--csv', *command], target.with_suffix('.csv'))
            row['ncu_full'] = str(base.with_name(base.name + '_full.ncu-rep'))
            row['ncu_source'] = str(base.with_name(base.name + '_source.ncu-rep'))
        if args.nsys:
            nsys = shutil.which('nsys')
            if not nsys: raise FileNotFoundError('nsys')
            base = report / 'nsys' / case.name
            base.parent.mkdir(parents=True, exist_ok=True)
            run([nsys, 'profile', '--force-overwrite=true', '--trace=cuda', '--sample=none',
                 '--cpuctxsw=none', '-o', str(base), *command, '--iterations', '10'], base.with_suffix('.log'))
            run([nsys, 'stats', '--force-export=true', '--report', 'cuda_gpu_kern_sum', '--format', 'csv',
                 str(base.with_suffix('.nsys-rep'))], base.with_suffix('.csv'))
            row['nsys_report'] = str(base.with_suffix('.nsys-rep'))
    payload = {'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__, 'iterations':args.iterations,
               'repeats':args.repeats,'warmup':20,'binaries':{name:{'path':str(path),
               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in paths.items()},'results':results}
    (report/'bench.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (report/'bench.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['case','implementation','median_ms','min_ms','max_ms','timing']);writer.writeheader()
        for row in results:
            for name in row['samples_ms']:
                writer.writerow({'case':row['case'],'implementation':name,'median_ms':row['median_ms'][name],
                                 'min_ms':row['min_ms'][name],'max_ms':row['max_ms'][name],'timing':row['timing']})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=Path('build/swin'))
    parser.add_argument('--config', default='Release')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--ncu', action='store_true')
    parser.add_argument('--nsys', action='store_true')
    parser.add_argument('--grouped-gemm', action='store_true')
    parser.add_argument('--block', action='store_true')
    parser.add_argument('--baseline-executable', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not check_sources():
        return 1
    import torch
    from window_attention_reference import artifact_root, benchmark_cases, configure_reference, executable, write_fixture

    if args.repeats <= 0: raise ValueError('positive repeats required')
    if args.iterations <= 0 or args.batch_size <= 0: raise ValueError('positive iterations and batch required')
    if args.block: return benchmark_block(args)
    if args.grouped_gemm:
        exe = executable(args.build_dir, args.config, 'swin_grouped_gemm')
        report = artifact_root(args.report_dir or args.build_dir / 'reports/grouped_gemm')
        # Verify every format before timing either format, including standalone invocations.
        run([str(exe)], report / 'verify.log')
        output = run([str(exe), '--iterations', str(args.iterations)], report / 'bench.log')
        rows = [{'format': fmt, 'runtime_ms': float(ms)} for fmt, ms in
                re.findall(r'(e[45]m[23]/e[45]m[23]) Runtime:\s*([0-9.eE+-]+)', output)]
        if len(rows) != 2: raise RuntimeError('missing grouped GEMM timings')
        with (report / 'bench.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['format', 'runtime_ms'])
            writer.writeheader(); writer.writerows(rows)
        if args.ncu:
            ncu = shutil.which('ncu') or shutil.which('ncu.bat')
            if not ncu: raise FileNotFoundError('ncu')
            run([ncu, '--force-overwrite', '--target-processes', 'all', '--set', 'full',
                 '--launch-count', '2', '--export', str(report / 'grouped_gemm'),
                 '--page', 'raw', '--csv', str(exe)], report / 'ncu.csv')
        if args.nsys:
            nsys = shutil.which('nsys')
            if not nsys: raise FileNotFoundError('nsys')
            base = report / 'grouped_gemm'
            run([nsys, 'profile', '--force-overwrite=true', '--trace=cuda', '--sample=none',
                 '--cpuctxsw=none', '-o', str(base), str(exe), '--iterations', '10'], report / 'nsys.log')
            run([nsys, 'stats', '--force-export=true', '--force-overwrite=true', '--report', 'cuda_gpu_kern_sum',
                 '--format', 'csv', str(base.with_suffix('.nsys-rep'))], report / 'nsys.csv')
        (report / 'bench.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(output, end='')
        return 0
    configure_reference()
    exe = executable(args.build_dir, args.config)
    cases = benchmark_cases()
    if args.case:
        unknown = set(args.case) - {c.name for c in cases}
        if unknown: raise ValueError(f'unknown cases: {sorted(unknown)}')
        cases = [c for c in cases if c.name in args.case]
    report = artifact_root(args.report_dir or args.build_dir / 'reports/window_attention')
    fixtures = artifact_root(args.build_dir / 'fixtures/window_attention')
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    from dataclasses import replace
    for case in cases:
        case = replace(case, batch=args.batch_size)
        directory, tensors, _ = write_fixture(case, fixtures)
        command = [str(exe), '--case-dir', str(directory)]
        # C++ validates these exact tensors before entering its timed loop.
        output = run([*command, '--iterations', str(args.iterations)], report / f'{case.name}.log')
        match = re.search(r'Runtime:\s*([0-9.eE+-]+)\s*ms', output)
        if not match: raise RuntimeError('native timing missing')
        reference_ms = pytorch_time(case, tensors, args.iterations)
        row = {'case': case.name, 'batch': args.batch_size, 'fused_ms': float(match[1]),
               'pytorch_ms': reference_ms, 'timing': 'eager_cuda_events', 'kernel_launches': 1}
        if args.ncu:
            ncu = shutil.which('ncu') or shutil.which('ncu.bat')
            if not ncu: raise FileNotFoundError('ncu')
            base = report / 'ncu' / case.name
            base.parent.mkdir(parents=True, exist_ok=True)
            run([ncu, '--force-overwrite', '--target-processes', 'all', '--set', 'full',
                 '--launch-count', '1', '--import-source', 'yes', '--export', str(base),
                 '--page', 'raw', '--csv', *command], base.with_suffix('.csv'))
            row['ncu_report'] = str(base.with_suffix('.ncu-rep'))
        if args.nsys:
            nsys = shutil.which('nsys')
            if not nsys: raise FileNotFoundError('nsys')
            base = report / 'nsys' / case.name
            base.parent.mkdir(parents=True, exist_ok=True)
            run([nsys, 'profile', '--force-overwrite=true', '--trace=cuda', '--sample=none',
                 '--cpuctxsw=none', '-o', str(base), *command, '--iterations', '10'], base.with_suffix('.log'))
            run([nsys, 'stats', '--report', 'cuda_gpu_kern_sum', '--format', 'csv',
                 str(base.with_suffix('.nsys-rep'))], base.with_suffix('.csv'))
            row['nsys_report'] = str(base.with_suffix('.nsys-rep'))
        rows.append(row)
        print(f"{case.name}: fused {row['fused_ms']:.4f} ms, PyTorch {reference_ms:.4f} ms", flush=True)
    (report / 'bench.json').write_text(json.dumps({'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'iterations': args.iterations, 'results': rows}, indent=2), encoding='utf-8')
    with (report / 'bench.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    print(f'Artifacts: {report}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
