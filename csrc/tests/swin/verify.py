"""Swin correctness gate, including the full fused DLSS4 attention forward."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from window_attention_reference import artifact_root, configure_reference, executable, verification_cases, write_fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=Path('build/swin'))
    parser.add_argument('--config', default='Release')
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--all-families', action='store_true')
    parser.add_argument('--block', action='store_true', help='full C32/ratio4 block including 720p cases')
    args = parser.parse_args()
    configure_reference()
    families = ('swin_window_index', 'swin_window_gather_scatter', 'swin_patch_embed', 'swin_mlp', 'swin_patch_merging')
    try:
        executable(args.build_dir, args.config, 'swin_grouped_gemm')
        families += ('swin_grouped_gemm',)
    except FileNotFoundError:
        if 'swin_grouped_gemm' in args.case:
            raise
    cases = verification_cases()
    fixture_writer = write_fixture
    family = 'window_attention'
    operator_name = 'swin_window_attention'
    if args.block:
        from block_reference import cases as block_cases, write_fixture as block_fixture
        cases = block_cases()
        fixture_writer = block_fixture
        family, operator_name = 'block', 'swin_block'
    if args.case:
        unknown = set(args.case) - set(families) - {operator_name} - {c.name for c in cases}
        if unknown: raise ValueError(f'unknown cases: {sorted(unknown)}')
    selected_families = [name for name in families if name in args.case or (args.all_families and not args.case)]
    if selected_families:
        for name in selected_families:
            suffix = '.exe' if sys.platform == 'win32' else ''
            base = args.build_dir / 'tests/swin'
            exe = base / args.config / (name + suffix)
            if not exe.exists(): exe = base / (name + suffix)
            subprocess.run([str(exe.resolve())], check=True)
            if name == 'swin_grouped_gemm':
                from grouped_gemm_artifacts import dump_code
                result = dump_code(exe.resolve(), artifact_root(args.build_dir / 'reports/grouped_gemm'))
                print('PTX/native instruction gate passed:', result['ptx'], flush=True)
    if args.case and operator_name not in args.case:
        cases = [c for c in cases if c.name in args.case]
    if not cases:
        print(f'VERIFY PASSED: {len(selected_families)} family gate(s)')
        return 0
    exe = args.executable.resolve() if args.executable else executable(args.build_dir, args.config, operator_name)
    root = artifact_root(args.build_dir / 'fixtures' / family)
    rows = []
    for index, case in enumerate(cases):
        directory, _, expected = fixture_writer(case, root)
        command = [str(exe), '--case-dir', str(directory), '--graph']
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout, result.stderr)
            raise RuntimeError(f'{case.name}: kernel/reference gate failed')
        actual = torch.from_numpy(np.fromfile(directory / 'actual.bin', dtype=np.float16).copy()).float()
        error = (actual - expected.float().cpu().flatten()).abs()
        if actual.numel() != expected.numel() or not torch.isfinite(error).all():
            raise RuntimeError(f'{case.name}: invalid output')
        mae, maximum = error.mean().item(), error.max().item()
        if mae > 1e-3 or maximum > 2e-2:
            raise RuntimeError(f'{case.name}: MAE={mae}, max_abs={maximum}')
        rows.append({'case': case.name, 'shape': [case.batch, case.height, case.width, case.channels],
                     'mae': mae, 'max_abs': maximum, 'graph_nodes': 1})
        print(f'[{index + 1}/{len(cases)}] PASS {case.name}: MAE={mae:.3g} max_abs={maximum:.3g}', flush=True)
    report = artifact_root(args.build_dir / 'reports' / family / 'verify.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({'seed': 2026, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                                 'mae_limit': 1e-3, 'max_abs_limit': 2e-2, 'cases': rows}, indent=2), encoding='utf-8')
    print(f'VERIFY PASSED: {len(rows)} {family} cases; report: {report}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
