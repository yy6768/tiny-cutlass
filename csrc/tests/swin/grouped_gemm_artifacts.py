"""Extract and validate the actual compiled grouped GEMM PTX and native code."""
from pathlib import Path
import json
import re
import shutil
import subprocess


def dump_code(exe: Path, directory: Path) -> dict:
    cuobjdump = shutil.which('cuobjdump')
    ptxas = shutil.which('ptxas')
    if not cuobjdump or not ptxas:
        raise FileNotFoundError('CUDA cuobjdump and ptxas must be on PATH')
    directory.mkdir(parents=True, exist_ok=True)
    dump = subprocess.check_output([cuobjdump, '--dump-ptx', str(exe)], text=True)
    modules = re.findall(r'(?ms)^\.version .*?(?=^Fatbin |\Z)', dump)
    if len(modules) != 1:
        raise RuntimeError(f'expected one PTX module, found {len(modules)}')
    ptx = modules[0].strip() + '\n'
    wanted = {'mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32',
              'mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32'}
    instructions = set(re.findall(r'mma\.sync\.aligned\.[^\s]+', ptx))
    if instructions != wanted or '.target sm_89' not in ptx:
        raise RuntimeError(f'unexpected PTX instructions: {instructions}')
    path = directory / 'grouped_gemm.ptx'
    path.write_text(ptx, encoding='utf-8')
    # Ensure the exported file is a complete, assemblable PTX module.
    subprocess.run([ptxas, '-arch=sm_89', str(path), '-o', str(directory / 'grouped_gemm.cubin')], check=True)
    sass = subprocess.check_output([cuobjdump, '--dump-sass', str(exe)], text=True)
    (directory / 'grouped_gemm.sass').write_text(sass, encoding='utf-8')
    if 'QMMA.16832.F32.E4M3.E4M3' not in sass or 'QMMA.16832.F32.E5M2.E5M2' not in sass:
        raise RuntimeError('native FP8 QMMA instructions missing')
    result = {'target': 'sm_89', 'ptx': str(path), 'instructions': sorted(instructions),
              'native_instructions': sorted(set(re.findall(r'QMMA\.[^\s]+', sass)))}
    (directory / 'instructions.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
