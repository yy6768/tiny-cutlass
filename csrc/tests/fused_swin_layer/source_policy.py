"""Local source gate; library internals are deliberately outside its scope."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / 'csrc/swin/fused_swin_layer'

# Method names are CUTLASS's device/operator contracts, with actual definition sources.
METHOD_SOURCES = {
    'operator': 'gemm/device/gemm.h',
    'can_implement': 'gemm/device/gemm.h',
    'get_workspace_size': 'gemm/device/gemm.h',
    'initialize': 'gemm/device/gemm.h',
    'run': 'gemm/device/gemm.h',
}


def check_source():
    errors = []
    for name, source in METHOD_SOURCES.items():
        text = (ROOT / '3rdparty/cutlass/include/cutlass' / source).read_text(encoding='utf-8')
        if not re.search(r'\b' + name + r'\s*\(', text):
            errors.append(f'CUTLASS method source missing: {name}')
    files = sorted(FAMILY.rglob('*.h'))
    if not files:
        errors.append('family has no implementation')
    for path in files:
        text = path.read_text(encoding='utf-8')
        code = re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)
        patterns = {
            'inline assembly': r'\b(?:asm|__asm__)\b',
            'CuTe/collective route': r'cute::|["<]cute/|cutlass/gemm/collective',
            'Torch ownership': r'\b(?:at|torch)::|ATen/',
            'local arch or layout or iterator class': r'\b(?:class|struct)\s+\w*(?:Iterator|Layout|SmemLayout|Arch)\w*',
            'local low-level include': r'#\s*include\s+"swin/[^"\n]*/(?:arch|smem_layout|iterator)/',
            'raw global kernel': r'\b__global__\b',
            'SIMT or WMMA substitution': r'OpClassSimt|nvcuda::wmma',
            'other architecture': r'\bSm(?:90|100|120)\b',
            'concrete dtype/arch name': r'\b(?:class|struct)\s+\w*(?:Fp16|BF16|Sm80|Sm89|NHWC)\w*',
            'old implementation dependency': r'#\s*include\s+"swin/window_attention/',
        }
        for label, pattern in patterns.items():
            if re.search(pattern, code):
                errors.append(f'{path.relative_to(ROOT)}: {label}')
        # Definitions with a return type; constructor/type names and operator() are separate contracts.
        definitions = re.findall(
            r'\b(?:void|bool|int|float|double|size_t|cutlass::Status)\s+([A-Za-z_]\w*)\s*\(', code)
        for name in definitions:
            if name not in METHOD_SOURCES:
                errors.append(f'{path.relative_to(ROOT)}: unregistered function {name}')
    if errors:
        raise RuntimeError('\n'.join(errors))
    print(f'SOURCE PASS: {len(files)} headers, CUTLASS method definitions and local boundaries checked')
    return files


if __name__ == '__main__':
    check_source()
