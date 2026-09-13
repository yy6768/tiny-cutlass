"""Independent PyTorch transcription of the pasted DLSS4 attention forward.

Missing helpers are explicit here: repeated reflection (singleton replication)
with mandatory RMSNorm before attention and preexpanded runtime position bias.
The reference never imports or consumes the kernel's gather/scatter tables.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    name: str
    batch: int = 1
    height: int = 8
    width: int = 12
    channels: int = 96
    groups: int = 3
    qk_channels: int = 32
    value_channels: int = 32
    shift_h: int = 0
    shift_w: int = 0
    separate: bool = False
    dense: bool = False
    rms_epsilon: float = 0.0009765625
    input_offset: float = 0.0
    output_bias: bool = True
    input_scale: float = 0.7
    bias_scale: float = 0.2
    zero_qk: bool = False


def verification_cases() -> list[Case]:
    cases = []
    for height, width in ((8, 12), (7, 9)):
        for separate in (False, True):
            for dense in (False, True):
                for sh, sw in ((0, 0), (2, 2), (0, 3), (1, 0)):
                    name = f"reflect_{height}x{width}_{'separate' if separate else 'shared'}_{'dense' if dense else 'relative'}_{sh}{sw}"
                    cases.append(Case(name, batch=2, height=height, width=width,
                                      separate=separate, dense=dense, shift_h=sh, shift_w=sw))
    for sh in range(4):
        for sw in range(4):
            cases.append(Case(f"single_window_shift_{sh}{sw}", height=4, width=4,
                              channels=24, groups=5, qk_channels=8, value_channels=16,
                              dense=True, separate=True, shift_h=sh, shift_w=sw,
                              bias_scale=2.0))
    for h, w in ((1, 1), (1, 7), (2, 3), (3, 1), (4, 4), (5, 11)):
        cases.append(Case(f"reflect_small_{h}x{w}", height=h, width=w, 
                          channels=40, groups=2, qk_channels=24, value_channels=40,
                          separate=True, shift_h=3, shift_w=1))
    cases.extend([
        Case("tails", channels=8, groups=1, qk_channels=8, value_channels=8),
        Case("different_dims", channels=48, groups=5, qk_channels=48, value_channels=24, separate=True),
        Case("max_heads", height=4, width=4, channels=32, groups=64, qk_channels=8, value_channels=8),
        Case("max_dimensions", height=4, width=8, channels=1024, groups=8, qk_channels=64, value_channels=64, separate=True),
        Case("large_logits", separate=True, input_scale=3.0, bias_scale=3.0, shift_h=2, shift_w=2),
        Case("bias_scale_order", zero_qk=True, bias_scale=4.0, dense=True),
        Case("relative_orientation", zero_qk=True, bias_scale=4.0),
        Case("null_output_bias", output_bias=False),
        Case("rms_nonzero_mean", input_offset=2.0),
        Case("rms_zero_input", input_scale=0.0),
        Case("rms_tiny_input", input_scale=0.0001, rms_epsilon=1e-6),
        Case("rms_constant_input", input_scale=0.0, input_offset=0.5, rms_epsilon=1e-5),
    ])
    return cases


def benchmark_cases() -> list[Case]:
    return [
        Case("window_attention_64x64", height=64, width=64, channels=96, groups=3),
        Case("window_attention_shift_64x64", height=64, width=64, channels=96, groups=3, shift_h=2, shift_w=2),
        Case("window_attention_reflect_63x65", height=63, width=65, channels=96, groups=3,  shift_h=2, shift_w=2),
        Case("window_attention_separate_32x32", height=32, width=32, channels=192, groups=6, separate=True),
    ]


def partition(x: torch.Tensor) -> torch.Tensor:
    b, h, w, c = x.shape
    return x.reshape(b, h // 4, 4, w // 4, 4, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, 16, c)


def reflect_axis(length: int, before: int, after: int, device: torch.device) -> torch.Tensor:
    # Construct the reflected sequence from an explicit mirrored period.
    if length == 1:
        return torch.zeros(before + length + after, dtype=torch.long, device=device)
    period = list(range(length)) + list(range(length - 2, 0, -1))
    values = [period[i % len(period)] for i in range(-before, length + after)]
    return torch.tensor(values, dtype=torch.long, device=device)


def reference(case: Case, tensors: dict[str, torch.Tensor], expanded_bias=None, axes=None) -> torch.Tensor:
    x = F.rms_norm(tensors['input'], (case.channels,), tensors['rms_weight'], eps=case.rms_epsilon)
    b, h, w, _ = x.shape
    sh, sw = case.shift_h, case.shift_w
    hp = math.ceil((h + sh) / 4) * 4
    wp = math.ceil((w + sw) / 4) * 4
    iy, ix = axes if axes is not None else (reflect_axis(h, sh, hp - h - sh, x.device),
                                          reflect_axis(w, sw, wp - w - sw, x.device))
    x = x.index_select(1, iy)
    x = x.index_select(2, ix)
    windows = partition(x)
    qkv = torch.einsum('ntc,goc->ntgo', windows, tensors['qkv_weight'])
    if case.separate:
        query, key, value = qkv.split([case.qk_channels, case.qk_channels, case.value_channels], -1)
        key = key.permute(0, 2, 1, 3).contiguous()
    else:
        query, value = qkv.split([case.qk_channels, case.value_channels], -1)
    query = query.permute(0, 2, 1, 3).contiguous()
    if not case.separate:
        key = query
    value = value.permute(0, 2, 1, 3).contiguous()
    scores = query @ key.transpose(-1, -2)
    if expanded_bias is not None:
        bias = expanded_bias.reshape(case.groups, 16, 16)
    elif case.dense:
        bias = tensors['position_bias']
    else:
        coords = torch.stack(torch.meshgrid(torch.arange(4, device=x.device), torch.arange(4, device=x.device), indexing='ij')).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous() + 3
        index = rel[..., 0] * 7 + rel[..., 1]
        bias = tensors['position_bias'][index.flatten()].view(16, 16, case.groups).permute(2, 0, 1)
    scores = scores + bias.unsqueeze(0)
    weights = torch.softmax(scores * (case.qk_channels ** -0.5), dim=-1)
    groups = (weights @ value).permute(0, 2, 1, 3).contiguous()
    hidden = groups.reshape(groups.shape[0], 16, case.groups * case.value_channels)
    hidden = F.linear(hidden, tensors['output_weight'], tensors['output_bias'] if case.output_bias else None)
    y = hidden.reshape(b, hp // 4, wp // 4, 4, 4, case.channels).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, case.channels)
    return y[:, sh:sh + h, sw:sw + w].contiguous()


def configure_reference() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def make_tensors(case: Case, device: str = 'cuda') -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(2026)
    def rand(shape, scale):
        return (torch.randn(shape, generator=gen) * scale).half().to(device)
    projected = (2 if case.separate else 1) * case.qk_channels + case.value_channels
    tensors = {
        'input': rand((case.batch, case.height, case.width, case.channels), case.input_scale),
        'rms_weight': rand((case.channels,), 0.25) + 1,
        'qkv_weight': rand((case.groups, projected, case.channels), 0.6 / math.sqrt(case.channels)),
        'position_bias': rand((case.groups, 16, 16) if case.dense else (49, case.groups), case.bias_scale),
        'output_weight': rand((case.channels, case.groups * case.value_channels), 0.6 / math.sqrt(case.groups * case.value_channels)),
        'output_bias': rand((case.channels,), 0.1),
    }
    tensors['input'] += case.input_offset
    if case.zero_qk:
        tensors['qkv_weight'][:, :projected - case.value_channels] = 0
    return tensors


def expand_position_bias(case: Case, compact: torch.Tensor) -> torch.Tensor:
    """Prepare once per weight update: 2D [G*16,16], reused by every window.

    Deliberately use scalar host indexing, independent of the reference's
    meshgrid/gather expression, to check query-minus-key orientation.
    """
    if case.dense:
        return compact.reshape(case.groups * 16, 16).contiguous()
    table = compact.cpu()
    expanded = torch.empty((case.groups * 16, 16), dtype=table.dtype)
    for g in range(case.groups):
        for q in range(16):
            for k in range(16):
                expanded[g * 16 + q, k] = table[(q // 4 - k // 4 + 3) * 7 + q % 4 - k % 4 + 3, g]
    return expanded.to(compact.device)


def write_fixture(case: Case, root: Path) -> tuple[Path, dict[str, torch.Tensor], torch.Tensor]:
    directory = root / case.name
    directory.mkdir(parents=True, exist_ok=True)
    tensors = make_tensors(case)
    with torch.inference_mode():
        expected = reference(case, tensors)
    if not bool(torch.isfinite(expected).all()):
        raise RuntimeError(f'nonfinite reference: {case.name}')
    fields = [case.batch, case.height, case.width, case.channels, case.groups, case.qk_channels,
              case.value_channels, case.shift_h, case.shift_w, int(case.separate), case.rms_epsilon, int(case.output_bias), int(not case.dense)]
    (directory / 'problem.txt').write_text(' '.join(map(str, fields)), encoding='utf-8')
    (directory / 'case.json').write_text(json.dumps(asdict(case), indent=2), encoding='utf-8')
    for name, tensor in tensors.items():
        tensor.cpu().contiguous().numpy().tofile(directory / f'{name}.bin')
    if not case.dense:
        tensors['position_bias'].cpu().numpy().tofile(directory / 'relative_bias.bin')
    expand_position_bias(case, tensors['position_bias']).cpu().numpy().tofile(directory / 'position_bias.bin')
    expected.float().cpu().numpy().tofile(directory / 'reference.bin')
    return directory, tensors, expected


def executable(build_dir: Path, config: str, target: str = 'swin_window_attention') -> Path:
    import sys
    name = target + ('.exe' if sys.platform == 'win32' else '')
    for path in (build_dir / 'tests/swin' / config / name, build_dir / 'tests/swin' / name):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(name)


def artifact_root(path: Path) -> Path:
    root = Path(__file__).resolve().parents[3] / 'build'
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'artifacts must remain under {root}')
    return path
