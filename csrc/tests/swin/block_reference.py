"""Independent full DLSS4 block reference and the requested 720p workload."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from window_attention_reference import (Case, make_tensors as attention_tensors,
    reference as attention_reference, expand_position_bias, reflect_axis)


def cases(full=True):
    result = [Case(f'block_shift_{h}{w}', height=5, width=7, channels=32, groups=1,
                   shift_h=h, shift_w=w, separate=(h % 2 == 1), dense=(w % 2 == 1))
              for h in range(4) for w in range(4)]
    result += [Case('block_singleton', height=1, width=1, channels=32, groups=1, shift_h=3, shift_w=3),
               Case('block_constant', height=4, width=4, channels=32, groups=1, input_scale=0, input_offset=2),
               Case('block_tiny', height=4, width=4, channels=32, groups=1, input_scale=1e-4, rms_epsilon=1e-6),
               Case('block_batch', batch=2, height=9, width=11, channels=32, groups=1, separate=True)]
    if full: result += benchmark_cases()
    return result


def benchmark_cases():
    return [Case('block_720x1280', height=720, width=1280, channels=32, groups=1),
            Case('block_shift_720x1280', height=720, width=1280, channels=32, groups=1, shift_h=2, shift_w=2),
            Case('block_separate_720x1280', height=720, width=1280, channels=32, groups=1, separate=True, shift_h=2, shift_w=2)]


def make_tensors(case):
    tensors = attention_tensors(case)
    gen = torch.Generator().manual_seed(2027)
    def rand(shape, scale): return (torch.randn(shape, generator=gen) * scale).half().cuda()
    tensors.update({
        'mlp_rms_weight': rand((32,), .25) + 1,
        'fc1_weight': rand((128,32), .6 / math.sqrt(32)),
        'fc1_bias': rand((128,), .1),
        'fc2_weight': rand((32,128), .6 / math.sqrt(128)),
        'fc2_bias': rand((32,), .1),
    })
    return tensors


def prepare(case, tensors):
    hp = math.ceil((case.height + case.shift_h) / 4) * 4
    wp = math.ceil((case.width + case.shift_w) / 4) * 4
    device=tensors['input'].device
    axes=(reflect_axis(case.height,case.shift_h,hp-case.height-case.shift_h,device),
          reflect_axis(case.width,case.shift_w,wp-case.width-case.shift_w,device))
    return expand_position_bias(case,tensors['position_bias']), axes


def reference(case, tensors, bias=None, axes=None):
    r = tensors['input'] + attention_reference(case, tensors, bias, axes)
    norm = F.rms_norm(r, (case.channels,), tensors['mlp_rms_weight'], eps=case.rms_epsilon)
    hidden = F.gelu(F.linear(norm, tensors['fc1_weight'], tensors['fc1_bias']), approximate='none')
    return r + F.linear(hidden, tensors['fc2_weight'], tensors['fc2_bias'])


def write_fixture(case, root: Path):
    directory=root/case.name;directory.mkdir(parents=True,exist_ok=True)
    tensors=make_tensors(case)
    with torch.inference_mode(): expected=reference(case,tensors)
    if not bool(torch.isfinite(expected).all()): raise RuntimeError('nonfinite block reference')
    fields=[case.batch,case.height,case.width,case.channels,case.groups,case.qk_channels,
            case.value_channels,case.shift_h,case.shift_w,int(case.separate),case.rms_epsilon,
            int(case.output_bias),int(not case.dense)]
    (directory/'problem.txt').write_text(' '.join(map(str,fields)),encoding='utf-8')
    (directory/'mlp.txt').write_text(f'4 {case.rms_epsilon}',encoding='utf-8')
    (directory/'case.json').write_text(json.dumps({**asdict(case),'mlp_ratio':4},indent=2),encoding='utf-8')
    for name,tensor in tensors.items(): tensor.cpu().contiguous().numpy().tofile(directory/f'{name}.bin')
    if not case.dense: tensors['position_bias'].cpu().numpy().tofile(directory/'relative_bias.bin')
    expand_position_bias(case,tensors['position_bias']).cpu().numpy().tofile(directory/'position_bias.bin')
    expected.float().cpu().numpy().tofile(directory/'reference.bin')
    return directory,tensors,expected
