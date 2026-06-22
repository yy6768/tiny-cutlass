from __future__ import annotations

import argparse
import json
import shutil
import time
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch


OFFICIAL_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/"
    "swin_tiny_patch4_window7_224.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--model-source", type=Path, default=Path("cache/swin_transformer.py"))
    parser.add_argument("--fallback-onnx", type=Path, default=Path("cache/swin.onnx"))
    parser.add_argument("--offline-onnx", type=Path)
    parser.add_argument("--hf-model-id", default="microsoft/swin-tiny-patch4-window7-224")
    parser.add_argument("--url", default=OFFICIAL_URL)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "tiny-cutlass-swin-checkpoint-fetch/1.0",
            "Accept": "application/octet-stream",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            print(f"downloading {url} (attempt {attempt})")
            with urllib.request.urlopen(request, timeout=120) as response:
                with tmp.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            tmp.replace(path)
            return
        except Exception as exc:  # pragma: no cover - network errors are external.
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(min(attempt * 2, 10))
    raise RuntimeError(f"failed to download {url}: {last_error}") from last_error


def load_swin_module(source: Path):
    import importlib.util
    import types

    try:
        import timm  # noqa: F401
    except ModuleNotFoundError:
        timm_mod = types.ModuleType("timm")
        models_mod = types.ModuleType("timm.models")
        layers_mod = types.ModuleType("timm.models.layers")

        class DropPath(torch.nn.Identity):
            def __init__(self, drop_prob: float = 0.0) -> None:
                super().__init__()
                self.drop_prob = drop_prob

        def to_2tuple(value):
            return value if isinstance(value, tuple) else (value, value)

        def trunc_normal_(tensor, mean=0.0, std=1.0):
            return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std)

        layers_mod.DropPath = DropPath
        layers_mod.to_2tuple = to_2tuple
        layers_mod.trunc_normal_ = trunc_normal_
        sys.modules["timm"] = timm_mod
        sys.modules["timm.models"] = models_mod
        sys.modules["timm.models.layers"] = layers_mod

    spec = importlib.util.spec_from_file_location("official_swin_transformer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"unexpected checkpoint format in {path}")
    return checkpoint


def export_onnx(model: torch.nn.Module, path: Path, batch_size: int) -> None:
    if path.exists():
        return
    model.eval()
    example = torch.randn(batch_size, 3, 224, 224)
    torch.onnx.export(
        model,
        example,
        path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )


def write_f32(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(array, dtype=np.float32).tofile(path)


def copy_if_needed(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def relpos_bias(attn: torch.nn.Module) -> np.ndarray:
    table = attn.relative_position_bias_table.detach()
    index = attn.relative_position_index.detach().reshape(-1)
    window_len = attn.window_size[0] * attn.window_size[1]
    bias = table[index].reshape(window_len, window_len, -1).permute(2, 0, 1)
    return bias.contiguous().cpu().numpy()


def relpos_bias_from_table(table: torch.Tensor, heads: int, window_size: int = 7) -> np.ndarray:
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    index = relative_coords.sum(-1).reshape(-1)
    bias = table.detach().cpu()[index].reshape(
        window_size * window_size,
        window_size * window_size,
        heads,
    )
    return bias.permute(2, 0, 1).contiguous().numpy()


def export_artifacts(
    model: torch.nn.Module,
    checkpoint: dict[str, torch.Tensor],
    out_dir: Path,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    image_nchw = torch.randn(batch_size, 3, 224, 224, generator=generator)
    image_nhwc = image_nchw.permute(0, 2, 3, 1).contiguous().numpy()
    write_f32(out_dir / "input_image_nhwc.f32", image_nhwc)

    patch = model.patch_embed
    write_f32(out_dir / "patch_embed_input_nhwc.f32", image_nhwc)
    write_f32(out_dir / "patch_embed_weight_oihw.f32", patch.proj.weight.detach().numpy())
    write_f32(out_dir / "patch_embed_bias.f32", patch.proj.bias.detach().numpy())
    write_f32(out_dir / "patch_embed_norm_weight.f32", patch.norm.weight.detach().numpy())
    write_f32(out_dir / "patch_embed_norm_bias.f32", patch.norm.bias.detach().numpy())

    stages: list[dict[str, object]] = []
    for stage_idx, layer in enumerate(model.layers):
      resolution = int(layer.input_resolution[0])
      dim = int(layer.dim)
      heads = int(layer.blocks[0].num_heads)
      head_dim = dim // heads

      x = torch.randn(
          batch_size,
          resolution,
          resolution,
          dim,
          generator=generator,
      ).contiguous()
      input_path = out_dir / f"stage{stage_idx}_input_nhwc.f32"
      write_f32(input_path, x.numpy())

      shifts = []
      seen: set[int] = set()
      for block in layer.blocks:
          shift = int(block.shift_size)
          if shift in seen:
              continue
          seen.add(shift)
          qkv_weight = block.attn.qkv.weight.detach().t().contiguous().numpy()
          proj_weight = block.attn.proj.weight.detach().t().contiguous().numpy()
          qkv_weight_path = out_dir / f"stage{stage_idx}_shift{shift}_qkv_weight.f32"
          qkv_bias_path = out_dir / f"stage{stage_idx}_shift{shift}_qkv_bias.f32"
          proj_weight_path = out_dir / f"stage{stage_idx}_shift{shift}_output_weight.f32"
          proj_bias_path = out_dir / f"stage{stage_idx}_shift{shift}_output_bias.f32"
          rel_bias_path = out_dir / f"stage{stage_idx}_shift{shift}_rel_bias.f32"
          write_f32(qkv_weight_path, qkv_weight)
          write_f32(qkv_bias_path, block.attn.qkv.bias.detach().numpy())
          write_f32(proj_weight_path, proj_weight)
          write_f32(proj_bias_path, block.attn.proj.bias.detach().numpy())
          write_f32(rel_bias_path, relpos_bias(block.attn))
          shifts.append(
              {
                  "shift_size": shift,
                  "use_mask": bool(shift != 0),
                  "qkv_weight": qkv_weight_path.name,
                  "qkv_bias": qkv_bias_path.name,
                  "output_weight": proj_weight_path.name,
                  "output_bias": proj_bias_path.name,
                  "rel_bias": rel_bias_path.name,
              }
          )

      stages.append(
          {
              "stage": stage_idx,
              "image_size": resolution,
              "window_size": int(layer.blocks[0].window_size),
              "head_number": heads,
              "head_size": head_dim,
              "input": input_path.name,
              "shifts": shifts,
          }
      )

    manifest = {
        "source": OFFICIAL_URL,
        "checkpoint": "swin_tiny_patch4_window7_224.pth",
        "onnx": "swin_tiny_patch4_window7_224.onnx",
        "batch_size": batch_size,
        "seed": seed,
        "note": (
            "Artifacts are exported from the official Swin-Tiny checkpoint. "
            "The current C++ harness validates implemented subpaths: "
            "PatchEmbed and WindowAttention layout/projection/attention/reverse."
        ),
        "patch_embed": {
            "batch_size": batch_size,
            "image_size": 224,
            "in_channels": 3,
            "input_channels_padded": 8,
            "embed_dim": 96,
            "patch_size": 4,
            "input": "patch_embed_input_nhwc.f32",
            "kernel": "patch_embed_weight_oihw.f32",
            "bias": "patch_embed_bias.f32",
            "gamma": "patch_embed_norm_weight.f32",
            "beta": "patch_embed_norm_bias.f32",
        },
        "stages": stages,
        "checkpoint_tensors": len(checkpoint),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_artifacts_from_hf(
    model_id: str,
    out_dir: Path,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    from transformers import SwinForImageClassification

    model = SwinForImageClassification.from_pretrained(model_id)
    model.eval()
    state = model.state_dict()
    out_dir.mkdir(parents=True, exist_ok=True)

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x).logits

    onnx_path = out_dir / "swin_tiny_patch4_window7_224.onnx"
    if not onnx_path.exists():
        torch.onnx.export(
            LogitsOnly(model),
            torch.randn(batch_size, 3, 224, 224),
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
        )

    rng = np.random.default_rng(seed)
    image_nhwc = rng.standard_normal((batch_size, 224, 224, 3), dtype=np.float32)
    write_f32(out_dir / "input_image_nhwc.f32", image_nhwc)
    write_f32(out_dir / "patch_embed_input_nhwc.f32", image_nhwc)
    write_f32(
        out_dir / "patch_embed_weight_oihw.f32",
        state["swin.embeddings.patch_embeddings.projection.weight"].cpu().numpy(),
    )
    write_f32(
        out_dir / "patch_embed_bias.f32",
        state["swin.embeddings.patch_embeddings.projection.bias"].cpu().numpy(),
    )
    write_f32(
        out_dir / "patch_embed_norm_weight.f32",
        state["swin.embeddings.norm.weight"].cpu().numpy(),
    )
    write_f32(
        out_dir / "patch_embed_norm_bias.f32",
        state["swin.embeddings.norm.bias"].cpu().numpy(),
    )

    stage_cfg = [
        (0, 56, 96, 3, [0, 1]),
        (1, 28, 192, 6, [0, 1]),
        (2, 14, 384, 12, [0, 1]),
        (3, 7, 768, 24, [0]),
    ]
    stages: list[dict[str, object]] = []
    for stage_idx, resolution, dim, heads, blocks in stage_cfg:
        x = rng.standard_normal(
            (batch_size, resolution, resolution, dim), dtype=np.float32
        )
        input_path = out_dir / f"stage{stage_idx}_input_nhwc.f32"
        write_f32(input_path, x)
        shifts = []
        for block_idx in blocks:
            shift = 0 if block_idx == 0 or resolution <= 7 else 3
            prefix = f"swin.encoder.layers.{stage_idx}.blocks.{block_idx}.attention"
            self_prefix = f"{prefix}.self"
            q_weight = state[f"{self_prefix}.query.weight"]
            k_weight = state[f"{self_prefix}.key.weight"]
            v_weight = state[f"{self_prefix}.value.weight"]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0).t().contiguous()
            qkv_bias = torch.cat(
                [
                    state[f"{self_prefix}.query.bias"],
                    state[f"{self_prefix}.key.bias"],
                    state[f"{self_prefix}.value.bias"],
                ],
                dim=0,
            ).contiguous()
            proj_weight = state[f"{prefix}.output.dense.weight"].t().contiguous()
            proj_bias = state[f"{prefix}.output.dense.bias"].contiguous()
            rel_table = state[f"{self_prefix}.relative_position_bias_table"]
            rel_bias = relpos_bias_from_table(rel_table, heads)

            qkv_weight_path = out_dir / f"stage{stage_idx}_shift{shift}_qkv_weight.f32"
            qkv_bias_path = out_dir / f"stage{stage_idx}_shift{shift}_qkv_bias.f32"
            proj_weight_path = out_dir / f"stage{stage_idx}_shift{shift}_output_weight.f32"
            proj_bias_path = out_dir / f"stage{stage_idx}_shift{shift}_output_bias.f32"
            rel_bias_path = out_dir / f"stage{stage_idx}_shift{shift}_rel_bias.f32"
            write_f32(qkv_weight_path, qkv_weight.cpu().numpy())
            write_f32(qkv_bias_path, qkv_bias.cpu().numpy())
            write_f32(proj_weight_path, proj_weight.cpu().numpy())
            write_f32(proj_bias_path, proj_bias.cpu().numpy())
            write_f32(rel_bias_path, rel_bias)
            shifts.append(
                {
                    "shift_size": shift,
                    "use_mask": bool(shift != 0),
                    "qkv_weight": qkv_weight_path.name,
                    "qkv_bias": qkv_bias_path.name,
                    "output_weight": proj_weight_path.name,
                    "output_bias": proj_bias_path.name,
                    "rel_bias": rel_bias_path.name,
                }
            )
        stages.append(
            {
                "stage": stage_idx,
                "image_size": resolution,
                "window_size": 7,
                "head_number": heads,
                "head_size": dim // heads,
                "input": input_path.name,
                "shifts": shifts,
            }
        )

    manifest = {
        "source": f"Hugging Face official model: {model_id}",
        "source_kind": "official_hf",
        "checkpoint": "huggingface_hub",
        "onnx": "swin_tiny_patch4_window7_224.onnx",
        "batch_size": batch_size,
        "seed": seed,
        "note": (
            "Artifacts were exported from Microsoft Hugging Face Swin-Tiny "
            "weights and cover the implemented PatchEmbed and WindowAttention "
            "subpaths."
        ),
        "patch_embed": {
            "batch_size": batch_size,
            "image_size": 224,
            "in_channels": 3,
            "input_channels_padded": 8,
            "embed_dim": 96,
            "patch_size": 4,
            "input": "patch_embed_input_nhwc.f32",
            "kernel": "patch_embed_weight_oihw.f32",
            "bias": "patch_embed_bias.f32",
            "gamma": "patch_embed_norm_weight.f32",
            "beta": "patch_embed_norm_bias.f32",
        },
        "stages": stages,
        "checkpoint_tensors": len(state),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def onnx_initializers(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    tensors = {
        init.name: numpy_helper.to_array(init).astype(np.float32, copy=False)
        for init in model.graph.initializer
    }
    aliases: dict[str, str] = {}
    for node in model.graph.node:
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            aliases[node.output[0]] = node.input[0]
    return tensors, aliases


def resolve_value(
    name: str,
    tensors: dict[str, np.ndarray],
    aliases: dict[str, str],
) -> np.ndarray:
    seen: set[str] = set()
    cur = name
    while cur in aliases:
        if cur in seen:
            raise RuntimeError(f"cyclic ONNX alias at {name}")
        seen.add(cur)
        cur = aliases[cur]
    if cur not in tensors:
        raise KeyError(f"ONNX value is not an initializer: {name} -> {cur}")
    return tensors[cur]


def find_node_input(
    onnx_path: Path,
    op_type: str,
    output_name: str,
    input_index: int,
) -> str:
    import onnx

    model = onnx.load(str(onnx_path))
    for node in model.graph.node:
        if node.op_type == op_type and output_name in node.output:
            return node.input[input_index]
    raise KeyError(f"could not find {op_type} node producing {output_name}")


def find_add_bias_input(onnx_path: Path, output_name: str) -> str:
    import onnx

    model = onnx.load(str(onnx_path))
    for node in model.graph.node:
        if node.op_type == "Add" and output_name in node.output:
            # PyTorch exporter emits Add(bias, matmul) for Linear.
            return node.input[0]
    raise KeyError(f"could not find Add node producing {output_name}")


def export_artifacts_from_onnx(
    onnx_path: Path,
    out_dir: Path,
    batch_size: int,
    seed: int,
    source_note: str,
) -> dict[str, object]:
    tensors, aliases = onnx_initializers(onnx_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_if_needed(onnx_path, out_dir / "swin_tiny_patch4_window7_224.onnx")

    rng = np.random.default_rng(seed)
    image_nhwc = rng.standard_normal((batch_size, 224, 224, 3), dtype=np.float32)
    write_f32(out_dir / "input_image_nhwc.f32", image_nhwc)
    write_f32(out_dir / "patch_embed_input_nhwc.f32", image_nhwc)
    write_f32(out_dir / "patch_embed_weight_oihw.f32", tensors["patch_embed.proj.weight"])
    write_f32(out_dir / "patch_embed_bias.f32", tensors["patch_embed.proj.bias"])
    write_f32(out_dir / "patch_embed_norm_weight.f32", tensors["patch_embed.norm.weight"])
    write_f32(out_dir / "patch_embed_norm_bias.f32", tensors["patch_embed.norm.bias"])

    stage_cfg = [
        (0, 56, 96, 3, [0, 1]),
        (1, 28, 192, 6, [0, 1]),
        (2, 14, 384, 12, [0, 1]),
        (3, 7, 768, 24, [0]),
    ]
    stages: list[dict[str, object]] = []
    for stage_idx, resolution, dim, heads, blocks in stage_cfg:
        x = rng.standard_normal(
            (batch_size, resolution, resolution, dim), dtype=np.float32
        )
        input_path = out_dir / f"stage{stage_idx}_input_nhwc.f32"
        write_f32(input_path, x)
        shifts = []
        for block_idx in blocks:
            shift = 0 if block_idx == 0 or resolution <= 7 else 3
            prefix = f"/layers.{stage_idx}/blocks.{block_idx}/attn"
            module_prefix = f"layers.{stage_idx}.blocks.{block_idx}.attn"
            qkv_weight_name = find_node_input(
                onnx_path,
                "MatMul",
                f"{prefix}/qkv/MatMul_output_0",
                1,
            )
            proj_weight_name = find_node_input(
                onnx_path,
                "MatMul",
                f"{prefix}/proj/MatMul_output_0",
                1,
            )
            rel_bias_name = find_node_input(
                onnx_path,
                "Add",
                f"{prefix}/Add_output_0",
                1,
            )
            qkv_bias = resolve_value(
                f"{module_prefix}.qkv.bias", tensors, aliases
            )
            proj_bias = resolve_value(
                find_add_bias_input(onnx_path, f"{prefix}/proj/Add_output_0"),
                tensors,
                aliases,
            )
            rel_bias = resolve_value(rel_bias_name, tensors, aliases)
            if rel_bias.ndim == 4 and rel_bias.shape[0] == 1:
                rel_bias = rel_bias[0]

            qkv_weight_path = out_dir / f"stage{stage_idx}_shift{shift}_qkv_weight.f32"
            qkv_bias_path = out_dir / f"stage{stage_idx}_shift{shift}_qkv_bias.f32"
            proj_weight_path = out_dir / f"stage{stage_idx}_shift{shift}_output_weight.f32"
            proj_bias_path = out_dir / f"stage{stage_idx}_shift{shift}_output_bias.f32"
            rel_bias_path = out_dir / f"stage{stage_idx}_shift{shift}_rel_bias.f32"
            write_f32(qkv_weight_path, resolve_value(qkv_weight_name, tensors, aliases))
            write_f32(qkv_bias_path, qkv_bias)
            write_f32(proj_weight_path, resolve_value(proj_weight_name, tensors, aliases))
            write_f32(proj_bias_path, proj_bias)
            write_f32(rel_bias_path, rel_bias)
            shifts.append(
                {
                    "shift_size": shift,
                    "use_mask": bool(shift != 0),
                    "qkv_weight": qkv_weight_path.name,
                    "qkv_bias": qkv_bias_path.name,
                    "output_weight": proj_weight_path.name,
                    "output_bias": proj_bias_path.name,
                    "rel_bias": rel_bias_path.name,
                }
            )
        stages.append(
            {
                "stage": stage_idx,
                "image_size": resolution,
                "window_size": 7,
                "head_number": heads,
                "head_size": dim // heads,
                "input": input_path.name,
                "shifts": shifts,
            }
        )

    manifest = {
        "source": source_note,
        "source_kind": "cached_onnx_fallback",
        "checkpoint": None,
        "onnx": "swin_tiny_patch4_window7_224.onnx",
        "batch_size": batch_size,
        "seed": seed,
        "note": (
            "Artifacts were extracted from an ONNX model because the official "
            "checkpoint download was not reachable from this environment. "
            "This keeps the implemented PatchEmbed and WindowAttention subpath "
            "verification/profile flow runnable, but it is not proof that a "
            "fresh official checkpoint download succeeded."
        ),
        "patch_embed": {
            "batch_size": batch_size,
            "image_size": 224,
            "in_channels": 3,
            "input_channels_padded": 8,
            "embed_dim": 96,
            "patch_size": 4,
            "input": "patch_embed_input_nhwc.f32",
            "kernel": "patch_embed_weight_oihw.f32",
            "bias": "patch_embed_bias.f32",
            "gamma": "patch_embed_norm_weight.f32",
            "beta": "patch_embed_norm_bias.f32",
        },
        "stages": stages,
        "checkpoint_tensors": None,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "swin_tiny_patch4_window7_224.pth"
    onnx_path = args.checkpoint_dir / "swin_tiny_patch4_window7_224.onnx"
    manifest_path = args.checkpoint_dir / "manifest.json"

    if (
        not args.force
        and args.offline_onnx is None
        and manifest_path.exists()
        and onnx_path.exists()
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"reuse manifest={manifest_path}")
        print(f"onnx={onnx_path}")
        print(f"source_kind={manifest.get('source_kind', 'official_checkpoint')}")
        print(f"stages={len(manifest.get('stages', []))}")
        return

    if args.offline_onnx is not None:
        manifest = export_artifacts_from_onnx(
            args.offline_onnx,
            args.checkpoint_dir,
            args.batch_size,
            args.seed,
            f"offline ONNX: {args.offline_onnx}",
        )
        print(f"onnx={args.checkpoint_dir / 'swin_tiny_patch4_window7_224.onnx'}")
        print(f"manifest={args.checkpoint_dir / 'manifest.json'}")
        print(f"stages={len(manifest['stages'])}")
        return

    try:
        download(args.url, checkpoint_path)
    except Exception as exc:
        print(f"official checkpoint download failed: {exc}")
        try:
            print(f"trying Hugging Face official model: {args.hf_model_id}")
            manifest = export_artifacts_from_hf(
                args.hf_model_id,
                args.checkpoint_dir,
                args.batch_size,
                args.seed,
            )
            print(f"onnx={args.checkpoint_dir / 'swin_tiny_patch4_window7_224.onnx'}")
            print(f"manifest={args.checkpoint_dir / 'manifest.json'}")
            print(f"source_kind={manifest.get('source_kind')}")
            print(f"stages={len(manifest['stages'])}")
            return
        except Exception as hf_exc:
            print(f"Hugging Face official model failed: {hf_exc}")
        if args.fallback_onnx.exists():
            print(f"using cached ONNX fallback: {args.fallback_onnx}")
            manifest = export_artifacts_from_onnx(
                args.fallback_onnx,
                args.checkpoint_dir,
                args.batch_size,
                args.seed,
                f"cached ONNX fallback after failed download: {args.fallback_onnx}",
            )
            print(f"onnx={args.checkpoint_dir / 'swin_tiny_patch4_window7_224.onnx'}")
            print(f"manifest={args.checkpoint_dir / 'manifest.json'}")
            print(f"stages={len(manifest['stages'])}")
            return
        raise
    checkpoint = load_checkpoint(checkpoint_path)
    module = load_swin_module(args.model_source)
    model = module.SwinTransformer()
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint load mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    export_onnx(model, onnx_path, args.batch_size)
    manifest = export_artifacts(
        model,
        checkpoint,
        args.checkpoint_dir,
        args.batch_size,
        args.seed,
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"onnx={onnx_path}")
    print(f"manifest={args.checkpoint_dir / 'manifest.json'}")
    print(f"stages={len(manifest['stages'])}")


if __name__ == "__main__":
    main()
