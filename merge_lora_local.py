#!/usr/bin/env python3
"""
Stream-merge PEFT LoRA wrappers in runpod-pipeline-final/ → runpod-pipeline-merged/.

The DPO save left PEFT wrapping intact:
  - Standard module-LoRA on q/k/v/o (rank 16)
  - Parameter-LoRA on MoE experts.gate_up_proj (inner) and experts.down_proj (outer)

Merge formula: W_new = W_base + scaling * (B @ A), with scaling = alpha/r = 32/16 = 2.0.
For MoE param-LoRA, B@A produces a per-expert delta broadcast across all 128 experts.

Output renames:
  *.base_layer.weight              -> *.weight                  (standard linear)
  *.experts.base_layer.base_layer.gate_up_proj -> *.experts.gate_up_proj
  *.experts.base_layer.base_layer.down_proj    -> *.experts.down_proj
  All lora_A/lora_B tensors are dropped.
"""

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("runpod-pipeline-final")
DST = Path("runpod-pipeline-merged")

LORA_ALPHA = 32
LORA_R = 16
SCALING = LORA_ALPHA / LORA_R  # 2.0


def classify_tensors(weight_map: dict[str, str]):
    """
    Walk all input tensor names; produce a list of merge groups + passthrough names.

    A merge group is one of:
      ("std",   out_name, base_name,                lora_A_name, lora_B_name)
      ("param", out_name, base_name,                lora_A_name, lora_B_name)
        — base_name has shape [E, ...]; we broadcast B@A across E experts.

    Passthrough: tensors with no LoRA wrapping that we copy as-is (with rename if needed).
    """
    keys = set(weight_map.keys())
    groups = []
    consumed: set[str] = set()

    # Standard module-LoRA: <prefix>.base_layer.weight + <prefix>.lora_A.default.weight + <prefix>.lora_B.default.weight
    # Match attention projections (q/k/v/o). Avoid matching the doubly-nested MoE experts case.
    for k in sorted(keys):
        if not k.endswith(".base_layer.weight"):
            continue
        prefix = k[: -len(".base_layer.weight")]
        a = f"{prefix}.lora_A.default.weight"
        b = f"{prefix}.lora_B.default.weight"
        if a in keys and b in keys:
            out = f"{prefix}.weight"
            groups.append(("std", out, k, a, b))
            consumed.update([k, a, b])

    # Parameter-LoRA on MoE experts (inner = gate_up_proj, outer = down_proj).
    # Inner base:  <prefix>.experts.base_layer.base_layer.gate_up_proj
    # Inner LoRA:  <prefix>.experts.base_layer.lora_A.default.weight + .lora_B.default.weight
    # Outer base:  <prefix>.experts.base_layer.base_layer.down_proj
    # Outer LoRA:  <prefix>.experts.lora_A.default.weight + .lora_B.default.weight
    for k in sorted(keys):
        if k.endswith(".experts.base_layer.base_layer.gate_up_proj"):
            prefix = k[: -len(".experts.base_layer.base_layer.gate_up_proj")]
            a = f"{prefix}.experts.base_layer.lora_A.default.weight"
            b = f"{prefix}.experts.base_layer.lora_B.default.weight"
            out = f"{prefix}.experts.gate_up_proj"
            if a in keys and b in keys:
                groups.append(("param", out, k, a, b))
                consumed.update([k, a, b])
        elif k.endswith(".experts.base_layer.base_layer.down_proj"):
            prefix = k[: -len(".experts.base_layer.base_layer.down_proj")]
            a = f"{prefix}.experts.lora_A.default.weight"
            b = f"{prefix}.experts.lora_B.default.weight"
            out = f"{prefix}.experts.down_proj"
            if a in keys and b in keys:
                groups.append(("param", out, k, a, b))
                consumed.update([k, a, b])

    passthrough = sorted(keys - consumed)
    return groups, passthrough


def open_for(name: str, weight_map: dict[str, str]):
    return SRC / weight_map[name]


def load_tensor(name: str, weight_map: dict[str, str]) -> torch.Tensor:
    path = open_for(name, weight_map)
    with safe_open(str(path), framework="pt") as f:
        return f.get_tensor(name)


def merge_std(base: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Standard LoRA: W' = W + scaling * (B @ A). All ops in F32, output BF16."""
    base32 = base.to(torch.float32)
    delta = (B.to(torch.float32) @ A.to(torch.float32)) * SCALING
    return (base32 + delta).to(torch.bfloat16)


def merge_param(base: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Parameter-LoRA on a 3D MoE expert tensor [E, out, in].
    B@A produces [out, in] which broadcasts across E experts.
    """
    base32 = base.to(torch.float32)  # [E, out, in]
    delta = (B.to(torch.float32) @ A.to(torch.float32)) * SCALING  # [out, in]
    # Broadcast across leading expert dim
    return (base32 + delta.unsqueeze(0)).to(torch.bfloat16)


def assign_shards(out_names: list[str], shard_max_bytes: int, sizes: dict[str, int]) -> list[list[str]]:
    """Greedy bin-pack output tensors into shards by byte size."""
    shards: list[list[str]] = [[]]
    cur = 0
    for name in out_names:
        sz = sizes[name]
        if cur + sz > shard_max_bytes and shards[-1]:
            shards.append([])
            cur = 0
        shards[-1].append(name)
        cur += sz
    return shards


def build_output_plan(weight_map: dict[str, str]):
    """Return (groups, passthrough, planned_output) — planned_output is dict[out_name, ('group'|'pass', payload)]."""
    groups, passthrough = classify_tensors(weight_map)
    plan: dict[str, tuple] = {}
    for g in groups:
        kind, out, base, a, b = g
        plan[out] = (kind, base, a, b)
    for p in passthrough:
        plan[p] = ("pass", p)
    return groups, passthrough, plan


def compute_output_sizes(plan: dict, weight_map: dict[str, str]) -> dict[str, int]:
    """For sharding: estimate bytes per output tensor (BF16 = 2 bytes/elem)."""
    sizes = {}
    for out, payload in plan.items():
        kind = payload[0]
        if kind in ("std", "param"):
            base_name = payload[1]
            with safe_open(str(open_for(base_name, weight_map)), framework="pt") as f:
                meta = f.get_slice(base_name).get_shape()
            n = 1
            for d in meta:
                n *= d
            sizes[out] = n * 2  # output is BF16
        else:  # pass
            with safe_open(str(open_for(out, weight_map)), framework="pt") as f:
                t = f.get_slice(out)
                shape = t.get_shape()
                dtype_str = t.get_dtype()
            n = 1
            for d in shape:
                n *= d
            bpe = {"BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1}.get(dtype_str, 4)
            sizes[out] = n * bpe
    return sizes


def sanity_check_layer0(weight_map: dict[str, str]):
    """Verify layer-0 merge produces sensible shapes and magnitudes."""
    print("=== SANITY CHECK: layer 0 ===")
    # Standard q_proj
    base = load_tensor("model.layers.0.self_attn.q_proj.base_layer.weight", weight_map)
    A = load_tensor("model.layers.0.self_attn.q_proj.lora_A.default.weight", weight_map)
    B = load_tensor("model.layers.0.self_attn.q_proj.lora_B.default.weight", weight_map)
    merged = merge_std(base, A, B)
    diff = (merged.float() - base.float()).abs()
    print(f"q_proj merged shape={tuple(merged.shape)} dtype={merged.dtype}")
    print(f"  base abs mean = {base.float().abs().mean().item():.4e}")
    print(f"  merged abs mean = {merged.float().abs().mean().item():.4e}")
    print(f"  |merged - base| mean = {diff.mean().item():.4e}, max = {diff.max().item():.4e}")
    assert merged.shape == base.shape, f"q_proj shape mismatch: {merged.shape} vs {base.shape}"
    assert merged.dtype == torch.bfloat16

    # Param-LoRA gate_up_proj (inner)
    base = load_tensor("model.layers.0.mlp.experts.base_layer.base_layer.gate_up_proj", weight_map)
    A = load_tensor("model.layers.0.mlp.experts.base_layer.lora_A.default.weight", weight_map)
    B = load_tensor("model.layers.0.mlp.experts.base_layer.lora_B.default.weight", weight_map)
    merged = merge_param(base, A, B)
    diff = (merged.float() - base.float()).abs()
    print(f"gate_up_proj merged shape={tuple(merged.shape)} dtype={merged.dtype}")
    print(f"  base abs mean = {base.float().abs().mean().item():.4e}")
    print(f"  merged abs mean = {merged.float().abs().mean().item():.4e}")
    print(f"  |merged - base| mean = {diff.mean().item():.4e}, max = {diff.max().item():.4e}")
    assert merged.shape == base.shape, f"gate_up_proj shape mismatch: {merged.shape} vs {base.shape}"
    assert merged.dtype == torch.bfloat16

    # Param-LoRA down_proj (outer)
    base = load_tensor("model.layers.0.mlp.experts.base_layer.base_layer.down_proj", weight_map)
    A = load_tensor("model.layers.0.mlp.experts.lora_A.default.weight", weight_map)
    B = load_tensor("model.layers.0.mlp.experts.lora_B.default.weight", weight_map)
    merged = merge_param(base, A, B)
    diff = (merged.float() - base.float()).abs()
    print(f"down_proj merged shape={tuple(merged.shape)} dtype={merged.dtype}")
    print(f"  base abs mean = {base.float().abs().mean().item():.4e}")
    print(f"  merged abs mean = {merged.float().abs().mean().item():.4e}")
    print(f"  |merged - base| mean = {diff.mean().item():.4e}, max = {diff.max().item():.4e}")
    assert merged.shape == base.shape, f"down_proj shape mismatch: {merged.shape} vs {base.shape}"
    assert merged.dtype == torch.bfloat16

    print("=== SANITY CHECK PASSED ===\n")


def materialize(out_name: str, payload: tuple, weight_map: dict[str, str]) -> torch.Tensor:
    kind = payload[0]
    if kind == "std":
        _, base_name, a_name, b_name = payload
        return merge_std(
            load_tensor(base_name, weight_map),
            load_tensor(a_name, weight_map),
            load_tensor(b_name, weight_map),
        )
    if kind == "param":
        _, base_name, a_name, b_name = payload
        return merge_param(
            load_tensor(base_name, weight_map),
            load_tensor(a_name, weight_map),
            load_tensor(b_name, weight_map),
        )
    if kind == "pass":
        (_, in_name) = payload
        return load_tensor(in_name, weight_map)
    raise ValueError(f"Unknown kind: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-only", action="store_true", help="Run sanity check on layer 0 then exit")
    ap.add_argument("--shard-bytes", type=int, default=20_000_000_000, help="Max bytes per output shard (default 20 GB)")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"Source dir missing: {SRC}", file=sys.stderr)
        sys.exit(1)

    with (SRC / "model.safetensors.index.json").open() as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    total_size = idx.get("metadata", {}).get("total_size")
    print(f"Input: {len(weight_map)} tensors, declared total_size={total_size}")

    sanity_check_layer0(weight_map)
    if args.sanity_only:
        return

    DST.mkdir(parents=True, exist_ok=True)

    groups, passthrough, plan = build_output_plan(weight_map)
    print(f"Plan: {len(groups)} merged tensors, {len(passthrough)} passthrough, {len(plan)} total outputs")

    # Sort outputs in a stable, layer-aware order so each shard contains contiguous content
    def sort_key(name: str):
        parts = name.split(".")
        layer = -1
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer = int(parts[i + 1])
                except ValueError:
                    pass
                break
        # embed/lm_head/norm at the head, then layers in order, then anything else
        if name == "model.embed_tokens.weight":
            head = 0
        elif name == "lm_head.weight":
            head = 9_999_999  # at end
        elif name == "model.norm.weight":
            head = 9_999_998  # near end
        else:
            head = 1 + layer * 1000  # by layer
        return (head, name)

    out_names = sorted(plan.keys(), key=sort_key)
    sizes = compute_output_sizes(plan, weight_map)
    total_out = sum(sizes.values())
    print(f"Estimated output total: {total_out / 1e9:.1f} GB")

    shards = assign_shards(out_names, args.shard_bytes, sizes)
    n_shards = len(shards)
    print(f"Will write {n_shards} shards (max {args.shard_bytes / 1e9:.0f} GB each)")

    new_weight_map: dict[str, str] = {}
    new_total_size = 0

    for i, shard in enumerate(shards, start=1):
        shard_name = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_path = DST / shard_name
        print(f"\n--- Shard {i}/{n_shards}: {shard_name} ({len(shard)} tensors) ---")
        shard_dict: dict[str, torch.Tensor] = {}
        for name in shard:
            payload = plan[name]
            t = materialize(name, payload, weight_map)
            shard_dict[name] = t.contiguous()
            new_weight_map[name] = shard_name
            new_total_size += t.numel() * t.element_size()
            kind = payload[0]
            print(f"  [{kind}] {name}  shape={tuple(t.shape)}  dtype={t.dtype}")
        save_file(shard_dict, str(shard_path), metadata={"format": "pt"})
        del shard_dict
        gc.collect()
        sz = shard_path.stat().st_size
        print(f"  wrote {sz / 1e9:.2f} GB")

    new_index = {
        "metadata": {"total_size": new_total_size},
        "weight_map": new_weight_map,
    }
    with (DST / "model.safetensors.index.json").open("w") as f:
        json.dump(new_index, f, indent=2)
    print(f"\nWrote new index with {len(new_weight_map)} tensors, total_size={new_total_size}")

    # Copy passthrough non-tensor files
    for name in ["chat_template.jinja", "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"]:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, DST / name)
            print(f"copied {name}")

    print(f"\nDone. Merged model written to {DST}/")


if __name__ == "__main__":
    main()
