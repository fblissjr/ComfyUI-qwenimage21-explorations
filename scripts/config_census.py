#!/usr/bin/env python3
"""Config census of a quantized safetensors checkpoint. No GPU, no ComfyUI, no torch.

Prints "N layers, K configs, and which module roles hold each", with in/out
features beside the role column.

Why the shape of the output is what it is
-----------------------------------------
Three tidy cross-model stories were built and dissolved while this was being
worked out, and every one died the same way: a K greater than one appeared, the
interesting explanation was reached for, and the boring one -- shape happens to
identify that module uniquely -- was checkable from the header already open. So
the feature columns are not optional decoration. They hand the reader the
confound instead of waiting to be asked.

Do NOT audit `full_precision_matrix_mult` by payload length. Length classes do
not partition by content: `convrot_groupsize: 64` and `: 16` are both the same
byte length, and a shorter format name buys back exactly the bytes the flag
costs, so a flagged nvfp4 layer can hide inside a class of unflagged int8 ones.
This reads every blob. They are tens of bytes; the whole scan is trivial.

A layer marked `full_precision_matrix_mult` dequantizes and runs the matmul in
full precision -- ComfyUI's `can_use_quantized_matmul` requires the flag to be
absent -- so a checkpoint carrying it is storage-only however good its format.

Usage:  python scripts/config_census.py <file.safetensors> [...]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import sys
from pathlib import Path

def read_header(path: Path) -> tuple[dict, dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {}) or {}
    return header, meta, 8 + n


def normalise_role(key: str) -> str:
    return re.sub(r"(^|\.)(layers|blocks)\.\d+\.", r"\1\2.N.", key)


def configs_from_file(path: Path) -> tuple[dict[str, dict], dict]:
    """Return {module_path: config} plus the raw header. Covers both mechanisms."""
    header, meta, data0 = read_header(path)
    out: dict[str, dict] = {}

    if "_quantization_metadata" in meta:
        # Newer files omit per-layer tensors and carry one metadata blob.
        try:
            layers = json.loads(meta["_quantization_metadata"]).get("layers", {})
            out.update(layers)
        except (json.JSONDecodeError, AttributeError):
            pass

    cq = {k: v["data_offsets"] for k, v in header.items() if k.endswith(".comfy_quant")}
    if cq:
        lo = min(o[0] for o in cq.values())
        hi = max(o[1] for o in cq.values())
        with path.open("rb") as f:
            f.seek(data0 + lo)
            raw = f.read(hi - lo)
        for key, (s, e) in cq.items():
            try:
                out[key[: -len(".comfy_quant")]] = json.loads(raw[s - lo : e - lo].decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    return out, header


def census(path: Path) -> int:
    configs, header = configs_from_file(path)
    print(f"\n=== {path.name} ===")
    if not configs:
        print("  no quantization metadata (neither .comfy_quant tensors nor _quantization_metadata)")
        return 0

    by_config: dict[str, list[str]] = collections.defaultdict(list)
    for module, conf in configs.items():
        by_config[json.dumps(conf, sort_keys=True)].append(module)

    flagged = [m for m, c in configs.items() if c.get("full_precision_matrix_mult")]
    print(f"  {len(configs)} quantized layers, {len(by_config)} configs")

    for blob in sorted(by_config, key=lambda b: -len(by_config[b])):
        modules = by_config[blob]
        print(f"\n  [{len(modules)} layers] {blob}")
        roles: dict[str, list[str]] = collections.defaultdict(list)
        for m in modules:
            roles[normalise_role(m)].append(m)
        for role in sorted(roles):
            sample = roles[role][0]
            w = header.get(f"{sample}.weight", {})
            shape = w.get("shape") or []
            feat = ""
            if len(shape) == 2:
                # A packed int4 weight stores K/2 columns; the note keeps that visible.
                feat = f"  out={shape[0]:<7d} in={shape[1]:<7d} {w.get('dtype','')}"
            print(f"      x{len(roles[role]):<4d} {role:<62s}{feat}")

    print()
    if flagged:
        print(f"  !! full_precision_matrix_mult on {len(flagged)} layer(s) -- STORAGE ONLY, kernels will not fire")
        for m in flagged[:8]:
            print(f"       {m}")
        if len(flagged) > 8:
            print(f"       ... and {len(flagged) - 8} more")
        return 1
    print("  full_precision_matrix_mult: absent on all layers (full scan) -- kernels can fire")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    rc = 0
    for path in args.files:
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            rc = 2
            continue
        rc |= census(path)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
