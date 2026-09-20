#!/usr/bin/env python3
"""Where ComfyUI's two views of a reference image stop agreeing. No GPU, no torch.

`TextEncodeQwenImage21` resizes each reference once and hands the SAME size to
the text encoder and to the VAE, which is what 2.1 requires: a vision slot
stands for a fixed group of latent tokens, so the encoder's slot count and the
VAE's latent grid are two views of one number.

Then the encoder branch is resized a second time. `process_qwen2vl_images`
applies its own floor and ceiling, and those are transformers' Qwen2-VL library
defaults rather than the bounds this checkpoint declares in its processor
config. When a clamp fires, the encoder reads a different scale than the VAE
encodes -- and because the 2.1 path drops the vision tokens from the
conditioning, there is no count left to check, so nothing raises.

This prints the cases where the two disagree. Both functions below are
transcribed from the source named above them; re-read them if either moves.

Usage:  python scripts/refview_bounds.py [--all]
"""

from __future__ import annotations

import argparse
import math

# comfy/text_encoders/qwen_vl.py::process_qwen2vl_images defaults, which
# comfy/text_encoders/qwen3vl.py does not override. Equal to transformers'
# Qwen2VLImageProcessor defaults (56*56 and 28*28*1280).
COMFY_MIN_PIXELS = 3136
COMFY_MAX_PIXELS = 12845056
# <models>/Qwen-Image-2.1/processor/preprocessor_config.json, which declares
# image_processor_type Qwen2VLImageProcessorFast and its own size dict.
# transformers reads size.shortest_edge as min_pixels and longest_edge as max.
CKPT_MIN_PIXELS = 65536
CKPT_MAX_PIXELS = 16777216
FACTOR = 32  # patch_size 16 * merge_size 2


def node_size(resolution: int, src_w: int, src_h: int) -> tuple[int, int]:
    """comfy_extras/nodes_qwen.py::TextEncodeQwenImage21.execute."""
    if resolution > 0:
        ratio = src_w / src_h
        w = round(math.sqrt(resolution * resolution * ratio) / FACTOR) * FACTOR
        h = round(math.sqrt(resolution * resolution / ratio) / FACTOR) * FACTOR
    else:
        w, h = round(src_w / FACTOR) * FACTOR, round(src_h / FACTOR) * FACTOR
    return max(FACTOR, w), max(FACTOR, h)


def clamp(w: int, h: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """comfy/text_encoders/qwen_vl.py::process_qwen2vl_images, sizing only."""
    h_bar, w_bar = round(h / FACTOR) * FACTOR, round(w / FACTOR) * FACTOR
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(FACTOR, math.floor(h / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(w / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / FACTOR) * FACTOR
        w_bar = math.ceil(w * beta / FACTOR) * FACTOR
    return w_bar, h_bar


CASES = [
    ("node default", 1024, 1024, 1024),
    ("node default, 16:9", 1024, 1920, 1080),
    ("widget maximum", 4096, 1024, 1024),
    ("widget minimum step", 32, 1024, 1024),
    ("passthrough, large source", 0, 6000, 4000),
    ("passthrough, small source", 0, 40, 40),
    ("passthrough, ordinary source", 0, 1024, 1024),
    ("passthrough, source under the checkpoint floor", 0, 200, 200),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="print agreeing cases too")
    args = ap.parse_args()

    print(f"{'case':<48}{'res':>6}{'node':>12}{'encoder':>12}{'vs checkpoint':>16}")
    split = 0
    for name, res, sw, sh in CASES:
        w, h = node_size(res, sw, sh)
        ew, eh = clamp(w, h, COMFY_MIN_PIXELS, COMFY_MAX_PIXELS)
        cw, ch = clamp(w, h, CKPT_MIN_PIXELS, CKPT_MAX_PIXELS)
        internal = (ew, eh) != (w, h)
        external = (ew, eh) != (cw, ch)
        if not (internal or external) and not args.all:
            continue
        split += internal or external
        note = "same" if not external else f"{cw}x{ch}"
        flag = "  SPLIT" if internal else ""
        print(f"{name:<48}{res:>6}{f'{w}x{h}':>12}{f'{ew}x{eh}':>12}{note:>16}{flag}")

    print()
    print("SPLIT = the encoder read a different scale than the VAE encoded; nothing raises.")
    print("'vs checkpoint' = what the encoder branch would be under the bounds the")
    print("checkpoint's own processor config declares, where that differs from ComfyUI's.")
    print(f"\n{split} of {len(CASES)} cases differ. The node default is not one of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
