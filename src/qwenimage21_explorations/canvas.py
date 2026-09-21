"""The canvas an expander answer asks for, at the area the graph already uses.

Both expanders pick the output shape: `wh_ratio` (e.g. "16:9") when the answer
is a new composition, and on edit `ratio_follow` (e.g. "<image2>") when the
output inherits that reference's frame. The model README renders `wh_ratio`
from its fixed WH_RATIO_TO_SIZE table (2048x2048 for 1:1), while diffusers,
sglang, DiffSynth-Studio and vllm-omni all default to 1024x1024 and the official
ComfyUI graphs to a 1-megapixel ResolutionSelector. So the owner chose the ratio
at the graph's own area -- an expanded run costs what an unexpanded one does.

Core's `ResolutionSelector` does ratio-at-area, but takes the ratio as a combo of
fixed labels ("16:9 (Widescreen)"), which a free-form answer cannot feed.

Pure arithmetic: importable without ComfyUI.
"""

from __future__ import annotations

import math
import re

RATIO_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")
FOLLOW_RE = re.compile(r"^\s*<image(\d+)>\s*$")
#: Sides snap to this, as core's encode node and the README's table both do.
MULTIPLE = 32


def _snap(x: float) -> int:
    return max(MULTIPLE, round(x / MULTIPLE) * MULTIPLE)


def encoded_size(width: int, height: int, resolution: int) -> tuple[int, int]:
    """A reference's size after `TextEncodeQwenImage21` resizes it.

    Mirrors `comfy_extras/nodes_qwen.py`: about resolution**2 pixels at the
    reference's aspect, or its own size when resolution is 0, sides to 32.
    """
    if resolution > 0:
        ratio = width / height
        return _snap(math.sqrt(resolution * resolution * ratio)), _snap(math.sqrt(resolution * resolution / ratio))
    return _snap(width), _snap(height)


def choose(wh_ratio: str, ratio_follow: str, width: int, height: int,
           refs: list[tuple[int, int]], resolution: int) -> tuple[int, int]:
    """(width, height) for the latent.

    Without references the base is `width` x `height`; with them it is the
    first reference as the encoder sized it, which is core's own edit latent.
    `ratio_follow` naming a reference wins, then `wh_ratio` at the base's area,
    then the base unchanged -- so an answer with neither changes nothing.
    """
    sizes = [encoded_size(w, h, resolution) for w, h in refs]
    base = sizes[0] if sizes else (width, height)
    m = FOLLOW_RE.match(ratio_follow or "")
    if m and 1 <= int(m.group(1)) <= len(sizes):
        return sizes[int(m.group(1)) - 1]
    m = RATIO_RE.match(wh_ratio or "")
    if m and int(m.group(1)) > 0 and int(m.group(2)) > 0:
        a, b = int(m.group(1)), int(m.group(2))
        area = base[0] * base[1]
        return _snap(math.sqrt(area * a / b)), _snap(math.sqrt(area * b / a))
    return base
