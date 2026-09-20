"""Image token accounting for the Qwen-Image 2.1 prompt expanders.

An image is one `<|image_pad|>` in the rendered text and hundreds to thousands
of positions in `input_ids`. That expansion is where token accounting goes
wrong, in two directions we have now seen both of:

* A server under-reported it. heylook stamped `prompt_tokens` from a one-token
  continuation seed, so every image-bearing request reported 1 (fixed upstream
  2026-09-20; the fix was verified against exactly the arithmetic below).
* Tooling can miss it entirely. The predecessor repo's two token debuggers
  counted literal `<|image_pad|>` substrings, so their "vision token count" was
  1 per image regardless of resolution -- the quantity that actually varies was
  never measured.

With up to 16 reference images accepted by `TextEncodeQwenImage21`, sequence
length is the most likely surprise in this pipeline, so it gets a function
rather than an assumption.

Constants come from the checkpoints' `processor_config.json`: `patch_size` 16,
`merge_size` 2. Verified live: a 448px square contributes exactly 196 positions
and an 896px square exactly 784.
"""

from __future__ import annotations

import math

PATCH_SIZE = 16
MERGE_SIZE = 2
#: Side length must land on a multiple of this for the merge to divide evenly.
ALIGNMENT = PATCH_SIZE * MERGE_SIZE  # 32


def align(px: int, alignment: int = ALIGNMENT) -> int:
    """Round a side length up to the processor's grid, minimum one cell."""
    return max(alignment, int(math.ceil(px / alignment) * alignment))


def image_tokens(width: int, height: int, *, aligned: bool = False) -> int:
    """Positions one still image contributes to `input_ids`.

    `(W/patch * H/patch) / merge**2`. Set `aligned=True` if the dimensions are
    already on the 32px grid and should not be rounded again.
    """
    w = width if aligned else align(width)
    h = height if aligned else align(height)
    return (w // PATCH_SIZE) * (h // PATCH_SIZE) // (MERGE_SIZE ** 2)


def fit_to_pixel_budget(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Scale down to `max_pixels`, preserving aspect, landing on the grid.

    Mirrors the cap the reference implementation applies to source images
    (`pe_core.Profile.image_max_pixels`).
    """
    if width * height <= max_pixels:
        return align(width), align(height)
    scale = math.sqrt(max_pixels / (width * height))
    return align(int(width * scale)), align(int(height * scale))


def estimate_input_tokens(
    *, system_tokens: int, brief_tokens: int, images: list[tuple[int, int]] | None = None,
    wrapper_tokens: int = 18,
) -> dict:
    """Predict an image-bearing request's input token count, by part.

    Useful because a server may not report it, and because knowing which part
    dominates is what tells you whether a token cap is about to truncate. The
    `wrapper_tokens` default covers the chat-template scaffolding (role markers,
    the vision block delimiters, the generation prompt) and is approximate --
    the exact figure is whatever the tokenizer produces for the rendered string.
    """
    per_image = [image_tokens(w, h) for w, h in (images or [])]
    total = system_tokens + brief_tokens + sum(per_image) + wrapper_tokens
    return {
        "system": system_tokens,
        "brief": brief_tokens,
        "images": per_image,
        "images_total": sum(per_image),
        "wrapper": wrapper_tokens,
        "total": total,
    }
