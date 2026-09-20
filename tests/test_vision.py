"""Image-pad expansion is where token accounting goes wrong in both directions:
a server under-reported it, and the predecessor repo's debuggers never measured
it at all. These pin the arithmetic against values verified live."""

import pytest

from qwenimage21_explorations.vision import (
    ALIGNMENT, align, estimate_input_tokens, fit_to_pixel_budget, image_tokens,
)


@pytest.mark.parametrize("side,expected", [(448, 196), (896, 784)])
def test_square_expansion_matches_live_measurement(side, expected):
    # (side/16)**2 / 4 -- confirmed against real request deltas on the server
    assert image_tokens(side, side) == expected


def test_expansion_scales_with_area_not_side():
    assert image_tokens(896, 896) == 4 * image_tokens(448, 448)


def test_dimensions_round_up_to_the_merge_grid():
    assert align(1) == ALIGNMENT
    assert align(33) == 64
    assert align(64) == 64


def test_aligned_flag_skips_rounding():
    assert image_tokens(448, 448, aligned=True) == image_tokens(448, 448)


def test_pixel_budget_preserves_aspect_and_grid():
    w, h = fit_to_pixel_budget(4000, 2000, 1024 * 1024)
    assert w % ALIGNMENT == 0 and h % ALIGNMENT == 0
    assert w * h <= 1024 * 1024 * 1.1  # grid rounding can nudge it slightly
    assert 1.8 < w / h < 2.2


def test_small_image_is_only_aligned_not_scaled():
    assert fit_to_pixel_budget(100, 100, 1024 * 1024) == (align(100), align(100))


def test_one_large_image_can_outweigh_the_system_prompt():
    # I21's system prompt is ~4510 tokens. A single 1MP reference is comparable,
    # which is why a token cap has to account for images, not just the prompt.
    est = estimate_input_tokens(system_tokens=4510, brief_tokens=15,
                                images=[fit_to_pixel_budget(1024, 1024, 1024 * 1024)])
    assert est["images_total"] > 1000
    assert est["total"] == sum((est["system"], est["brief"], est["images_total"], est["wrapper"]))
