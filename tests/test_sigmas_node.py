"""The Sigmas node reads the grid the sampler samples, not the one it is handed.

EmptyLatentImage emits 4 channels at 1/8 and tags the latent with that ratio;
the sampler rescales an empty latent to 2.1's 64 channels at 1/16 before it
samples (comfy/sample.py::fix_empty_latent_channels). Read raw, a 1024x1024
canvas counted four times its tokens and every t2i graph sampled on a shift
meant for a canvas four times the area. Found by a same-seed control.
"""

import torch


def sigmas(nodes, latent, steps=25):
    return nodes.Sigmas.execute(latent, steps).args[0]


def test_an_empty_generic_latent_gets_the_schedule_of_the_grid_it_becomes(nodes):
    native = {"samples": torch.zeros(1, 64, 64, 64)}
    generic = {"samples": torch.zeros(1, 4, 128, 128), "downscale_ratio_spacial": 8}
    assert torch.equal(sigmas(nodes, native), sigmas(nodes, generic))


def test_a_non_square_canvas_converts_each_side(nodes):
    native = {"samples": torch.zeros(1, 64, 52, 78)}          # 1248x832
    generic = {"samples": torch.zeros(1, 4, 104, 156), "downscale_ratio_spacial": 8}
    assert torch.equal(sigmas(nodes, native), sigmas(nodes, generic))


def test_a_filled_latent_is_read_as_given(nodes):
    """The sampler rescales only an empty latent, so a filled one keeps its grid."""
    filled = {"samples": torch.ones(1, 4, 128, 128), "downscale_ratio_spacial": 8}
    as_given = {"samples": torch.ones(1, 64, 128, 128)}
    assert torch.equal(sigmas(nodes, filled), sigmas(nodes, as_given))
