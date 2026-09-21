"""The release's sigma schedule.

Core gets two things wrong here and both are silent: a constant shift where the
config asks for a dynamic one, and no terminal stretch at all. These pin the
behaviour that replaces it, including the ordering mistake that would make the
stretch a no-op while still looking implemented.
"""

import math

import pytest

from qwenimage21_explorations import sigmas as S


def tokens(w, h):
    return S.latent_tokens(w, h)


def test_latent_cell_is_one_token():
    # VAE 16x, consumed unpatched, so a 1024 canvas is a 64x64 grid.
    assert tokens(1024, 1024) == 64 * 64


def test_mu_matches_the_affine_form_the_config_describes():
    slope = (S.MAX_SHIFT - S.BASE_SHIFT) / (S.MAX_SEQ_LEN - S.BASE_SEQ_LEN)
    for seq in (256, 1024, 4096, 8192, 16384):
        expected = seq * slope + S.BASE_SHIFT - slope * S.BASE_SEQ_LEN
        assert S.dynamic_mu(seq) == pytest.approx(expected)


def test_mu_hits_the_declared_endpoints():
    assert S.dynamic_mu(S.BASE_SEQ_LEN) == pytest.approx(S.BASE_SHIFT)
    assert S.dynamic_mu(S.MAX_SEQ_LEN) == pytest.approx(S.MAX_SHIFT)


def test_mu_moves_with_the_canvas():
    """Core's constant is the thing this replaces, so it had better not be constant."""
    small, large = S.dynamic_mu(tokens(512, 512)), S.dynamic_mu(tokens(2048, 2048))
    assert small < S.dynamic_mu(tokens(1024, 1024)) < large


@pytest.mark.parametrize("steps", [1, 4, 25, 40])
def test_schedule_length_is_steps_plus_the_trailing_zero(steps):
    s = S.schedule(steps, tokens(1024, 1024))
    assert len(s) == steps + 1
    assert s[-1] == 0.0


def test_one_step_has_no_span_to_stretch():
    """Degenerate, and the reference divides by zero here. Leave the curve alone."""
    assert S.schedule(1, tokens(1024, 1024)) == [pytest.approx(1.0), 0.0]


@pytest.mark.parametrize("steps", [4, 25, 40])
def test_terminal_lands_on_the_configured_value(steps):
    """The ordering test. Stretching AFTER appending zero leaves this at core's tail."""
    s = S.schedule(steps, tokens(1024, 1024))
    assert s[-2] == pytest.approx(S.SHIFT_TERMINAL)


def test_disabling_the_stretch_reproduces_the_unstretched_curve():
    seq = tokens(1024, 1024)
    plain = S.schedule(25, seq, shift_terminal=None)
    mu = math.exp(S.dynamic_mu(seq))
    step = (1.0 - 1.0 / 25) / 24
    expected = [mu / (mu + (1.0 / (1.0 - i * step) - 1.0)) for i in range(25)]
    assert plain[:-1] == pytest.approx(expected)
    assert plain[-2] != pytest.approx(S.SHIFT_TERMINAL)


def test_schedule_starts_at_one_and_descends():
    s = S.schedule(25, tokens(1024, 1024))
    assert s[0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(s, s[1:]))


def test_a_larger_canvas_holds_more_noise_at_every_step():
    """What the dynamic shift is for: more target tokens, more shift."""
    small = S.schedule(25, tokens(768, 768), shift_terminal=None)
    large = S.schedule(25, tokens(1536, 1536), shift_terminal=None)
    assert all(b > a for a, b in zip(small[1:-1], large[1:-1]))


def test_denoise_keeps_the_tail_of_a_longer_schedule():
    full = S.schedule(50, tokens(1024, 1024))
    half = S.schedule(25, tokens(1024, 1024), denoise=0.5)
    assert len(half) == 26
    assert half == pytest.approx(full[-26:])
    assert half[0] < 1.0


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_no_denoise_is_an_empty_schedule(bad):
    assert S.schedule(25, tokens(1024, 1024), denoise=bad) == []


def test_overrides_reach_the_maths():
    """The constants are the checkpoint's; another checkpoint may declare others."""
    a = S.dynamic_mu(4096)
    b = S.dynamic_mu(4096, max_seq_len=4096, max_shift=1.15)
    assert a != pytest.approx(b)
