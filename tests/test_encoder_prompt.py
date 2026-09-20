"""The conditioning encoder's chat string, which is a different contract from the expanders'.

Two properties carry real failure modes behind them, both seen in sibling
projects: a system turn that goes missing takes most of the sequence with it,
and a template run through `str.format` breaks on braces the user typed.
"""

import pytest

from qwenimage21_explorations.chat import ENCODER_SYSTEM, render_encoder_prompt

IM_START = "<|im_start|>"


def test_canonical_system_is_what_the_implementations_send():
    assert ENCODER_SYSTEM == "Comprehend and analyze the provided prompt."


@pytest.mark.parametrize("system", ["", "   ", "\n"])
def test_blank_system_falls_back_rather_than_omitting_the_turn(system):
    # ComfyUI drops everything before the SECOND <|im_start|>. With no system
    # turn that marker is the assistant's, and the user turn is cut with it.
    out = render_encoder_prompt("a corgi", system=system)
    assert out.count(IM_START) == 3
    assert ENCODER_SYSTEM in out


def test_second_marker_starts_the_user_turn():
    out = render_encoder_prompt("a corgi")
    second = out.index(IM_START, out.index(IM_START) + 1)
    assert out[second:].startswith(f"{IM_START}user\n")


@pytest.mark.parametrize("text", ["{}", "a {cat} on a {{mat}}", "100% {0}"])
def test_braces_in_the_prompt_survive(text):
    assert text in render_encoder_prompt(text)


@pytest.mark.parametrize("system", ["describe {this}", "{}"])
def test_braces_in_the_system_prompt_survive(system):
    # The reason this module assembles the string instead of formatting a
    # template: core's `template.format(text)` raises on these.
    assert system in render_encoder_prompt("a corgi", system=system)


def test_images_are_numbered_from_one_and_precede_the_prompt():
    out = render_encoder_prompt("swap them", n_images=3)
    body = out.split(f"{IM_START}user\n")[1]
    assert body.startswith("<image1>")
    for i in (1, 2, 3):
        assert f"<image{i}>" in out
    assert body.index("<image3>") < body.index("swap them")


def test_no_vision_block_without_images():
    assert "<|image_pad|>" not in render_encoder_prompt("a corgi")


def test_empty_prompt_becomes_a_space():
    assert f"{IM_START}user\n {'<|im_end|>'}" in render_encoder_prompt("")


def test_ends_at_the_assistant_marker_with_no_think_block():
    out = render_encoder_prompt("a corgi")
    assert out.endswith(f"{IM_START}assistant\n")
    assert "<think>" not in out
