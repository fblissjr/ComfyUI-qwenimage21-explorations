"""Assert on the FINAL prompt string, never on the flags that produced it.

This file exists because of one bug in the predecessor repo that survived eight
releases undetected: its encoder built a full `<|im_start|>...` string and the
framework tokenizer then wrapped it a second time. Nothing errored. The
conditioning was silently nested one level deeper than intended and merely read
as a mediocre prompt. It was found only once a formatted-prompt output was added
-- in the same release that fixed it.

Our defence is that ComfyUI's `Qwen35ImageTokenizer.tokenize_with_weights` takes
its `skip_template` branch on `text.startswith('<|im_start|>')`. That is a
string test, so anything that perturbs the first character -- a stray newline, a
BOM, an "improvement" that strips or pads -- silently re-enables wrapping. These
tests pin the properties that check depends on.
"""

import re

import pytest

from qwenimage21_explorations.chat import Conversation, render_generation_prompt

IM_START = "<|im_start|>"


def conv(thinking=True, n=0, system="SYS", brief="b"):
    return Conversation(system=system, thinking=thinking).with_brief(brief, n_images=n)


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("n", [0, 1, 3])
def test_no_leading_whitespace_before_the_marker(thinking, n):
    # The skip_template branch is `startswith`, not `strip().startswith`.
    out = render_generation_prompt(conv(thinking=thinking, n=n))
    assert out.startswith(IM_START)
    assert out[0] != "﻿"


def test_exactly_one_system_turn():
    out = render_generation_prompt(conv())
    assert out.count(f"{IM_START}system") == 1


def test_turn_markers_balance_with_one_open_assistant_turn():
    # Every turn closes except the final assistant turn, which the model continues.
    out = render_generation_prompt(conv(n=2))
    assert out.count(IM_START) == out.count("<|im_end|>") + 1
    assert out.rstrip().endswith(("<think>", "</think>"))


def test_thinking_on_leaves_the_block_open():
    out = render_generation_prompt(conv(thinking=True))
    assert out.count("<think>") == 1
    assert "</think>" not in out


def test_thinking_off_closes_it_exactly_once():
    out = render_generation_prompt(conv(thinking=False))
    assert out.count("<think>") == 1
    assert out.count("</think>") == 1


def test_no_duplicated_headers_from_a_double_wrap():
    # What a double-wrap looks like: a second system header, or a system turn
    # appearing after the user turn.
    out = render_generation_prompt(conv())
    assert not re.search(rf"{re.escape(IM_START)}system.*{re.escape(IM_START)}system", out, re.DOTALL)
    assert out.index(f"{IM_START}system") < out.index(f"{IM_START}user")


def test_system_text_is_not_repeated_in_the_body():
    marker = "UNIQUE_SYSTEM_SENTINEL"
    out = render_generation_prompt(conv(system=marker))
    assert out.count(marker) == 1


def test_special_tokens_survive_comfy_tokenizer_as_single_ids():
    """The predecessor's think tags silently became subwords under a bundled
    tokenizer that lacked them, so every thinking experiment tested text shaped
    like a think block. Our generation prompt ends on an open <think>; if that
    subworded, the model would be somewhere it was never trained."""
    comfy = pytest.importorskip("comfy.text_encoders.qwen35",
                                reason="needs ComfyUI on sys.path")
    raw = comfy.tokenizer(model_type="qwen35_9b")().qwen35_9b.tokenizer
    expected = {
        "<|im_start|>": 248045, "<|im_end|>": 248046,
        "<think>": 248068, "</think>": 248069,
        "<|vision_start|>": 248053, "<|vision_end|>": 248054,
        "<|image_pad|>": 248056,
    }
    for token, want in expected.items():
        got = raw.encode(token)
        got = got.ids if hasattr(got, "ids") else got
        assert got == [want], f"{token} tokenized as {got}, expected single id {want}"
