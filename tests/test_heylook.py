"""heylook returns thinking as its own content block; ComfyUI returns it inline.

One parser has to grade both, so the backend folds the two shapes into one.
These pin that fold, and the truncation signal the harness depends on.
"""

from qwenimage21_explorations.answer import parse_and_grade
from qwenimage21_explorations.backends.heylook import normalise_response


def payload(blocks, stop="end_turn", usage=None):
    return {"content": blocks, "stop_reason": stop, "usage": usage or {}}


def test_separate_thinking_block_is_folded():
    r = normalise_response(payload([
        {"type": "thinking", "thinking": "reasoning here"},
        {"type": "text", "text": '{"rewritten_prompt":"x","wh_ratio":"1:1"}'},
    ]))
    assert r.thinking == "reasoning here"
    assert r.text.startswith("{")
    # re-rendered inline, the shared parser grades it like a ComfyUI answer
    assert parse_and_grade(r.as_inline(), task="t2i").contract_ok


def test_thinking_block_using_text_key():
    # some servers put the trace in `text` on a thinking-typed block
    r = normalise_response(payload([{"type": "thinking", "text": "t"}, {"type": "text", "text": "{}"}]))
    assert r.thinking == "t"


def test_no_thinking_block_round_trips_unchanged():
    r = normalise_response(payload([{"type": "text", "text": "{}"}]))
    assert r.thinking == ""
    assert r.as_inline() == "{}"


def test_truncation_is_surfaced():
    # max_tokens below the trace length cuts mid-thinking; downstream that reads
    # as unparseable JSON, which looks exactly like a quantization fault.
    r = normalise_response(payload([{"type": "thinking", "thinking": "cut off"}], stop="max_tokens"))
    assert r.truncated
    assert not parse_and_grade(r.as_inline(), task="t2i").parse_ok


def test_usage_is_carried():
    r = normalise_response(payload([{"type": "text", "text": "{}"}],
                                   usage={"input_tokens": 2449, "output_tokens": 1219}))
    assert (r.input_tokens, r.output_tokens) == (2449, 1219)
