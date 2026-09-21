"""heylook returns thinking as its own content block; ComfyUI returns it inline.

One contract has to grade both, so these pin the normalisation, the truncation
signal the harness depends on, and the pixel cap the server will not apply.
"""

import base64
import io

import pytest

from qwenimage21_explorations.answer import parse_and_grade
from qwenimage21_explorations.backends import heylook
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


def test_the_pixel_cap_is_opt_out():
    """A pre-sized image should reach the server untouched.

    The cap is an area, so an image on the encoder's own 32-pixel grid can sit a
    fraction of a percent above it. Resampling then buys nothing and costs the
    grid: 1376x768, which is what the encode node produces for a 16:9 reference
    at its default, is 0.4% over.
    """
    from PIL import Image

    def sent(cap):
        blob = heylook.encode_image(Image.new("RGB", (1376, 768)), cap)
        raw = base64.b64decode(blob["source"]["data"])
        return Image.open(io.BytesIO(raw)).size

    assert sent(0) == (1376, 768)
    assert sent(heylook.DEFAULT_MAX_PIXELS) != (1376, 768)


def test_truncated_reads_the_sentinel_the_api_declares():
    """Zero truncated rows is only evidence if this reads the right value.

    The live response schema declares stop_reason as one of end_turn,
    max_tokens, stop_sequence. If the server renamed the middle one, the
    harness would report a truncated trace as a clean answer and a quantization
    comparison would inherit the mistake.
    """
    declared = ("end_turn", "max_tokens", "stop_sequence")
    got = {r: normalise_response(payload([{"type": "text", "text": "{}"}], stop=r)).truncated
           for r in declared}
    assert got == {"end_turn": False, "max_tokens": True, "stop_sequence": False}


def test_a_failed_request_is_cancelled_server_side(monkeypatch):
    """Hanging up does not stop a non-streaming run; it blocks the queue behind it.

    So a timeout has to send the explicit DELETE, keyed by the same
    X-Request-ID the POST carried.
    """
    seen = {}

    def fake_post(url, **kw):
        seen["post"] = (url, kw["headers"]["X-Request-ID"])
        raise heylook.requests.Timeout("too slow")

    def fake_delete(url, **kw):
        seen["delete"] = url

    monkeypatch.setattr(heylook.requests, "post", fake_post)
    monkeypatch.setattr(heylook.requests, "delete", fake_delete)

    with pytest.raises(heylook.requests.Timeout):
        heylook.generate(base_url="http://h/", model="m", system="s", brief="b",
                         temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                         presence_penalty=0.0, max_tokens=10)

    rid = seen["post"][1]
    assert seen["delete"].endswith(f"/v1/requests/{rid}"), seen


def test_a_delivered_request_is_not_cancelled(monkeypatch):
    class Ok:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return payload([{"type": "text", "text": "{}"}])

    calls = []
    monkeypatch.setattr(heylook.requests, "post", lambda url, **kw: Ok())
    monkeypatch.setattr(heylook.requests, "delete", lambda url, **kw: calls.append(url))
    heylook.generate(base_url="http://h", model="m", system="s", brief="b",
                     temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                     presence_penalty=0.0, max_tokens=10)
    assert calls == []
