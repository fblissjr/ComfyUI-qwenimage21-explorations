"""The expander's request planning and the live preset check, without ComfyUI.

`check_presets` is what `scripts/check_presets_live.py` runs against a real
heylook: the node's tests stub the server, so they hold its contract as it was
when they were written. A preset's stored depth sent to a model with no depth
control passed every stubbed test and was a 400 on the real server.
"""

import pytest
import requests

from qwenimage21_explorations import expander
from qwenimage21_explorations.backends import heylook

PRESETS = [
    {"id": "p1", "name": "plain", "params": {"enable_thinking": True}},
    {"id": "p2", "name": "deep", "params": {"enable_thinking": True, "reasoning_effort": "medium"}},
]
MODELS = [
    {"id": expander.HEYLOOK_MODELS["t2i"], "capabilities": ["thinking"],
     "engine": {"thinking": {"switch": "enable_thinking", "depth": None}}},
    {"id": "q", "capabilities": ["thinking", "reasoning_effort"],
     "engine": {"thinking": {"switch": "enable_thinking", "depth": {
         "values": ["xhigh", "medium", "low"], "aliases": {}, "default": "xhigh", "unknown": "raises", "off": []}}}},
]


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(heylook, "list_models", lambda base_url, timeout=30: MODELS)
    monkeypatch.setattr(heylook, "list_presets", lambda base_url, timeout=30: PRESETS)


def test_a_blank_model_is_the_trained_expander_for_the_task(server):
    assert expander.plan(task="edit", base_url="http://h").model == expander.HEYLOOK_MODELS["edit"]


def test_a_preset_depth_is_dropped_where_the_model_has_none(server):
    assert "reasoning_effort" not in expander.plan(task="t2i", base_url="http://h", preset="deep").extra


def test_a_preset_depth_is_kept_where_the_model_offers_it(server):
    assert expander.plan(task="t2i", base_url="http://h", model="q", preset="deep").extra == {
        "reasoning_effort": "medium"}


def test_every_preset_is_sent_once_per_task_and_model_with_one_token(server):
    sent = []

    def send(**kw):
        sent.append(kw)

    rows = expander.check_presets("http://h", tasks=["t2i"], models=["", "q"], send=send)
    assert [(r.preset, r.model) for r in rows] == [
        ("plain", expander.HEYLOOK_MODELS["t2i"]), ("plain", "q"),
        ("deep", expander.HEYLOOK_MODELS["t2i"]), ("deep", "q")]
    assert all(kw["max_tokens"] == 1 for kw in sent)
    assert [(kw["extra"] or {}).get("reasoning_effort") for kw in sent] == [None, None, None, "medium"]
    assert all(r.error is None for r in rows)


def test_a_refused_request_is_reported_with_the_server_s_reason(server):
    def send(**kw):
        if (kw["extra"] or {}).get("reasoning_effort"):
            raise requests.HTTPError("400 from http://h/v1/messages: {\"detail\":\"no depth\"}")

    rows = expander.check_presets("http://h", tasks=["t2i"], models=["q"], send=send)
    assert [(r.preset, r.error) for r in rows] == [
        ("plain", None), ("deep", "400 from http://h/v1/messages: {\"detail\":\"no depth\"}")]


def test_presets_are_found_by_id_so_a_doubled_name_is_still_checked(monkeypatch):
    monkeypatch.setattr(heylook, "list_models", lambda base_url, timeout=30: MODELS)
    monkeypatch.setattr(heylook, "list_presets", lambda base_url, timeout=30: [
        {"id": "a", "name": "dupe", "params": {}}, {"id": "b", "name": "DUPE", "params": {}}])
    rows = expander.check_presets("http://h", tasks=["t2i"], models=[""], send=lambda **kw: None)
    assert [r.error for r in rows] == [None, None]
