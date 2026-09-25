"""The heylook lookups a browser makes through ComfyUI.

heylook sends no CORS headers (v2.0.123), so a front end on another origin
cannot read its presets or models. These routes fetch them server-side, as the
expander node already does, and hand back only what a picker reads.
"""

import asyncio
import json

import requests

from qwenimage21_explorations import routes
from qwenimage21_explorations.backends import heylook


class Req:
    def __init__(self, **query):
        self.query = query


def call(handler, **query):
    r = asyncio.run(handler(Req(**query)))
    return r.status, json.loads(r.text)


def test_presets_keep_only_what_a_picker_reads():
    raw = [{"id": "a1", "name": "qwen_image-edit", "system_prompt": "long", "created_at": "t",
            "params": {"enable_thinking": True, "max_tokens": 16000}}]
    assert heylook.browser_presets(raw) == [
        {"id": "a1", "name": "qwen_image-edit", "params": {"enable_thinking": True, "max_tokens": 16000}}]


def test_models_keep_only_id_and_capabilities():
    raw = [{"id": "m", "capabilities": ["chat", "vision"], "sampler_defaults": {}, "engine": {}}]
    assert heylook.browser_models(raw) == [{"id": "m", "capabilities": ["chat", "vision"]}]


def test_the_presets_route_answers_in_heylook_s_shape(monkeypatch):
    seen = []
    monkeypatch.setattr(heylook, "list_presets", lambda url, timeout=30: seen.append(url) or
                        [{"id": "a1", "name": "p", "system_prompt": "x", "params": {}}])
    status, body = call(routes.presets, base_url="http://box:8080/")
    assert status == 200
    assert body == {"presets": [{"id": "a1", "name": "p", "params": {}}]}
    assert seen == ["http://box:8080/"]


def test_the_models_route_answers_in_heylook_s_shape(monkeypatch):
    monkeypatch.setattr(heylook, "list_models", lambda url, timeout=30: [{"id": "m", "capabilities": ["vision"]}])
    assert call(routes.models, base_url="http://box:8080") == (200, {"data": [{"id": "m", "capabilities": ["vision"]}]})


def test_no_address_is_a_400_not_a_request(monkeypatch):
    monkeypatch.setattr(heylook, "list_presets", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    status, body = call(routes.presets, base_url="  ")
    assert status == 400 and "base_url" in body["error"]


def test_an_unreachable_server_is_a_502_naming_why(monkeypatch):
    def down(url, timeout=30):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(heylook, "list_models", down)
    status, body = call(routes.models, base_url="http://box:8080")
    assert status == 502 and "refused" in body["error"]
