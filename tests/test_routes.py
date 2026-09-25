"""The routes on ComfyUI's server: heylook lookups and input thumbnails.

heylook sends no CORS headers (v2.0.123), so no web page on another origin can
read its presets or models. These routes fetch them server-side, as the
expander node already does, and hand back only ids, names, params and
capabilities.
"""

import asyncio
import json

import requests

from qwenimage21_explorations import routes
from qwenimage21_explorations.backends import heylook


class Req:
    headers: dict = {}

    def __init__(self, **query):
        self.query = query


def call(handler, **query):
    r = asyncio.run(handler(Req(**query)))
    return r.status, json.loads(r.text)


def test_presets_keep_ids_names_and_params_only():
    raw = [{"id": "a1", "name": "qwen_image-edit", "system_prompt": "long", "created_at": "t",
            "params": {"enable_thinking": True, "max_tokens": 16000}}]
    assert heylook.summarise_presets(raw) == [
        {"id": "a1", "name": "qwen_image-edit", "params": {"enable_thinking": True, "max_tokens": 16000}}]


def test_models_keep_only_id_and_capabilities():
    raw = [{"id": "m", "capabilities": ["chat", "vision"], "sampler_defaults": {}, "engine": {}}]
    assert heylook.summarise_models(raw) == [{"id": "m", "capabilities": ["chat", "vision"]}]


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


# ---------------------------------------------------------------------------
# Thumbnails: /view re-encodes the full image on every preview request, so a
# listing of a few hundred inputs cost megabytes and seconds of server time.


def thumb(tmp_path, **query):
    from PIL import Image
    Image.new("RGB", (1600, 900), (200, 40, 40)).save(tmp_path / "wide.png")
    handler = routes.thumbnail_handler(lambda: str(tmp_path))
    r = asyncio.run(handler(Req(**query)))
    return r


def test_a_thumbnail_is_small_and_keeps_its_aspect(tmp_path):
    import io
    from PIL import Image
    r = thumb(tmp_path, filename="wide.png", size="256")
    assert r.status == 200 and r.content_type == "image/webp"
    im = Image.open(io.BytesIO(r.body))
    assert im.size == (256, 144)


def test_the_size_is_clamped(tmp_path):
    import io
    from PIL import Image
    r = thumb(tmp_path, filename="wide.png", size="99999")
    assert max(Image.open(io.BytesIO(r.body)).size) == routes.THUMB_MAX


def test_a_name_outside_the_input_folder_is_refused(tmp_path):
    (tmp_path.parent / "secret.png").write_bytes(b"x")
    for name in ("../secret.png", "/etc/passwd", "sub/../../secret.png"):
        assert thumb(tmp_path, filename=name).status == 404, name


def test_a_missing_file_is_a_404(tmp_path):
    assert thumb(tmp_path, filename="nope.png").status == 404


def test_an_unchanged_file_revalidates_without_a_body(tmp_path):
    first = thumb(tmp_path, filename="wide.png")
    etag = first.headers["ETag"]
    handler = routes.thumbnail_handler(lambda: str(tmp_path))

    class Cond(Req):
        headers = {"If-None-Match": etag}

    again = asyncio.run(handler(Cond(filename="wide.png")))
    assert again.status == 304
