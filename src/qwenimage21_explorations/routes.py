"""HTTP routes on ComfyUI's server: heylook lookups, and input thumbnails.

**heylook.**
heylook sends no CORS headers since v2.0.123, on purpose: its API is
unauthenticated, and a cross-origin grant let any page the owner opened drive
it. So no web page on another origin can read its presets or models. These
fetch them server-side, as the expander node already does when it runs, and
return only ids, names, params and capabilities -- not the presets' system
prompts.

They add no reach: anyone who can call ComfyUI can already queue the expander
node against any `base_url`, and it fetches `/v1/presets` from there.

**Thumbnails.** Core's `/view` re-encodes the full image on every `preview`
request and never resizes, so listing a few hundred inputs as previews costs
tens of kilobytes and a full-size encode each. `/qwenimage21/thumb` answers a
small WebP of one file in the input folder, contained the way `/view` contains
its paths, with an ETag so an unchanged file revalidates without a body.

Handlers only; the extension's `on_load` registers them on ComfyUI's server,
so this module imports without it.
"""

from __future__ import annotations

import asyncio
import io
import os

import requests
from aiohttp import web
from PIL import Image, ImageOps

from qwenimage21_explorations.backends import heylook


async def _lookup(request, fetch, shape, key: str) -> web.Response:
    base_url = request.query.get("base_url", "").strip()
    if not base_url:
        return web.json_response({"error": "base_url is required"}, status=400)
    try:
        rows = await asyncio.to_thread(fetch, base_url)
    except (requests.RequestException, ValueError) as exc:
        return web.json_response({"error": f"{base_url}: {exc}"}, status=502)
    return web.json_response({key: shape(rows)})


async def presets(request) -> web.Response:
    return await _lookup(request, heylook.list_presets, heylook.summarise_presets, "presets")


async def models(request) -> web.Response:
    return await _lookup(request, heylook.list_models, heylook.summarise_models, "data")


#: Longest side of a thumbnail, in pixels: the default, and the clamp on what a caller asks.
THUMB_DEFAULT = 256
THUMB_MIN = 32
THUMB_MAX = 512


def _input_file(root: str, name: str) -> str | None:
    """`name` inside `root`, or None. The containment test core's `/view` uses."""
    root = os.path.abspath(root)
    path = os.path.abspath(os.path.join(root, name))
    if os.path.commonpath((path, root)) != root or not os.path.isfile(path):
        return None
    return path


def _thumb_bytes(path: str, size: int) -> bytes:
    with Image.open(path) as src:
        src.draft("RGB", (size, size))   # a JPEG decodes at a fraction of full size
        im = ImageOps.exif_transpose(src)
    im.thumbnail((size, size))
    im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=70)
    return buf.getvalue()


def thumbnail_handler(input_dir):
    """The thumbnail route over `input_dir()`, read per request as core reads it."""
    async def thumbnail(request) -> web.Response:
        path = _input_file(input_dir(), request.query.get("filename", ""))
        if path is None:
            return web.Response(status=404)
        try:
            size = int(request.query.get("size", THUMB_DEFAULT))
        except ValueError:
            size = THUMB_DEFAULT
        size = max(THUMB_MIN, min(size, THUMB_MAX))
        st = os.stat(path)
        headers = {"ETag": f'"{st.st_mtime_ns:x}-{st.st_size:x}-{size}"', "Cache-Control": "no-cache"}
        if request.headers.get("If-None-Match") == headers["ETag"]:
            return web.Response(status=304, headers=headers)
        try:
            body = await asyncio.to_thread(_thumb_bytes, path, size)
        except OSError:                  # not an image PIL can read
            return web.Response(status=415)
        return web.Response(body=body, content_type="image/webp", headers=headers)
    return thumbnail


def register(routes, input_dir) -> None:
    routes.get("/qwenimage21/heylook/presets")(presets)
    routes.get("/qwenimage21/heylook/models")(models)
    routes.get("/qwenimage21/thumb")(thumbnail_handler(input_dir))
