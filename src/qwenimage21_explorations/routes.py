"""The heylook lookups a front end makes through ComfyUI.

heylook sends no CORS headers since v2.0.123, on purpose: its API is
unauthenticated, and a cross-origin grant let any page the owner opened drive
it. So a browser on another origin cannot read its presets or models. These
fetch them server-side, as the expander node already does when it runs, and
return only what a picker reads -- not the presets' system prompts.

They add no reach: anyone who can call ComfyUI can already queue the expander
node against any `base_url`, and it fetches `/v1/presets` from there.

Handlers only; the extension's `on_load` registers them on ComfyUI's server,
so this module imports without it.
"""

from __future__ import annotations

import asyncio

import requests
from aiohttp import web

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
    return await _lookup(request, heylook.list_presets, heylook.browser_presets, "presets")


async def models(request) -> web.Response:
    return await _lookup(request, heylook.list_models, heylook.browser_models, "data")


def register(routes) -> None:
    routes.get("/qwenimage21/heylook/presets")(presets)
    routes.get("/qwenimage21/heylook/models")(models)
