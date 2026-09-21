"""heylook (heylookitsanllm) client for the prompt expanders.

`/v1/messages` is Anthropic Messages-conformant. The OpenAI-compatible
`/v1/chat/completions` was removed in heylook 1.79.66, so this is the route.

Two behaviours worth knowing before comparing a heylook row to a ComfyUI row:

* **Thinking comes back as its own content block**, not as inline ``<think>``
  tags. `normalise_response` folds both shapes into one (thinking, text) pair so
  the same parser grades either backend.
* **The server's per-model `sampler_defaults` are not the reference settings.**
  Leaving `top_k`, `presence_penalty` or `max_tokens` unset does not reproduce
  `<qwen-image-2.1-repo>/prompt_rewrite/pe_core.py::PROFILES`. A `max_tokens`
  below the trace length truncates mid-thinking, and a truncated trace parses as
  invalid JSON -- indistinguishable downstream from a quantization fault. Every
  request here sends them explicitly.

The server does not resize images (`/v1/capabilities`: "No server-side resize --
clients downscale before sending"), so `encode_image` caps pixels to match
training.

**Block order does not place the image markers on this path**, unlike
`chat.py`, which positions them itself for ComfyUI. The server coalesces every
text part of a message into one run and then places N markers by the model
family's own convention. So interleaving labels between images would not
survive: a per-image label belongs inside the single text block. Images also
have to ride on user turns, which is the only kind this builds.

The wire spelling is `thinking`, not `enable_thinking`. The request model
ignores unknown fields, so the wrong spelling is dropped in silence and the run
succeeds on the server's default -- the reason this sends the name the schema
declares rather than the one the config files use.
"""

from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_MAX_PIXELS = 1024 * 1024  # pe_core.Profile.image_max_pixels


@dataclass
class Response:
    thinking: str
    text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int

    @property
    def truncated(self) -> bool:
        """A trace cut off by the token cap. Reads downstream as bad JSON."""
        return self.stop_reason == "max_tokens"

    def as_inline(self) -> str:
        """Re-render in ComfyUI's inline shape, so one parser grades both."""
        return f"<think>\n{self.thinking}\n</think>\n\n{self.text}" if self.thinking else self.text


#: A preset's `params` use the server's INTERNAL spellings, and the wire refuses
#: some of them outright -- `enable_thinking` is a 422 there and appears in most
#: of the stored presets. Expanding a preset means translating, not forwarding.
PRESET_ALIASES = {"enable_thinking": "thinking", "max_new_tokens": "max_tokens"}


def list_presets(base_url: str, timeout: int = 30) -> list[dict]:
    """The user presets stored on the server. Names are not unique; ids are."""
    r = requests.get(f"{base_url.rstrip('/')}/v1/presets", timeout=timeout)
    r.raise_for_status()
    return r.json().get("presets", [])


def find_preset(presets: list[dict], wanted: str) -> dict:
    """By id first, then by name, case-insensitively. Raises naming what exists."""
    want = wanted.strip()
    for p in presets:
        if p.get("id") == want:
            return p
    matches = [p for p in presets if (p.get("name") or "").lower() == want.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        names = ", ".join(sorted((p.get("name") or "?") for p in presets)) or "none stored"
        raise ValueError(f"no preset named {wanted!r}. Available: {names}")
    ids = ", ".join(p["id"] for p in matches)
    raise ValueError(f"{len(matches)} presets are named {wanted!r}; use an id: {ids}")


def expand_preset(preset: dict) -> tuple[dict, str]:
    """A preset to (wire sampler fields, system prompt).

    The server never receives a preset: `preset` as a request field is a
    deliberate 422, because named sampler bundles were removed in v2.0.30. The
    UI expands presets client-side into explicit fields and so does this.
    """
    params = preset.get("params") or {}
    fields = {PRESET_ALIASES.get(k, k): v for k, v in params.items()}
    return fields, (preset.get("system_prompt") or "")


def encode_image(source, max_pixels: int = DEFAULT_MAX_PIXELS) -> dict:
    """`source` is a path or an already-open PIL image (what a ComfyUI node has).

    `max_pixels` of 0 sends the image untouched. That is the right setting when
    it was already sized upstream: the cap is an area, so an image on the
    encoder's own 32-pixel grid can sit a fraction of a percent above it and
    earn a resample that only softens it and leaves the grid.
    """
    from PIL import Image

    img = source if hasattr(source, "convert") else Image.open(source)
    img = img.convert("RGB")
    w, h = img.size
    if max_pixels and w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(buf.getvalue()).decode(),
        },
    }


def normalise_response(payload: dict) -> Response:
    thinking, text = [], []
    for blk in payload.get("content", []):
        kind = blk.get("type")
        if kind == "thinking":
            thinking.append(blk.get("thinking") or blk.get("text") or "")
        elif kind == "text":
            text.append(blk.get("text", ""))
    usage = payload.get("usage") or {}
    return Response(
        thinking="\n".join(t for t in thinking if t).strip(),
        text="\n".join(text).strip(),
        stop_reason=payload.get("stop_reason", ""),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )


def generate(
    *,
    base_url: str,
    model: str,
    system: str,
    brief: str,
    images: list | None = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    max_tokens: int,
    thinking: bool | None = None,
    extra: dict | None = None,
    timeout: int = 900,
) -> Response:
    """One expansion. Images go FIRST in the user turn, in order.

    `thinking` is heylook's own bool, not Anthropic's config object; left None
    the server applies the model's default, which is on for both expanders.
    """
    content: list[dict] = [encode_image(p, max_pixels) for p in (images or [])]
    content.append({"type": "text", "text": brief})
    body = {
        "model": model,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "messages": [{"role": "user", "content": content}],
    }
    if thinking is not None:
        body["thinking"] = thinking
    # Sampler fields this signature does not name -- an expanded preset's
    # `reasoning_effort`, say. The request model refuses unknown fields, so a
    # wrong spelling here is a 422 rather than a silent drop.
    if extra:
        body.update(extra)
    # Hanging up does NOT cancel on this server: a non-streaming run writes
    # nothing until it finishes, so a timeout here leaves it generating and
    # blocking everything queued behind it. The explicit DELETE is the stop,
    # and X-Request-ID is the handle for it.
    request_id = uuid.uuid4().hex
    root = base_url.rstrip("/")
    delivered = False
    try:
        r = requests.post(f"{root}/v1/messages", json=body, timeout=timeout,
                          headers={"X-Request-ID": request_id})
        r.raise_for_status()
        out = normalise_response(r.json())
        delivered = True
        return out
    finally:
        if not delivered:
            try:
                requests.delete(f"{root}/v1/requests/{request_id}", timeout=30)
            except requests.RequestException:
                pass  # the run is already orphaned; failing to say so changes nothing
