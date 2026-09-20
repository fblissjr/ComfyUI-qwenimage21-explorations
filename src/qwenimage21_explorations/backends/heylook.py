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
"""

from __future__ import annotations

import base64
import io
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


def encode_image(path: str | Path, max_pixels: int = DEFAULT_MAX_PIXELS) -> dict:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
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
    images: list[str | Path] | None = None,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    max_tokens: int,
    timeout: int = 900,
) -> Response:
    """One expansion. Images go FIRST in the user turn, in order."""
    content: list[dict] = [encode_image(p) for p in (images or [])]
    content.append({"type": "text", "text": brief})
    body = {
        "model": model,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "messages": [{"role": "user", "content": content}],
    }
    r = requests.post(f"{base_url.rstrip('/')}/v1/messages", json=body, timeout=timeout)
    r.raise_for_status()
    return normalise_response(r.json())
