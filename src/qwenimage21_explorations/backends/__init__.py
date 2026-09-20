"""Inference backends for the prompt expanders.

Two concrete backends, deliberately not a provider abstraction: ComfyUI's
in-process `CLIP.generate` on CUDA, and a heylook (MLX) server over HTTP. They
differ in more than transport -- ComfyUI uses its own bundled qwen35 tokenizer
and template and ignores the checkpoint's `chat_template.jinja`, while heylook
applies that jinja server-side. What must match across them is the effective
prompt and the sampling settings, which is what the harness exists to hold
steady.
"""

from . import heylook

__all__ = ["heylook"]
