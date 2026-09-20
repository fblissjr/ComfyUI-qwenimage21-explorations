"""Qwen-Image 2.1 explorations: prompt-expander harness and quantization tooling.

`chat`, `answer`, `templates` and `profiles` are pure and importable without
ComfyUI, so the same construction drives the ComfyUI nodes, the heylook/MLX
backend, and the offline calibration and evaluation scripts under `scripts/`.
"""

from . import answer, chat, profiles, templates
from .profiles import GREEDY, PROFILES

__all__ = ["answer", "chat", "profiles", "templates", "PROFILES", "GREEDY"]
