"""ComfyUI-qwenimage21-explorations.

Probes for the ComfyUI V3 extension API and exposes `comfy_entrypoint`.
The `src/` package is importable without ComfyUI; only this shim needs it.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from comfy_api.latest import ComfyExtension  # noqa: F401 -- probe only
except ImportError:
    pass
else:
    from qwenimage21_explorations.nodes import comfy_entrypoint  # noqa: F401
