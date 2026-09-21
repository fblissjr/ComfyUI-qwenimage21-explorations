import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def nodes():
    """The node module, which needs ComfyUI importable. Skips where it is not."""
    root = Path(__file__).resolve().parents[3]   # this repo sits at ComfyUI/custom_nodes/<repo>/
    if not (root / "comfy").is_dir():
        pytest.skip("needs ComfyUI alongside this checkout")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("qwenimage21_explorations.nodes")
    except Exception as exc:                      # comfy present but not importable here
        pytest.skip(f"ComfyUI not importable: {type(exc).__name__}")
