"""ComfyUI's tokenization must match the checkpoint's, or the model receives a
segmentation it was never trained on.

The vocabulary and merges are identical; the pre-tokenizer is not. ComfyUI
builds a Qwen2Tokenizer whose regex matches `\\p{L}+` (letters), while the
Qwen3.5 checkpoints declare `[\\p{L}\\p{M}]+` (letters and combining marks). So
mark-heavy scripts split differently. See docs section 29.

The xfails document the known divergence and will start failing loudly -- as
XPASS -- when upstream fixes it, which is the point.
"""

import os
from pathlib import Path

import pytest

CKPT = Path(os.environ.get("PE_T2I_CKPT", "")) if os.environ.get("PE_T2I_CKPT") else None


@pytest.fixture(scope="module")
def tokenizers():
    if CKPT is None or not (CKPT / "tokenizer.json").is_file():
        pytest.skip("set PE_T2I_CKPT to a checkpoint directory")
    comfy = pytest.importorskip("comfy.text_encoders.qwen35", reason="needs ComfyUI on sys.path")
    from tokenizers import Tokenizer
    ckpt = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
    comfy_raw = comfy.tokenizer(model_type="qwen35_9b")().qwen35_9b.tokenizer
    return ckpt, comfy_raw


def _enc(tok, s):
    r = tok.encode(s)
    return r.ids if hasattr(r, "ids") else r


CASES = {
    "chinese": "把标题改成夏日特惠",
    "japanese": "タイトルを日本語で追加してください",
    "korean": "제목을 한국어로 추가해 주세요",
    "arabic": "أضف عنوانًا باللغة العربية",
    "french": "ajoute un titre en français",
    "cyrillic": "добавь заголовок по-русски",
    "emoji": "add a 🌧️ and a 🎸 to the sign",
    "json": '{"rewritten_prompt": "a corgi", "wh_ratio": "16:9"}',
    "thai": pytest.param("เพิ่มชื่อเรื่องภาษาไทย", marks=pytest.mark.xfail(
        reason="ComfyUI uses Qwen2's pre-tokenizer regex; marks split from base letters",
        strict=True)),
    "hindi": pytest.param("शीर्षक हिंदी में जोड़ें", marks=pytest.mark.xfail(
        reason="same combining-mark divergence as Thai", strict=True)),
}


@pytest.mark.parametrize("text", list(CASES.values()), ids=list(CASES))
def test_comfy_tokenization_matches_checkpoint(tokenizers, text):
    ckpt, comfy_raw = tokenizers
    assert _enc(ckpt, text) == _enc(comfy_raw, text)


def test_vocab_and_merges_are_identical(tokenizers):
    """Establishes that the divergence is NOT a stale vocabulary, so that nobody
    tries to fix it by bundling a different tokenizer."""
    import json
    ck = json.loads((CKPT / "tokenizer.json").read_text())["model"]
    import comfy.text_encoders.qwen35 as q
    base = Path(q.__file__).parent / "qwen35_tokenizer"
    cv = json.loads((base / "vocab.json").read_text())
    cm = [l for l in (base / "merges.txt").read_text().splitlines()
          if l.strip() and not l.startswith("#version")]
    assert len(cv) == len(ck["vocab"])
    norm = lambda x: tuple(x) if isinstance(x, list) else tuple(x.split(" ", 1))
    assert [norm(x) for x in ck["merges"]] == [norm(x) for x in cm]
