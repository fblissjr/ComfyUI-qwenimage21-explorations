"""Parse and grade a prompt expander's answer.

The expanders emit a thinking trace followed by a JSON object; an answer
with none is still the prompt, and only grades as unparseable. Syntactic
validity (`parse_ok`) is the weakest useful gate: the answer contract is
mode-dependent, and a degraded model that still emits valid JSON while breaking
the contract is exactly the failure a parse-only check misses.

The contract rules below are read out of the checkpoints' own system prompts.
`<models>/Qwen-Image-2.1-PE-I21/system_prompt.txt` is the authority for edit;
the t2i prompt carries the `wh_ratio`-only subset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
IMAGE_TAG_RE = re.compile(r"<image(\d+)>")
WH_RATIO_RE = re.compile(r"^\d+:\d+$")
# Resolution/ratio strings the prompt forbids inside rewritten_prompt.
RES_IN_PROMPT_RE = re.compile(r"\b\d{3,4}\s*[x×]\s*\d{3,4}\b|\b\d+:\d+\b|\b[248]K\b", re.IGNORECASE)


@dataclass
class Answer:
    raw: str
    thinking: str = ""
    body: str = ""
    data: dict | None = None
    parse_ok: bool = False
    rewritten_prompt: str = ""
    wh_ratio: str = ""
    ratio_follow: str = ""
    violations: list[str] = field(default_factory=list)

    @property
    def contract_ok(self) -> bool:
        return self.parse_ok and not self.violations


def split_thinking(text: str) -> tuple[str, str]:
    """Return (thinking, remainder). Handles a block left unclosed by a token cap."""
    m = THINK_RE.search(text)
    if not m:
        return "", text.strip()
    inner = m.group(0)
    inner = inner[len("<think>"):]
    if inner.endswith("</think>"):
        inner = inner[: -len("</think>")]
    return inner.strip(), (text[: m.start()] + text[m.end():]).strip()


def strip_fences(text: str) -> str:
    m = FENCE_RE.match(text)
    return m.group(1).strip() if m else text.strip()


def _first_json_object(text: str) -> str | None:
    """Slice the first balanced {...} run, tolerating prose either side."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_parts(thinking: str, body: str) -> Answer:
    """Parse an answer a backend already split for us.

    heylook returns thinking as its own content block. Re-joining it into
    ComfyUI's inline shape only to split it again is a round trip that can only
    lose, so the backend's own split is used directly and the contract grading
    below is shared either way.
    """
    ans = Answer(raw=body)
    ans.thinking, ans.body = thinking.strip(), body.strip()
    return _parse_body(ans)


def parse(text: str) -> Answer:
    """Parse an answer with the thinking still inline, as ComfyUI returns it."""
    ans = Answer(raw=text)
    ans.thinking, ans.body = split_thinking(text)
    return _parse_body(ans)


def _parse_body(ans: Answer) -> Answer:
    # Until a JSON object replaces it, the answer itself is the prompt, as in the
    # reference runner (pe_core.py::parse_answer). A system prompt without the
    # trained output format gets a good rewritten prompt in plain prose, and
    # dropping it sent the encoder nothing. parse_ok stays False; grade reports it.
    ans.rewritten_prompt = ans.body
    candidate = strip_fences(ans.body)
    blob = candidate if candidate.startswith("{") else (_first_json_object(candidate) or "")
    if not blob:
        return ans
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return ans
    if not isinstance(data, dict):
        return ans
    ans.data = data
    ans.parse_ok = True
    # Upstream names the field `rewritten_prompt`; accept `positive_prompt` too,
    # which is what the reference runners write into their output records.
    ans.rewritten_prompt = str(data.get("rewritten_prompt") or data.get("positive_prompt") or "")
    ans.wh_ratio = str(data.get("wh_ratio") or "")
    ans.ratio_follow = str(data.get("ratio_follow") or "")
    return ans


def grade(ans: Answer, *, task: str, n_images: int = 0) -> Answer:
    """Apply the mode-dependent contract. Mutates and returns `ans`.

    `task` is "t2i" or "edit". `n_images` is how many images were supplied.
    """
    v = ans.violations
    if not ans.parse_ok:
        v.append("json:unparseable")
        return ans

    if not ans.rewritten_prompt.strip():
        v.append("field:rewritten_prompt_empty")

    tags = {int(n) for n in IMAGE_TAG_RE.findall(ans.rewritten_prompt)}
    if task == "edit":
        # Image Reference Rules: tags mandatory at N>=2, forbidden at N==1.
        if n_images >= 2 and not tags:
            v.append("tags:missing_at_multi_image")
        if n_images == 1 and tags:
            v.append("tags:present_at_single_image")
        if tags and max(tags) > max(n_images, 1):
            v.append(f"tags:out_of_range(max={max(tags)},n={n_images})")
        # wh_ratio and ratio_follow are mutually exclusive: exactly one carries a value.
        both = bool(ans.wh_ratio) and bool(ans.ratio_follow)
        neither = not ans.wh_ratio and not ans.ratio_follow
        if both:
            v.append("ratio:both_set")
        if neither:
            v.append("ratio:neither_set")
        if ans.ratio_follow and not IMAGE_TAG_RE.fullmatch(ans.ratio_follow):
            v.append(f"ratio_follow:malformed({ans.ratio_follow!r})")
    else:
        if ans.ratio_follow:
            v.append("ratio_follow:set_on_t2i")
        if not ans.wh_ratio:
            v.append("ratio:wh_ratio_missing")

    if ans.wh_ratio and not WH_RATIO_RE.match(ans.wh_ratio):
        v.append(f"wh_ratio:malformed({ans.wh_ratio!r})")

    # "Never include any resolution or aspect ratio information in rewritten_prompt."
    hit = RES_IN_PROMPT_RE.search(ans.rewritten_prompt)
    if hit:
        v.append(f"prompt:carries_resolution({hit.group(0)!r})")

    return ans


def parse_and_grade(text: str, *, task: str, n_images: int = 0) -> Answer:
    return grade(parse(text), task=task, n_images=n_images)


def grade_parts(thinking: str, body: str, *, task: str, n_images: int = 0) -> Answer:
    """The same contract, for a backend that split the thinking itself."""
    return grade(parse_parts(thinking, body), task=task, n_images=n_images)
