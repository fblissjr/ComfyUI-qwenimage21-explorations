"""System-prompt resolution for the Qwen-Image 2.1 prompt expanders.

Resolution order deliberately mirrors `<qwen-image-2.1-repo>/prompt_rewrite/pe_core.py::load_system_prompt`:
an explicit override wins, else the checkpoint's own `system_prompt.txt`.

Preferring the file that ships *inside* the checkpoint is upstream's choice and
the reason is worth repeating: the answer contract is part of what the weights
were trained on, so a prompt travelling with the weights cannot drift out of
sync with them. Swapping checkpoints and forgetting to swap the prompt fails
silently -- fluent output, wrong contract.

The two canonical prompts are NOT vendored here. They ship with the checkpoints
and are byte-identical to `<qwen-image-2.1-repo>/prompt_rewrite/prompts/`
(verified by sha256, 2026-09-20). Copying them in would create a third copy that
can drift, and they carry the upstream model licence. `templates/` is for local
variants only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TASKS = ("t2i", "edit")
_FRONTMATTER_DELIM = "---"


@dataclass(frozen=True)
class SystemPrompt:
    text: str
    source: str
    meta: dict


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return {}, content
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_DELIM:
            meta = {}
            for line in lines[1:i]:
                if ":" in line:
                    k, _, val = line.partition(":")
                    v = val.strip()
                    meta[k.strip()] = (
                        True if v.lower() == "true" else False if v.lower() == "false" else v
                    )
            return meta, "\n".join(lines[i + 1 :]).strip()
    return {}, content


def from_checkpoint(ckpt_dir: str | Path) -> SystemPrompt:
    path = Path(ckpt_dir) / "system_prompt.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"no system_prompt.txt in {Path(ckpt_dir).name}. Each task has its own; "
            "they are not interchangeable."
        )
    return SystemPrompt(path.read_text(encoding="utf-8").strip(), "checkpoint", {})


def from_template_file(path: str | Path) -> SystemPrompt:
    p = Path(path)
    meta, body = _parse_frontmatter(p.read_text(encoding="utf-8"))
    return SystemPrompt(body.strip(), "template", meta)


def local_templates(templates_dir: str | Path) -> dict[str, Path]:
    d = Path(templates_dir)
    return {p.stem: p for p in sorted(d.glob("*.md"))} if d.is_dir() else {}


def resolve(
    *,
    explicit_text: str = "",
    template_path: str | Path | None = None,
    ckpt_dir: str | Path | None = None,
) -> SystemPrompt:
    if explicit_text.strip():
        return SystemPrompt(explicit_text.strip(), "explicit", {})
    if template_path:
        return from_template_file(template_path)
    if ckpt_dir:
        return from_checkpoint(ckpt_dir)
    raise ValueError(
        "no system prompt: give explicit text, a template file, or a checkpoint directory"
    )
