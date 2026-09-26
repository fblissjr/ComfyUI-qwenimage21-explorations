"""The prompt expander's request, planned without ComfyUI.

`QwenImage21PEExpand` and `scripts/check_presets_live.py` both build the
heylook request here, so the check sends what the node sends. Only the
generation call and the images stay in the node.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import requests

from . import templates
from .backends import heylook
from .profiles import GREEDY, PROFILES, with_preset

#: heylook serves the two expanders under these names; see the smoke script.
HEYLOOK_MODELS = {"t2i": "Qwen-Image-2.1-PE-T21-mlx", "edit": "Qwen-Image-2.1-PE-I21-mlx"}


@dataclass
class Plan:
    """What one expansion sends, less the brief and the images."""
    model: str
    system: str
    system_source: str
    thinking: bool | None
    extra: dict
    profile: dict


def thinking_controls(base_url: str, model_id: str) -> dict | None:
    """The model's thinking controls from the server, or None when they cannot
    be had -- unreachable, not listed, or not reported -- so the caller sends
    what it has and heylook judges it."""
    try:
        rows = heylook.list_models(base_url)
    except Exception as e:  # the generate call reports an unreachable server itself
        logging.info("[QwenImage21PEExpand] model lookup failed, depth sent unchecked: %s", e)
        return None
    row = next((r for r in rows if r.get("id") == model_id), None)
    return heylook.model_thinking(row) if row else None


def plan(*, task: str, base_url: str, model: str = "", sampling: str = "reference", preset: str = "",
         checkpoint_dir: str = "", template_path: Path | None = None, system_override: str = "",
         thinking: bool = True, reasoning_effort: str = "", presets: list[dict] | None = None) -> Plan:
    """The request the node sends for these inputs. `presets` is the server's
    list when the caller already has it; otherwise it is fetched when a preset
    is named."""
    preset_fields, preset_system = {}, ""
    if preset.strip():
        found = heylook.find_preset(presets if presets is not None else heylook.list_presets(base_url), preset)
        preset_fields, preset_system = heylook.expand_preset(found)

    # Order: raw text, then a local template, then the preset, then the
    # checkpoint, then none at all. A preset beats the checkpoint because nearly every stored
    # one carries a system prompt and picking it is the point -- silently
    # preferring the checkpoint would ignore exactly what was asked for.
    # `system_source` is an output so the winner is never a guess.
    resolved = templates.resolve_with_preset(
        explicit_text=system_override,
        template_path=template_path,
        preset_text=preset_system,
        ckpt_dir=checkpoint_dir.strip() or None,
    )

    profile, extra = with_preset((GREEDY if sampling == "greedy" else PROFILES)[task], preset_fields)
    thinking = profile.pop("thinking", thinking)
    model_id = model.strip() or HEYLOOK_MODELS[task]
    # Typed beats the preset's. Either is checked against the model's own
    # controls, which cost a lookup only when there is a depth to check.
    typed = reasoning_effort.strip()
    effort = typed or str(extra.pop("reasoning_effort", "") or "").strip()
    if effort:
        sent = heylook.depth_to_send(effort, thinking_controls(base_url, model_id),
                                     thinking=thinking is not False, typed=bool(typed))
        if sent:
            extra["reasoning_effort"] = sent
        else:
            logging.warning("[QwenImage21PEExpand] reasoning_effort %r not sent to %s: it does not "
                            "reach the model (thinking off, or not a depth the model offers)",
                            effort, model_id)
    return Plan(model=model_id, system=resolved.text, system_source=resolved.source,
                thinking=thinking, extra=extra, profile=profile)


#: The brief a preset check sends. Its content does not matter: heylook checks
#: the request's fields before it loads or generates anything.
CHECK_BRIEF = "a red apple on a wooden table"


@dataclass
class CheckRow:
    preset: str
    task: str
    model: str
    thinking: bool | None
    depth: str | None
    error: str | None


def check_presets(base_url: str, *, tasks: list[str], models: list[str], send=None,
                  timeout: int = 300) -> list[CheckRow]:
    """Send every stored preset's request, as the node plans it, once per task
    and model (blank is the task's trained expander), capped at one token.

    Every row whose request the server refused carries its reason. heylook has
    no validate-only call, so a request it accepts loads the model and produces
    that one token. Presets are named by id, as the node takes them, so two
    presets sharing a name are both checked.
    """
    send = send or heylook.generate
    presets = heylook.list_presets(base_url)
    rows = []
    for p in presets:
        for task in tasks:
            for model in models:
                plan_ = plan(task=task, base_url=base_url, model=model, preset=p["id"], presets=presets)
                error = None
                try:
                    send(base_url=base_url, model=plan_.model, system=plan_.system, brief=CHECK_BRIEF,
                         images=[], thinking=plan_.thinking, extra=plan_.extra or None,
                         timeout=timeout, **{**plan_.profile, "max_tokens": 1})
                except (requests.RequestException, ValueError) as e:
                    error = str(e)
                rows.append(CheckRow(preset=p.get("name") or p["id"], task=task, model=plan_.model,
                                     thinking=plan_.thinking, depth=plan_.extra.get("reasoning_effort"),
                                     error=error))
    return rows
