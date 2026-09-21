"""Reference sampling settings, from `<qwen-image-2.1-repo>/prompt_rewrite/pe_core.py::PROFILES`.

Their home is a constant rather than prose because a harness that relies on any
backend's defaults is not running the reference configuration. heylook serves
these models with a different top_k, presence_penalty and max_tokens; ComfyUI's
TextGenerate defaults differ again on min_p and repetition_penalty. A max_tokens
below the thinking trace truncates mid-stream, and a truncated trace parses as
invalid JSON -- indistinguishable downstream from a quantization fault.

Pure data: importable without ComfyUI, so scripts and nodes share one source.
"""

from __future__ import annotations

#: The reference configuration, mirroring `pe_core.py::PROFILES` exactly.
#: Nothing here may diverge from upstream; `tests/test_profiles_match_upstream.py`
#: reads that file and fails if it does.
REFERENCE: dict[str, dict] = {
    "t2i": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=1.5, max_tokens=16256),
    "edit": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                 presence_penalty=0.0, max_tokens=24000),
}

#: Deliberate divergences from the reference, each with the reason it exists.
#: The test above fails on any difference that is NOT declared here, so a drift
#: stays visible while a decision stays possible.
OVERRIDES: dict[str, dict] = {
    # Reasoned, the owner, 2026-09-20: raised to head off a truncated thinking
    # trace. Upstream's t2i cap is 16256 and a trace above it would be cut
    # mid-stream, which parses as invalid JSON and reads downstream exactly
    # like a quantization fault. Costs nothing when the trace is shorter.
    "t2i": dict(max_tokens=24000),
}

#: What the harness actually sends.
PROFILES: dict[str, dict] = {
    task: {**values, **OVERRIDES.get(task, {})} for task, values in REFERENCE.items()
}

# Measurement configuration. You cannot A/B a quant on sampled text: with
# sampling on, two runs of the SAME arm disagree. ComfyUI's TextGenerate exposes
# a real greedy path (`sampling_mode: "off"` -> do_sample=False), which is
# preferable to emulating determinism with a tiny temperature -- top_k=1 is
# already argmax, so the epsilon buys nothing and scales logits 100x before a
# softmax that no longer matters.
GREEDY: dict[str, dict] = {
    task: {**p, "temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0,
           "presence_penalty": 0.0}
    for task, p in PROFILES.items()
}
