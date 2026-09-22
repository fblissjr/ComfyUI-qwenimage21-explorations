# One expansion, stage by stage: code, owner, guard, reference

last updated: 2026-09-22

The cross-index. For each stage of turning a user's brief into a rewritten
prompt: **our code**, the **document that owns** it, the **check that would go
red** if it broke, and the **implementation to compare against** when you need
to know what the stage should do.

**Written by a person.** There is no generator here, and nothing regenerates
this table from the code — if a stage moves, this page goes stale silently.

**"nothing" in the guard column is the useful entry.** It is not a to-do list.
It is there so nobody mistakes an unguarded stage for a guarded one, and so
that adding a check is a decision someone takes rather than a gap someone
assumed was already filled.

Compare-against names the implementation whose reading is most useful at that
stage, not the one that is right. [`references.md`](references.md) says what
each checkout is and is not evidence of.

---

## Building the request

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| system prompt resolution | `templates.py::resolve`, the `PESystemPrompt` node | the module docstring; [`../../templates/README.md`](../../templates/README.md) | `tests/test_templates.py` covers the order, including where a server preset sits in it. Written 2026-09-20; this cell read **nothing** until then | `coderef/Qwen-Image-2.1/prompt_rewrite/pe_core.py::load_system_prompt`, which it mirrors deliberately |
| sampling settings | `profiles.py` | the module docstring | `tests/test_profiles_match_upstream.py` reads `PROFILES` back against `pe_core.py` with `ast`, and skips where `coderef/` is absent. *(Corrected 2026-09-22: this cell said nothing did; the test went in with `69c39a5` on 2026-09-20)* | `pe_core.py::PROFILES`. Both backends' own defaults differ from it — [`../quantization-strategy.md`](../quantization-strategy.md) section 17 |
| chat string | `chat.py::render_generation_prompt`, the `PEPrompt` node | the module docstring | `tests/test_chat.py` pins the two places ComfyUI's bundled template diverges from the trained format; `tests/test_prompt_structure.py` asserts on the final string rather than on the flags that produced it, which is the shape of check the predecessor's double-wrap bug needed | each checkpoint's own `chat_template.jinja` |
| reference image sizing | **not ours.** Core's `TextEncodeQwenImage21` does one shared resize, then core's encoder path resizes again | [`sizing.md`](sizing.md) | **nothing.** No check compares the two views, and by the time they could differ the count that would reveal it has been dropped | [`sizing.md`](sizing.md)'s posture table: two implementations make it impossible, two raise |
| image token accounting | `vision.py` | the module docstring; [`../quantization-strategy.md`](../quantization-strategy.md) section 25 for how the arithmetic was checked against a live server | `tests/test_vision.py`, pinned against values verified live | each checkpoint's `processor_config.json` |

## Generation

Two backends, and they differ in more than transport. `backends/__init__.py`
owns why there is no provider abstraction over them.

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| tokenization, ComfyUI path | **not ours.** Core's bundled qwen35 tokenizer | [`../quantization-strategy.md`](../quantization-strategy.md) section 29 | `tests/test_tokenizer_parity.py`, against each checkpoint's own tokenizer. It **xfails strictly**, so it turns into XPASS the day upstream honours the checkpoint's pre-tokenizer regex. Needs the checkpoints on disk | each checkpoint's `tokenizer.json`, which declares the regex core ignores |
| generation, ComfyUI path | **not ours.** Core's `TextGenerate` on `CLIP.generate`, driven by the string `PEPrompt` builds | [`../quantization-strategy.md`](../quantization-strategy.md) section 17 for what differs from the reference settings; section 26 for why its hardcoded stop set lands safely here | **nothing.** No test exercises this path, and the check that makes our string authoritative is a `startswith` test inside ComfyUI's tokenizer — `tests/test_prompt_structure.py` pins the properties that test depends on, not the test itself | `ComfyUI/comfy/text_encoders/qwen35.py` |
| generation, heylook path, in a graph | `PEExpand` (`QwenImage21PEExpand`), which folds resolve, request, parse and grade into one node | the module docstrings it calls | `tests/` cover every pure part it calls; **nothing** covers the node itself, which needs a live server | no reference implementation runs this stage at all — [`upstream.md`](upstream.md) section 4 |
| generation, heylook path | `backends/heylook.py::generate` | the module docstring; [`../quantization-strategy.md`](../quantization-strategy.md) section 17 | `tests/test_heylook.py` covers response normalisation and the truncation flag. **Nothing** covers the request, which needs a live server | the server's `/v1/capabilities` and its per-model `sampler_defaults`, which are what the harness exists to override |

## Reading the answer

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| thinking / body split | `answer.py::split_thinking` | the module docstring | `tests/test_answer.py`, including a trace left unclosed by a token cap | — |
| JSON parse | `answer.py::parse` | the module docstring | `tests/test_answer.py`: fences, prose either side, the field alias | the reference runners' output records, which write the field under a different name than the system prompts use |
| contract grade | `answer.py::grade`, the `PEParse` node | [`contract.md`](contract.md). The authority is each checkpoint's own `system_prompt.txt`, which is not in this repo | `tests/test_answer.py` covers the violation classes | **nothing to compare against**, and [`upstream.md`](upstream.md) section 4 is why: no engine that implements 2.1 wires an expander, so grading one's output is a stage nobody else runs |

## Downstream, and off the run path

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| conditioning from the rewritten prompt | core's `TextEncodeQwenImage21`, or ours: `EncodeStructured`, which opens its fixed parts and defaults to reproducing it | [`sizing.md`](sizing.md) section 7 for what ours exposes; [`../quantization-strategy.md`](../quantization-strategy.md) section 29 for the encoder's own tokenizer, checked because this is the path that actually reaches the image; [`../bridge-encoder-findings.md`](../bridge-encoder-findings.md) for what the predecessor learned on the older stack | `tests/test_tokenizer_parity.py` covers the encoder's tokenizer against the pipeline's own and found no divergence; `tests/test_encoder_prompt.py` pins the chat string ours builds. **Nothing** compares ours against core's output | `ComfyUI/comfy/text_encoders/qwen_image21.py`, and [`upstream.md`](upstream.md) for the four other implementations of this stage |
| checkpoint census | `scripts/config_census.py` | its own docstring, which carries why the output has the columns it has | its exit code is the guard, and the scan is exhaustive by construction — the docstring says why sampling by payload length does not work | — |
| end-to-end run | `scripts/smoke_heylook.py` | [`../quantization-strategy.md`](../quantization-strategy.md) section 25 | its exit code: it fails unless every completed row is contract-clean | upstream's example briefs, which it runs verbatim, typos and mixed languages preserved |

---

## Three facts that cut across every stage

**Sampled text cannot A/B a change.** With sampling on, two runs of the *same*
arm disagree, so a difference between two arms is not evidence about either.
`profiles.py::GREEDY` is the measurement configuration and owns the reasoning,
including why a real greedy path is preferred over emulating determinism with a
tiny temperature.

**The two backends are not one backend with two transports.** ComfyUI uses its
own bundled tokenizer and template and never reads the checkpoint's
`chat_template.jinja`; heylook applies that jinja server-side. What the harness
holds steady across them is the effective prompt and the sampling settings —
that is the whole job. `backends/__init__.py` owns this.

**The run records are gitignored.** `scripts/smoke_heylook.py` writes under
`data/`, which is not tracked. So the tracked home of any result from a run is
the section of [`../quantization-strategy.md`](../quantization-strategy.md)
that records it, not the JSONL — and a fresh checkout has the script and the
conclusion but not the rows.
