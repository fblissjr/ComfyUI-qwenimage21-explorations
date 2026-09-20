# ComfyUI-QwenImageWanBridge: the record of what failed

Mined read-only from the retired reference checkout. Nothing in that repo was modified.

**Repo root for every path below:**
`coderef/ComfyUI-QwenImageWanBridge`
All paths after this point are relative to that root.

**Convention used throughout.** Every "why" is tagged:
- **Stated** means the repo says it, and the quote is reproduced.
- **Reconstructed** means I inferred it, and the inference is named.
- **Record silent** means no reason exists in the checkout. I have not supplied one.

---

## 0. Read this first: what this checkout can and cannot tell you

**The reasons for archiving survived. The archived code did not.**

Four locations that the documentation cites repeatedly are not present:

| Cited at | Path named | State |
|---|---|---|
| `CLAUDE.md:19`, `:94`, `:605` | `nodes/archive/` (wrapper, c2c subdirs) | absent |
| `CLAUDE.md:21`, `:38`, `:39`, `:545`, `:613` | `internal/` | absent |
| `CLAUDE.md:89` | `explorations/20251003_diffsynth_spatial_token_analysis.md` | absent |
| brief | `llm-dit-experiments/` | present but **empty** |

`CHANGELOG.md:507` and `:525` describe moving nodes *into* `nodes/archive/`; that directory does not exist here. So the changelog is the only surviving account of the archived work. I cite the line that names each missing path, never the target as though I had opened it.

**Git history is unusable as evidence.** `git log` returns two commits (`98d9cbc fix`, `ac4e99e ComfyUI-QwenImageWanBridge - fresh history`). Everything predates a squash. This has a consequence that constrains the whole report: **a file being absent does not mean it was deleted for a reason.** Where the changelog or a code comment states a reason, I quote it. Where only absence exists, I say the record is silent rather than inventing a motive.

**No experimental results exist anywhere in this checkout.** See section 9. This is the single largest gap and it changes how you should read the prompting/template material.

---

## 1. The trap that would cost you the most: the double-wrapped chat template

**Stated.** `CHANGELOG.md:169-173`, v2.9.9:

> **Double-Wrapping Bug**
> - All Z-Image encoders were double-wrapping prompts with the chat template
> - Our nodes format the prompt, then ComfyUI's tokenizer wrapped it again
> - Now pass `llama_template="{}"` to bypass automatic wrapping
> - Affects: ZImageTextEncoder, ZImageTextEncoderSimple, ZImageTurnBuilder

This is the highest-value finding in the repo for a new Qwen-Image 2.1 harness. If you hand ComfyUI's CLIP tokenizer a string you have already wrapped in `<|im_start|>system ... <|im_end|>`, ComfyUI wraps it again. Nothing errors. The conditioning is silently wrong and the images are merely somewhat worse, which is indistinguishable from a bad prompt.

Three things follow:

1. `llama_template="{}"` is the escape hatch when you are formatting the chat template yourself.
2. It was introduced with the Z-Image encoders (`CHANGELOG.md:463`, v2.9.0) and fixed in v2.9.9 (`:169-173`). It survived that span because nothing surfaced the encoded string. (The changelog carries no dates, so how long that was in wall time is not recoverable from this checkout.)
3. The fix arrived in the same release that added a `formatted_prompt` / `debug_output` on every encoder (`CHANGELOG.md:162-167`). **Reconstructed:** the bug was found because the owner made the encoded text visible, not by reasoning about it. A `formatted_prompt` output is cheap and is the instrument that catches this class of bug.

Related, same class: `CHANGELOG.md:399-416` (v2.9.2) records that `template_preset` was accepted as a parameter but **never used** in Python — the node relied entirely on a JavaScript auto-fill to populate `system_prompt`, so if the JS failed (browser cache, error) the template silently did nothing. Stated at `:409-411`:

> Previously, `template_preset` was received but never used - relied entirely on JS auto-fill
> Now works even if JS fails (browser cache, errors, etc.)

**Lesson for the new harness:** any ComfyUI node whose behavior depends on a JS widget-fill has a silent-no-op failure mode. Put the fallback in Python.

---

## 2. Silent divergences from the reference implementations

These are not bugs. They are places where ComfyUI, diffusers and DiffSynth each do something different, nothing errors, and the outputs differ. Each one cost a release to find.

**Padding tokens.** `CHANGELOG.md:56-81`, v2.9.11. Stated, with the comparison table at `:69-73`:

| Implementation | Filters Padding? |
|---|---|
| diffusers (HuggingFace) | Yes |
| DiffSynth | Yes |
| ComfyUI (stock) | No |

> Both official reference implementations filter padding tokens before sending to the DiT. Our default now matches this behavior.

The mechanism is at `:78` — `embeddings[mask.bool()]`. Stock ComfyUI sends the full padded sequence plus mask. **If you are comparing your harness against a diffusers reference and the images differ, check this before anything else.**

**`enable_thinking` is inverted from what the name suggests.** `CHANGELOG.md:474-476`, v2.9.0, stated:

> - The `enable_thinking` parameter is counterintuitive:
>   - `enable_thinking=True` = NO think block (this is what diffusers uses)
>   - `enable_thinking=False` = ADD think block

And at `:477`: "So ComfyUI is NOT missing anything - both produce the same output." This is explicitly logged as a **corrected** analysis (`:471` heading "Key Finding (Corrected Analysis)"), i.e. the owner's first read of the situation was wrong. `nodes/docs/z_image_analysis.md:238-240` records the correction:

> **Original claim**: ComfyUI omits thinking tokens
> **Reality**: ComfyUI matches diffusers exactly (no thinking tokens in either)
> **Status**: No fix needed

**Do not retry:** "ComfyUI is dropping the thinking tokens." It is not. That was chased and refuted.

**The bundled tokenizer is the wrong tokenizer for `<think>`.** `CHANGELOG.md:492-494`, v2.9.0, under a heading that is itself the finding — "Known Gaps vs Diffusers (Cannot Fix)":

> - **Tokenizer difference**: ComfyUI bundles Qwen2.5-VL tokenizer; `<think>` becomes subwords not special token

`nodes/docs/z_image_analysis.md:209-212` adds the cost estimate for fixing it (copy `tokenizer_config.json`, modify the encoder) and the owner's own verdict: "This is complex and may not help." **Record is silent on whether it was ever attempted.** Treat "swap in the correct tokenizer to get `<think>` as one token" as unexplored, not as refuted.

---

## 3. Token dropping: what it is for, and the failure that produced it

**Stated.** `CHANGELOG.md:880-884` (v2.1) and `:861-878` (v2.2). The motivating failure, verbatim at `:883`:

> - **System prompt appearing in generated images** - Template formatting issue

The fix sequence, at `:865-867` and `:877-878`:

> - System prompt included during encoding for context
> - First 34/64 embeddings dropped after encoding
> - Prevents text contamination while maintaining quality
> ...
> - Token dropping happens after encoding (was missing entirely)

Live values are in `nodes/qwen_processor_v2.py:25-30` (the constants, not prose): `text_to_image: 34`, `image_edit / multi_image_edit / inpainting: 64`. `CLAUDE.md:569` adds the condition: "Only applied when system_prompt is provided."

Two traps here:

- Drop too few and the system prompt renders as literal text in the image.
- The drop index is **mode-dependent**, so if your vision-token placement and your drop index disagree you corrupt the conditioning. This is exactly the bug that forced the template-builder-to-encoder `mode` wire in v2.6.2, stated at `CHANGELOG.md:655`: "Prevents vision token placement/drop index mismatches."

**Reconstructed, and worth saying plainly:** the entire template-connection refactor in v2.7.0 (`CHANGELOG.md:606-635`, a declared breaking change that required users to delete and recreate nodes) exists because mode, system prompt and vision-token placement were three separately-wired values that could disagree. The owner's own words at `CHANGELOG.md:598`: "Old multi-connection system (mode + system_prompt) no longer works as it was getting convoluted and confusing." **If your new harness has a mode concept, make it one value, carried on one wire.**

---

## 4. The encoder-reuse trap, and the node that died on it

This is the most technically specific dead end in the repo and the one most likely to look attractive again.

**The idea.** Reuse the Qwen2.5-VL weights ComfyUI has already loaded as a CLIP, add only an `lm_head`, and get text generation (prompt expansion) without loading a second copy of the model. Stated at `nodes/hunyuan_video_prompt_expander.py:4-7`:

> Uses the already-loaded Qwen2.5-VL weights for LLM-based prompt expansion.
> Extracts weights from ComfyUI's CLIP and adds only the lm_head for generation.
>
> This approach reuses ~14GB of already-loaded weights, only loading ~50MB extra for lm_head.

**How far it got.** A full node (`nodes/hunyuan_video_prompt_expander.py`), plus a dedicated key-mapping module (`nodes/qwen_state_dict_mapping.py`), plus a documented generation config matching the official HunyuanVideo rewriter (`:22-29`).

**Why it stopped. Stated**, `CHANGELOG.md:548`:

> **HunyuanVideoPromptExpander Node** - not working, likely not going to pursue due to better alternatives

That is the whole stated reason. It does not say *what* broke. But `nodes/qwen_state_dict_mapping.py:421-489` documents an architectural blocker that would be sufficient, in a `GOTCHAS` string literal. The two load-bearing ones:

`nodes/qwen_state_dict_mapping.py:424-428`:

> ### 1. LM Head Missing in ComfyUI
> ComfyUI's Qwen implementation is encoder-only and does NOT include the lm_head.
> - Missing key: `lm_head.weight`
> - Solution: Tie weights with embed_tokens (they share weights in Qwen)

`nodes/qwen_state_dict_mapping.py:448-456`:

> **CRITICAL:** These are architecturally different!
> - ComfyUI uses SwiGLU (3 linear layers)
> - Transformers uses standard MLP (2 linear layers with GELU)
>
> The weight shapes are different:
> - ComfyUI: gate_proj and up_proj each have shape (hidden, intermediate)
> - Transformers: fc1 has shape (hidden, hidden_dim)
>
> **This means vision encoder weights may NOT be directly transferable!**

The same warning appears inline at `nodes/qwen_state_dict_mapping.py:177-180`.

**Reconstructed link:** the changelog does not say the vision-MLP mismatch killed the expander, and I am not claiming it did. But the two artifacts are consistent: a node built on "extract ComfyUI's weights into a transformers model" and a module documenting that the vision half of exactly that transfer is architecturally invalid.

**Portable warnings for a Qwen VLM harness:**
- ComfyUI's Qwen2.5-VL is **encoder-only**. There is no `lm_head` in the state dict. You cannot generate text from it without adding one.
- Prefixes differ: ComfyUI `model.layers.X.*` and `visual.*`; transformers `model.language_model.layers.X.*` and `model.visual.*` (`:458-464`).
- ComfyUI may carry `scaled_fp8` and `_quantization_metadata` keys that must be filtered before handing a state dict to transformers (`:484-488`).
- Layer counts to sanity-check against: language model 28 layers, vision encoder 32 blocks (`:471-473`).

---

## 5. Monkey-patching ComfyUI core: done, regretted, made opt-in

**Stated.** `CHANGELOG.md:518-522`, v2.8.2, closing GitHub issue #11:

> **Debug patches no longer applied by default** ([#11](...))
> - Removed automatic monkey-patching of `QwenImageTransformer2DModel._forward`
> - Debug patches are now opt-in via `QWEN_ENABLE_DEBUG_PATCHES=true` environment variable
> - Eliminates wrapper overhead on forward passes for all users

The patch code survives at `nodes/debug_patch.py` and patches `comfy.model_base.QwenImage.extra_conds` (`nodes/debug_patch.py:26-28`). The current gate is at `__init__.py:374-381`.

**Lesson:** the repo shipped a custom-node package that patched ComfyUI's sampling pipeline on import, for every user, by default. It took an external bug report to reverse it. If your new harness wants end-to-end tracing of reference latents, build it behind an env var from day one.

---

## 6. Two unbounded-growth bugs, same release

**Stated.** `CHANGELOG.md:527` (the wrapper DiT) and `:530-532` (the debug controller), both under issue #12:

> - Addresses VRAM leak concerns (unbounded RoPE cache, unmanaged model memory)

> **Debug controller RAM leak fixed** ([#12](...))
> - Replaced unbounded lists with `deque` ring buffers (auto-evict oldest entries)
> - Limits: 1000 execution logs, 500 memory snapshots, 200 errors, 100 perf entries/component

`CLAUDE.md:96` names the RoPE cache as the reason the custom DiT was archived: "Custom DiT implementation (`models/qwen_image_dit.py`) - unbounded RoPE cache". That file and directory are **absent from this checkout**; the reason is all that survives.

The ring-buffer limits now live as constants near `nodes/qwen_debug_controller.py:29`.

**Lesson:** a RoPE cache keyed on resolution grows without bound in a node graph where users change resolution per run. Both the debug collector and the DiT cache hit this. Bound anything you key on user-variable shapes.

---

## 7. The resolution double-scaling trap

Not a dead end, but a trap that cost two releases and is directly relevant.

**Stated.** `CLAUDE.md:116-123`, marked IMPORTANT in the source:

> **Two Separate Paths (IMPORTANT):**
> - **Vision Encoder (VL Model)**: 384x384 area-based scaling (hardcoded, not configurable)
> - **VAE Encoder (Generation Model)**: Configurable via `vae_max_dimension` parameter

Alignment differs between the two paths: 32-pixel for VAE, 28-pixel for the vision encoder (`CLAUDE.md:135-136`). Z-Image differs again at 16-pixel (`CLAUDE.md:524`).

Two recorded failures on this:

- **Zoom-out.** `CHANGELOG.md:669-676`, v2.6.1. Default `area_1024` scaling shrank large inputs, and the model then produced zoomed-out images. The default became `preserve_resolution`. The old mode is annotated at `:706-712` as "legacy, not recommended" with three explicit failure modes.
- **Double scaling.** `CHANGELOG.md:658-660`, v2.6.2. The batch node scaled, then the encoder scaled again. The fix was metadata propagation: `qwen_pre_scaled`, `qwen_scaling_mode`, `qwen_batch_strategy` (`:665`), so the encoder could tell it had already been done.

**Lesson:** if two nodes can both legitimately scale, one of them must be able to detect that the other already did. Tag the tensor.

---

## 8. Structured LLM output rendering as literal text in the image

**Stated.** `CHANGELOG.md:206-208`, v2.9.8:

> When prompts contain JSON-style formatting like `{"subject": "a cat", "style": "photo"}`, the double-quoted key names can appear as literal text in the generated image. This filter strips quotes to prevent text rendering in images.

Two mitigations were built, which tells you the first one was not enough:
1. `strip_key_quotes` toggle on the encoders, later widened from keys-only to all quotes (`CHANGELOG.md:197-198`).
2. Anti-text-rendering instructions injected into the `<think>` block of the `json_structured` / `yaml_structured` / `markdown_structured` templates (`CHANGELOG.md:103-107`): "Added explicit 'CRITICAL: Do NOT render any text, labels, keys' instructions".

**Highly relevant** if the new harness pipes an LLM's structured output into conditioning. The model will happily render your schema.

---

## 9. The largest gap: a full experimental program, designed, never recorded

This is the finding I would most want a successor to know.

The repo contains an elaborate, statistically-literate experimental design to answer one question: **do system prompts actually influence generation once the system tokens are dropped after encoding?**

- `experiments/EXPERIMENT_METHODOLOGY.md` (390 lines): four prioritized experiments, pre-registered expected results for both the "templates work" and "templates don't work" cases (`:103-133`), power analysis and required sample sizes (`:268-277`), an interpretation table for cosine similarity (`:288-295`), and explicit conclusion criteria (`:341-358`).
- `experiments/system_prompt_influence_experiments.py` — the implementation.
- `nodes/template_influence_analyzer.py` — an in-ComfyUI version, "This node tests whether system prompts actually affect embeddings" (`:4`), with six calibrated comparison templates including a deliberate horror/comedy semantic-opposite pair (`:33-40`).
- `nodes/docs/hunyuanvideo_prompting_experiments.md` — nine parts of A/B protocols, a scoring rubric (`:570`), and a "Results Documentation Template" (`:661-690`).
- `nodes/docs/attention_based_prompt_expansion.md` (459 lines) — a hypothesis that the encoder's own attention could substitute for running a separate 235B rewriter, with an A/B design and three pre-registered outcome bands (`:312-331`).

**No results file exists anywhere in the checkout.** I searched for result/output artifacts and found none; the only `Observations` headings in the repo are the empty template at `nodes/docs/hunyuanvideo_prompting_experiments.md:682` and a placeholder at `nodes/docs/experimental/hunyuanvideo_nsfw_boundary_user_prompts.md:457`. `nodes/docs/z_image_turbo_workflow_analysis.md:69` is an analysis of an official workflow, not an experiment result.

**Record is silent on why.** There is no note saying the experiments were run, abandoned, or found inconclusive. Note the precise scope of the claim: no *recorded* results exist **in this checkout**. `internal/` is named at `CLAUDE.md:21` as the home for "Internal documentation and analysis" and is absent, so results may have lived there. That does not weaken the finding for your purposes, since you cannot act on results you do not have.

Two consequences, and they point in opposite directions:

1. **Do not assume the template system was validated.** The repo ships 144 Z-Image templates (`CLAUDE.md:537`) and 39 HunyuanVideo templates (`CLAUDE.md:357`) on top of a mechanism whose effectiveness the owner built an entire apparatus to test and then did not record testing. `experiments/EXPERIMENT_METHODOLOGY.md:382-390` even pre-committed to the outcome: "Consider removing template system (simplify codebase)."
2. **The design itself is reusable and is the best thing in the repo for you.** If the new harness needs to answer "does this system prompt do anything," the methodology document is a finished protocol. **Reconstructed judgment, not stated:** running it is probably higher value than any code port.

Related and unresolved in the same way: `nodes/docs/attention_based_prompt_expansion.md:281` states the core doubt honestly and never resolves it:

> This approximation may or may not hold. The DiT learned to decode **explicit** detail embeddings, not **implied** detail embeddings.

---

## 10. Things marked broken that were never fixed

Every marker below is quoted from the checkout. Grouped by what kind of signal it is, because the task rightly distinguishes them.

### 10a. Tried and did not work

**Mask-based inpainting (QwenMaskProcessor + QwenInpaintSampler).** The most heavily-marked feature in the repo. Markers at `CLAUDE.md:140`, `:218`, `:291`, all reading "NOT YET WORKING / TESTED", and `nodes/docs/QwenInpaintSampler.md:6-7`:

> **STATUS: EXPERIMENTAL - NOT FULLY TESTED AND LIKELY NOT WORKING**
> Inpainting workflow is experimental and may not work as expected or at all. There's likely better alternatives out there.

Known issues are enumerated at `nodes/docs/QwenInpaintSampler.md:205-208`: overcomplicated, padding crop may cause shape errors, multiple 5D tensor conversions, and a simpler alternative exists. It is still registered and loads (`__init__.py:117-118`). See also 10c.

**Spatial coordinate tokens (QwenSpatialTokenGenerator).** `README.md:202`, stated:

> - **QwenSpatialTokenGenerator**: Visual editor for spatial tokens that don't seem to do much of anything right now

Marked deprecated at `CLAUDE.md:92`. The stated technical reason is at `CLAUDE.md:89` and `:612` — the tokens exist in the tokenizer (`<|box_start|>`, `<|quad_start|>`, `<|object_ref_start|>`; see `nodes/qwen_token_debugger.py:24-31`) but DiffSynth does not use them, so the model has no training signal for them. `CHANGELOG.md:913-917` logs the status as "Experimental, low priority - spatial tokens effectiveness unclear". The analysis doc it points to (`explorations/20251003_diffsynth_spatial_token_analysis.md`, named at `CLAUDE.md:89`) is **absent**.

The scale of this dead end is worth noting: `nodes/qwen_spatial_token_generator.py` is the largest node file in the repo, and two of the five archived JavaScript files (`web/js/archive/qwen_spatial_interface.js`, `qwen_spatial_mask_interface.js`) are its UI. Substantial investment, explicit negative result.

**Do not retry:** driving Qwen-Image spatial control through coordinate tokens in the prompt. Use masks.

**HunyuanVideoPromptExpander.** See section 4. Stated dead at `CHANGELOG.md:548`.

**Multi-frame Wan bridge.** See section 11.

### 10b. Built, never validated, then archived

This is its own category and the changelog is unusually candid about it.

**The 11 wrapper nodes** (`CHANGELOG.md:756-791`, v2.5). An entire parallel implementation: transformers-based DiT loader, VL loader, VAE loader, processor, 2x2 patch packing following DiffSynth's `model_fn_qwen_image`, a FlowMatch sampler with resolution-aware dynamic shift, edit-latent injection bypassing ComfyUI's conditioning system. No DiffSynth dependency. The section heading itself is the verdict: "## v2.5 update - Wrapper Nodes (NOT WORKING YET - Experimental)".

Status, stated at `CHANGELOG.md:789-791`:

> - **Not fully tested** - Wrapper nodes are complete but haven't been validated with actual models
> - **Use Standard Nodes** - Recommended for production use

Archived three minor versions later, `CHANGELOG.md:524-527`, for VRAM leaks. `CLAUDE.md:95` restates: "Wrapper nodes (transformers/diffusers) - had VRAM leak issues, not production ready."

**Reconstructed:** a complete alternative implementation was written, never run against real weights, and then retired on a memory-management concern. Note the ordering — it was archived for a problem found by inspection, not by testing.

**Lesson, and it is the most expensive one here in engineer-hours:** the repo tried twice to go around ComfyUI's native Qwen support (the wrapper stack, then the custom DiT) and both times ended up recommending the native path. `CLAUDE.md:603` is the settled position: "Dependencies: None"; `CHANGELOG.md:528`: "Main workflow uses ComfyUI's native Qwen support - no wrapper nodes needed." `CHANGELOG.md:949` lists "Native implementation attempts (wrapper approach preferred)" under Removed/Archived, which read alongside `:524` means both directions were tried and both were dropped.

**EliGen entity control.** Marked untested at `CLAUDE.md:88`, `:91`, `:158`, and `CHANGELOG.md:919-923` ("Experimental, untested with current models"). The stated blocker at `CLAUDE.md:155-159` is architectural, and it is a genuinely useful warning:

> **DiffSynth Alternative (Not implemented and no plans to do so):**
> - EliGen uses attention masking INSIDE the DiT (requires model access)
> - Multi-entity with isolated attention per region
> - QwenEliGenEntityControl node exists but untested
> - Would need ComfyUI DiT integration (may not be possible)

**Do not retry casually:** per-region attention masking inside the DiT from a ComfyUI custom node. The owner assessed it as possibly impossible and did not attempt it.

### 10c. Worked, but was not worth it

**QwenInpaintSampler as an engineering decision**, separate from whether it functions. `nodes/docs/QwenInpaintSampler.md:22`:

> **This is a 548-line implementation for a simple blending operation.** Consider using standard KSampler + LatentCompositeMasked instead for simpler workflows.

Repeated at `CLAUDE.md:614` and `nodes/docs/QwenInpaintSampler.md:224`. The blend is a direct port of the diffusers formula `final = (1-mask)*original + mask*generated` (`CHANGELOG.md:738`). The verdict is that standard ComfyUI nodes achieve the same result.

**Do not retry:** porting a diffusers pipeline wholesale when the useful part is one line of tensor arithmetic. Check what ComfyUI already composes.

**QwenMultiReferenceHandler.** `nodes/qwen_multi_reference.py:60-62`, stated in the node's own `DESCRIPTION`:

> """[DEPRECATED] Use Image Batch node instead.
> This node is kept for backward compatibility.
> For new workflows, use Image Batch to combine multiple images."""

`TITLE` was changed to `[DEPRECATED] Multi-Reference Composer` (`:59`), it is unregistered at `__init__.py:143`, and the file is still on disk. It was replaced by `QwenImageBatch`, which itself exists because of a stated third-party failure at `CHANGELOG.md:658`: "KJNodes ImageBatchMulti black image issue (empty inputs)".

**C2C Vision Bridge.** `CHANGELOG.md:503-511`, v2.8.3, stated:

> **C2C Vision Bridge nodes archived**
> - Moved to `nodes/archive/c2c/` (3 nodes: QwenC2CBridgeLoader, QwenC2CCacheExtractor, QwenC2CVisionEnhancer)
> - Feature was experimental and not providing practical value

Cleanest "worked but wasn't useful" signal in the repo. The code is gone (that archive path does not exist here); the names suggest a KV-cache extraction and vision-enhancement path, but **the record does not say what C2C stood for or what it did**, and I am not going to guess.

**Five JavaScript interfaces.** `CHANGELOG.md:265-273`, v2.9.6, moved to `web/js/archive/` — `qwen_spatial_interface.js`, `qwen_spatial_mask_interface.js`, `qwen_testing_interface.js`, `qwen_token_analyzer.js`, `qwen_token_visualizer.js`. Stated reason, `:272`: "Eliminates deprecation warnings from ComfyUI". These are the UIs for the spatial-token and token-analysis work. Substantial code, retired because the ComfyUI frontend API moved under them.

**Lesson:** custom ComfyUI JS front-ends are a depreciating asset. The repo's surviving JS is small and does one thing (template auto-fill).

### 10d. Scope, not failure — do not upgrade these to failures

Three comments in `__init__.py` are scope statements. They say nothing about whether the code worked:

- `__init__.py:139`: `# Qwen-WAN Bridge nodes removed - not related to Qwen-Image-Edit`
- `__init__.py:141`: `# V2V nodes removed - not core functionality`
- `__init__.py:143`: `# Multi-Reference handler deprecated - use Image Batch node instead`

`CHANGELOG.md:945-950` lists alongside these: "WAN keyframe nodes", "Video-to-video nodes", "Dual encoding architecture (overcomplicated)", "Native implementation attempts (wrapper approach preferred)", "Inpainting nodes (broken, low priority)". Only the last carries a failure verdict. **"Dual encoding architecture (overcomplicated)" is the entire surviving record of that attempt** — no file, no further note.

### 10e. Dead code: two debug tools that cannot load

**Verified by static evidence, not by running anything** (ComfyUI's `folder_paths` and `comfy.*` are not importable outside ComfyUI, so an import test would prove nothing).

`nodes/qwen_validator.py:9-10` imports:

```python
from .qwen_config import QwenConfig
from .qwen_logger import QwenLogger
```

`nodes/qwen_config.py` and `nodes/qwen_logger.py` **do not exist in this checkout**. A repo-wide grep finds importers of both and a definition of neither. `nodes/qwen_debug_controller.py:19-24` imports `QwenLogger` inside a `try/except ImportError` whose fallback is a bare `from qwen_logger import QwenLogger` — the same missing module, so both branches fail.

Consequence: **QwenDebugController cannot import in this checkout**, and `qwen_validator.py` is unreachable dead code (nothing imports `ReferenceValidator`). Yet `CLAUDE.md:621` lists QwenDebugController under Debug Features and `README.md:197` lists it as a helper node.

**Do not conclude it was always broken** — `qwen_config.py` / `qwen_logger.py` may have existed before the history squash and been dropped from the reference copy. The record is silent.

**The portable lesson is the failure mode, not the missing file.** `__init__.py` wraps every single node registration in `try/except Exception` and prints `[QwenImageWanBridge] Failed to load X: {e}` (pattern repeated at `__init__.py:29`, `:39`, `:56`, `:68`, `:90`, `:102`, `:120`, `:137`, `:161`, `:180`, `:192`, and more). A node that is entirely broken degrades to one line in a console nobody reads, while the documentation continues to advertise it. If your new harness uses this pattern, add a summary at the end of `__init__.py` that reports the count of failed registrations loudly, or make load failures fatal in development.

---

## 11. The Wan bridge attempts (lower value — compact by request)

Nine research files in `nodes/research/`, **all dated the same day (2025-08-19)**. This was one burst of work, not a months-long sequence. There are no commits to order them by, so the sequence below is **reconstructed from their module docstrings**, which describe each other's failures.

The central technical problem, stated at `nodes/research/qwen_wan_diagnostic.py:2`:

> Diagnostic node to understand why 1 frame works but multiple frames don't

That framing appears again at `nodes/research/qwen_wan_minimal_bridge.py:3` ("Focus on why single frame works but multi-frame doesn't"). The problem was never resolved in this checkout.

| File | The idea | What it changed |
|---|---|---|
| `qwen_wan_bridge_v2.py:1-4` | "Exact replication of `WanVideoImageToVideoEncode` / But using Qwen latents directly instead of VAE encoding" | Copy Kijai's wrapper logic verbatim, substitute the latent source |
| `qwen_wan_minimal_bridge.py:1-3` | "Preserve first frame exactly" | Strip out all normalization. `:60` comment: "CRITICAL: Preserve first frame EXACTLY as Qwen provides" |
| `qwen_wan_native_bridge.py:1-3` | "Works with standard KSampler, not Kijai's wrapper" | Change the consumer, not the bridge |
| `qwen_wan_t2i_bridge.py:11-14` | Treat Qwen output as if it were Wan's own T2I output | Reframe the latent's provenance; adds `direct` / `normalized` / `scaled` modes |
| `qwen_wan_t2v_bridge.py:11-13` | "T2V models are more flexible and might handle the latent better" | Abandon I2V; inject the Qwen latent as a target or init noise for T2V |
| `qwen_wan_unified_bridge.py:12-15` | Align latents **and** text embeddings (Qwen CLIP to Wan UMT5) | Escalates from latent-space to joint latent-plus-text alignment |
| `qwen_wan_diagnostic.py:10-12` | "Test incrementally: 1, 5, 9... frames to find breaking point" | Stop building; instrument |
| `qwen_wan_parameter_sweep.py:2-3` | "Find the right combination that actually works" | Stop reasoning; brute-force denoise/CFG/sampler/resolution |
| `qwen_wan_whisk_bridge.py:9-17` | Google Whisk pattern: image to VLM to editable text to video. "This avoids direct latent transfer issues entirely!" | **Give up on latent transfer.** Go through natural language instead |

**Reconstructed sequence:** replicate exactly, then strip everything, then change the consumer, then reframe the latent, then change the model class, then escalate to joint text alignment, then instrument, then sweep, then abandon the approach entirely and route through text. Note that `qwen_wan_whisk_bridge.py:50` is a stub — `"[VLM would analyze image here with prompt: {}]"` — so even the give-up option was never finished.

**Why it stopped. Stated**, `__init__.py:139`: "Qwen-WAN Bridge nodes removed - not related to Qwen-Image-Edit". That is a scope statement. The technical failure (multi-frame) and the scope decision are two separate things and the record gives one sentence for each. Nothing says the multi-frame problem was solved.

The surviving production Wan bridge (`nodes/qwen_wan_bridge.py`, registered at `__init__.py:78-88`) is still marked `CLAUDE.md:207`: "### Wan Video Bridge (QwenWanBridge) - EXPERIMENTAL AND LIKELY NOT USEFUL YET".

**Transferable beyond Wan, and this is the only part worth carrying:** when nine variants of the same bridge exist, the shape of the sequence is the finding. Replicate-exactly and strip-everything are the two cheap probes; if neither works, instrument before building variant three. And the escape hatch that the owner reached last — go through natural language instead of trying to transfer latents between two models' spaces — was reachable from the start.

---

## 12. What the debug tooling was built to catch

Four instruments, four bug classes. For each, the bug is **stated** unless noted.

**`nodes/debug_patch.py` — reference latents vanishing between encoder and model.** Stated at `:2`: "Debug patch for ComfyUI sampling pipeline - traces reference latents end-to-end". It patches `comfy.model_base.QwenImage.extra_conds` (`:26-28`) and looks for a `trace_id` the encoder stamped onto the conditioning (`:38-40`). **Reconstructed:** you build a trace-ID mechanism when conditioning is arriving at the model in a state you cannot explain from the encoder's side. `CLAUDE.md:611` and `CHANGELOG.md:658-660` both describe latents being silently transformed between nodes.

**`nodes/qwen_token_debugger.py` — special tokens not being what you think they are.** The file's core is a hard-coded token-ID table (`:16-50`) covering vision, spatial, chat, control, FIM and tool tokens. **Reconstructed:** you hard-code `<|vision_start|> = 151652` when you have been burned by a token that tokenized into subwords instead of a single special token. This is the same class of bug as the `<think>` subword finding at `CHANGELOG.md:494` and the double-wrapping bug at `:169`. **Bug class: you cannot see the token stream, so you cannot tell a correct prompt from a malformed one.** Note `README.md:199` files this node under "Available but likely not working or deprecated".

**`nodes/qwen_validator.py` — prompts referring to images that are not there.** Stated at `:1-4`: "Validates Picture references in prompts". It extracts `Picture N` references and checks N against the number of images actually supplied (`:51-80`), with modes off/warn/error/verbose. **Bug class:** the multi-image format is `Picture 1: <|vision_start|>...` (`nodes/qwen_processor_v2.py:71`), so a prompt saying "the woman in Picture 3" with two images supplied silently refers to nothing. Related guidance at `CLAUDE.md:271-272` recommends describing subjects semantically rather than by image number. Dead code — see 10e.

**`nodes/qwen_debug_controller.py` — no single place to see what any of it was doing.** Stated at `CHANGELOG.md:844-856` (v2.3): multi-level debugging, performance profiling, memory tracking, component filtering, JSON export. Its own fix at `:530-532` is the RAM leak from section 6. **Bug class: observability across a node graph.** Also `:858-859`, a real ComfyUI-specific trap — the first version spammed the console on every run, and silencing it by default was itself logged as a fix: "Debug patches now run silently unless explicitly enabled / No more console spam during normal operation."

**Summary for the new harness.** The four instruments correspond to: (1) conditioning mutating between nodes, (2) the token stream being invisible, (3) prompt-to-image-count mismatch, (4) no cross-node observability. Of these, **(2) is the one I would build first** — it is the instrument that eventually caught the most expensive bug in the repo (section 1), and it is cheap: a `formatted_prompt` string output on every encoder.

---

## 13. Two reversals in the record — both sides cited, neither resolved

The changelog is a contemporaneous log, not a settled account. It contradicts itself in two places where understanding changed. A reader should know each question was answered twice, differently.

**`ZImageTextEncoderSimple` was removed as redundant, then re-added.**
- Removed, `CHANGELOG.md:351-354` (v2.9.4): "Redundant - same functionality as ZImageTextEncoder with `template_preset='none'` / Use ZImageTextEncoder instead (matches diffusers by default)".
- Re-added five patch versions later, `CHANGELOG.md:156-160` (v2.9.9): "Simplified encoder for quick use - ideal for negative prompts ... Lighter weight for simple encoding tasks".
- Present in the final `CLAUDE.md:440`.

**The 512-token limit was removed as unnecessary, then reinstated as a warning.**
- Removed, `CHANGELOG.md:385-388` (v2.9.3): "ComfyUI natively handles unlimited context (`max_length=99999999`) / Qwen3-4B supports 40K tokens (`max_position_embeddings: 40960`) / The 512 limit was unnecessarily restrictive".
- Reinstated as a soft warning in the final release, `CHANGELOG.md:8-15` (v2.9.12): "Warning when exceeding 512 tokens (reference implementations truncate at 512)" and, at `:15`, "ComfyUI allows longer sequences, but results may differ from reference implementations for prompts exceeding 512 tokens."

**The reconciliation is real and worth internalizing:** both statements are true. ComfyUI can encode long sequences; diffusers and DiffSynth truncate at 512. So a long prompt works in ComfyUI and is not reproducible against the reference. If your new harness cares about matching a reference implementation, 512 is a real boundary even though nothing enforces it.

---

## 14. Hard-won lessons stated in the source

Items not covered above where the repo explains a non-obvious choice in its own words.

**Why face swapping fails, with a mechanism.** `nodes/docs/qwen_prompt_cookbook.md:298-303`:

> The model is trained for:
> 1. **Identity preservation** (keeping the same face)
> 2. **Style transfer** (changing everything BUT the face)
> 3. **Multi-image composition** (placing people in scenes)
>
> Face swapping is the INVERSE of identity preservation, which explains poor results.

The workaround, `:305-310`, is to replace the whole person rather than the face. Face-only replacement is rated "Success rate: Very low, random" (`:291-294`). This is a good example of a negative result with a stated mechanism rather than just an observation.

**Capabilities the model does not have.** `nodes/docs/qwen_prompt_cookbook.md:209-230`, stated: cannot sharpen blurry images or remove out-of-focus blur (and "Prompts like 'in sharp focus' or 'crisp details' don't help"); struggles with true worm's-eye view; cannot reliably convert a low-angle shot to eye-level. Also `:234`: Chinese prompts often give finer control than English.

**Z-Image CFG.** `CLAUDE.md:471`, stated and non-obvious:

> **Important:** Z-Image uses Decoupled DMD - CFG is baked in during training. Use CFG=1.0 at inference (no guidance scaling). Negative prompts have no effect at CFG=1, so use `ConditioningZeroOut` instead of encoding text.

Encoding a negative prompt for a CFG=1 model is wasted work that looks like it is doing something.

**Z-Image multi-turn is off-distribution.** `nodes/docs/z_image_analysis.md:230`: "Z-Image is primarily trained on single-turn prompts. Multi-turn is experimental." Note the repo built a whole `ZImageTurnBuilder` chain anyway (`CLAUDE.md:446-457`).

**Thinking and assistant content are weighted below the user prompt.** `README.md:58`, stated as an observation under an example image:

> replacing a class (sloth instead of cat) often results in a mix - thinking/assistant are weighted lower than user prompt

**Deep-copy conversation state.** `CHANGELOG.md:341-343`, v2.9.4, labelled Critical:

> **Critical: Shallow Copy Bug in ZImageMessageChain**
> - `list()` creates new list but message dicts were still shared references
> - Now uses `copy.deepcopy()` to prevent conversation corruption

Live at `nodes/z_image_encoder.py:409` ("Deep copy to avoid shared references"). In a ComfyUI graph, one conversation object can be consumed by several downstream nodes; a shallow copy lets one branch mutate another's history.

**Empty dict passes a `is not None` check.** `CHANGELOG.md:345-347`, same release. An optional `conversation_override` arriving as `{}` was treated as present. Small, but exactly the kind of thing ComfyUI's optional-input model produces.

**Debug output rendered as markdown and hid the thing being debugged.** `CHANGELOG.md:248-252` (v2.9.6) and `:175-177` (v2.9.9): `<think>` tags were being swallowed by HTML rendering in Preview nodes, so the debug output looked like the tags were missing when they were present. Fixes: escape angle brackets, wrap in a code fence, add an explicit "Think Tag Check: Contains '<think>': True/False" line, and log to the server console as well (`CLAUDE.md:624-626`). **A debugging tool that lies is worse than none.**

**VAE substitution is not free.** `CLAUDE.md:523`, stated: "Using non-official VAEs (like Wan2.1-upscale2x) is experimental - same tensor shape but different scaling factors may cause color shifts." Same tensor shape, different `scaling_factor` / `shift_factor`, silent color shift. `ZImageWanVAEDecode` exists as an explicit experiment in this (`CLAUDE.md:86`, `README.md:200`) and is marked "for testing only".

---

## 15. Condensed: do-not-retry list for a Qwen-Image 2.1 harness

Ordered by what it would cost to rediscover.

1. **Do not let ComfyUI's tokenizer wrap a prompt you already wrapped.** Pass `llama_template="{}"`. Expose a `formatted_prompt` output so you can see it. (`CHANGELOG.md:169-173`)
2. **Do not assume stock ComfyUI matches diffusers.** Padding filtering differs (`CHANGELOG.md:56-81`); the bundled tokenizer differs (`:494`); the reference truncates at 512 and ComfyUI does not (`:15`).
3. **Do not chase "ComfyUI omits the thinking tokens because `enable_thinking` is off."** Investigated and refuted: the flag is inverted, and ComfyUI matches diffusers. (`CHANGELOG.md:474-477`, `nodes/docs/z_image_analysis.md:238-240`)
   - **Separate and still open:** ComfyUI's bundled Qwen2.5-VL tokenizer splits `<think>` into subwords rather than emitting one special token. Logged as a known gap, never fixed, and the owner judged a tokenizer swap "complex and may not help." Unexplored, not refuted. (`CHANGELOG.md:492-494`, `nodes/docs/z_image_analysis.md:209-212`)
4. **Do not try to reuse ComfyUI's loaded Qwen2.5-VL as a generative LM.** Encoder-only, no `lm_head`, and the vision MLP is architecturally incompatible with transformers. (`nodes/qwen_state_dict_mapping.py:421-489`)
5. **Do not drive spatial control through coordinate tokens.** The tokens exist; the model was not trained on them via this path. Use masks. (`README.md:202`, `CLAUDE.md:89`)
6. **Do not port a diffusers pipeline wholesale for a one-line blend.** (`nodes/docs/QwenInpaintSampler.md:22`)
7. **Do not build a wrapper stack around ComfyUI's native Qwen support.** Tried twice, retired twice. (`CHANGELOG.md:524-527`, `:949`)
8. **Do not monkey-patch ComfyUI core on import.** Gate it behind an env var from the start. (`CHANGELOG.md:518-522`)
9. **Do not key an unbounded cache on user-variable shapes** (RoPE by resolution, debug logs by run). (`CHANGELOG.md:527`, `:530-532`)
10. **Do not let two nodes both scale.** Tag the tensor when one has. (`CHANGELOG.md:658-660`, `:665`)
11. **Do not rely on a JS widget-fill for behavior.** Put the fallback in Python. (`CHANGELOG.md:169-171` of v2.9.2 section)
12. **Do not swallow node registration failures into a console line.** (`__init__.py`, the repeated `try/except Exception` pattern; consequence visible at 10e)
13. **Do not pipe raw structured LLM output into conditioning** without stripping quotes. The schema renders into the image. (`CHANGELOG.md:206-208`)
14. **Do not treat the 144 Z-Image / 39 HunyuanVideo templates as validated.** The experiment designed to validate them has no recorded results. (section 9)

---

## 16. Where the record is silent

Stated here so these are not mistaken for gaps in the search.

- **Why the prompt expander stopped.** "not working" and "better alternatives" (`CHANGELOG.md:548`); no failure mode given.
- **Whether the multi-frame Wan problem was ever diagnosed.** Diagnostic and sweep tools exist; no results.
- **What C2C was.** Three node names and "not providing practical value" (`CHANGELOG.md:507-510`); code absent.
- **What the "dual encoding architecture" was.** One word, "overcomplicated" (`CHANGELOG.md:948`); no file, no further note.
- **Whether any template-influence experiment was ever run.** Full design, no results. (section 9)
- **Whether `qwen_config.py` / `qwen_logger.py` ever existed.** They are imported and absent; history is squashed. (10e)
- **What `llm-dit-experiments/` held.** The directory is present and empty. Named in the brief, so stated explicitly rather than omitted.
- **Why the `## Implementation Decisions` section of `CLAUDE.md:607` is empty.** A heading with no body, immediately followed by `## Known Issues`.
