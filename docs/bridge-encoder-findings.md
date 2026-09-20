# Bridge encoder findings: Qwen VLM as a conditioning text encoder

Mined from the retired `ComfyUI-QwenImageWanBridge` reference checkout, encoder/conditioning
side only. Read-only pass; nothing in the source repo was modified.

**Repo root for all citations below:**
`coderef/ComfyUI-QwenImageWanBridge/`
All `path:line` pointers are relative to that root.

**Target it was built for:** Qwen-Image-Edit-2509 with Qwen2.5-VL-7B (3584-dim).
**Target we are building for:** Qwen-Image 2.1 with Qwen3-VL-8B.
Treat every finding as describing the older stack unless the "Applies to 2.1?" line says otherwise.

---

## Read this first: the evidence base is thinner than it looks

The repo reads like it has an experimental record. It does not. Calibration up front,
because it changes how you should weight everything below.

**`experiments/EXPERIMENT_METHODOLOGY.md` is a plan, not results.** It is a well-built
experimental design for whether system prompts survive token dropping, with metrics,
power analysis and conclusion criteria. Every number in it sits under a heading that
reads `### Expected Results`, split into "If Templates WORK" and "If Templates DON'T
WORK" blocks (`experiments/EXPERIMENT_METHODOLOGY.md:103-133`, `:164-188`). Those are
hypothesised tables. They are not observations.

The experiments were never run, or at least never recorded here:

- The runner exists (`experiments/system_prompt_influence_experiments.py`,
  `experiments/quick_embedding_test.py`) and is complete enough to execute.
- The declared output directory `experiment_outputs/` with
  `embedding_analysis_results.json` (`experiments/EXPERIMENT_METHODOLOGY.md:330-337`)
  does not exist anywhere in the checkout. No results JSON of any kind exists.
- `llm-dit-experiments/` is an empty directory.

Do not quote a cosine similarity or an L2 distance from that document as if the bridge
measured it. Nobody did.

**Evidence labels used below:**

| Label | Meaning |
|---|---|
| Logged | Console output from a real run, pasted into a doc |
| Observed | A concrete failure or fix described with specific inputs, in a dated changelog entry |
| Reasoned | An argument from a reference implementation, with the reasoning visible |
| Asserted | A claim in a docstring or tooltip with no visible derivation |
| Unmeasured (self-declared) | The repo itself says it does not know |

**Second calibration point: the docs drifted off the code.** Several encoder docs
describe a parameter surface the final code does not have. `nodes/docs/QwenVLTextEncoder.md:47-52`
and `nodes/docs/resolution_tradeoffs.md:122-133` document `scaling_mode`
(`preserve_resolution` / `max_dimension_1024` / `area_1024`) and `batch_strategy`
(`max_dimensions` / `first_image`). Neither exists in the shipped code. The encoder has
`vae_max_dimension` (`nodes/qwen_vl_encoder.py:168-179`) and the batch node has
`batch_alignment` with `match_smallest` / `match_first` / `match_largest`
(`nodes/qwen_image_batch.py:85-111`). `nodes/docs/QwenVLTextEncoder.md:10` also still
instructs connecting separate `mode` and `system_prompt` outputs, which v2.7.0 replaced
with a single `template_output` (`CHANGELOG.md:606-614`). Where a finding comes from a
doc, I say which generation of the code it describes.

---

## Finding 1: The token-drop index is a hardcoded integer coupled to a system prompt the user can freely edit

**What they found / built.** Conditioning is produced by encoding the full chat template
including the system turn, then slicing a fixed number of embeddings off the front of the
sequence. The indices are a hardcoded dict:

```
"text_to_image": 34,  "image_edit": 64,  "multi_image_edit": 64,  "inpainting": 64
```
(`nodes/qwen_processor_v2.py:25-30`)

Applied post-encoding by slicing the sequence dimension (`nodes/qwen_vl_encoder.py:487-500`),
which is the DiffSynth order: encode with the system turn present so the user tokens attend
to it, then drop (`experiments/EXPERIMENT_METHODOLOGY.md:11-36`).

**The structural problem, which the repo never names.** 34 and 64 are token counts of
DiffSynth's *specific* system prompt strings. But the gate for applying them is merely
"is `system_prompt` non-empty":

```python
if self.processor and system_prompt:
    formatted_text = self.processor.format_template(text, system_prompt, vision_tokens)
    drop_idx = self.processor.get_drop_index(mode)
```
(`nodes/qwen_vl_encoder.py:455-459`)

Nothing re-derives the index from the actual tokenized prefix. And the Template Builder
hands users arbitrary system text. The shipped templates already break the assumption:
`nodes/templates/default_edit.md:6` carries the long DiffSynth edit prompt, while
`nodes/templates/minimal_edit.md:6` is one short sentence ("Make only the specific changes
requested. Preserve all other aspects of the original image exactly."). Both declare
`mode: image_edit`, so both get `drop_idx = 64`. A shorter system turn means the slice runs
past the system tokens and eats the head of the user turn; a longer one leaves system
tokens in the conditioning. Silently, with no warning path. The only guard is a
"not enough tokens to drop" log (`nodes/qwen_vl_encoder.py:499-500`).

Worth noting the inverse case is handled inconsistently too: with no system prompt the code
sets `drop_idx = 0` and skips dropping entirely (`nodes/qwen_vl_encoder.py:460-465`), with a
comment conceding DiffSynth always has a template so always drops.

**Evidence class.** The coupling is **demonstrated in shipped data**, not hypothesised: two
templates in the repo declare the same `mode: image_edit` with visibly different system
prompt lengths (`nodes/templates/default_edit.md:6` vs `nodes/templates/minimal_edit.md:6`),
and both route to `drop_idx = 64` through a gate that tests only non-emptiness. The divergence
is real and present in the released templates. What is uninvestigated is the downstream effect
on generated output, which the repo never examined. The origin of dropping at all is Observed:
v2.1 fixed "System prompt appearing in generated images" (`CHANGELOG.md:880-884`), and
v2.2 added dropping because it "was missing entirely" (`CHANGELOG.md:861-878`). So the
failure mode dropping prevents is real and was seen in output.

**Applies to 2.1?** The *hazard* is designed out, and that is the lesson. In
`comfy/text_encoders/qwen_image21.py` the template is fixed in code and the system turn is
removed structurally rather than by a magic integer, so there is no user-editable string that
can desynchronise a slice. Do not port 34/64. They are artifacts of a specific prompt string
for a different model's tokenizer.

**What it means for us.** If our harness ever exposes a configurable system prompt, the drop
boundary must be derived, not constant: tokenize the prefix and slice by measured length, or
better, keep 2.1's structural approach and never let the boundary be a number at all. If we
compare our conditioning against a reference implementation and see a constant offset in
sequence length, this class of bug is the first thing to check.

---

## Finding 2: Vision tokens were only ever counted as marker strings, never as expanded tokens

**What they found.** Both token-inspection tools count literal substrings in the prompt
text. `nodes/qwen_token_debugger.py:143-162` counts regex matches of `<|vision_start|>`,
`<|image_pad|>`, `<|vision_end|>` and friends; `nodes/qwen_token_analyzer_standalone.py`
does the same. The encoder's own debug output reports character counts, not token counts
(`nodes/qwen_vl_encoder.py:576-581`).

**What went wrong, and it is the interesting part.** A single `<|image_pad|>` in the prompt
string is not one token after processing. The processor expands it to one token per merged
vision patch, so the real count scales with image resolution. Nothing in this repo computes
that expansion. I grepped for it specifically: there is no `grid_thw`, no `merge_size`, no
patch-count arithmetic anywhere in the encoder path. The consequence is that their
"vision_tokens" output (`nodes/qwen_token_debugger.py:76-78`, returned as an INT) reports
1 per image regardless of whether the image contributed a handful of tokens or hundreds.

This is exactly the divergence the task asked about: their token counts could not diverge
from expectations, because they never measured the quantity that varies. The debugger
validates *syntax* (matched start/end pairs, coordinate well-formedness at
`nodes/qwen_token_debugger.py:271-337`) and is genuinely useful for that. It is not a token
budget tool, despite its name.

The one place real tokens are counted is unrelated to vision: v2.9.12 added counting against
a 512 reference limit using ComfyUI's bundled tokenizer, and it applies only to the Z-Image
(Qwen3-4B, text-only) encoders (`CHANGELOG.md:3-31`).

**Evidence class.** Asserted / structural. Readable from code. No measurement exists.

**Applies to 2.1?** The gap applies more, not less. Qwen3-VL is dynamic-resolution, so the
per-image token count varies with the resolution we feed it. Also note the hardcoded token
IDs in `nodes/qwen_token_debugger.py:16-51` (151652 for `<|vision_start|>`, 151655 for
`<|image_pad|>`, etc.) are Qwen2.5-VL vocabulary. Do not carry those integers over to
Qwen3-VL without re-deriving them from the tokenizer.

**What it means for us.** Build the vision-token accounting the bridge never had, and build
it early. Our harness should report, per reference image, the actual expanded token count
and the running sequence total, derived from the processor's grid output rather than from
string matching. With 2.1 accepting up to 16 reference images, sequence length is the
resource most likely to surprise us, and it is invisible to every tool in this repo.

---

## Finding 3: Two separate resolution paths for the same image, and this is the cleanest idea in the repo

**What they found.** Each input image is resized twice, to different targets, for different
consumers:

- **Vision path**: scaled to a 384x384 target *area* preserving aspect ratio, then aligned
  to 28px (`nodes/qwen_vl_encoder.py:226-251`). Feeds the VLM for semantic understanding.
- **VAE path**: capped at `vae_max_dimension` (default 2048), aligned to 32px
  (`nodes/qwen_vl_encoder.py:202-224`). Produces `reference_latents` for pixel detail.

The two are deliberately decoupled, and `CLAUDE.md:115-125` states the separation as a rule
with an explicit "IMPORTANT" marker. The alignment constants differ because the consumers
differ: 28 is the ViT patch grid, 32 is the VAE requirement.

**Evidence class.** Reasoned, and architecturally sound. The 384x384 figure itself is
Asserted, see Finding 4.

**Applies to 2.1?** The principle ports directly and is worth preserving explicitly. 2.1
still has both consumers: the VLM sees pixels to understand, and the DiT receives reference
latents. The specific constants do not port (Qwen3-VL's patch/merge geometry and the 2.1 VAE
need re-deriving).

**What it means for us.** Keep semantic resolution and latent resolution as two independently
controllable knobs in the harness rather than one "image size". They trade off against
different budgets: vision resolution against sequence length, VAE resolution against VRAM and
output detail. Collapsing them into one parameter, which is what most ComfyUI nodes do, makes
both untunable.

---

## Finding 4: The production 384x384 rule contradicts the reference smart_resize the repo also implements

**What they found, twice, differently.** The shipped encoder and batch node hardcode a fixed
384x384 target area (`nodes/qwen_vl_encoder.py:240`, `nodes/qwen_image_batch.py:164`), with
the justification "Model's trained resolution" appearing only as an inline comment.

But the deprecated multi-reference node implements the actual Qwen VL contract:

```python
def qwen_smart_resize(self, width, height,
                      min_pixels=4*28*28, max_pixels=16384*28*28, factor=28)
```
(`nodes/qwen_multi_reference.py:64-101`)

That is the official dynamic-resolution algorithm from `qwen_vl_utils.process_vision_info`,
with a pixel *budget* (a floor and a ceiling) rather than a fixed square, plus a MAX_RATIO
guard. The same file also implements DiffSynth's 32px/1MP variant
(`nodes/qwen_multi_reference.py:103-118`).

So the repo contains both the reference implementation and a fixed-square heuristic, and
shipped the heuristic. The node holding the reference version is marked
`[DEPRECATED]` at `nodes/qwen_multi_reference.py:1-14`. The two were never reconciled and
the repo never explains the choice.

The only stated reason for 384 is a docstring: "Vision encoder hardcoded to 384px (model's
trained resolution). Higher values cause object duplication and scaling artifacts"
(`nodes/qwen_image_batch.py:185-186`). "Object duplication" is a specific, plausible failure
signature, which makes me think something was genuinely seen. But there is no run, no image,
no dated entry behind it.

**Evidence class.** Asserted, and contradicted within the same repo. Flagging rather than
endorsing either rule. Qwen2.5-VL is a dynamic-resolution model and I found no citation
anywhere in the repo for a fixed 384x384 trained resolution.

**Applies to 2.1?** The fixed-square rule should not port. Qwen3-VL is also dynamic
resolution, so a single hardcoded square is the wrong shape of answer. The pixel-budget
formulation (`min_pixels` / `max_pixels` in units of the patch area) is the one that ports
conceptually, with constants re-derived from Qwen3-VL's preprocessor config.

**What it means for us.** Take the vision-side resize rule from Qwen3-VL's own
`preprocessor_config.json` / `qwen_vl_utils`, not from this repo. If we do end up capping
vision resolution below the model default, treat "object duplication at higher vision
resolution" as a hypothesis worth a deliberate A/B, since it is the only quality claim here
specific enough to test.

**One number not to blur:** `nodes/docs/resolution_tradeoffs.md:178` says "3584px =
16,384 tokens". That is the DiT/latent-side budget, not the vision-encoder token budget.
The repo uses "tokens" for both. Keep them separate in our accounting.

---

## Finding 5: Aggressive area-based downscaling caused a visible zoom-out; preserving input resolution fixed it

**What they found.** This is the best-documented quality finding in the repo. Large edit
inputs were being scaled to a fixed ~1MP area, which visibly zoomed the subject out. The
v2.6.1 entry gives the concrete case: a 1477x2056 input was being scaled to 864x1216
(0.59x), and the fix was to default to preserving input resolution with only 32px alignment,
giving 1472x2048 (`CHANGELOG.md:669-676`). The entry lays out behaviour across four input
size classes (`CHANGELOG.md:713-728`).

**Evidence class.** Observed. It is a dated changelog entry describing a specific reported
issue with specific input and output dimensions and a named fix. That is the strongest
evidence class present in this repo. It is still not a controlled measurement: no A/B, no
image, no metric.

**Important caveat on mechanism.** The `scaling_mode` parameter this entry introduced no
longer exists. The final code replaced it with `vae_max_dimension` (default 2048), which
caps the long edge rather than normalising area (`nodes/qwen_vl_encoder.py:202-224`). The
*finding* survives the refactor; the *mechanism the docs describe* does not. Anything you
read in `nodes/docs/resolution_tradeoffs.md` or `nodes/docs/QwenImageBatch.md` about
`scaling_mode` describes the v2.6.1-era code.

**The repo is explicit that it did not measure the other direction.** From
`nodes/docs/resolution_tradeoffs.md:41`: "We don't actually know if 1024x1024 produces
better results than 1328x1328 or 2048x2048 - just that it uses less VRAM." That is an
honest Unmeasured (self-declared), and it is the correct reading of the whole resolution
section: they established that *aggressive downscaling hurts*, not that *more pixels help*.

**Applies to 2.1?** The underlying mechanism (area-normalising an edit reference changes the
subject's apparent scale relative to the canvas) is model-independent and should still apply.
2.1's reference handling differs enough that the specific numbers do not transfer.

**What it means for us.** Default to preserving reference-image resolution, capping only for
VRAM, and treat any area-normalisation as a change that alters composition rather than just
cost. If we add a resolution knob, the thing to measure is the direction the bridge never
tested: whether encoding above ~1MP actually improves fidelity, or only spends VRAM.

---

## Finding 6: Batching forces every reference image to one shared resolution, and the repo logged what that costs

**What they found.** `QwenImageBatch` computes per-image ideal dimensions, then collapses
them to a single target for the whole batch via `batch_alignment`
(`nodes/qwen_image_batch.py:228-255`), because the images become one stacked tensor
(`torch.cat`, `nodes/qwen_image_batch.py:304`). Three strategies, defaulting to
`match_smallest` for VRAM safety. Aspect ratio is not preserved per image; images are
stretched to the common target.

There is a nice detail at `nodes/qwen_image_batch.py:251-255`: after picking a final VAE
size, the vision dimensions are *recomputed from the final VAE aspect ratio* rather than
carried from the per-image calculation, explicitly to stop "chimera" dimensions where the
two paths disagree on aspect.

**The logged record.** `nodes/docs/QwenImageBatch.md:219-229` pastes real console output:
three inputs (1024x1024, 1024x1024, 1328x1024) all scaled to 1344x1024, with the node's own
aspect-distortion report showing the two square images taking AR 1.00 -> 1.31 while the third
moved 1.30 -> 1.31. That is a genuine Logged artifact, and it quantifies the cost of
uniform batching: to accommodate one wider image, both square images were stretched ~31%.

**Evidence class.** Logged for the distortion behaviour. The VRAM guidance around it
("4+ images may cause VRAM issues", optimal 1-3, `nodes/qwen_image_batch.py:326-327`,
`CLAUDE.md` known issues) is Asserted.

**Applies to 2.1?** Mostly superseded, and that is good news. `TextEncodeQwenImage21` takes
up to 16 reference images as separate inputs, so there is no tensor-stacking constraint
forcing a shared resolution. The bridge's whole batching apparatus is a workaround for a
limitation 2.1 does not have.

**What it means for us.** Do not carry over uniform-resolution batching. Keep reference
images at their own aspect ratios and resolutions. The transferable warning is the failure
mode: if any part of our harness stacks references into one tensor for convenience, it will
silently reintroduce this distortion, and the bridge's log shows it is not subtle.

---

## Finding 7: Reference images were labelled "Picture N:" for the encoder, but users were told not to address them by number

**What they found.** Multi-image prompts get positional labels injected before each vision
span:

```
Picture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2: ...
```
(`nodes/qwen_vl_encoder.py:410-416`, `nodes/qwen_processor_v2.py:68-72`)

Two placements, by mode (`nodes/qwen_vl_encoder.py:396-447`):
- `image_edit`: vision tokens go *before* the user prompt; labels only when 2+ images and
  `auto_label` is on. Single image gets no label at all.
- `multi_image_edit`: vision tokens go *inside* the user prompt, labels always on, matching
  DiffSynth's `encode_prompt_edit_multi` (`CHANGELOG.md:647-650`).

**The tension, which is the actual finding.** The encoder emits numbered handles, but the
repo's own usage guidance says not to use them. From the headshot-swap recipe in `CLAUDE.md`:
"Describe subjects semantically, not by image numbers", with the worked example phrasing
everything as "the face of [person in scene]" rather than "Picture 1".

Reinforcing this, the deprecated multi-reference node removed its index-addressing mode
outright with the comment "The 'index' mode is removed as it's not supported by the model"
(`nodes/qwen_multi_reference.py:24`).

Read together: the labels are structural scaffolding that delimits which pixels belong to
which image. They are not addressable handles the prompt can reliably refer to.

**Evidence class.** The format is Reasoned (copied from DiffSynth). The
"don't address by number" guidance is Asserted, but it appears in two independent places
and one of them is a removal of a feature that did not work, which is weak corroboration.

**Applies to 2.1?** Directly, and this is the finding I would most want carried forward.
2.1 prefixes `<imageN>` reference blocks per image. The bridge's experience says: treat those
as delimiters, and write prompts that identify references by their content, not by their
index. Whether Qwen3-VL is better at index addressing than Qwen2.5-VL was is an open question
and a cheap thing for us to test.

**What it means for us.** When building prompts for multi-reference edits, describe each
reference semantically. If we want to validate index addressing, that is a clean, small
experiment: same references, prompt by index vs prompt by description, compare.

---

## Finding 8: On the conditioning tap, they made no choice at all for the Qwen-Image path

**What the code does.** The bridge does not select a layer, and does not touch normalisation.
It calls ComfyUI and takes what comes back:

```python
tokens = clip.tokenize(formatted_text, images=vision_images if vision_images else [])
conditioning = clip.encode_from_tokens_scheduled(tokens)
```
(`nodes/qwen_vl_encoder.py:479-483`)

Everything the bridge does to conditioning happens *after* that call: the front-slice
(Finding 1), and attaching reference latents. There is no hidden-state index, no
`output_hidden_states`, no layer-norm toggle anywhere in the Qwen-Image encoder path. So the
honest answer to "which layer, normed or un-normed, did they find it mattered?" is: they
never made the choice, so they could not have found it mattered.

**The one layer choice in the repo is a different model.** `hidden_states[-2]` (second to
last) appears only in the Z-Image / Qwen3-4B path: `nodes/docs/z_image_encoder.md:676` lists
it as the embedding layer used, and `nodes/docs/z_image_analysis.md:108-113` quotes diffusers'
`pipeline_z_image.py` doing `.hidden_states[-2]`. `CHANGELOG.md:50` records "Verified
`hidden_states[-2]` extraction matches reference implementations" — that is verification by
reading the reference source, not an A/B against `[-1]`. Different model, different
convention, and it should not be generalised to the Qwen-Image path.

**What the repo does know about the tap's surroundings.** `nodes/qwen_state_dict_mapping.py`
maps ComfyUI's Qwen structure against HuggingFace's and documents two relevant facts:
ComfyUI's language model ends in `model.norm`, an RMSNorm described as "(final norm)"
(`nodes/qwen_state_dict_mapping.py:43`), and there is no `lm_head` at all —
"lm_head: Linear (NOT in ComfyUI - encoder only!)" (`nodes/qwen_state_dict_mapping.py:119`).
So the normed/un-normed question in this stack is precisely whether that final RMSNorm is
applied, and the bridge never interrogated it.

**Adjacent and genuinely useful: padding handling.** For Z-Image they did establish that both
reference implementations filter padding tokens by attention mask before the DiT, while stock
ComfyUI passes the full padded sequence plus a mask (`CHANGELOG.md:56-81`, with an
implementation comparison table at `:69-75`). They added `filter_padding`, default on, doing
`embeddings[mask.bool()]`. And they traced the downstream consequence: in ComfyUI's Z-Image
DiT the mask reaches `context_refiner` but the main transformer layers receive `mask=None`
(`nodes/docs/z_image_analysis.md:130`). That means unfiltered padding is not merely inert
there, it is attended to.

**Evidence class.** The absence of a layer choice is a fact about the code. The
`hidden_states[-2]` and padding findings are Reasoned (verified by reading reference
implementations), explicitly not measured.

**Applies to 2.1?** The layer/norm question: nothing to port, which is a legitimate result.
2.1 fixes `layer_norm_hidden_state = False` (un-normed final hidden state) and the bridge
offers no evidence for or against. The padding finding may still apply and is worth checking
against 2.1's encoder.

**What it means for us.** Two things. First, treat the un-normed choice in
`qwen_image21.py` as unvalidated by this prior work; if we want to know whether it matters,
that is an experiment nobody here ran. Second, the padding question is a real, cheap check
for our harness: confirm whether our conditioning path filters by attention mask or passes
padded sequence plus mask, and whether anything downstream actually honours the mask. The
bridge found that last part is where the assumption breaks.

---

## Finding 9: Reference latents ride alongside conditioning as a separate channel, which 2.1 has since fused

**What they found.** Reference images travel two parallel routes to the model. The vision
route goes through the VLM and becomes part of the embedding sequence. The pixel route goes
through the VAE and is bolted onto the conditioning dict:

```python
conditioning = node_helpers.conditioning_set_values(
    conditioning, {"reference_latents": ref_latents}, append=True)
```
(`nodes/qwen_vl_encoder.py:502-508`)

Each image is VAE-encoded separately and appended as a list (`nodes/qwen_vl_encoder.py:385-386`).
The generation canvas is entirely independent: an empty 16-channel latent from
`QwenVLEmptyLatent` (`nodes/qwen_vl_helpers.py:129-138`). So references and canvas never meet
until the DiT.

**Friction they hit with this design.** Mismatched reference latent shapes were a live
problem. The standard encoder can only warn about it
(`nodes/qwen_vl_encoder.py:516-521`, collecting unique shapes and emitting
"This may cause generation issues"). The advanced encoder had to add real workarounds:
padding latents to even dimensions for patch processing
(`nodes/qwen_vl_encoder_advanced.py:31-60`) and adding a time dimension for the 5D Wan21
latent format (`nodes/qwen_vl_encoder_advanced.py:425-430`). On top of that sits a monkey
patch to ComfyUI's RoPE so batches with differing image sizes take max dimensions across the
batch rather than the first element's (`nodes/qwen_vl_encoder.py:36-62`, attributed to a
DiffSynth commit).

**Evidence class.** Observed. These are fixes for errors that actually fired, recorded in
`CHANGELOG.md:816-823` ("Dimension mismatch auto-handling - No more errors from mismatched
resolutions") and `CHANGELOG.md:900-902`.

**Applies to 2.1?** Different point of assembly. Per the task brief, 2.1's
`encode_token_weights` swaps vision spans for reference latents inside the encoder, whereas
the bridge attaches them post-hoc via `conditioning_set_values` and leaves reconciliation to
the DiT. I have not read `qwen_image21.py`, so I am not claiming what that difference does to
the shape-mismatch surface; I am only noting that the two designs assemble the channels at
different stages.

**What it means for us.** The transferable part is the catalogue of failure modes, not a
verdict on either design. If our harness ever assembles reference latents itself, the two
failures with prior art here are mismatched per-image latent shapes and odd (non-even) latent
dimensions in patch processing. Worth checking whether 2.1's path can produce either.

---

## Finding 10: The 512-token reference limit, and where it does and does not apply

**What they found.** Reference implementations (diffusers, DiffSynth) truncate text at 512
tokens; ComfyUI does not, so prompts above that diverge from reference behaviour
(`CHANGELOG.md:13-15`). They added a counter with a warning above the limit
(`CHANGELOG.md:3-31`, sample output at `:19-31`).

**Scope caveat.** This landed only on the Z-Image (Qwen3-4B) encoders
(`CHANGELOG.md:11`). The Qwen-Image encoder in `nodes/qwen_vl_encoder.py` never got it; its
debug output still reports characters, not tokens (`nodes/qwen_vl_encoder.py:576-581`). So
the Qwen-Image path in this repo has no token budget awareness whatsoever, which compounds
Finding 2.

**Evidence class.** Reasoned, from reading reference implementations.

**Applies to 2.1?** Needs re-checking rather than porting. Whether 2.1's reference pipeline
truncates, and at what length, is a property of that pipeline, and with 16 possible reference
images the vision-side expansion (Finding 2) will dominate sequence length long before prose
does.

**What it means for us.** Verify 2.1's actual truncation behaviour against its reference
implementation rather than assuming 512. And whatever the limit is, count vision expansion
against it, not just text.

---

## Summary table

| # | Finding | Evidence | Applies to 2.1? |
|---|---|---|---|
| 1 | Drop index is a constant coupled to an editable system prompt | Reasoned; origin Observed | Hazard designed out; do not port 34/64 |
| 2 | Vision tokens counted as strings, never as expanded tokens | Asserted / structural gap | Gap applies more; token IDs are stale |
| 3 | Vision and VAE resolution as two decoupled paths | Reasoned | Principle ports; constants do not |
| 4 | Fixed 384x384 contradicts the reference smart_resize in the same repo | Asserted, self-contradicting | Fixed square does not port; use pixel budget |
| 5 | Area-normalising large edit inputs caused visible zoom-out | Observed (dated, specific dims) | Mechanism still applies; numbers do not |
| 6 | Uniform batch resolution distorts aspect ratios | Logged (console output) | Superseded; 2.1 takes separate refs |
| 7 | "Picture N:" labels are delimiters, not addressable handles | Asserted, corroborated twice | Applies directly to `<imageN>` |
| 8 | No layer/norm choice was ever made for the Qwen-Image path | Fact about code | Nothing to port; padding check is worth doing |
| 9 | Reference latents as a side channel, with recurring shape fixes | Observed | Superseded; argues for 2.1's fused design |
| 10 | 512-token reference limit, Z-Image path only | Reasoned | Re-verify for 2.1 |

## Things I would not carry over

- Any number from `experiments/EXPERIMENT_METHODOLOGY.md`. Hypothetical.
- The 34 / 64 drop indices (`nodes/qwen_processor_v2.py:25-30`). Tokenizer- and
  prompt-specific.
- Qwen2.5-VL special token IDs (`nodes/qwen_token_debugger.py:16-51`). Wrong vocabulary.
- The fixed 384x384 vision target. Uncited, and contradicted in-repo.
- Uniform-resolution batching in `nodes/qwen_image_batch.py`. Solves a problem 2.1 lacks.
- `hidden_states[-2]`. Correct for Qwen3-4B / Z-Image, not a statement about the
  Qwen-Image path.
- Spatial coordinate tokens (`nodes/qwen_spatial_token_generator.py`). Marked deprecated,
  "not used by DiffSynth", effectiveness explicitly unclear (`CHANGELOG.md:911-917`).
  Flagging only because it sits in the scoped file list; the dead-end record is another
  agent's scope.

## Open questions this repo raises but does not answer

1. Does the un-normed final hidden state (2.1's `layer_norm_hidden_state = False`) differ
   materially from the normed one? No prior evidence either way.
2. Does capping vision resolution below the model default actually prevent "object
   duplication" (`nodes/qwen_image_batch.py:185-186`), or was that misattributed?
3. Can Qwen3-VL reliably address references by index, where Qwen2.5-VL apparently could not
   (`nodes/qwen_multi_reference.py:24`)?
4. Does encoding references above ~1MP improve fidelity, or only cost VRAM? The repo
   explicitly does not know (`nodes/docs/resolution_tradeoffs.md:41`).
5. Does 2.1's conditioning path filter padding by attention mask, and does anything
   downstream honour the mask? The bridge found ComfyUI's Z-Image DiT largely does not
   (`nodes/docs/z_image_analysis.md:130`).
