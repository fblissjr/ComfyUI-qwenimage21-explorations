# Reference image sizing: one number, two readers, five postures

last updated: 2026-09-20

What resizes a reference image, how many times, and what happens when the two
readers of it disagree. Read [`upstream.md`](upstream.md) first for the shared
conditioning contract; this page owns the one place the implementations take
materially different risks.

---

## 1. Why this is a contract in 2.1 and was a knob in H3

In 2.1 a vision-language image slot stands for a fixed group of latent tokens
(`coderef/diffusers/src/diffusers/models/transformers/transformer_qwenimage21.py::_IMG_TOKENS_PER_SLOT`).
So the encoder's slot count and the VAE's latent grid are **two views of one
number**, and every implementation resizes each reference once and hands the
same size to both readers.

This is the part worth pausing on if you know the sister project. H3 ships a
deliberate knob for showing the text encoder a *different* view of a reference
than the VAE encodes — `MiniMaxH3AppendRefImage.qwen_view`, with `separate` and
its own short edge — and
`coderef/ComfyUI-h3-explorations/docs/h3_references.md` justifies it explicitly:
**"nothing indexes a Qwen token against a latent patch."** In H3 that is true.

**In 2.1 it is not true, and that is the whole difference.** A slot does index
against latent tokens. diffusers raises when the counts disagree, in
`build_token_metadata`; sglang raises from `build_layout`. The H3 knob's
justification does not transfer, so the H3 knob does not transfer either — not
because it was wrong there, but because the thing it relied on is gone here.

## 2. The five postures

Every implementation resizes once itself. The question is what it does about
the **processor's own second resize**, which applies a floor and a ceiling
afterwards and can move the encoder's view off the shared number.

| implementation | posture toward the second resize | if a clamp would fire |
|---|---|---|
| LightX2V | **turns it off**: passes `do_resize=False`, and says in place that a second resize can change the slot count | cannot happen |
| DiffSynth-Studio | **pre-empts it**: reads the processor's own floor (`get_processor_min_pixels`) and folds it into its own sizing, so its resize already satisfies the processor | cannot happen |
| diffusers | neither | **raises** in `build_token_metadata` |
| sglang | neither | **raises** from `build_layout` |
| ComfyUI core | neither, **and it has dropped the vision tokens by then**, so no count survives to check | **nothing happens.** The encoder read one scale, the VAE encoded another, and the geometry — which comes from the latents — is unaffected |

ComfyUI's row is the finding. It is not that ComfyUI resizes carelessly: the
node does the one shared resize deliberately and says so in a comment. It is
that ComfyUI is the only one of the five that **cannot** notice the divergence,
because its bookkeeping strategy ([`upstream.md`](upstream.md) section 3) throws
away the quantity the other two check.

## 3. ComfyUI clamps with the library's numbers, not the checkpoint's

The second resize is `ComfyUI/comfy/text_encoders/qwen_vl.py::process_qwen2vl_images`.
Its floor and ceiling are defaults, and `qwen3vl.py` does not override them.
Those defaults equal **transformers' Qwen2-VL image-processor defaults**.

This checkpoint declares its own, and they are different numbers. Its processor
config names `Qwen2VLImageProcessorFast` and carries a `size` dict, which
transformers reads as the pixel floor and ceiling — a reading DiffSynth-Studio
independently confirms by using `shortest_edge` as an area floor.

The homes, so neither is copied into prose here:

| bound | where it lives |
|---|---|
| what ComfyUI applies | `ComfyUI/comfy/text_encoders/qwen_vl.py::process_qwen2vl_images` default arguments, and `scripts/refview_bounds.py::COMFY_MIN_PIXELS` / `COMFY_MAX_PIXELS` |
| what the checkpoint declares | `<models>/Qwen-Image-2.1/processor/preprocessor_config.json`, its `size` dict; mirrored at `scripts/refview_bounds.py::CKPT_MIN_PIXELS` / `CKPT_MAX_PIXELS` |

### Transformers or hardcoded? Both, and differently per subsystem

Worth separating, because the two halves of "ComfyUI ignores the checkpoint's
config" fail for different reasons:

- **Text.** ComfyUI *does* use transformers — it imports `Qwen2Tokenizer` and
  calls `from_pretrained` on **its own bundled vocabulary directory**
  (`ComfyUI/comfy/text_encoders/qwen25_tokenizer/` for this encoder), never on
  the checkpoint. So the class is real and the vocabulary is a copy. The
  tokenizer class is chosen in ComfyUI's code, so whatever class or
  pre-tokenizer the checkpoint declares cannot reach it — which is the
  mechanism behind [`../quantization-strategy.md`](../quantization-strategy.md)
  section 29.
- **Images.** No transformers at all. `process_qwen2vl_images` is ComfyUI's own
  reimplementation in torch, and its bounds are constants in that function's
  signature that happen to equal the transformers library defaults.

So: a real transformers class pointed at a bundled vocabulary on one side, and
an independent reimplementation carrying copied constants on the other. Neither
reads the checkpoint.

**For this encoder that costs nothing, and it was checked rather than assumed.**
The bundled vocabulary carries the same added-token set as the checkpoint's
processor, same entries and same ids, including every vision and turn marker;
neither side declares a `pretokenize_regex`, so the Qwen2 default really is the
right one here. Base vocabulary and merge rules were compared in
[`../quantization-strategy.md`](../quantization-strategy.md) section 29 and are
pinned by `tests/test_tokenizer_parity.py`. The one config value that differs is
the declared context length, and it never binds: ComfyUI sets its own
effectively unbounded length when it constructs the tokenizer, so nothing
truncates at either number. **There is no encoder-side vocabulary gap** — the
tokenizer problem is confined to the expanders.

**This is the same class as [`../quantization-strategy.md`](../quantization-strategy.md)
section 29** — ComfyUI substituting Qwen2-family defaults for what the
checkpoint declares, now found in two subsystems: the tokenizer's
pre-tokenizer regex, and the image processor's bounds.

**The two are not equally severe, and it matters which you act on.** Section 29
bites **at default settings**, on scripts the edit prompt gives a worked example
for. This one is **clean at the node default** and only reachable at the widget's
extremes. Section 29 is the one to report upstream; this one is a thing to know
before turning a knob.

## 4. Where it is actually reachable

`scripts/refview_bounds.py` prints the cases where the encoder's view and the
VAE's view part company, and where ComfyUI's bounds differ from the
checkpoint's. Run it rather than trusting a summary:

```
python scripts/refview_bounds.py
```

The shape of the answer: the node default is clean, and the divergence is
reachable from the widget's top end, its bottom step, and the pass-through mode
with a source outside the bounds. The script is the authority for which.

## 5. What the impact is, and who already owns the experiment

**Not known here, and the honest answer is that the sister project built the rig
and never scored it.** H3 has a reference-view ablation — `bench/refview2_arms.json`,
arms across several scenes — for exactly this question: whether a coarser or
finer encoder view of a reference changes the result. Its own wiki records the
split as **an option, unmeasured**, and its `h3_references.md` says whether any
arm helps "is unmeasured and is the owner's matched-seed comparison to judge."

So the mechanism is understood on both sides and the effect size is measured on
neither. What can be said without a render:

- **Geometry is not at risk.** The DiT takes its layout from the reference
  latents and the slot indices, not from the encoder's grid, so a divergence
  changes what the encoder *read*, not where anything lands.
- **It is not nothing either.** The vision tokens are dropped from the
  conditioning, but they are dropped *after* the language model has attended to
  them, so the retained text positions carry the encoder's reading of the image
  at whatever scale it saw.
- **The direction is knowable, the magnitude is not.** Above the ceiling
  ComfyUI shows the encoder less detail than the checkpoint's own bounds allow;
  at the bottom it shows it less than the checkpoint would insist on.

If this becomes worth settling, the H3 ablation is the closest prior art for
how to arm it, and [`../quantization-strategy.md`](../quantization-strategy.md)
section 13b is this repo's rule about deciding the measurement before building.


## 6. Many references: what each implementation targets

Every implementation sizes **each reference independently**, preserving that
image's own aspect, at a shared target *area*, and **upscales a small reference
to reach it**. ComfyUI does this too. There is no batching-forces-one-size
behaviour here and no never-upscale clamp — the thing that had to be fixed in
H3 is not present in this path.

What differs is **which number the area comes from**, and **which reference
sets the canvas**:

| implementation | reference area is | canvas, when not given explicitly |
|---|---|---|
| sglang | **the output canvas's area** | required from the caller |
| DiffSynth-Studio | **the output canvas's area** | required from the caller |
| diffusers | a separate `output_resolution` parameter | derived from the **last** image's aspect |
| LightX2V | a separate `resolution` config value | the **last** image's size |
| ComfyUI core | a separate `resolution` widget | the node emits a latent at the **first** reference's size |

**Mixed aspect ratios are a first-class case, not a tolerated one.** Each
reference keeps its own geometry all the way through: core's DiT reads `h, w`
per reference in `build_sequence` and lays out that reference's own grid, with
a deliberate half-token adjustment where a reference's grid parity differs from
the target's (`ComfyUI/comfy/ldm/qwen_image21/model.py`). Nothing forces
references to a common size or a common aspect. Nothing needs building for this.

**And the area decoupling is a widget value, not a missing feature.** ComfyUI's
per-reference sizing is the same computation sglang applies, differing only in
which area it is given. Set the `resolution` widget to the geometric mean of the
canvas you will actually sample at and the two agree exactly:
`scripts/refview_bounds.py --parity` checks that over a spread of canvases and
reference shapes, including extreme ratios.

Two consequences, neither of them a defect on its own:

- **ComfyUI is the odd one out on which reference sets the canvas.** diffusers
  and LightX2V take the last; ComfyUI takes the first. With references of
  different aspect ratios, the same inputs give a differently shaped output.
- **In ComfyUI the reference area is decoupled from the canvas.** At the node's
  defaults they agree, because the node emits a latent sized from a reference
  it sized itself. Wire a differently sized latent instead and the references
  stay at the widget's area, where sglang and DiffSynth would have followed the
  canvas. The coupling is a convention of the default wiring, not a contract.

## 7. What a custom node could and could not change

Recorded because the question comes up, not as a recommendation — this repo's
standing rule is to adapt to existing nodes rather than add them.

**Reachable from a node**, because they are decisions the node makes before
core sees anything: which reference sets the canvas; sizing references to a
supplied canvas area instead of a widget; expanding a batched input into
several references instead of taking its first frame; and requiring a VAE
rather than silently falling through to encoder-only conditioning.

**Pre-emptable but not suppressible:** the second resize. `process_qwen2vl_images`
is called inside core's encoder path with no arguments a caller can set
(`ComfyUI/comfy/text_encoders/qwen3vl.py`), so a node cannot turn it off the way
LightX2V does. It *can* do what DiffSynth-Studio does — keep its own sizing
inside the bounds core will apply, so the clamps never fire.

**Not a node fix at all:** the tokenizer's pre-tokenizer
([`../quantization-strategy.md`](../quantization-strategy.md) section 29). It is
decided where the tokenizer is constructed, and the fix belongs upstream.

### Where that leaves it

**Two of the four reachable items dissolve on inspection.** Canvas-coupled
reference sizing is the `resolution` widget set to the canvas's geometric mean,
checkable with `--parity` above. Mixed aspect ratios already work. Batch
expansion is a wiring choice with sockets to spare.

**One is left, and it is not really a sizing question.** The VAE input is
optional, so leaving it unwired does not fail — it selects the encoder-only
conditioning mode, which is a deliberate path with no signal that you are on
it. That is the only item here that produces a confidently wrong result from an
ordinary mistake.

**Decided 2026-09-20: a structured encode node was built anyway, as a research
surface rather than as a fix.** `QwenImage21EncodeStructured` exposes the
system turn, `keep_vision` and which reference sets the canvas, with defaults
that reproduce the stock node. Editing the system prompt is off-distribution
and the node says so in its own description — the point is to be able to A/B
it, not to recommend it. [`decisions.md`](decisions.md).
