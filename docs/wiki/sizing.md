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
