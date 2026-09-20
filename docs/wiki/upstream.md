# How the other implementations run 2.1

last updated: 2026-09-20

Six checkouts were read on 2026-09-20 on two questions: how a text-to-image
request is built, and what changes when condition images are added. **Five of
them implement Qwen-Image 2.1**; vllm-omni does not, at the revision read, and
appears here only in section 4 where its absence is the finding. The revisions
read are in [`references.md`](references.md), and they moved under this page
the same day it was written — re-read before quoting.

**This is the wiki's one owner page.** Every other page routes to a document
under `docs/` that owns its facts. This one states them, because no document
under `docs/` owns the cross-implementation read and inventing one to route to
would be ceremony. If that changes, this page moves and the wiki goes back to
being purely a router.

**Nothing here was run.** This is a source read. It compares how the
conditioning is constructed, not output quality and not speed. sglang's
cookbook at `coderef/sglang/docs/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1.mdx`
carries a dated measured table with its own conditions; read it there rather
than expecting numbers here.

**The system prompt on this page is the conditioning encoder's, not the
expanders'.** They are different models doing different jobs, and crossing them
is a real hazard — the diffusers pipeline metadata ships a `chat_template.jinja`
belonging to the encoder ([`../quantization-strategy.md`](../quantization-strategy.md)
section 18). Everything the expanders' prompts govern is
[`contract.md`](contract.md).

---

## 1. The conditioning contract

Every implementation below builds the same string and asks the encoder for the
same tensor. This is the part a reader can rely on.

| question | what they do | where to read one |
|---|---|---|
| system prompt | one short fixed sentence, byte-identical across all five | `coderef/diffusers/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`, `QwenImage21Pipeline.sys_prompt` |
| template | system turn, user turn, then an **open** assistant turn the encoder never continues — the generation prompt's shape, read for hidden states rather than sampled | same file, `prompt_template_t2i` |
| condition images | one `<imageN><|vision_start|><|image_pad|><|vision_end|>` block per image, **1-indexed, space-joined, ahead of the prompt text**. Order is the order the model reads them | same file, `prompt_template_ti2i` |
| empty prompt | replaced with a single space, because Qwen has no bos token and an empty string leaves the encoder nothing to read | diffusers and sglang both; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/qwen_image21.py` |
| padding | left, as trained | same |
| the system turn is dropped from the hidden states | **always, and always derived** — never a constant. See below |
| guidance | off by default; 2.1 is meant to be sampled without it. Turning it on doubles the work per step | the `true_cfg_scale` default in the diffusers pipeline signature |

### The drop index is derived by every one of them

Four of the five tokenize the system message and take its length; ComfyUI core
instead scans the token stream for the **second** `<|im_start|>` and drops
everything before it, adjusting for each image expanded ahead of it
(`ComfyUI/comfy/text_encoders/qwen_image21.py`,
`QwenImage21TEModel.encode_token_weights`).

Different mechanisms, one rule: **nobody hardcodes it.** That is worth stating
plainly because the predecessor repo did hardcode it, against a user-editable
prompt, and [`../bridge-encoder-findings.md`](../bridge-encoder-findings.md)
Finding 1 is the record of what that cost. Five independent implementations
declining to do it is about as settled as a design question gets here.

### They all want the pre-final-norm hidden state, by mechanisms of unequal robustness

The transformer was trained on the last decoder layer's output **before** the
text encoder's final RMSNorm. All five target that tensor. How they get it is
not equally safe, and this is the row to read carefully rather than to read as
unanimity:

| implementation | mechanism | what it depends on |
|---|---|---|
| diffusers | forward hook on the norm returning its input | nothing — works on either transformers major |
| DiffSynth-Studio | forward hook, same shape | same |
| ComfyUI core | `layer_norm_hidden_state = False` | its own encoder stack |
| LightX2V | never applies a final norm at all | its hand-written layer stack having no such step. Construction, not a choice |
| sglang | takes `hidden_states[-1]` as transformers 4.57 returns it | **the transformers version.** diffusers' own comment states that from 5.0 that entry is tied to the normalized `last_hidden_state` |

diffusers additionally carries a TODO to replace its hook with a config flag
once `tie_last_hidden_states=False` ships upstream — see
[`next_steps.md`](next_steps.md), because that will change the correct
mechanism for everyone and will do it quietly.

## 2. What changes for edit

| question | what they do |
|---|---|
| resize | **one resize feeds both the vision tower and the VAE**, onto a 32-pixel grid, sized from the target area and the image's own aspect. Not two paths, not two resolutions. **ComfyUI does this one shared resize too, and then runs a second pass on the encoder branch alone** — see [`sizing.md`](sizing.md), which owns that divergence |
| why it must be one | each vision-language image slot stands for a 2x2 group of latent tokens (`coderef/diffusers/src/diffusers/models/transformers/transformer_qwenimage21.py::_IMG_TOKENS_PER_SLOT`), so the encoder's slot count and the VAE's latent grid are two views of one number. diffusers **raises** when they disagree, in `build_token_metadata` |
| the hazard this creates | letting the processor resize again after your own resize can change the slot count. The five take **four different postures** toward it, and that is [`sizing.md`](sizing.md)'s subject |
| transparency | the image is taken as RGBA. The alpha is composited over white **for the vision tower only**; the VAE keeps all four channels. **All five do this**, ComfyUI core in `TextEncodeQwenImage21` itself *(corrected 2026-09-20: this row previously said ComfyUI was the exception, which was wrong — the node does it inline and says so in a comment)* |
| condition latents | VAE-encoded, normalized by per-channel mean and std, packed, and concatenated into **one joint sequence** with the target. Not a side channel |
| block structure | block-causal: each image block is internally bidirectional, later blocks and the target attend to earlier ones. **Block boundaries come from the shape list, not from runs in the image mask** — two adjacent condition images form one run and must stay two blocks, or they would attend to each other bidirectionally |
| static prefix | text and condition-image keys and values do not change across steps, so the first step prefills them and later steps recompute only the target's tokens. diffusers and sglang both carry a cache for this; in ComfyUI it is a node |

## 3. Where they genuinely diverge

**Vision-slot bookkeeping — three strategies for one fusion.** Each keeps a
different thing in the encoder's output sequence and expands it somewhere else:

- **diffusers** keeps every image-pad position, then expands each four-fold in
  the transformer and drops the latents into those positions.
- **sglang** collapses each image's run to a single slot at encode time
  (`collapse_image_slots`), then expands that slot to the full latent grid in a
  precomputed layout.
- **ComfyUI core** drops the vision tokens from the conditioning entirely and
  records an insertion index per image (`image_slots`) for the DiT to fill.

Same destination, three sets of bookkeeping. This matters when porting: an
index or a count taken from one of them means something different in another.

**ComfyUI core has a mode the others do not expose.** Its `keep_vision` option
keeps the vision tokens in the conditioning, so an image conditions **through
the text encoder alone** with no VAE latents. No reference implementation
offers this, and none of them can be consulted about what it does. It is the
one place this repo's path can do something the references cannot, and it is
therefore also the one place their agreement buys us nothing.

## 4. What none of them does

**Not one of the five wires in a prompt expander.** Every engine takes the
prompt as given. Upstream ships the expanders as a *separate* codebase with its
own runner and server (`coderef/Qwen-Image-2.1/prompt_rewrite/`) and recommends
them in its README for best results, but the integration is left to the caller.

**And it is not that serving engines lack the integration point.** vllm-omni
carries a first-class one: `prompt_expand_func`, a hook the engine collects
from whichever stage client provides it
(`coderef/vllm-omni/vllm_omni/engine/omni_engine_base.py`). Several model
families register one — Bagel, Ming-Flash-Omni, MiniMax-Music3, Audex. **The
Qwen-Image pipelines register none**, and that checkout has no Qwen-Image 2.1
support at all at the revision read, so it is not an implementation of this
model so much as evidence about the shape of the gap.

So the mechanism is proven in a serving engine and nobody has pointed it at
this model family. That is the gap this repo's harness sits in, and it explains
why there is no reference implementation to compare `answer.py::grade` against
([`stages.md`](stages.md)): grading an expander's output is not a stage anyone
else runs.

## 5. What this settles for us

- **The predecessor's two-resolution idea is not available in 2.1.**
  [`../bridge-encoder-findings.md`](../bridge-encoder-findings.md) Finding 3
  calls separate vision and VAE resolution paths the cleanest idea in that
  repo. In 2.1 the two are one number, and the reference implementation raises
  when they disagree. The idea does not port; recorded in
  [`decisions.md`](decisions.md).
- **The transparency gap is real, and now sharper.** The image pipeline handles
  RGBA end to end, deliberately and in four implementations. So
  [`../quantization-strategy.md`](../quantization-strategy.md) section 23's
  finding — that neither expander's system prompt has any vocabulary for
  transparency — is a gap between a working capability and its expander, not a
  capability nobody has.
- **Our chat construction is the expanders' format, and it is not this one.**
  `chat.py` builds a generation prompt to be *sampled from*, ending in an open
  `<think>` block. The template on this page is read for hidden states and
  continues nothing. Both are `<|im_start|>` strings and they are not
  interchangeable.
