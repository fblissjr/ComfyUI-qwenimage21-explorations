# Sampling: steps, guidance, and the sigma schedule

last updated: 2026-09-22

What the example graphs set, where those values come from, and the two places
ComfyUI's schedule departs from the release. Numbers are not repeated here:
`scripts/sigma_schedule.py` prints them, and `--widgets` prints the fix.

---

## 1. What the release specifies

The checkpoint's `scheduler/scheduler_config.json` is the authority. It asks
for a `FlowMatchEulerDiscreteScheduler` with **`use_dynamic_shifting: true`**,
an exponential time shift, its own token bounds and shift range, and a
**`shift_terminal`**. Dynamic shifting means the shift exponent, mu, is an
affine function of the target's latent token count rather than a constant.

## 2. What each implementation does

| | steps | guidance | mu | `shift_terminal` |
|---|---|---|---|---|
| diffusers | 40 | off by default | computed from the target latents' length | applied by its scheduler |
| LightX2V | 40 (config) | off (`enable_cfg: false`) | computed, in its own scheduler | applied explicitly |
| DiffSynth-Studio | 40 (`pipelines/qwen_image_21.py::QwenImage21Pipeline.__call__`) | off (`cfg_scale` defaults to 1) | computed, passed as `dynamic_shift_len` | its scheduler's |
| sglang | 40 (`configs/sample/qwenimage21.py::QwenImage21SamplingParams`) | off by default | computed from the canvas | its cloned scheduler's |
| vllm-omni, unmerged `qwen-image-2.1` branch | 50 (the pipeline's fallback, and its recipe's) | off (`true_cfg_scale` defaults to 1) | computed from the target's packed latent length | the checkpoint's scheduler, loaded as shipped |
| **ComfyUI** | **25** in the official graph | off (`cfg 1.0`) | **a constant**, `supported_models.py::QwenImage21`'s `sampling_settings["shift"]` | **absent from ComfyUI entirely** |

**Guidance is off everywhere, and that is not a default anyone should change
casually** — 2.1 is meant to be sampled without it, and turning it on doubles
the work per step.

**On steps, ComfyUI's official graph is the outlier and it is a deliberate
one.** Every other implementation defaults higher: diffusers, LightX2V,
DiffSynth-Studio and sglang to 40, vllm-omni's branch to 50. The Comfy-Org
graph ships 25. **This repo's t2i graphs now ship 40** (the owner,
2026-09-22), after a sweep found t2i still converging at 25. *(Corrected 2026-09-22: this table gave DiffSynth-Studio's and
sglang's steps as the caller's. Both already defaulted to 40 at the revisions
[`references.md`](references.md) records, so the outlier was starker than the
page said. See [`decisions.md`](decisions.md).)*

**t2i has not converged at 25, on either canvas swept.**
`scripts/steps_sweep.py` rendered the two generated t2i prompts, three seeds
each, at several step counts against a high-step reference, with the owner's
sage node in the graph. Distance to the reference falls at every step count
up to 50, with no plateau at 25, and falls at the layout scale as well as at
full resolution. By eye the extra steps refine detail and leave the
composition alone. Cost grows linearly with steps. Conditions, gates and the
table: `python scripts/steps_sweep.py --report data/steps_sweep/2026-09-22`.
This replaces the 2026-09-20 sweep's t2i half, which is *withdrawn*: those
graphs sampled on a schedule set for four times their canvas (see
[`decisions.md`](decisions.md)).

**Edit converges as far as the edit leaves the model free, and 20 is short of
it for the freer edits.** The same instrument ran the four upstream edit
samples, three seeds each, sage on, each at the canvas its expander answer
chose. The text-translation edit is nearly flat across 16 to 40. The head
turn and the flag scene keep moving. The two-subject composite moves most, at
the layout scale too, and on one seed its 16- and 20-step renders placed the
subjects and props differently from 25 onward; another seed held one layout
throughout. The 100-step reference is a convergence target, not a quality
bar: on one seed it grew an extra ear. Table and gates:
`python scripts/steps_sweep.py --report data/steps_sweep/2026-09-22_edit`.
This supersedes the 2026-09-20 finding that edit "barely moves", which was one
scene, a local recolor, the most constrained kind. **The templates carry one value
per mode** rather than the official graph's single
figure —
`scripts/build_example_workflows.py::STEPS` holds them, and the reasoning and
its limits are in [`decisions.md`](decisions.md). Conditions and the caveats are in
[`decisions.md`](decisions.md); re-derive with `scripts/sigma_schedule.py` for
the schedules and a sweep for the renders.

### The one thing nobody disagrees about

**mu is derived from the target image's latent grid, and nothing else.** Not
the reference images, not their count, not their shapes, not the encoder, not
the step count. All four implementations were read on this point and all four
feed it the target only — diffusers the target latents' own length, sglang and
LightX2V the target canvas, DiffSynth the target latents' height times width.
vllm-omni's branch, read 2026-09-22, makes five: the packed target latents'
length (`prepare_timesteps`).
Reference latents ride in the same joint sequence but never enter the schedule.

## 3. ComfyUI's two departures

**The shift is a constant.** It equals the computed mu at 1024x1024 and drifts
in both directions from there — less shift than the reference below that
canvas, more above it. At 1024x1024 the resulting schedule matches the
reference to within rounding; at other canvases it does not.

**`shift_terminal` is not implemented.** Grep finds the key nowhere in
ComfyUI. The reference stretches the schedule so its last non-zero sigma lands
on the configured terminal; ComfyUI's runs to its own tail and then to zero.
This one is resolution-independent and is the larger of the two departures at
1024x1024, where the dynamic half already agrees.

Run `scripts/sigma_schedule.py --sigmas` for both, at whatever step count.

## 4. Closing the gap

**`QwenImage21Sigmas` builds the schedule the config asks for** and emits
`SIGMAS` for `SamplerCustomAdvanced`. It reads the shift from the **latent it
is given**, so the canvas it sizes for cannot disagree with the canvas the
sampler receives, and it applies the terminal stretch. The checkpoint's
scheduler values are its defaults rather than literals, so a checkpoint
declaring different ones is served by the same node. The arithmetic is
`src/qwenimage21_explorations/sigmas.py`, importable and tested without
ComfyUI.

`terminal_mode` selects how the schedule ends, so the ordering is a stated
choice rather than something a reader has to get right:

| mode | what it does | whose behaviour |
|---|---|---|
| `release` | stretch the curve, then end at zero | the checkpoint's |
| `off` | no stretch | core's |
| `stop_short` | the schedule **ends at the terminal** and never reaches zero, so the sampler leaves that much noise | nobody's — off-distribution |

`stop_short` is exactly what applying the stretch after the trailing zero
produces. It is offered as a named mode because it is worth being able to see,
and because naming it is what stops it happening by accident.

**And the accident is guarded, not merely tested.**
`sigmas.stretch_to_terminal` raises on a curve that already ends at zero, so
the wrong order fails loudly at the one place it could be written. The test
suite pins the modes apart as well, but the precondition is what makes the
mistake unwriteable.

### The stock node can do the dynamic half, but on a coincidence

`ModelSamplingFlux` reproduces the dynamic mu exactly, at every canvas, given
the two widget values `scripts/sigma_schedule.py --widgets` prints. It works
because its hardcoded token count is `width * height / 256`, which is Flux's
VAE-8-plus-patch-2 geometry and also, by coincidence, 2.1's VAE-16 unpatched
geometry; its wrong sequence bound then absorbs into the two shift widgets.

**Recorded as a fact, not a recommendation.** Two unrelated geometries
agreeing is not a contract, the widget values are magic numbers with no
on-screen reason, and it still leaves the terminal stretch undone. Reach for it
only to sanity-check the node above against stock machinery.

### Correcting something said earlier

An earlier draft of this page said `shift_terminal` "cannot" be done in
ComfyUI. That was wrong. **No stock node does it**, which is the true claim;
the transform itself is three lines on the sigma vector, and nothing about
ComfyUI's sampler prevents it.

**A second correction, to the failure mode.** That draft also said stretching
after the zero was appended "changes nothing". It changes quite a lot: the
trailing zero is itself stretched to the terminal, so the schedule never
reaches zero and **the sampler stops with that much noise still in the image**.
It is a visible defect rather than a silent one, which is better, but it is not
a no-op. `scripts/sigma_schedule.py` and `tests/test_sigmas.py` both carry the
real numbers.

## 5. Tying the schedule to something else

The question comes up because 2.1's joint sequence carries more than the
target: text, and one block per reference image. Could the schedule track any
of that?

**Nothing in the release or in four implementations suggests it should**, and
that is the strongest thing that can be said today. The schedule governs how
much noise is removed per step from the target tokens, and the reference
tokens are static context — prefilled once and cached, never denoised
([`upstream.md`](upstream.md) section 2). A schedule that moved with the
reference count would be changing the target's trajectory because of tokens
that are not on it.

That makes it an experiment rather than a fix, and it needs a stated
prediction before it is worth a render. Two candidates that are at least
coherent:

- **Reference-aware shift.** Hypothesis: with heavy reference conditioning the
  early steps are more constrained, so less shift is needed. Nothing supports
  this; the falsifier is that a matched pair at one seed shows no difference.
- **Terminal tied to edit strength.** More defensible, because `shift_terminal`
  already exists as a knob the release sets and ComfyUI ignores. Getting it to
  the release's value is the control that has to come first.

**The order matters.** Two known departures from the reference are open, both
with a correct target to hit. Measuring a novel schedule against a baseline
that is already off in two known ways would confound the two.
[`../quantization-strategy.md`](../quantization-strategy.md) section 13b is
this repo's rule about deciding the measurement before building.

**The hook is already built.** `QwenImage21Sigmas` emits `SIGMAS`, so an
experiment is a change to `sigmas.py`'s arithmetic and not a patched sampler.
Anything tried there should be graded against that node at its defaults, which
is the release's own schedule, rather than against core's.
