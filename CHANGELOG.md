# Changelog

## 0.2.0

### Added

- `src/qwenimage21_explorations/sage.py` and `QwenImage21SageAttention` --
  routes 2.1's unmasked image attention through SageAttention's INT8-QK /
  FP8-PV kernel, via ComfyUI's `optimized_attention_override`. The module
  docstring is the owner document: what the model hands an override, why the
  policy gates on the K length rather than Q, and what it declines.

  The shape is favorable and was read rather than assumed: head_dim 128 is
  sage's native size, and 2.1's image segments -- the references and the
  target -- pass no mask. Only the short text segments carry one, and they are
  declined by default. Installing the override leaves the prefix K/V cache on;
  the model's `hooked` guard does not list it.

  Two things it deliberately does not do, both stated in the docstring rather
  than left to be discovered: it does not claim the memory saving the H3 pack's
  forward patch gets, because an override cannot reach the caller's q/k/v, and
  `auto` does not carry H3's rotated quantizer, which was graded on H3
  captures and is ungraded here.

  Masked calls, when `sage_masked` opts them in, go to
  `sageattn_qk_int8_pv_fp16_triton` rather than to whatever `sage_mode` names.
  Not a preference: the sm89 fp8++ general-mask path serves only a trailing
  window of the keys and silently ignores the mask before it, which a causal
  mask -- what 2.1's text segments carry -- falls outside of. Found while
  smoke-testing this node; characterised in the sage fork at
  `tests/repros/repro_fp8_mask_window.py` and recorded under its "Known kernel
  bugs".

  **No speed claim.** The kernel has been exercised at 2.1's shapes and is
  correct there; nothing has been timed, and there is no A/B. The one that
  would produce a number is named in the docstring.

- `scripts/smoke_sage.py` -- runs the real kernel at the two shapes the model
  produces, prints each beside SDPA, and exits non-zero if a call falls back
  silently or a masked call does not honour its mask. The mask check is
  structural rather than an rtol threshold: under a causal mask row 0 attends
  to one key, so its output must BE that key's value, and an approximate
  kernel has nothing to hide behind. Forcing masked calls back onto the fp8
  path was confirmed to turn it red.

- `tests/test_sage.py` -- the routing policy and the override's reshape
  contract, on stubbed kernels, with no CUDA context. Seven deliberate
  mutations were run against it; the first pass caught that the mask branch
  was dead (every masked case was also short, so deleting the branch left the
  suite green) and that the head-major branch could not tell Q from K. Both
  now have a test that decides them.

### Changed

- A server preset with no system prompt overrides nothing, and when nothing
  else supplies one the expander sends no `system` field and reports
  `system_source` "none", instead of failing the run. A graph with no preset
  and no checkpoint runs the same way rather than raising.
- A preset can raise the profile's `max_tokens` but no longer lower it, so a
  general-chat preset cannot undo the t2i cap that prevents truncated traces
  (`profiles.py::with_preset`).

## 0.1.0

First working harness for the Qwen-Image 2.1 prompt expanders, plus the
research that shaped it.

### Added

- `src/qwenimage21_explorations/` — pure modules, importable without ComfyUI:
  - `chat.py` builds the chat string the expanders were trained on. Ends the
    generation prompt with an OPEN `<think>` block when thinking is on, which
    ComfyUI's bundled template does not do.
  - `answer.py` splits the thinking trace, parses the JSON, and grades the
    mode-dependent contract. `parse_ok` is syntax; `contract_ok` catches a model
    emitting valid JSON while breaking the rules.
  - `templates.py` resolves a system prompt, preferring the checkpoint's own
    copy, mirroring upstream's order.
  - `profiles.py` holds the reference sampling settings and a greedy variant for
    comparison runs.
  - `backends/heylook.py` — Anthropic-conformant client; folds heylook's
    separate thinking block into ComfyUI's inline shape so one parser grades
    both.
- Three ComfyUI nodes (V3 schema) wrapping those modules, adding nothing.
- Two more nodes, and example graphs for both modes:
  - `QwenImage21PEExpand` — the expander as one chat-shaped node: resolves the
    system prompt, calls heylook with the reference sampling profile, parses and
    grades the answer. Names a truncated response rather than letting it read
    downstream as a model fault.
  - `QwenImage21EncodeStructured` — the encode node with its fixed parts opened
    up (system turn, `keep_vision`, which reference sets the canvas). Defaults
    reproduce the stock node; changing the system prompt is off-distribution.
  - `example_workflows/` plus `scripts/build_example_workflows.py`, which
    regenerates them, `--check`s them against disk, and with `--server`
    validates node types, input names and widget counts against a running
    ComfyUI. Both graphs pass ComfyUI's own prompt validation.
  - `chat.render_encoder_prompt` assembles the encoder's chat string instead of
    formatting a template, which breaks on a system prompt containing braces.
- `scripts/config_census.py` — reads any quantized checkpoint with no GPU, no
  ComfyUI and no torch. Reports layers, configs and module roles with in/out
  features, and exits non-zero on `full_precision_matrix_mult`.
- `scripts/smoke_heylook.py` — end-to-end run over the upstream example briefs.
- `docs/wiki/` — hand-written router pages, linked from the README: the stage
  cross-index with its guard column, the answer contract's authority chain,
  dated decisions and withdrawals, the `coderef/` map, and the open items
  gathered out of the long documents. Nothing generates or checks them.
- `docs/wiki/upstream.md` — how diffusers, sglang, DiffSynth-Studio, LightX2V
  and ComfyUI core each run 2.1 for text-to-image and for edit: the conditioning
  contract they agree on, the three bookkeeping strategies they use for one
  fusion, and the expander stage none of them runs. The wiki's one owner page.
- `QwenImage21Sigmas` — the schedule the checkpoint's scheduler config asks
  for: dynamic shift read from the latent's own shape, plus the terminal
  stretch core has no equivalent for. Emits `SIGMAS` for
  `SamplerCustomAdvanced`. `terminal_mode` names how the schedule ends —
  `release` (the checkpoint's), `off` (core's), `stop_short` (ends at the
  terminal, never reaching zero) — so the ordering is a choice rather than
  something to get right silently, and `stretch_to_terminal` raises on a curve
  that already ends at zero. Arithmetic in
  `src/qwenimage21_explorations/sigmas.py`, importable without ComfyUI.
- The example graphs now sample through `SamplerCustomAdvanced` with that node,
  because `KSampler` builds its own sigmas. One latent feeds both the sampler
  and the schedule, so the shift cannot be computed for a different canvas than
  the one being sampled.
- `prompt_bank/` — prompts in the register the expanders actually emit. Real
  outputs marked `source: generated` are the yardstick; `hand-written` entries
  are held to their shape. A brief and a rewritten prompt are different
  registers and the encoder only sees the second, so driving a render from a
  brief is off-distribution.
- `docs/wiki/sampling.md` and `scripts/sigma_schedule.py` — the sigma schedule:
  what the release's scheduler config asks for, what each implementation does,
  and ComfyUI's two departures (a constant shift where the release asks for a
  dynamic one, and no `shift_terminal` at all). `--widgets` prints the stock
  `ModelSamplingFlux` values that close the dynamic half exactly.
- `docs/wiki/sizing.md` and `scripts/refview_bounds.py` — reference image
  sizing: why 2.1 couples the encoder's and the VAE's view of a reference when
  H3 did not, the five postures toward the processor's second resize, and the
  bounds ComfyUI applies against the ones the checkpoint declares. The script
  prints where the two views part company; no GPU, no torch.
- `docs/quantization-strategy.md` and two research reports.
- 24 tests.

### Notes

- The quantization line is parked; see the banner in
  `docs/quantization-strategy.md` for why and what would reopen it.
- The canonical system prompts are deliberately not vendored. They ship with the
  checkpoints, and a third copy can drift silently.
