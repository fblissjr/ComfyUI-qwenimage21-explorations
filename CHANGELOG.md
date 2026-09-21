# Changelog

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
