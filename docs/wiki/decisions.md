# Decisions, withdrawals and corrections

last updated: 2026-09-20

Written by hand. One entry per decision the owner made, or per claim that was
corrected or withdrawn: what changed, what it used to say, where it lives now.
Newest first.

Older and finer-grained history is not copied here:

- [`../../CHANGELOG.md`](../../CHANGELOG.md): every change, by version.
- `git log`: the commit messages carry the reasoning for code changes.
- [`../quantization-strategy.md`](../quantization-strategy.md): its numbered
  sections are the working record, and **later sections withdraw earlier ones
  in place rather than editing them away**. Section 21 against section 20 is
  the worked example. Read a section together with anything that cites it.

**This repo is days old and everything below happened on one day.** The single
date heading is accurate, not an artifact of a young page.

---

## 2026-09-20

- **The quantization line is parked** (the owner). The prompt-expander path
  stays on the verified-bf16 conversions served over the LAN; no quantized
  artifact is being built. In one line: the format ComfyUI can execute fast is
  produced by a data-free quantizer, so the calibration work has nothing to
  feed, and no measurement has shown a quality problem worth solving. The
  banner at the top of [`../quantization-strategy.md`](../quantization-strategy.md)
  is the authority, including **what would reopen it** and the explicit list of
  what is *not* parked: the harness in `src/`, the census in `scripts/`, and the
  capability gaps.

- **Section 20's "tool choice splits by destination" verdict is withdrawn by
  section 21.** It rested on llm-compressor's W4A16 artifact giving no kernel on
  this card. Verified on this machine that an accelerated W4A16 path exists in
  the installed `comfy_kitchen`; it is simply not registered in core, and a
  custom node can register a format at runtime — the predecessor project's AWQ
  loader does exactly that. **What core natively supports therefore bounds only
  what works with zero custom code.** Both sections stand as written; section 21
  says which parts of 20 no longer hold, and names the genuine remaining
  unknown (whether the artifact repacks cleanly into the kernel's expected
  layout). In [`next_steps.md`](next_steps.md).

- **ComfyUI mis-tokenizes mark-heavy scripts for these expanders, and it is not
  a stale tokenizer bundle.** The bundled vocabulary and merge rules are
  identical to the checkpoints', so shipping a different bundle would change
  nothing. What differs is the pre-tokenizer: core builds a Qwen2 tokenizer
  whose hardcoded regex matches letters, while these checkpoints declare letters
  **and combining marks**, so Thai and Devanagari split where they should not.
  Round-trip is safe both ways — nothing is corrupted; the model receives a
  segmentation it was never trained on, on exactly the case the edit prompt
  gives a worked example for. The heylook path is unaffected, and **the
  conditioning encoder is unaffected and was checked the same way** — it
  declares no custom regex, so core's choice is correct for it. The fix belongs
  upstream and is not ours to carry. Strategy section 29;
  `tests/test_tokenizer_parity.py` xfails strictly, so it becomes XPASS the day
  upstream fixes it.

- **Calibration data is the priority surface, not the recipe** (the owner, via
  h3guy). It is the input with the most control and the least prior art, and it
  is where the previous effort was weakest — that effort spent its time on the
  recipe, and the recipe is not where it lost. Strategy section 16;
  [`../calibration-research.md`](../calibration-research.md) is the work that
  followed from it.

- **The canonical system prompts are deliberately not vendored.** They ship
  inside each checkpoint and were verified byte-identical to the upstream
  copies. A third copy can drift out of sync with the weights, and that failure
  is silent: fluent output graded against the wrong contract. `templates/` holds
  local variants only, and a variant there is a hypothesis — whether these
  models are prompt-agnostic at all is untested.
  [`../../templates/README.md`](../../templates/README.md), and
  [`contract.md`](contract.md) for what the choice costs.

- **Comparison runs use a real greedy path, not a small temperature.**
  `profiles.py::GREEDY`. You cannot A/B a quantized candidate on sampled text,
  and emulating determinism with an epsilon temperature buys nothing that
  argmax does not already give. Reference sampling lives in `PROFILES` beside
  it, as a constant rather than in prose, because a harness relying on any
  backend's defaults is not running the reference configuration.

- **The two backends are deliberately not a provider abstraction**
  (`backends/__init__.py`). They differ in which tokenizer and template apply,
  not only in transport. What the harness holds steady across them is the
  effective prompt and the sampling settings.

- **The bf16 reference arm is not a clean sheet, and that is the useful part of
  it.** A row of the first end-to-end run fails a tag rule with its ratio fields
  correct; precision was verified from the conversions' own configs, so this is
  the model's behaviour and not conversion damage. **Any acceptance threshold
  set against an assumed-perfect reference would have been wrong from the first
  run.** Strategy section 25; [`contract.md`](contract.md) section 4 for the
  rule tension behind it.

- **Client-side image-token prediction is kept even though the server bug that
  motivated it was fixed upstream.** A server that reports a number is not the
  same as a number we can check, and the predecessor's tooling shows how easily
  this particular quantity goes unmeasured. `vision.py`; strategy section 25.

- **The earlier claim that region masking happens downstream at the latent
  stage was wrong for this model.** Region selection is native and in-band —
  drawn or painted onto the reference image, or passed as another image.
  Corrected in place at strategy section 23, which also records that the
  expanders' system prompts have no vocabulary for any of it.

- **The predecessor repo's experimental record does not exist, and nothing from
  it may be quoted as measured.** Its methodology document reads like results
  but is a plan: every number sits under an "expected results" heading, the
  declared output directory is absent, and no results file exists in the
  checkout. Three separate measurement rigs were built there and none produced
  a recorded result. [`../bridge-prompting-findings.md`](../bridge-prompting-findings.md)
  headline and claim ledger; [`../bridge-encoder-findings.md`](../bridge-encoder-findings.md)
  opens with the same calibration.

- **Two findings from that repo changed code here rather than staying in a
  document.** Never hardcode a token boundary against an editable system prompt
  — it produced a silent slice there, and `templates/` exists precisely to allow
  prompt variants. And assert on the final prompt string, never on the flags
  that produced it: a double-wrapped template went undetected there until a
  formatted-prompt output was added, in the same release that fixed it — no
  error, just conditioning nested one level deeper than intended, reading as a
  mediocre prompt. `vision.py` and `tests/test_prompt_structure.py` are the
  responses; strategy sections 27 and 28.
