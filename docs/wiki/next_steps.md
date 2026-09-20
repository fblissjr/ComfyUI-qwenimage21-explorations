# Next steps

last updated: 2026-09-20

**A written page, not generated. It says what to do next and where the
reasoning lives; it restates none of it.** Kept to pointers on purpose: when it
disagrees with the document it points at, that document is right and this page
is stale.

The open items in this repo are real but scattered: each long document under
`docs/` ends with its own list of what it could not settle. This page is the
entry point to those lists.

---

## Held by the park

The quantization line is parked by the owner — [`decisions.md`](decisions.md).
Nothing below in this section is work to pick up; it is here so that reopening
the line starts from the right place rather than from the top.

- **Start at tool choice, then the corpus arithmetic.** The banner at the top
  of [`../quantization-strategy.md`](../quantization-strategy.md) names the two
  sections to read in order, and says which of the two is the non-obvious part
  that still stands.
- **The genuine remaining unknown is repacking**, not format availability:
  whether a portable AWQ artifact converts cleanly into the accelerated
  layout's expected packing order, zero-point convention and group axis.
  Strategy section 21, which is also what withdrew the earlier verdict.
- **Decide the acceptance measurement and the baseline arm before building
  anything.** Strategy section 13b. The reference arm is already known not to
  be a clean sheet ([`contract.md`](contract.md) section 4), which is exactly
  the input that decision needs.
- **The one real unknown on the recipe side is a run, not a reading**: the
  sequential pipeline has never been exercised on a dense model of this
  architecture here. Strategy section 20, "Open unknown".

## Not parked

- **The three capability gaps between the image model and its own expanders.**
  Region selection is native and in-band, and the expanders' system prompts
  have no vocabulary for any of it; strategy section 23 predicts a specific
  failure for the annotation case and says why a mask image has no category in
  the prompt. **Untested**, and the document argues they belong in any eval
  corpus as their own strata. This is the most concrete open work here that
  needs no quantization decision.
- **The qwen35 pre-tokenizer divergence is worth reporting upstream**, and the
  report is the whole of the work — the vocabulary is already correct, so there
  is nothing to carry here. It affects every qwen35 model on any mark-heavy
  script, not only these checkpoints. Strategy section 29, "The fix";
  [`decisions.md`](decisions.md) for what it does and does not break.

- **A prediction is already written down and is cheap to test.**
  [`../convrot-research.md`](../convrot-research.md), "Things to verify before
  relying on them", item 1 states in advance what every layer's group size
  should be and what a violation would mean. `scripts/config_census.py` is the
  instrument, and it needs no GPU.
- **The rest of that document's verification list** is web and kernel reading,
  including one claim that came from a search summary rather than a paper body.
- **What the calibration note could not verify**:
  [`../calibration-research.md`](../calibration-research.md) section 11. The
  first item gates several of its own recommendations, and it resolves as a
  by-product of generating any corpus — so it is cheap once anything runs.
- **What the predecessor raised and did not answer**:
  [`../bridge-encoder-findings.md`](../bridge-encoder-findings.md), "Open
  questions this repo raises but does not answer". These are encoder-side and
  become live if work moves downstream of the expanders.
- **Budget before looping.** Strategy section 25's "Cost, for planning": output
  length varies widely across rows and image-bearing rows are slow on the
  current backend, so a stratified corpus is not a quick loop. Prefer the
  greedy configuration for anything being compared.

## Guards that do not exist

Not a to-do list — the repo's habit is that a new check needs a real instance it
would have caught. Recorded so that nobody reads an unguarded thing as guarded.
[`stages.md`](stages.md)'s guard column is the full table; these are the ones a
reader is most likely to assume are covered.

- **`templates.py`'s resolution order** and **`profiles.py`'s reference values
  against upstream**. Both are conventions the code implements, and a drift in
  either is silent and reads downstream as a model problem.
- **The ComfyUI generation path.** No test exercises it; what makes our string
  authoritative there is a `startswith` test inside ComfyUI's own tokenizer.
  `tests/test_prompt_structure.py` pins the properties that test depends on,
  which is as close as this repo gets.
- **The links on this wiki.** Nothing checks that they resolve, and nothing
  lists the documents under `docs/` that no link reaches —
  [`index.md`](index.md).
