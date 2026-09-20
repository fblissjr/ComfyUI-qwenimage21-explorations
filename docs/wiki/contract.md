# The answer contract: who states each rule, and what the harness encodes

last updated: 2026-09-20

The expanders emit a thinking trace and then a JSON object. Three things claim
to govern that object, and they are not the same thing. This page ranks them
and says which rules came from where, so that a failing row can be attributed
to the model, to the harness's reading, or to the rules pulling against each
other.

Written by hand. `answer.py`'s docstring owns the code side; this page owns the
question of authority.

---

## 1. The authority is not in this repo

Ranked, highest first:

1. **Each checkpoint's own `system_prompt.txt`.** It ships inside the
   checkpoint, and the answer contract is part of what the weights were trained
   on. The edit checkpoint's prompt is the authority for edit mode; the t2i one
   carries a narrower subset. Verified byte-identical to the upstream repo's
   copies by sha256 — see [`../../templates/README.md`](../../templates/README.md).
2. **`answer.py::grade`** — this repo's *reading* of those prompts, encoded as
   violation classes. Everything in section 3 below that is marked "reading" is
   a judgement someone made, not a quote.
3. **Everything else**, including this page and
   [`../quantization-strategy.md`](../quantization-strategy.md) section 22's
   edit-mode taxonomy, which was read out of the edit prompt and is a
   description of it.

**Consequence worth stating once.** The authority is not tracked here, on
purpose: a third copy drifts, and the drift failure is silent — fluent output
graded against the wrong contract. The cost of that choice is that **you cannot
check the harness against the authority without the checkpoints on disk.** When
a violation class looks wrong, open the checkpoint's `system_prompt.txt` and
read the rule; do not argue with `grade()` from this page.

## 2. The two gates are different questions

`parse_ok` is syntax: something JSON-shaped came back. It is the weakest useful
gate, and a degraded model that still emits well-formed JSON while breaking the
rules passes it.

`contract_ok` is the mode-dependent contract. It is the gate that exists
because the first failure mode a quantized candidate would plausibly show is
fluent-but-non-conformant output, which parse-only checks wave through.

Neither is a quality judgement. Both are satisfiable by a bad rewrite.

## 3. What is stated, and what is a reading

| rule | source | in the code |
|---|---|---|
| the rewrite may not carry resolution or aspect-ratio information | **stated** by the prompt, near-verbatim | `prompt:carries_resolution`. The rule is stated; **the detector is a regex**, so it is this repo's approximation of a rule expressed in prose, and it can fire on text that merely looks like a ratio |
| at one image, tags are forbidden in the rewrite — refer to the image naturally | **stated** by the edit prompt | `tags:present_at_single_image` |
| at two or more images, every image is tagged and its role stated | **stated** by the edit prompt's Image Reference Rules | `tags:missing_at_multi_image` covers the tagging half. **The role half is not checked** — see section 5 |
| a tag may not name an image that was not supplied | **reading.** Follows from the tags addressing supplied images, not stated as a rule | `tags:out_of_range` |
| exactly one of the two ratio fields carries a value, in edit mode | **reading** of how the edit prompt uses them | `ratio:both_set`, `ratio:neither_set` |
| the follow field, when set, is exactly an image tag | **reading** | `ratio_follow:malformed` |
| t2i carries the ratio field and not the follow field | **reading**, from the t2i prompt carrying the narrower subset | `ratio_follow:set_on_t2i`, `ratio:wh_ratio_missing` |
| the rewrite is non-empty | **reading.** No prompt states it; nothing useful survives without it | `field:rewritten_prompt_empty` |

`answer.py::parse` also accepts a second field name for the rewrite, because
the reference runners write their output records under a different key than the
system prompts use. That is tolerance at the boundary, not a contract rule.

## 4. The stated rules pull against each other, and the reference arm shows it

At a single image the edit prompt forbids tags in the rewrite, while the ratio
follow field is itself an image tag. The rule and the field want opposite
things at the same input.

This is not hypothetical. The first end-to-end run, on verified-bf16 weights,
had a row fail exactly this rule with its ratio fields correct — recorded in
[`../quantization-strategy.md`](../quantization-strategy.md) section 25, which
also establishes that it is the model's own behaviour and not conversion
damage.

**What follows from it:** the reference arm is not a clean sheet, and any
acceptance threshold set against an assumed-perfect reference is wrong before
the first candidate runs. That consequence is section 25's; it is repeated here
because a reader arriving at the contract from a failing row needs it.

## 5. What `grade()` does not check

Named because an unchecked rule that reads as checked is the failure this page
exists to prevent:

- **Whether a stated role is actually stated.** At two or more images the
  prompt requires each image's role — which is the canvas, which supplies
  material. Only the presence of tags is checked.
- **Anything about the rewrite's quality**, its language, or whether it
  preserved the user's own action verbs and spatial relations.
- **The thinking trace.** Its content is never inspected. Its *length* is the
  thing that bites, by way of the token cap — [`stages.md`](stages.md).
- **The edit-mode taxonomy.** [`../quantization-strategy.md`](../quantization-strategy.md)
  section 22 reads eight modes with distinguishing contracts out of the edit
  prompt. `grade()` applies one contract to all of them.
- **Region selection and transparency.** The prompts have no vocabulary for
  either, which is a gap in the prompts rather than in the grader — section 23,
  and [`next_steps.md`](next_steps.md).
