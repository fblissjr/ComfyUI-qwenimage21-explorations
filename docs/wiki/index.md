# The wiki: where to start, and who owns each answer

last updated: 2026-09-20

Written by hand. A router, not an authority: it states no fact about these
models that is not owned somewhere else, and where it disagrees with an owner
the owner is right.

**One exception, declared rather than left to be noticed.**
[`upstream.md`](upstream.md) is an owner page living in a router directory: it
states the cross-implementation read because no document under `docs/` owns
that, and inventing one to route to would be ceremony. Every other page here
routes.

**Most of what is written down here is research, not results.** It was read out
of the checkpoints, out of ComfyUI core, out of the upstream repo, and out of a
retired predecessor repo. There has been one end-to-end run, and its result is
recorded in [`../quantization-strategy.md`](../quantization-strategy.md)
section 25. Each page below says which kind of thing it is holding, and the
documents themselves tag their claims.

## Written pages

| page | what it is for |
|---|---|
| [`stages.md`](stages.md) | one row per stage of one expansion: our code, the document that owns it, the check that would go red if it broke, and the implementation to compare against. The guard column is the point |
| [`contract.md`](contract.md) | **the answer contract.** What the checkpoints' system prompts state, what `answer.grade()` encodes as this repo's reading of them, and what neither covers. The authority is not in this repo, which is the first thing to know about it |
| [`decisions.md`](decisions.md) | what was decided, withdrawn or corrected, and when. Newest first, one line per decision with where it lives now |
| [`upstream.md`](upstream.md) | **how the other implementations run 2.1**, for text-to-image and for edit: what five of them agree on, the three bookkeeping strategies they use for one fusion, and the stage none of them runs. The wiki's one owner page |
| [`sizing.md`](sizing.md) | **reference image sizing**: why one number feeds two readers in 2.1, the five postures toward the processor's second resize, and the one place ComfyUI cannot notice a divergence. Start here if you know H3's `qwen_view` knob |
| [`sampling.md`](sampling.md) | **steps, guidance and the sigma schedule**: what each implementation sets, the two places ComfyUI's schedule departs from the release, and the half of that gap a stock node already closes |
| [`references.md`](references.md) | the checkouts under `coderef/`: what each one is for, and what it is not evidence of. Read before proposing a borrow |
| [`next_steps.md`](next_steps.md) | **what to do next, and only pointers to why.** The open items are scattered across the long documents; this is the entry point to them |

## Read these before you start

| file | what it answers |
|---|---|
| [`../quantization-strategy.md`](../quantization-strategy.md) | the spine. Working notes, numbered sections, cited from everywhere else — including by its own later sections, which withdraw earlier ones rather than editing them away. The banner at the top says what is parked, what is not, and what would reopen it |
| [`../calibration-research.md`](../calibration-research.md) | what a calibration corpus for these checkpoints would have to look like, and the specific degeneracy an ordinary row shape has here. Section 11 is what its author could not verify |
| [`../convrot-research.md`](../convrot-research.md) | what ConvRot is, and what the literature does and does not say about rotating vision towers and hybrid linear-attention models. Every claim carries an evidence tag; the closing section lists what to verify before relying on it |
| [`../bridge-dead-ends.md`](../bridge-dead-ends.md) | the predecessor repo's record of what failed, with each "why" tagged stated, reconstructed, or record silent. Section 15 is the do-not-retry list |
| [`../bridge-prompting-findings.md`](../bridge-prompting-findings.md) | the predecessor's prompting work, with a claim ledger. Its headline is that three measurement rigs were built and none produced a recorded result |
| [`../bridge-encoder-findings.md`](../bridge-encoder-findings.md) | the predecessor's encoder and conditioning work, and the two warnings from it that changed code here |
| [`../../templates/README.md`](../../templates/README.md) | why the canonical system prompts are not in this repo, and what `templates/` is for instead |
| [`../../CHANGELOG.md`](../../CHANGELOG.md) | every change, by version |

## Code and directories

| path | the rule |
|---|---|
| `src/qwenimage21_explorations/` | pure modules, importable without ComfyUI. **Their docstrings are the owner documents for their own behaviour** and carry the reasons; read the module before any page here that summarises it |
| `src/qwenimage21_explorations/nodes/` | ComfyUI V3 nodes wrapping those modules, adding nothing. The node list is append-only: saved graphs match widget values by index |
| `scripts/config_census.py` | reads a quantized checkpoint's header with no GPU, no ComfyUI and no torch. Exits non-zero on `full_precision_matrix_mult`, the flag that makes a checkpoint storage-only however good its format |
| `scripts/smoke_heylook.py` | the end-to-end run, over the upstream example briefs verbatim |
| `tests/` | the run command is in [`../../README.md`](../../README.md). What they do and do not cover is [`stages.md`](stages.md)'s guard column |
| `templates/` | local system-prompt variants only. A variant there is a hypothesis, not a drop-in |
| `coderef/` | reference checkouts. Gitignored, machine-local, part symlink. Port from them, never import — [`references.md`](references.md) |
| `internal/`, `data/` | gitignored: reference material and run records. A tracked document citing either is citing something a fresh checkout does not have |

## What nothing checks

There is no generator behind this wiki and no reachability check. **Nothing
verifies that the links on these pages resolve, and nothing prints the
documents under `docs/` that no link from here reaches.** Both are hand
discipline.

Said plainly rather than left for a reader to discover, because it is the same
move as [`stages.md`](stages.md)'s guard column: an unguarded thing that reads
as guarded is worse than one that says so.
