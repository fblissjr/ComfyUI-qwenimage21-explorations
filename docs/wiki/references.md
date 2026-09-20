# The checkouts under `coderef/`: what each one is good for

last updated: 2026-09-20

`coderef/` holds the reference checkouts. `ls -l coderef/` is the list of what
is currently on disk — some symlinks, some real clones — and this page is what
each one is *for*: what it implements, what has been read out of it, and what
it is not evidence of.

**Written by a person.**

**`coderef/` is gitignored and machine-local.** Nothing under it is repo
content, and several entries are symlinks that resolve only on the machine that
made them. This matters beyond housekeeping: the three `bridge-*` documents
under `docs/` are **tracked** and cite `coderef/ComfyUI-QwenImageWanBridge`
paths throughout, so **a fresh checkout has those findings but cannot follow
their pointers.** That is a known and accepted property of those documents —
they were written to carry the finding, with the pointer as provenance for
whoever has the clone.

**Do not import Python from `coderef/`.** Port from it. A bench script that
requires a clone and prepends it to `sys.path` is unrunnable on a box that has
the package and no checkout.

**A revision is an observation point, not a pin.** Everything below was read on
the date in this page's header. A checkout moves under you; re-read before
quoting.

---

## The upstream reference

| checkout | what it is | reach for it when |
|---|---|---|
| `coderef/Qwen-Image-2.1` | **the vendor's own repo.** `prompt_rewrite/pe_core.py` is the reference runner this harness mirrors: `load_system_prompt` for resolution order, `PROFILES` for the sampling settings, `Profile.image_max_pixels` for the input cap. `prompt_rewrite/prompts/` holds the canonical system prompts, and `prompt_rewrite/data/` the example briefs that `scripts/smoke_heylook.py` runs verbatim | you need to know what the reference implementation actually does at a stage, or you are changing anything in `profiles.py` or `templates.py` |

Two roots the documents here also cite, neither of them under `coderef/`:
`ComfyUI/` itself, the install this node lives in, which is the authority for
what the ComfyUI path does; and `<models>/`, wherever the checkpoints live on
your machine, which is the authority for the system prompts, the chat template
and the processor config. `<qwen-image-2.1-repo>` in the older documents means
the first row above.

## The tool and the serving path

| checkout | what it is | reach for it when |
|---|---|---|
| `coderef/llm-compressor` | the quantization library the calibration work reasons about: what a calibration batch actually does, what AWQ and GPTQ observe, how the collator handles ragged rows, what the library's own examples do | you are reading [`../calibration-research.md`](../calibration-research.md) and want to check a claim, or the quant line reopens |
| `coderef/heylook-models` | the mlx-vlm conversions served over the LAN, config files only. **Not a reference implementation** — it is the artifact the current reference arm runs on, and its configs are what "verified bf16" was verified against | you need to know what precision an arm actually ran at, or whether a conversion carries the trained template unmodified |

## The predecessor, and the sister project

| checkout | what it is | reach for it when |
|---|---|---|
| `coderef/ComfyUI-QwenImageWanBridge` | **the retired predecessor.** Built for the older stack — an earlier edit model with an earlier VL encoder — so treat every finding as describing that stack unless the mining document says otherwise. Three tracked documents mine it: [`../bridge-dead-ends.md`](../bridge-dead-ends.md), [`../bridge-prompting-findings.md`](../bridge-prompting-findings.md), [`../bridge-encoder-findings.md`](../bridge-encoder-findings.md) | you are about to build something it already tried. Read the do-not-retry list first |
| `coderef/ComfyUI-h3-explorations` | **the sister project**, and the source of this wiki's shape. Where the prior quantization effort's record lives, and where the house conventions these documents follow were worked out | you want the prior effort's evidence, or a convention question about how these documents are written |

### What the predecessor is not evidence of

Stated here because it is a property of the *reference*, and because the
checkout reads like it has more behind it than it does:

- **Its experimental record does not exist.** Three measurement rigs were
  built; none produced a recorded result. Its methodology document's numbers
  sit under "expected results" headings and are hypotheses. Do not quote a
  similarity or a distance from it as measured.
- **Its archived code is gone, and only the reasons survived.** Several
  directories its own documentation cites repeatedly are absent from the
  checkout, one present but empty. The mining documents say which.
- **Some of its findings are about parameters that no longer exist**, even
  where the mechanism still applies. Each mined finding carries an
  "applies to 2.1?" judgement; read that line before carrying anything across.
