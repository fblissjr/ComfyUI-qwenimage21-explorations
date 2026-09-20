# ComfyUI-qwenimage21-explorations

Research repo for Qwen-Image 2.1: a harness for its two prompt-expander models,
and tooling for quantizing them and its text encoder.

Nothing here is established beyond one machine (an RTX 4090, ComfyUI at
`c194dd00`) and a set of checkpoint reads. Claims carry pointers; re-derive
rather than trust.

## What's here

```
src/qwenimage21_explorations/   importable, no ComfyUI dependency
  chat.py       builds the chat string the expanders were trained on
  answer.py     parses the answer and grades it against the task contract
  templates.py  resolves a system prompt (checkpoint copy preferred)
  nodes/        thin ComfyUI wrappers, adding nothing of their own
scripts/        run directly; no GPU, no ComfyUI
docs/           findings, with pointers
templates/      local system-prompt variants (canonical ones are NOT vendored)
data/           run outputs and local artifacts (gitignored)
coderef/        reference checkouts (gitignored) -- port from, never import
internal/       unshared notes (gitignored)
```

## The models

Both expanders are fine-tuned **Qwen3.5-VL 9B** (`model_type: qwen3_5`, hybrid
GatedDeltaNet/full attention), one for text-to-image and one for image editing.
They are not the conditioning text encoder, which is Qwen3-VL 8B and shares no
architecture, vocabulary or tokenizer with them.

## Two things the nodes exist to get right

**The generation prompt ends with an OPEN `<think>` block.** That is the trained
format -- both official runners pass `enable_thinking=True`, and the
checkpoints' `chat_template.jinja` opens the block rather than closing it.
ComfyUI's bundled qwen35 template appends nothing there, and its thinking-off
form differs in whitespace too. Because our string starts with `<|im_start|>`,
ComfyUI passes it through untouched.

**`parse_ok` is the weakest useful gate.** The answer contract is
mode-dependent: `<imageN>` tags are mandatory at two or more images and
forbidden at one, `wh_ratio` and `ratio_follow` are mutually exclusive on edit,
and resolution strings must never appear in `rewritten_prompt`. A degraded
model that still emits valid JSON while breaking those rules passes a syntax
check. `QwenImage21PEParse` reports both.

## Check a checkpoint yourself

No GPU, no ComfyUI, no torch:

```
python scripts/config_census.py <any-quantized.safetensors>
```

Prints how many layers are quantized, how many distinct configs they hold, and
which module roles hold each -- with in/out features beside the role, because a
"role" that turns out to be uniquely identified by shape is the confound that
dissolved three separate cross-model stories while this was being written.

Exits non-zero if any layer carries `full_precision_matrix_mult`, which makes a
checkpoint storage-only however good its format.

## Tests

```
uv run --with pytest python -m pytest tests/ -q
```

## Licence

MIT, see LICENSE. The models, their system prompts and the upstream repo carry
their own licences.
