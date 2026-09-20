# Calibration corpus design for the Qwen-Image-2.1 prompt expanders and the 2.1 text encoder

Research note. 2026-09-20. Scope: calibration data only. Recipes, schemes and
feasibility are another agent's task; where a corpus decision constrains a recipe
decision I say so and stop there.

Everything below that is a code claim carries a path and a line number. Everything
that is an external claim carries a URL and a label for how much evidence stands
behind it. Where I could not verify something I say so rather than asserting it;
section 11 collects those.

The headline finding is in section 5. It is arithmetic, not analogy: for these two
checkpoints a calibration row built the ordinary way (system prompt plus user
brief, no assistant turn) is roughly 99 percent boilerplate by token count, and
because AWQ pools its channel statistic over tokens, a 512-row corpus of that shape
carries about as much distinct signal as a single example. That is the specific
degeneracy this model family invites and it is invisible to every check that looks
at row counts.

---

## 1. Measured inputs

These are the numbers the rest of the note reasons from. Re-derive them with:

```bash
python -c "
from tokenizers import Tokenizer
import pathlib
for name, d in [('T21','<models>/Qwen-Image-2.1-PE/Qwen-Image-2.1-PE-T21'),
                ('I21','<models>/Qwen-Image-2.1-PE/Qwen-Image-2.1-PE-I21')]:
    tok = Tokenizer.from_file(d+'/tokenizer.json')
    sp = pathlib.Path(d+'/system_prompt.txt').read_text().strip()
    print(name, len(tok.encode(sp, add_special_tokens=False).ids), len(sp))
"
```

Conditions: measured 2026-09-20, against each checkpoint's own `tokenizer.json`
and `system_prompt.txt` as they sit on disk today, using the `tokenizers` build in
`ComfyUI/.venv` (the llm-compressor `.venv` has no `tokenizers` or
`transformers` installed).

| quantity | T21 (t2i) | I21 (edit) |
| --- | --- | --- |
| `system_prompt.txt` tokens | 2426 | 4509 |
| `system_prompt.txt` chars | 9907 | 17708 |

User briefs, same tokenizer, taken from the shipped example inputs at
`<qwen-image-2.1-repo>/prompt_rewrite/data/`:

| brief | tokens |
| --- | --- |
| `a cat` | 2 |
| `一只在雨中弹吉他的柯基` | 8 |
| `cyberpunk noodle stall at night, wide shot` | 11 |
| `a vintage travel poster for Kyoto in autumn with the text "KYOTO" across the top` | 18 |
| `can u this image so that the head twist to the left while the body and arms not moved` | 19 |

Vision tokens per image, derived from `processor_config.json` (`patch_size: 16`,
`merge_size: 2`) and confirmed against the live processor: one token per 32x32
pixel block, so `(H/32) * (W/32)`. A 1024x768 image gives grid `[1, 48, 64]` and
768 image-pad tokens in `input_ids`. `pe_core.py::Profile.image_max_pixels` caps
source images at 1024*1024 pixels, so the ceiling is 1024 vision tokens per image;
`processor_config.json` sets `size.shortest_edge: 65536` (a pixel count, not an
edge length), giving a floor of 64.

---

## 2. What llm-compressor actually does with a calibration batch

### 2.1 How a dataset is specified

`DatasetArguments` in
`the llm-compressor checkout/src/llmcompressor/args/dataset_arguments.py`
is the whole surface. The fields that matter here:

- `dataset` (line 98) accepts a name string, a HF `Dataset`/`DatasetDict`, or a
  pre-built `DataLoader`. Passing an in-memory `Dataset` is the path for a custom
  corpus.
- `num_calibration_samples` (line 141), default 512.
- `max_seq_length` (line 110). Read the help text carefully: "Sequences longer
  than this will be truncated". There is no warning when truncation removes the
  part of a row you cared about.
- `data_collator` (line 79), default `"truncation"`.
- `use_loss_mask` (line 276): "Whether to use the 'loss_mask' field from the batch
  for AWQ loss calculation. When True, only tokens where loss_mask=1 contribute to
  the AWQ optimization objective."
- `shuffle_calibration_samples` (line 145), default True.
- `pipeline` (line 233), default `"independent"`; the registry infers
  `"sequential"` whenever any modifier declares `requires_calibration_data`
  (`pipelines/registry.py:56-60`).

If the dataset already has an `input_ids` column it is passed straight through
without re-tokenization (`datasets/utils.py:94-96`). That is the hook for a
fully pre-processed corpus, and it is the one to use here, because the
preprocessing for these models is not expressible as a `preprocessing_func` over
a text column once images are involved.

### 2.2 The collator silently truncates to the shortest row in the batch

`DataCollatorWithTruncation` (`datasets/utils.py:299-316`) takes the minimum
length across the batch and, if `max_seq_length` is set, the minimum of that and
`max_seq_length`, then slices every feature to it. With `batch_size=1` only the
`max_seq_length` cap applies. With `batch_size > 1` one short row truncates the
whole batch. The `loss_mask` key is truncated alongside `input_ids` (line 306), so
it stays aligned, but a row whose assistant turn falls past the cut ends up with an
all-zero mask and contributes nothing to the AWQ statistic while still being
counted as a sample.

### 2.3 What AWQ observes

Two separate things, both in
`the llm-compressor checkout/src/llmcompressor/modifiers/transform/awq/base.py`:

The smoothing statistic, at lines 415-447. A forward hook on each balance layer
takes `args[0].abs()`, flattens to `[tokens, hidden]`, optionally filters rows by
the loss mask, then accumulates a running sum and count:

```python
new_sum = masked_activations.float().sum(dim=0).cpu()
new_count = torch.tensor(masked_activations.size(0)).cpu()
```

So the statistic is a per-input-channel mean of `|x|`, pooled over every token that
passes the mask, across the entire corpus. It is token-weighted. A row's influence
on the smoothing scale is proportional to how many of its tokens survive the mask,
not to it being one row out of N.

The grid-search objective, at lines 763-797. Mean squared error between the
parent module's fp16 output and its output with quantized weights, restricted to
masked token positions:

```python
token_mask = loss_mask.to(fp16_batch.device) == 1  # (batch, seq)
fp16_masked = fp16_batch[token_mask]
```

Also token-weighted, also mask-filtered.

The mask never changes what the model sees. It is applied after the forward pass,
inside the hook, to decide which positions enter the statistic. Everything in the
row is still in the KV context and still conditions the activations at every later
position. That distinction is the whole of section 5.

One guard to be aware of: `use_loss_mask=True` raises if any resolved mapping is an
`up_proj -> down_proj` pair whose balance layer name contains `.experts.`
(`awq/base.py:214-222`, predicate at lines 886-903). These checkpoints are dense
(`config.json` has `mlp_only_layers: []` and no expert config), so this should not
fire, but it is resolved at runtime against actual module names and is worth
confirming on the first run rather than assuming.

### 2.4 What GPTQ observes, and the thing that matters

`modifiers/gptq/helpers.py:40-76`:

```python
num_added = inp.shape[0]
...
num_samples += num_added
inp = math.sqrt(2) * inp.to(dtype=GPTQ_PRECISION)
hessian += inp.matmul(inp.t())
```

Two things follow. First, `num_added` is the batch dimension, not the token count,
so the Hessian is normalized by number of sequences while being accumulated over
tokens; a longer row contributes proportionally more mass. Second, and more
important for corpus design: there is no mask. Grepping `loss_mask` across
`src/llmcompressor/` returns hits only in `awq/base.py`, the two pipelines,
`datasets/utils.py`, `args/dataset_arguments.py` and `core/state.py`. GPTQ's
Hessian sees every token in the row, system prompt included, with equal weight.

The corpus consequence, stated plainly: steering the statistic toward the
generation distribution is available under AWQ and is not available under GPTQ.
Under GPTQ the only lever is the token composition of the rows themselves.

### 2.5 What static activation quantization observes

`observers/min_max.py`. `MinMaxObserver` keeps a running elementwise min and max
across calls (lines 36-41); `_get_min_max` is `amin`/`amax` over the batch and last
dim (lines 71-74). The moving-average variant lerps with
`averaging_constant`, default 0.01 (line 57).

For any scheme with static activation scales, one outlier row sets the range for
the whole tensor and 511 well-chosen rows cannot pull it back. The dominant corpus
property in that regime is absence of outliers, not representativeness. For dynamic
per-token activation quantization the corpus does not touch activation scales at
all. Say which regime a claim applies to before acting on it.

### 2.6 Multimodal handling

There is no multimodal-specific machinery in the dataset layer. The pattern in
every VLM example is: call the processor yourself, produce whatever keys it emits,
and pass a trivial collator that unsqueezes a single sample
(`examples/multimodal_vision/qwen3_vl_example.py:69-72`):

```python
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}
```

Note this collator bypasses `DataCollatorWithTruncation` entirely, so
`max_seq_length` no longer truncates; truncation has to happen in your own
preprocessing (`max_length=...`, `truncation=True` on the processor call, as at
lines 60-62 of that file).

The sequential pipeline reads the mask once from the intermediates cache
(`pipelines/sequential/pipeline.py:123-131`) and sets `current_batch_idx` before
each forward (line 152), which is what lets the AWQ hook find the right mask.

### 2.7 Hybrid linear attention

Both `qwen3_5` and `qwen3_next` are supported, and the support is specifically
about the 3-to-1 `linear_attention`/`full_attention` interleave in `config.json`:

- AWQ: `modifiers/transform/awq/dynamic_mappings.py:225-229` registers
  `Qwen3_5ForConditionalGeneration` (this model's architecture) to
  `build_hybrid_attention_mappings`.
- SmoothQuant:
  `modifiers/transform/smoothquant/dynamic_mappings.py:47-93, 129-134`. The builder
  reads `layer_types` and restricts the attention `input_layernorm` regex to
  full-attention layer indices, because "linear-attention layers do not expose
  q/k/v projections" (lines 51-57).
- Tracing: `tracing_ignore` defaults include `_update_linear_attn_mask` and
  `_update_mamba_mask` (`args/dataset_arguments.py:206, 216`).
- Norms: `Qwen3_5RMSNorm` is in the offset-norm list
  (`modeling/offset_norm.py:60-62`).

Corpus implication, and it is a real one: only 8 of the 32 layers have q/k/v
projections to smooth, and the GatedDeltaNet layers carry recurrent state that
evolves along the sequence. A short row does not exercise the same state regime as
a long one. I found nothing in this repo that measures that, and nothing in the
literature either (section 4.4). Treat sequence length as a variable to be checked
rather than a number to copy from a Llama example.

### 2.8 What the library's own examples do

| example | data | samples | seq len | assistant turn? |
| --- | --- | --- | --- | --- |
| `examples/multimodal_vision/qwen3_vl_example.py` | flickr30k, fixed question "What does the image show?" | 512 | 2048 | no, `add_generation_prompt=True` |
| `examples/multimodal_vision/qwen_2_5_vl_example.py` | same | 512 | 2048 | no |
| `examples/awq/qwen3_next_thinking_example.py` | Magpie-Reasoning-V2 CoT | 256 | 4096 | yes, masked |
| `examples/awq/llama_example_with_masking.py` | ultrachat_200k | 256 | 512 | yes, masked |
| `src/llmcompressor/transformers/data/flickr_30k.py:54-75` | flickr30k | n/a | n/a | yes, the caption |

The VLM examples and the registry's own flickr wrapper disagree about whether to
include an assistant turn, which is a fair signal that nobody has settled it.

The thinking example is the closest prior art to our problem and it states its
rationale in a comment (`qwen3_next_thinking_example.py:11-14`):

> Using a reasoning dataset with loss_mask ensures AWQ weight scales are fit to
> the thinking-mode activation distribution.

It also filters all-zero masks (line 84), and it carries a caveat at lines 60-64
about `add_generation_prompt=True` injecting a `<think>\n` primer absent from the
full conversation, which would over-count `prompt_len`. That caveat does not apply
to these checkpoints. See section 6.3; I verified the opposite empirically.

The docs page `docs/steps/choosing-dataset.md:27-48` says domain alignment matters,
recommends 128 to 512 samples, and offers ultrachat / open-platypus / wikitext /
c4. No guidance on system prompts, thinking traces or structured output.

---

## 3. Published practice: VLMs

Sample counts and sources are consistent across the literature, and the consistency
is mostly inherited rather than re-derived.

- 128 image-caption pairs is the traditional baseline; 512 is the recent drift
  upward. Sources are COCO Captions, usually the ShareGPT4V re-captioned variant,
  or flickr30k.
  ([MBQ](https://arxiv.org/html/2412.19509v1),
  [VLM best practices](https://arxiv.org/html/2601.15287v1))
- The one VLM-calibration finding with a mechanism behind it is MBQ's. Evidence:
  measured gradients, not folklore.

  > "the average absolute gradient value of the language token features is over
  > 10x larger than that of vision tokens"

  > "treating vision tokens and language tokens equally during calibration may
  > over-emphasize the insensitive vision tokens, resulting in notable accuracy
  > loss"

  ([MBQ, arXiv 2412.19509](https://arxiv.org/html/2412.19509v1))

  This matters directly for the edit model: at the 1M-pixel cap a single source
  image contributes up to 1024 tokens, and a multi-image edit contributes several
  times that. Under an unmasked statistic those tokens compete with the ~20-token
  brief and the assistant turn on equal footing.

- Whether to quantize the vision tower is contested. The llm-compressor README
  asserts "compressing these parameters offers little benefit"
  (`examples/multimodal_vision/README.md:22`); the best-practices paper measures
  the opposite, that "ViT and LLM exhibit comparable importance in model
  performance, despite significant differences in parameter size"
  ([arXiv 2601.15287](https://arxiv.org/html/2601.15287v1)). This is out of scope
  here because the owner has already decided to exclude the vision tower, but note
  that the decision makes the vision tokens' role purely one of conditioning: they
  enter the language path as embeddings and the language-path statistics do see
  them.

- I found nothing on multi-image calibration specifically. Every source I read
  calibrates on one image per row. The edit model takes 1..N. Treat N > 1 coverage
  as unsupported territory.

---

## 4. Published practice: instruction-tuned and structured-output models

### 4.1 Distribution matching has direct evidence, and it is method-dependent

The AWQ paper's own ablation is the cleanest primary result. Two Pile subsets
(PubMed Abstracts, Enron Emails), OPT-6.7B, INT3-g128, each subset used as
calibration and evaluated on both:

> "Using the same calibration and evaluation distribution works best... when using
> a different calibration distribution, AWQ only increases the perplexity by
> 0.5-0.6, while GPTQ has 2.3-4.9 worse perplexity."

> "AWQ is less sensitive to the calibration set distribution since it only measures
> the average activation scale from the calibration set, which is more
> generalizable across different dataset distributions."

([AWQ, arXiv 2306.00978](https://arxiv.org/pdf/2306.00978))

So matching helps both, and it helps GPTQ roughly four to eight times more than it
helps AWQ. Evidence quality: good. Caveats: one small dense model, one bit-width,
perplexity only, two domains that differ in topic rather than in conversational
structure. Nothing in it speaks to structural mismatch of the kind we have here
(same topic, completely different position in the chat template).

### 4.2 Calibration data variance is larger than commonly assumed

Williams and Aletras, ACL 2024, is the systematic study.

> "we find substantial variations in downstream task performance, contrasting
> existing work that suggests a greater level of robustness to the calibration
> data"

Accuracy on RTE for a pruned LLaMA-7B moved by nearly 10 points depending on the
random seed used to sample C4. Setup: 128 examples of 2048 tokens each, 262,144
tokens per calibration set, ten non-overlapping sets per source dataset, five
sources (C4, CNN-DM, RedPajama, RefinedWeb, Wikipedia). Methods: GPTQ, SpQR,
SparseGPT, Wanda. Quantization dispersed less than pruning.
([arXiv 2311.09755](https://arxiv.org/abs/2311.09755),
[ACL Anthology](https://aclanthology.org/2024.acl-long.544/))

Their recommendations, verbatim from section 5: release the calibration data;
evaluate across several calibration sets during development to see how sensitive
you are; manually inspect randomly sampled calibration data to remove anomalous
examples.

I checked specifically and they do not test matched versus mismatched deployment
distributions, and they do not test AWQ or SmoothQuant. Do not cite this paper for
the domain-matching claim; cite AWQ's ablation for that.

The practical reading for us: run the same recipe against two or three corpus
variants and compare, because a single run tells you nothing about whether you
landed on a good draw or a bad one.

### 4.3 Reasoning models degrade disproportionately

Multiple 2025-2026 papers report that reasoning models lose more to low-bit PTQ
than non-reasoning models of the same size, with the failure showing up as longer
chains and overthinking rather than as incoherence
([arXiv 2504.04823](https://arxiv.org/html/2504.04823v2),
[arXiv 2606.00206](https://arxiv.org/html/2606.00206v1)). One source notes
calibration sequence length being raised to 4096 for reasoning models to avoid
extreme K-cache outliers.

Evidence quality: the degradation finding is well replicated. The claim that
calibrating on reasoning traces fixes it is not something I found measured
anywhere. llm-compressor asserts it in a comment
(`examples/awq/qwen3_next_thinking_example.py:11-14`) and the mechanism is sound,
but I could find no ablation. Treat it as a well-motivated hypothesis, not a
result.

The failure mode is relevant on its own terms: a quantized expander that thinks
longer will hit `max_new_tokens` (16256 for t2i, 24000 for edit, from
`pe_core.py::PROFILES`) and emit an unterminated thinking block, which
`pe_core.py::split_thinking` returns as an empty answer and `parse_answer` marks
`parse_ok: False`. That gives a free, sharp evaluation signal. Use it.

### 4.4 Long fixed system prompts: no evidence exists

I searched for prior work on including a long fixed system prompt in calibration
rows and found none. The closest anything comes is generic advice that calibration
"might involve creating attention masks, position IDs, or specific prompt templates
if the model uses them"
([apxml course notes](https://apxml.com/courses/quantized-llm-deployment/chapter-1-advanced-llm-quantization-fundamentals/calibration-data-selection),
a tutorial, not a primary source).

Nothing measures it. Nothing discusses the token-budget consequence. Nothing
addresses hybrid linear-attention architectures with respect to calibration
sequence length either. The owner's read that this is untapped is correct, and it
means section 5 is reasoning from mechanism, with its confidence stated
accordingly.

---

## 5. The core question: system prompt and expected output in the rows

Short answer: put the whole thing in the row, and mask the statistic to the
assistant turn. Those are two separate decisions and conflating them is the
mistake.

### 5.1 The system prompt must be in the row

The activations at every output-token position are conditioned on the system prompt
sitting in the KV context. In the full-attention layers the output positions attend
to it directly; in the GatedDeltaNet layers the recurrent state at the output
positions is a function of having consumed it. There is no configuration in which
this model produces a token without 2426 or 4509 tokens of instruction behind it.
A corpus that omits the system prompt collects activations from a distribution the
deployed model never enters. Confidence: high. This is not an appeal to
distribution-matching as a heuristic; it is that the conditioning context is part
of what the activation is.

### 5.2 The system prompt must not dominate the statistic

Here is the arithmetic, and this is the part that is specific to these checkpoints
rather than general advice.

Build a t2i row the way the library's VLM examples build one: system plus user
brief plus `add_generation_prompt=True`, no assistant turn. From section 1 that is
2426 + ~11 + a handful of control tokens. Call it 2450, of which 2426 (99.1 percent)
is a byte-identical prefix shared with every other row in the corpus.

Now recall from section 2.3 that AWQ's smoothing statistic is
`sum(|x|, dim=0) / count` pooled over tokens. Collect 512 such rows and you have
roughly 1.25 million system-prompt token positions and roughly 5.6 thousand
brief token positions. The pooled per-channel mean is, to three digits, the system
prompt's per-channel mean. Whether you collected 8 rows or 512 rows changes almost
nothing about the resulting scales.

That is a degenerate corpus wearing the costume of a diverse one. Every check you
would normally run passes: 512 distinct rows, 512 distinct briefs, good coverage of
styles and languages, no duplicates. And the statistic it produces is the statistic
of one example.

The same arithmetic applies to GPTQ, where the Hessian is `sum(x x^T)` over all
tokens (section 2.4) and there is no mask to escape with.

For the edit model the ratio is less extreme but still lopsided: 4509 system tokens
plus 64 to 1024 vision tokens per image plus a ~20-token brief. The brief is under
half a percent of the prompt. And per MBQ (section 3), the vision tokens competing
for the remaining share are the least sensitive tokens in the sequence.

### 5.3 Therefore: full rows, masked to the assistant turn

Put system prompt, images, brief, thinking trace and JSON answer all in the row.
Build a `loss_mask` that is 0 across the prompt and 1 across the assistant turn.
Pass `use_loss_mask=True`.

What this buys, mechanically:

- The forward pass is the deployment forward pass. Every conditioning effect of the
  system prompt is present in the activations at the positions that count.
- The AWQ channel statistic is computed over generation-position activations only,
  which is the distribution the model is in when it is doing the job.
- Row count starts meaning something again, because each row now contributes a
  distinct thousand-ish masked tokens rather than a shared 2426.
- MBQ's modality-imbalance problem is solved for free: vision tokens live in the
  prompt and are excluded by construction. No separate modality weighting needed.

The recipe consequence, which I flag and leave to the other agent: this corpus
design is only fully exploitable under AWQ. Under GPTQ the mask is ignored and the
Hessian is dominated by the system prompt regardless. That is a corpus fact about
recipe choice, not a recipe recommendation.

### 5.4 Confidence and what would falsify it

Confidence that the system prompt belongs in the row: high. Confidence that the
statistic should be masked to the assistant turn: moderate-to-high on mechanism,
low on published support, because nothing in the literature measures it (section
4.4).

The honest uncertainty is this. Masking to the assistant turn optimizes the scales
for generation-position activations. But the prompt positions are also computed at
inference, and errors there propagate into the KV state that generation reads. It
is conceivable that a small share of prompt-position weight is better than zero. I
have no evidence either way, and no principled way to pick a ratio. If someone wants
to explore it, the shape of the experiment is a mask of 1.0 on assistant tokens and
a small constant on prompt tokens, which the current code does not support (the
mask is used as a boolean, `awq/base.py:432-434` and `784`).

Falsifiers, cheapest first:

1. Run AWQ twice with an identical recipe, once on system-included masked rows and
   once on generic instruction data, and compare the per-layer smoothing scale
   vectors by cosine similarity. No eval harness needed. If the scales come out
   near-identical, corpus design is not the lever for this model and this entire
   note is misdirected effort. That is a genuinely useful thing to learn early.
2. Same two quants, evaluate on held-out briefs: `parse_ok` rate from
   `pe_core.py::parse_answer`, thinking-trace length distribution, and similarity of
   `rewritten_prompt` to the fp16 model's output on the same briefs. If masked
   calibration does not beat generic on `parse_ok` and length, the masking argument
   is wrong even if the scales differ.
3. Ablate the mask alone: same rows, `use_loss_mask` on versus off. This isolates
   section 5.2's claim from section 5.1's. If off is as good as on, the
   token-budget argument is wrong and the cheaper corpus wins.

Run 1 before building the full corpus. It is a few hours and it can save the rest.

---

## 6. Corpus A: the prompt expanders

Two corpora, same construction, different system prompts and different inputs. Do
not share rows between them; `pe_core.py::load_system_prompt:111-134` is explicit
that the prompts are not interchangeable and that mixing them fails silently.

### 6.1 Row structure

A row is the full two-turn conversation plus the assistant answer:

```
system:    <the checkpoint's own system_prompt.txt, stripped>
user:      [images in order, first]  +  the brief text
assistant: <think>\n{thinking trace}\n</think>\n\n{"rewritten_prompt": ..., ...}
```

Build the messages with `pe_core.py::build_messages:195-216` so image ordering
matches deployment (images first, in order, because the system prompt addresses
them as `<image1>`, `<image2>` and reordering silently re-points every reference).
Then append the assistant message.

Rendered, that produces exactly what the deployed model sees followed by what it
would have emitted. The chat template puts the thinking block in place for the
final assistant turn only (`chat_template.jinja`, the
`loop.index0 > ns.last_query_index` branch), which is correct for a single-turn
row.

Fields per row: `input_ids`, `attention_mask`, `loss_mask`, plus the multimodal
keys for the edit model. Measured, `Qwen3VLProcessor` emits exactly
`attention_mask`, `image_grid_thw`, `input_ids`, `mm_token_type_ids` and
`pixel_values`. Prune each row to the keys the model's `forward` accepts plus
`loss_mask`, and check that on row zero rather than discovering it at hour three of
a run. See failure mode 3 in section 9 for why `loss_mask` survives the pruning.

### 6.2 Thinking is on by default, and that is load-bearing

`chat_template.jinja` ends with:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```

`enable_thinking` undefined means thinking on. Both official runners pass
`enable_thinking=True` explicitly, which is the same thing. A calibration row
without a thinking trace is calibrating the model for a mode it never runs in, and
the thinking trace is the bulk of the generated tokens.

### 6.3 Building the mask: a correction to the library's example

`examples/awq/qwen3_next_thinking_example.py:60-64` warns that rendering the
prompt with `add_generation_prompt=True` injects a `<think>\n` primer that is
absent from the full conversation, so `prompt_len` would be over-counted, and
therefore renders the user turn without the generation prompt.

That caveat is wrong for these checkpoints. I verified with jinja2 against the
shipped `chat_template.jinja` that the generation-prompt render is an exact byte
prefix of the full-conversation render:

```
GEN  ==> '<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nBRIEF<|im_end|>\n<|im_start|>assistant\n<think>\n'
FULL ==> '<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nBRIEF<|im_end|>\n<|im_start|>assistant\n<think>\nTHINK\n</think>\n\n{"rewritten_prompt": "X", "wh_ratio": "3:2"}<|im_end|>\n'
prefix match: True
```

And at token level, with an image, through the real `Qwen3VLProcessor`: gen 790
tokens, full 815 tokens, `torch.equal(full[:790], gen)` is True.

So `prompt_len` is exactly the length of the generation-prompt render, and the mask
is `[0] * prompt_len + [1] * (total_len - prompt_len)`. This is the simplest
construction and it is correct here. Do not copy the workaround from the qwen3_next
example; it would under-count by the primer.

### 6.4 The mask must be built from processor output, not from the text render

This is the trap I would most expect someone to hit. The processor expands
`<|image_pad|>` inline in `input_ids`. Measured: a 1024x768 image renders as one
`<|image_pad|>` in the text but becomes 768 tokens in `input_ids`, and the
rendered gen prompt came out at 790 tokens where the text-level token count would
be around 22.

A mask built by tokenizing the rendered text without images would be roughly 22
long against an 815-long sequence. `DataCollatorWithTruncation` truncates every key
to the minimum length across the listed keys (`datasets/utils.py:306-314`), so
`input_ids` would be silently chopped to 22 tokens and the run would complete with
a corpus of fragments.

Correct construction for both models:

```python
prompt_len = processor(text=[render(sys + user, add_generation_prompt=True)],
                       images=imgs, padding=False)["input_ids"].shape[-1]
full = processor(text=[render(sys + user + assistant, add_generation_prompt=False)],
                 images=imgs, padding=False)
total_len = full["input_ids"].shape[-1]
loss_mask = [0] * prompt_len + [1] * (total_len - prompt_len)
```

Then assert `len(loss_mask) == total_len` per row and drop rows where
`not any(loss_mask)`, as the library's own example does at
`qwen3_next_thinking_example.py:84`.

### 6.5 Sequence length

Do not copy `MAX_SEQUENCE_LENGTH = 2048` from the VLM examples. T21's system prompt
alone is 2426 tokens and I21's is 4509. That cap truncates mid-system-prompt, the
mask goes all-zero, nothing errors, and the run calibrates on boilerplate
fragments.

Floors, reasoned from section 1 plus the expected answer size (the t2i system
prompt asks for "about twenty sentences and four to five hundred words", so roughly
550 to 700 tokens of JSON payload, plus a thinking trace of unknown length):

| model | system | vision | brief | answer | floor |
| --- | --- | --- | --- | --- | --- |
| T21 t2i | 2426 | 0 | ~15 | 700 + think | 6144 |
| I21 edit | 4509 | 64-1024 per image | ~20 | 700 + think | 12288 |

Set the floor from the corpus rather than guessing: generate the answers first,
measure the actual full-row length distribution, and set the cap above its 95th
percentile. Drop rows above the cap rather than truncating them, because a
truncated row is a row whose answer is cut off mid-JSON, which is not a sample of
anything.

The hybrid-attention concern from section 2.7 argues the same way: longer rows
exercise more of the GatedDeltaNet state trajectory, and these rows are naturally
long. There is no reason to shorten them here.

### 6.6 Sample count

Start at 256 per checkpoint. Reasoning, in order of weight:

- Published practice is 128 to 512 for both VLMs and text models (section 3, 4.2),
  and llm-compressor's own docs say 128 to 512 (`docs/steps/choosing-dataset.md:46`).
- With the mask on, each row contributes roughly a thousand masked tokens, so 256
  rows is roughly 256k masked tokens, which is about the same masked-token budget
  as Williams and Aletras' 128 x 2048 = 262,144. That is a defensible anchor.
- Cost scales with total row length, not masked length, and these rows are three to
  six times longer than a typical calibration row. 256 is the affordable end.
- Williams and Aletras' variance result (section 4.2) says the draw matters more
  than the count past a point. Two independent 256-row draws compared against each
  other is worth more than one 512-row draw.

If the falsifier in section 5.4 shows large scale differences between corpora,
raise to 512 and compare. If it shows small differences, the count was never the
lever.

### 6.7 Images for the edit model

Requirements, in order:

1. Every row needs at least one image; `pe_core.py::resolve_image_paths:155-176`
   raises if an edit case has none.
2. Downscale with `pe_core.py::load_image:179-192` at
   `Profile.image_max_pixels = 1024*1024`, matching training. Do not hand the
   processor full-resolution images; `processor_config.json` would allow up to
   16.7M pixels, which is a different token count and a different activation
   regime from deployment.
3. Cover the multi-image case. The system prompt has a whole section on
   `<image1>`/`<image2>` tagging rules that only activates at N >= 2, and the
   `ratio_follow` field only has meaningful values there. A corpus of
   single-image rows never exercises it. Target roughly 70 percent single-image,
   20 percent two-image, 10 percent three-or-more, adjusted to whatever the real
   usage mix is if that is known.
4. Cover aspect ratios and content types, because `wh_ratio`/`ratio_follow`
   selection depends on the input's shape, and because vision-token count scales
   with pixel count. A corpus of square 1024x1024 images gives every row exactly
   1024 vision tokens and never exercises the shorter ones.
5. Cover images with text in them, in more than one script. The system prompt's
   language decision (B) branches on the dominant language of text in the image,
   and that branch is a large share of what the edit model does.

Source: the owner's own images, whatever actually gets edited in practice. Five
sample images ship at `prompt_rewrite/data/images/` which is enough to smoke-test
the pipeline and nowhere near enough to calibrate. Public alternatives that fit the
edit-instruction shape rather than the captioning shape would be preferable to
COCO, but I have not verified the availability or licensing of any specific one, so
I am not naming one.

### 6.8 Where the briefs come from, and where the answers come from

The briefs are the input half. The answers have to be generated.

Generate with the fp16 checkpoint at production sampling. `pe_core.py::PROFILES`
sets `presence_penalty` to 1.5 for t2i and 0.0 for edit, with a note at
`pe_core.py:50-53` that they are not interchangeable and that a wrong penalty
"does not fail, it quietly changes the distribution you sample from". The runners
at `run_transformers.py` and `run_vllm.py` already do this; produce the corpus with
them and keep the output JSONL.

Filter on `parse_ok` using `pe_core.py::parse_answer:288-320`, and audit with
`pe_core.py::report_parse_failures:365-378`. A row whose answer did not parse is a
row whose assistant turn is not an example of the output contract.

Two consequences to accept with open eyes:

- This is self-distillation. The corpus inherits the fp16 model's failure modes,
  and the quantized model is being fit to reproduce them. That is what you want for
  a prompt expander, where fidelity to the fp16 behavior is the goal, but it means
  the corpus cannot correct anything the fp16 model does badly.
- `parse_ok` is a syntax check, not a quality check. It keeps fluent-but-wrong
  answers: a `rewritten_prompt` that dropped a fixed element from the brief, an
  ignored aspect ratio, an edit directive that leaked into untargeted content. A
  second filter pass is worth the effort. Cheap mechanical checks that follow
  directly from the system prompts: does every quoted string from the brief appear
  in the answer (step 1 of the t2i prompt says copy them character for character);
  is `wh_ratio` in the allowed set; for edit, is exactly one of
  `wh_ratio`/`ratio_follow` non-empty (the system prompt says they are mutually
  exclusive); for multi-image edit, does the answer use `<imageN>` tags rather than
  natural-language references (the system prompt calls this mandatory and
  non-negotiable).

### 6.9 Assessment of the owner's existing prompt collections

Neither collection is usable as-is, and for the same underlying reason: both are
output-side artifacts, and what this corpus needs is input-side briefs.

`coderef/shrug-prompter/templates/styles/` (250+ files). These are
system prompts, not user briefs. The README is explicit: "styles/ — 250+
single-purpose style / system-prompt snippets. Each file's body is a ready-to-use
system prompt." A sample body (`styles/art_deco.md`) reads "The roaring twenties
distilled into visual language, where geometry becomes poetry and luxury feels like
a birthright..." — atmospheric prose meant to steer a VLM, not a request a user
types. Dropping these into the user turn puts a system-prompt-register paragraph
where a 2-to-19-token brief belongs, which is off-distribution in exactly the way
this note is trying to avoid.

They are useful indirectly: the style vocabulary they carry is a good source of
coverage axes (which styles, which media, which eras should appear across the
corpus). `styles/rewriter/` has three files that are closer in kind, but they are
rewriter system prompts, still the wrong side.

`coderef/ComfyUI-h3-explorations/prompt_bank/`. 139 entries per
`bank.json`, modes `t2va` (84), `ref2va` (34), `i2va`/`fl2va`/`l2va` (7 each).
These are video prompts with `integrated_multimodal_description`,
`overall_soundscape` and `non_diegetic_music` fields, graded on a "17k+5 frame
grid". Wrong modality, wrong output contract, and again the long structured
register of an expanded prompt rather than a brief.

What is actually authoritative about the register is the shipped examples at
`prompt_rewrite/data/t2i_example.jsonl` and `edit_example.jsonl`. Read them
closely; they are short and they are specific:

- 2 to 19 tokens.
- Mixed Chinese and English, sometimes with no English at all.
- Ungrammatical and untidy: "can u this image so that the head twist to the left
  while the body and arms not moved" is a real shipped example, missing a verb.
- The edit file carries a `task_type` field with values `single_scene_complex`,
  `text_edit`, `basic_edit`. That is a coverage axis handed to you.

A corpus of clean well-formed English briefs is off-distribution against this. The
properties to preserve deliberately: the length distribution, the language mix
(and specifically Chinese-language briefs, since both system prompts branch on
input language), lowercase and typos and missing punctuation, and `task_type`
coverage for edit.

My recommendation for sourcing briefs: write them, stratified against a coverage
matrix, rather than harvesting them. The axes that the system prompts themselves
tell you matter:

For t2i (from `system_prompt.txt` steps 1, 2, 6): whether the brief fixes text
strings to be rendered; whether it states an aspect ratio; whether it names a
style; brief length (very short versus paragraph); language; whether the subject
is a single-subject frame or a region-divided layout (the step 5 branch); and
presence of job-instructions-rather-than-content ("use double quotes", "4K") which
step 1 says to obey silently and never echo.

For edit (from its system prompt): number of images; the intent branch (change this
picture versus new picture of this subject); whether the instruction gives exact
text or names a target language or neither (the three-way decision B); whether the
image contains text and in which script; whether a size or ratio is stated;
`task_type`; language of the instruction.

A few hundred stratified briefs is a day of work and it is the single highest-value
artifact here, because it is also the holdout source and the evaluation set.

---

## 7. Corpus B: the conditioning text encoder

This is a different problem. Do not carry the section 5 answer over to it.

### 7.1 What the runtime shape actually is

From `ComfyUI/comfy/text_encoders/qwen_image21.py`:

- The template is short and fixed (line 9-10):
  `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n`.
  Nothing like a 2426-token system prompt, so the token-budget degeneracy of
  section 5.2 does not arise and its whole argument is inapplicable.
- With images, the user turn is prefixed with
  `<image1><|vision_start|><|image_pad|><|vision_end|> <image2>...` (lines 22-24).
- There is no generation. One prefill, and a hidden state is tapped.
- `encode_token_weights` (lines 47-77) drops the system turn and the vision spans
  from the output, after the forward pass. They are present in the forward pass and
  do condition everything downstream of them, so they belong in the calibration
  row. Dropping them from the row would be a different computation.
- `layer_norm_hidden_state = False` (line 34) with the comment "last layer without
  the final RMSNorm: transformers 4.57 hidden_states[-1], which Qwen's results are
  tuned to (5.x norms it)". The conditioning vector is a pre-final-norm hidden
  state. Quantization error in the last decoder layer lands on it with no
  normalization to absorb the scale. That is a recipe consideration (the last
  layers may warrant different treatment) which I note and leave alone.

### 7.2 The corpus is the expanders' output, not their input

This is the connection worth making explicitly. In the deployed pipeline the text
encoder's input is a rewritten prompt produced by one of the PE models: roughly 400
to 500 words of observational present-tense description with quoted text strings
in it, per the t2i system prompt's "Throughout / Size" section. It is not a user
brief, and it is not generic web text or instruction data.

So the encoder's calibration corpus should be the `rewritten_prompt` field of the
PE corpus from section 6, plus whatever else actually reaches this encoder in the
owner's workflows (hand-written prompts, prompts from other sources). This is free:
you generate it while building corpus A.

One mismatch to name and then set aside: those `rewritten_prompt` strings come from
the fp16 expanders, while deployment will feed this encoder the quantized
expanders' output. I expect the difference to be immaterial, since the whole point
of corpus A is that the two distributions coincide. But it makes a useful
diagnostic: if the encoder's calibration statistics shift measurably between fp16
and quantized PE output, that is a finding about the PE quant, not about the
encoder.

Add coverage for the register properties that the encoder will see and that a
PE-only corpus might under-sample: prompts with heavy quoted text in non-Latin
scripts, very short prompts (a user bypassing the expander), and negative prompts
if they are encoded through the same path.

### 7.3 Row structure, counts, images

Rows: the encoder's own template applied to the prompt text, with images when the
workflow has them, built the way `QwenImage21Tokenizer.tokenize_with_weights`
builds them so the `<imageN>` prefix and vision block placement match. No assistant
turn.

On masking, the honest position is weaker than I first wrote it, and the parallel
to section 5 is worth stating rather than glossing.

The discriminator in section 5 was never "is there generation". It was: is there a
subset of positions whose activations the downstream consumer actually reads? For
the encoder there is. `encode_token_weights` builds a `keep` mask and drops both
the system turn and every vision span from `out` before the DiT sees it
(`qwen_image21.py:60-71`). The system turn is about a dozen tokens and does not
matter. The vision spans do: up to 1024 tokens per image, which on an image-bearing
row can be most of the sequence. In reference-latent mode they are discarded from
the output entirely (line 66), which is structurally the same situation as the PE
system prompt, and it is the MBQ modality-imbalance case from section 3 showing up
again.

So the same question applies, and I have no more evidence here than I had there.
The relevant quantity is the vision-token share of image-bearing rows, which
depends on the workflow mix and on image sizes and which I have not measured. The
`keep_vision` branch at line 66 sharpens it: reference-latent mode discards the
vision spans, keep-vision mode retains them, and the two therefore differ on
exactly this. That reinforces treating them as two corpora rather than one.

Default recommendation: `use_loss_mask` off, on the grounds that the tapped hidden
states for retained positions are a per-position output with no pooling, and no
single subset is obviously the target. But run the section 5.4 falsifier in its
encoder form before accepting it: compare per-layer scales with the mask off
against a mask that zeroes vision spans, on an image-heavy corpus. If they differ,
the question is live and my default is the wrong answer.

Count: 256 to 512. This is the standard regime and there is no reason here to
deviate from it; rows are much shorter than corpus A's so the cost is lower.

Sequence length: 2048 is probably adequate for text-only rows (a 500-word prompt is
roughly 700 tokens plus template) but must be checked against the actual measured
distribution, and must be raised for image-bearing rows by the vision-token count.

Images: the split between text-only and image-bearing rows should match the
workflow mix, because the two produce very different sequences. Note the branch at
`qwen_image21.py:66` — when reference latents are coming the vision tokens are
dropped from the output, and when there is no VAE they stay and the image
conditions through the text encoder alone. Those are two different deployment
modes. If both are used, both belong in the corpus, and I would want to know the
split before fixing a ratio. Vision tokens here follow the same `(H/32)*(W/32)`
arithmetic as section 1, since this is also a Qwen3-VL processor, though I did not
verify that against this specific checkpoint's processor config.

Vision tower excluded from quantization per the owner's decision. That does not
remove vision tokens from the language path; they still enter as embeddings and
still shape the language-path statistics, which is why image-bearing rows are
needed at all.

---

## 8. Holdout, and the near-duplicate trap

### 8.1 On the prior history

I could not find it. Grepping `coderef/ComfyUI-h3-explorations/docs/`
for "near-duplicate", "near duplicate", "holdout" and "hold-out" returns exactly one
hit, `evidence.md:206`, and it is about audio token holdouts, unrelated. Rather
than reconstruct a history I do not have, here is the mechanical procedure and the
reason the usual version of it fails on this specific corpus.

### 8.2 Why the standard check would pass a broken corpus here

The instinct is to dedup on the rendered calibration row. Do not. Every t2i row
shares an identical 2426-token prefix, so any whole-row similarity metric reports
something near 99 percent for every pair in the corpus. The metric is saturated: it
cannot distinguish "these two rows are genuinely near-identical" from "these two
rows share the system prompt, as all rows do". A near-duplicate check run that way
either flags everything or, after you tune the threshold to stop it flagging
everything, flags nothing. It would have passed a corpus of 512 copies of the same
brief.

### 8.3 What to do instead

Dedup on the brief text and, for edit, on the image, never on the rendered row.

- Exact: normalize (casefold, collapse whitespace, strip punctuation) and hash.
  Catches the byte-different-but-identical case that bit the prior effort.
- Near: character n-gram shingling with MinHash or simple Jaccard over 3-grams,
  threshold tuned by eyeballing the flagged pairs. On 2-to-19-token strings this
  is fast and the false positives are easy to adjudicate by hand.
- Semantic: embed the briefs with any sentence encoder and flag high-cosine pairs.
  This is what catches "a cat in the rain playing guitar" against
  "一只在雨中弹吉他的柯基", which is the same brief in two languages and which
  neither exact nor n-gram catches.
- Images, for edit: perceptual hash (pHash or dHash) on the downscaled image, plus
  an embedding check. Two crops of the same photo are byte-different and visually
  near-identical, which is precisely the trap named in the task.
- Cross-field: a row can be a near-duplicate through its answer even when brief and
  image differ. Run the same check over `rewritten_prompt`.

### 8.4 Splitting

Split by source, not by random draw. A random split of a stratified corpus puts
siblings from the same stratum on both sides, and siblings are exactly what a
near-duplicate check is trying to separate. Concretely:

- Assign each brief a provenance key: which style concept, which `task_type`, which
  source image, which authoring session. Split on the key, so a whole key lands
  entirely in calibration or entirely in holdout.
- For edit, an image and all briefs written against it go to the same side. A
  holdout brief over a calibration image is not held out.
- Hold out whole strata as well as within-stratum keys, so the holdout can answer
  both "does it generalize within what it saw" and "does it generalize to a
  category it did not see". A quant that is good at the first and bad at the second
  has overfit the corpus, and you cannot detect that with a random split.
- Freeze and record the split. Williams and Aletras' first recommendation is to
  release the calibration data precisely so that randomness stops being an
  unexamined source of variation. Write the split file, put its hash somewhere
  durable, and never regenerate it.

Size: a few hundred holdout briefs is plenty, since evaluation here is per-brief
and the signals (parse rate, length, similarity to fp16 output) are cheap.

---

## 9. Failure modes: corpora that look fine and produce a bad quant

Ordered by how likely I think they are here.

1. Truncation eats the row, silently. `MAX_SEQUENCE_LENGTH = 2048` copied from the
   VLM examples is below both system prompts. The row truncates mid-instruction,
   the assistant turn is gone, the mask is all-zero, the run completes without a
   warning, and the quant is calibrated on fragments of boilerplate. Gate:
   assert every row's length is under the cap before it enters the corpus, and drop
   rows where `not any(loss_mask)` (`qwen3_next_thinking_example.py:84`). Log both
   counts; if either is non-zero you have a bug, not a filter.

2. The mask is misaligned with the image-expanded `input_ids`. A mask built from
   the text render is roughly 22 entries against an 815-token sequence. The
   collator then truncates everything to 22 (`datasets/utils.py:306-314`) and you
   calibrate on the first 22 tokens of each row. Gate: assert
   `len(loss_mask) == len(input_ids)` per row.

3. Row keys the forward pass will not accept. The VLM examples' collator passes
   every key in the row dict straight through
   (`examples/multimodal_vision/qwen3_vl_example.py:69-72`), and `loss_mask` rides
   along with the processor's five keys. No example in this repo combines
   `loss_mask` with a multimodal processor's key set: the thinking example is
   text-only and the VLM examples use no mask. Prune each row to the keys the
   model's `forward` signature accepts, and keep `loss_mask` while doing it, since
   the pipeline fetches it separately from the intermediates cache
   (`pipelines/sequential/pipeline.py:126-129`) rather than through
   `subgraph.input_names`. Verify on the first row.

4. `LengthAwareSampler` quietly overrides your coverage mix. With
   `shuffle_calibration_samples=False` you get `LengthAwareSampler`
   (`datasets/utils.py:293-296`), whose order is
   `argsort(lengths, descending=True)` (line 352) and which yields
   `order[:num_samples]` (line 393). Section 6.8 has you generating and filtering,
   so the built corpus will be larger than the target and this selection becomes
   live. Multi-image edit rows are systematically the longest, so shuffle-off would
   preferentially select them and silently replace the 70/20/10 mix from section
   6.7 with something closer to all-multi-image. Fix: stratify and filter down to
   exactly the target count, then set `num_calibration_samples` to the dataset
   length so selection is a no-op. Leaving shuffle on is also fine here; the
   shuffle warning at lines 282-290 only concerns `batch_size > 1`. Pick one
   deliberately, because the default points the wrong way for this corpus.

5. System-prompt degeneracy. The corpus has no assistant turns, or has them but
   `use_loss_mask` is left off (it defaults to False,
   `args/dataset_arguments.py:276`). The statistic is the system prompt's. Section
   5.2. This is the one that looks most like a healthy corpus.

6. Wrong sampling parameters when generating the answers. `presence_penalty` is
   1.5 for t2i and 0.0 for edit; using the wrong one, or a library default of 0,
   changes the distribution the answers are drawn from without failing
   (`pe_core.py:50-53`). The corpus then teaches the quant a generation
   distribution the deployment never produces.

7. Corpus/holdout leakage through near-duplicates. Section 8. Symptom: holdout
   numbers that look good and deployment behavior that does not.

8. Images not downscaled to the training cap. Full-resolution images through a
   processor that allows 16.7M pixels produce several times the deployment vision
   token count, shifting both the token budget and the vision-token activation
   statistics. `pe_core.py::load_image` exists for exactly this; use it.

9. Answers filtered on syntax only. `parse_ok` keeps fluent-but-wrong answers
   (section 6.8). The quant is fit to reproduce contract violations.

10. Edit corpus is all single-image. The `<imageN>` tagging rules and the
   `ratio_follow` field are never exercised, and multi-image sequences are two to
   four times longer than single-image ones, which is a different regime for the
   GatedDeltaNet state. The quant degrades specifically on multi-image edits, which
   is a minority of your holdout if the holdout has the same skew.

11. All-English, all-clean briefs. Both system prompts branch hard on input
   language. A corpus that never contains a Chinese brief never exercises the
   branch, and the language-decision logic is a large share of the edit model's
   job.

12. Shuffling with batch size above 1. `shuffle_calibration_samples` defaults True
    and the code warns that shuffling with `batch_size > 1` combined with
    truncation "will delete a large number of tokens"
    (`datasets/utils.py:282-290`). Given how variable these row lengths are, use
    `batch_size=1`. With `batch_size=1` the shuffle flag no longer affects
    truncation at all, and its only remaining effect is which sampler you get,
    which is failure mode 4.

13. One outlier row sets a static activation range. Only applies to static
    activation schemes (section 2.5). A row with an unusual image or a pathological
    thinking trace can widen a tensor's range permanently. Williams and Aletras'
    third recommendation, manual inspection of the sampled rows, is cheap insurance
    at 256 rows.

14. Drawing conclusions from one run. The variance result in section 4.2 says a
    single corpus draw tells you little. Build two independent draws and compare
    before believing either.

---

## 10. Summary of recommendations

| decision | t2i (T21) | edit (I21) | text encoder |
| --- | --- | --- | --- |
| system prompt in row | yes, the checkpoint's own | yes, the checkpoint's own | yes, the short fixed one |
| assistant turn in row | yes, thinking + JSON | yes, thinking + JSON | no, there is no generation |
| `use_loss_mask` | True, assistant turn only | True, assistant turn only | False by default, but see 7.3 |
| rows | 256 to start | 256 to start | 256 to 512 |
| seq len | measure first, floor ~6144 | measure first, floor ~12288 | measure first, ~2048 text-only |
| `batch_size` | 1 | 1 | 1 |
| `num_calibration_samples` | = dataset length, so selection is a no-op (failure mode 4) | same | same |
| brief source | stratified, hand-written | stratified, hand-written, with `task_type` | the expanders' `rewritten_prompt` output |
| answers | generated fp16 at `PROFILES['t2i']`, filtered | generated fp16 at `PROFILES['edit']`, filtered | n/a |
| images | none | 1..N, capped at 1M px, mixed count and ratio | mixed, matched to workflow |
| dedup on | brief text | brief text + image pHash | prompt text |
| split by | provenance key and whole strata | provenance key, image-grouped | provenance key |

Do the section 5.4 falsifier first. It is cheap and it tells you whether the rest
is worth building.

---

## 11. What I could not verify

- Thinking-trace length. Every sequence-length recommendation depends on it and I
  have no measurement. You will know once you generate the corpus. The floors in
  section 6.5 are lower bounds assuming a modest trace; measure and raise.
- Whether masking to the assistant turn actually improves the quant. Mechanism is
  in section 5.3, the library asserts it in a comment, no published ablation
  exists (section 4.4). Section 5.4 is how to find out.
- Whether including a fraction of prompt-position tokens in the statistic would beat
  masking them out entirely. No evidence, and the current code's mask is boolean so
  it cannot express the intermediate.
- The MoE guard at `awq/base.py:214-222`. These are dense checkpoints so it should
  not fire, but it resolves against runtime module names and I did not load the
  model.
- The text encoder's vision-token arithmetic. I applied the same `(H/32)*(W/32)`
  formula as the PE models on the grounds that it is a Qwen3-VL processor, but I
  did not check the 8B encoder's own processor config.
- The deployment split between the encoder's reference-latent mode and its
  vision-tokens-retained mode (`qwen_image21.py:66`). I recommended matching the
  corpus to the workflow mix without knowing what that mix is.
- Whether masking vision spans out of the encoder's statistic would help
  (section 7.3). Same open question as section 5.4, same absence of evidence, and
  I did not measure the vision-token share of image-bearing encoder rows.
- Whether the prior quantization effort's near-duplicate problem is documented
  anywhere. My grep across the h3 docs found nothing relevant (section 8.1).
- Any public dataset of image-edit instructions suitable as a brief source. I did
  not find and verify one, so I did not name one.

---

## Sources

- [AWQ: Activation-aware Weight Quantization, arXiv 2306.00978](https://arxiv.org/pdf/2306.00978)
- [On the Impact of Calibration Data in Post-training Quantization and Pruning, arXiv 2311.09755](https://arxiv.org/abs/2311.09755) / [ACL 2024](https://aclanthology.org/2024.acl-long.544/)
- [MBQ: Modality-Balanced Quantization for Large Vision-Language Models, arXiv 2412.19509](https://arxiv.org/html/2412.19509v1)
- [Towards Understanding Best Practices for Quantization of Vision-Language Models, arXiv 2601.15287](https://arxiv.org/html/2601.15287v1)
- [Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models, arXiv 2504.04823](https://arxiv.org/html/2504.04823v2)
- [Quantized Reasoning Models Think They Need to Think Longer, but They Do Not, arXiv 2606.00206](https://arxiv.org/html/2606.00206v1)
- [Understanding and Selecting Calibration Data for LLM Quantization, OpenReview](https://openreview.net/forum?id=pfw3saHzGU) (could not fetch full text; login wall)
- [Calibration Data for LLM Quantization, apxml course notes](https://apxml.com/courses/quantized-llm-deployment/chapter-1-advanced-llm-quantization-fundamentals/calibration-data-selection) (tutorial, not primary)
