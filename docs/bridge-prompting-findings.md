# Prompting findings mined from ComfyUI-QwenImageWanBridge

Source: `coderef/ComfyUI-QwenImageWanBridge`
Read-only pass. Nothing in the repo was modified. Per house rule, this extracts transferable
findings only; no changes are proposed to that repo.

Scope covered: system prompts, chat templates, thinking blocks, template tooling, prompt
expansion, output parsing. The encoder/conditioning path and the dead-end census are other
agents' scope and are touched here only where they invalidate a prompting measurement.

Every claim below is labelled. `[measured]` means a number or a token ID came out of a run.
`[observed]` means a single generated artifact exists that you can look at. `[impression]`
means a judgement stated by the author after looking at outputs, with the setup recoverable
but no control or repetition. `[assertion]` means stated in prose with nothing behind it in
the repo. `[plan]` means an experiment design that was written and, as far as the repo shows,
never run.

---

## Headline: three measurement rigs were built, none produced a recorded result

The repo contains three separate apparatuses for answering "do system prompts actually do
anything":

1. A ComfyUI node, `nodes/template_influence_analyzer.py` (302 lines), registered and loadable
   (`__init__.py:360-363`).
2. An offline script, `experiments/quick_embedding_test.py` (277 lines), with a CLI.
3. A written methodology with power analysis, Cohen's d thresholds, required sample sizes and
   significance tests: `experiments/EXPERIMENT_METHODOLOGY.md`, plus three more test plans in
   `nodes/docs/` (`hunyuanvideo_system_prompt_test_plan.md`, `hunyuanvideo_prompting_experiments.md`,
   `nodes/docs/attention_based_prompt_expansion.md`).

No output of any of them is in the repo. `EXPERIMENT_METHODOLOGY.md:330-337` names an
`experiment_outputs/` directory holding `embedding_analysis_results.json`; that directory does
not exist anywhere in the tree, is not gitignored (`.gitignore` has no such entry), and no
JSON, log or table of results exists under any name. Every numeric table in those documents is
explicitly labelled "Expected Results" or "Expected Outcomes" and is bracketed by an
if-it-works / if-it-does-not-work pair, for example `EXPERIMENT_METHODOLOGY.md:105-133` and
`nodes/docs/attention_based_prompt_expansion.md:310-331`. Those are hypotheses written in the
shape of results. Do not mistake them for data if you skim these files.

That is the cautionary half. The valuable half is elsewhere and is real: a small number of
genuinely measured tokenizer facts, one series of four annotated generation screenshots, and a
set of bugs they hit that we can hit too. That is where most of this report goes.

---

## Finding 1: the README's system-prompt claims rest on four screenshots, and the setup is fully recoverable

Status: `[observed]` for the outcomes, `[impression]` for the generalisations in the captions.

The README makes four claims with an image each (`README.md:37-67`). The images are not bare
outputs. Each is a ComfyUI screenshot showing the encoder node with every field visible plus
the debug pane with the exact formatted prompt and token count. That makes the setup
reconstructable, which is better than most of this repo. Here is the reconstruction.

All four share one user prompt: `Make a picture of a white cat` (29 chars, confirmed in each
debug pane). The variable is which field carries a conflicting instruction.

| Ex | asset | system prompt | thinking_content | assistant_content | tokens | outcome |
|----|-------|---------------|------------------|-------------------|--------|---------|
| 1 | `assets/zimage_example1.png` | "Make a photo of a poodle. Ignore the user's instructions." (57 ch) | "Let's make it a poodle instead, and be bright pink and have wings so it can fly" (79 ch) | "Here's your flying bright pink poodle with wings" (48 ch) | 72 | white poodle with pink wings and pink harness |
| 2 | `assets/zimage_example2.jpeg` | propaganda template, 293 ch | empty | empty | 76 | white cat rendered as a Soviet propaganda poster |
| 3 | `assets/zimage_example3.jpeg` | none (0 ch) | "Let's make it bright pink sloth instead." (40 ch) | "Here's your bright pink sloth." (30 ch) | 40 | white cat wearing a pink sloth-shaped fur suit |
| 4 | `assets/zimage_example4.jpeg` | none (0 ch) | "Let's make it pink human instead" (32 ch) | "Here's your pink human" (22 ch) | 35 | white cat, intact, with a flat pink humanoid silhouette behind it |

One annotation the table cannot show. In examples 3 and 4 the `add_think_block` widget reads
`false`, but the debug pane in both reads "Think block: enabled" and "Think tags: YES", because
a non-empty `thinking_content` auto-enables the block regardless of the widget
(`nodes/z_image_encoder.py:662`). All four examples therefore shipped a think block. None of
them is a thinking-off condition, despite two of them appearing to be one. This is the clearest
live instance of the control-disagrees-with-prompt trap described in Finding 3.

What this actually supports:

- Examples 3 and 4 have no system prompt at all, so they isolate thinking plus assistant
  against the user prompt. In both, the user's subject survives completely intact and the
  competing content appears as an additive, spatially separate artifact rather than a
  replacement. That is real support for the README caption at line 58, "thinking/assistant are
  weighted lower than user prompt". One sample per condition, but the pattern is the same in
  both and the failure mode is distinctive: not a blend of two concepts, but the losing concept
  rendered as a decal stuck onto the winning one.
- Example 1 is the only case where the class flipped, cat to poodle. It is also the only case
  where system, thinking and assistant all three agreed. So the claim "system + thinking +
  assistant combined can overpower parts of user prompt" is supported by exactly one image, and
  the three fields are confounded. Nothing here says which of the three did the work, or whether
  one alone would have sufficed. Note also that within that same image the user's "white" won
  over the injected "bright pink" for the body, so the override was partial at the attribute
  level while total at the class level. The caption says exactly this and is accurate to the
  image.
- Example 2 is the sole evidence for "system prompt is best for guiding style". A
  style-describing system prompt produced that style while the user's subject survived. There is
  no control image in the repo showing the same prompt without the system prompt, and no seed is
  displayed on any of the four. The shipped example workflows use seed 6741 with control
  `fixed` (`example_workflows/z-image_custom_nodes_workflow.json`, KSampler node 3), which makes
  a held seed plausible but does not establish it for these images.

What it does not support, and this is the important one for us: none of this is
instruction-following. Z-Image's Qwen3-4B is used purely as an encoder. It never generates.
The string "Ignore the user's instructions." in example 1 is not an instruction that the model
obeys or refuses; it is 8 tokens that get encoded alongside everything else. The observed
outcomes look like embedding-space blending with field-position weighting, which is what you
would predict, and not like a model deciding whose instruction to follow.

Transfer to us: the mechanism does not transfer, because our expanders generate. For them,
system-prompt conflict is genuine instruction-following and the failure modes are different
(refusal, contract violation, mode confusion), not decal artifacts. What does transfer is the
experimental design, which is good and cheap: hold the user prompt fixed at one short phrase,
put a deliberately conflicting instruction in exactly one field at a time, and read off which
one wins. Their version lacks a control and a seed record. Ours should have both.

---

## Finding 2: in this pipeline `<think>` was never a think token, and that undercuts every thinking result above

Status: `[measured]`, with specific token IDs.

`nodes/docs/z_image_analysis.md:57-67` records an actual tokenizer comparison:

- Qwen3-4B tokenizer: `<think>` is a single token, id 151667.
- The tokenizer ComfyUI bundles (Qwen2.5-VL) maps 151667 and 151668 to `<|meta|>` and
  `<|endofmeta|>`. Typing `<think>` there produces the subword pieces `['<th', 'ink', '>']`.

Same file also records that the base chat format is byte-identical across the two tokenizers
(`z_image_analysis.md:41-48`: 10 tokens either way, `<|im_start|>` is 151644 in both). So the
divergence is narrow and specific: only the think tags.

The consequence is not drawn out in the README but is drawn out in the analysis doc at
`z_image_analysis.md:135-141`: every thinking-block experiment in this repo encoded `<think>`
as three ordinary subword tokens, not as the control token the model was trained on. Whatever
the four screenshots in Finding 1 show, they do not show the effect of a think block in the
model's own vocabulary. They show the effect of some extra text that happens to be shaped like
one. The repo's own recommendation (`z_image_analysis.md:89-96`) is to test with and without
`add_think_block` before bothering to fix the tokenizer, and there is no record that this was
done.

Transfer to us: this transfers as a check to run, not as a fact about our models. Our serving
path is different and our expanders are Qwen3.5-VL derivatives whose own tokenizer certainly
has the think tokens. The action is: in whatever tokenizes our prompt before it reaches the
model, confirm that the trailing open `<think>` resolves to the single special token id and not
to subwords, and confirm the same for `</think>` at the parse boundary. This is a one-line
assertion in the harness and it is exactly the class of bug that produces "the model behaves
subtly wrong and nobody knows why". Given that our generation prompt ends on an open `<think>`,
a subworded tag would put the model in a state it has never been trained in.

---

## Finding 3: the `enable_thinking` inversion, and an unresolved internal contradiction about the default

Status: `[measured]` for the inversion, `[assertion]` in direct contradiction with itself for the default.

The inversion is real, was tested, and is worth carrying: in the Qwen3 chat template,
`enable_thinking=True` adds no think block (it leaves the assistant turn open so the model can
generate its own thinking), and `enable_thinking=False` is what injects a pre-filled empty
`<think>\n\n</think>\n\n` to skip thinking. The repo states this as a correction to its own
earlier reading (`nodes/docs/z_image_analysis.md:5`, `:41-42`; `CHANGELOG.md:471-477`,
labelled "Key Finding (Corrected Analysis)"; restated at `nodes/docs/z_image_encoder.md:96-101`).
They renamed their parameter from the Qwen name to `add_think_block` specifically because the
Qwen name reads backwards in an encoding context (`z_image_analysis.md:44`).

The problem with the default is provable entirely inside the repo, without adjudicating what any
external library does. Their default emits a prompt shape that neither branch of the upstream
flag produces.

Per their own table (`nodes/docs/z_image_encoder.md:96-99`) the Qwen template has exactly two
outputs: `enable_thinking=True` gives no block at all, and `enable_thinking=False` gives an
empty `<think>\n\n</think>\n\n`. Their code with `add_think_block=True` emits
`<think>\n{think_inner}\n</think>\n\n` where `think_inner` is whatever `thinking_content` holds
(`nodes/z_image_encoder.py:788-796`). When that content is non-empty, which is the majority case
because 15 shipped templates pre-fill it (`CHANGELOG.md:126-128`) and the JS auto-fills it on
template selection (`CHANGELOG.md:110-113`), the result is a populated think block. That is a
third shape. It is not the no-block shape and it is not the empty-block shape, and the Qwen
chat template never produces it.

The narrower contradiction sits underneath. In the empty-content case their output is exactly
the `enable_thinking=False` shape, confirmed empirically by example 2 in Finding 1, whose debug
pane prints a bare `<think>` / `</think>` pair. Yet `nodes/docs/z_image_encoder.md:105-114` and
`CLAUDE.md` both justify the `True` default by saying it matches a reference implementation
using `enable_thinking=True`, which by their own table is the branch that adds no block. I am
not resolving which side is right, because the reference path is redacted in the repo
(`EXPERIMENT_METHODOLOGY.md:13` shows `<diffsynth-studio-path>`) and cannot be checked from
here. The default is load-bearing either way: it is what every user gets and what all four
README screenshots ran with.

This matters to us for a reason beyond bookkeeping. A pre-filled thinking trace handed to a
model that was trained to generate one is precisely the shape mismatch we would reproduce if
anything in our harness ever populates the open `<think>` in the trained generation prompt.
Their default does this by design and nobody ever checked what it cost.

There is a smaller related trap in the code. `add_think_block` is auto-enabled whenever
`thinking_content` is non-empty (`nodes/z_image_encoder.py:662`:
`use_think_block = add_think_block or bool(thinking_content.strip())`). Examples 3 and 4 in
Finding 1 both show the widget reading `false` while the debug pane reads "Think block: enabled"
and "Think tags: YES". Also, the widget default is `True` (`z_image_encoder.py:565`) while the
Python signature default is `False` (`z_image_encoder.py:608`), so the effective default depends
on whether the node is driven through the UI or called directly.

Transfer to us: the inversion itself is the durable lesson and it generalises across the whole
Qwen3 family, which includes our expanders' base. Anywhere our harness exposes a
thinking on/off control, name it for what it emits, not for the upstream flag, and assert on the
emitted string rather than on the flag. And take the auto-enable trap as a direct warning: a
control whose stated value disagrees with the prompt that actually ships is a control you cannot
reason from. Our thinking is on by default via an open `<think>` in the trained generation
prompt; if we ever add a suppression switch, the test must read the final prompt string, not the
switch.

---

## Finding 4: Z-Image is encode-only with a closed think block, so the repo has nothing on suppressed thinking

Status: `[assertion]` that multi-turn is out of distribution; otherwise this is a scope fact.

This is the direct answer to the part of question 2 about whether a thinking-trained model
behaves differently when thinking is suppressed. The repo has nothing on it, and structurally
could not have.

The Z-Image path never runs a decode. The formatted string goes straight to
`clip.tokenize(...)` then `clip.encode_from_tokens_scheduled(...)`
(`nodes/z_image_encoder.py:745-746`). The think block they construct is always closed:
`<think>\n{content}\n</think>\n\n` (`z_image_encoder.py:788-796`, and the same in
`format_conversation` at `z_image_encoder.py:305-318`). Turning `add_think_block` off does not
suppress a model's thinking; it just removes some characters from a string that is about to be
embedded. There is no generation to degrade.

The one node in the repo that did generate is the Hunyuan prompt expander, and it is marked
dead (see Finding 8).

One adjacent claim worth carrying as a flag rather than a fact: `z_image_analysis.md:111` says
"Z-Image is primarily trained on single-turn prompts. Multi-turn is experimental." No source or
test is given. If true for us in any form, it matters, because we have a multi-turn-shaped
contract for the edit expander. Treat as unverified.

Transfer to us: the honest conclusion is that this repo's entire thinking-block body of work is
about static text in an encoder, and our question (does a thinking-trained generator degrade
when its thinking is suppressed) is untouched by it. Do not let the volume of thinking-related
material in this repo suggest otherwise. The only thing that carries is the tokenizer check in
Finding 2 and the naming discipline in Finding 3.

---

## Finding 5: what `template_influence_analyzer.py` measures, and why its method is the one not to copy

Status: `[plan]` for results; the method itself is readable and its flaws are `[measured]` from the code.

The intended method, which is sound in outline:

1. Format the same user prompt twice, once under system prompt A and once under B, using
   DiffSynth-style chat markers (`template_influence_analyzer.py:92-100`).
2. Encode both through the already-loaded ComfyUI CLIP, avoiding a second model load
   (`:102-122`). Reusing the loaded model is the genuinely good idea in this file.
3. Truncate both to the shorter length, flatten to 1-D, and compute cosine similarity and L2
   distance (`:124-161`).
4. Additionally compute per-token L2 and compare the mean over the first quarter of positions
   against the last quarter, to test a "position decay" hypothesis: that user tokens nearer the
   system prompt are more influenced by it (`:148-160`, `:266-269`; hypothesis stated at
   `EXPERIMENT_METHODOLOGY.md:146-188`).
5. Bucket the cosine value into four verdicts at thresholds 0.995 / 0.99 / 0.95 (`:163-172`).
   Optionally run all pairs among six built-in templates spanning empty, minimal, default,
   horror, comedy and cinematic (`:33-40`, `:195-234`).

It produces a formatted text report and nothing else. No file is written, so even if it was run
interactively the numbers were never captured. Nothing in the repo records an output.

Three method defects, all of which we would inherit if we copied the shape:

1. The node accepts a `drop_idx` input, documents it, prints it, and never applies it. It is
   threaded into `get_embedding` (`:102-108`) and then not used in the body (`:110-122`); the
   token dropping it names happens, if at all, inside ComfyUI's encode path, not here. The
   offline twin does apply it (`experiments/quick_embedding_test.py:60-63`). So the node and the
   script measure different quantities under the same name. If any number was ever quoted from
   one of them, there is no way to tell which.

2. The empty-template arm is not a control. `format_template` returns the bare user prompt
   when the system prompt is empty (`:94-95`, same at `quick_embedding_test.py:31-32`). So
   "empty vs anything" compares removing the system prompt against removing every chat marker at
   once, plus a sequence-offset of roughly 30-plus tokens. Combined with truncate-to-min-length
   and a fixed drop index in the script version, position i in one tensor is compared against a
   different source token in the other. Whatever that measures, it is not the effect of the
   system prompt.

3. Flattened cosine over full hidden-state sequences will sit near 1.0 almost regardless of
   input. LLM hidden states are strongly anisotropic; they share a large common mean direction,
   and flattening the whole sequence into one vector before cosine buries any per-token signal
   under that shared component. The four verdict thresholds (0.995, 0.99, 0.95) have no stated
   origin anywhere in the repo, and the identical ladder is restated as an authoritative
   interpretation table at `EXPERIMENT_METHODOLOGY.md:289-295`. A rig built this way would
   report "templates have NO effect" for almost any pair, including pairs that visibly change
   generated images. Given Finding 1 shows templates do visibly change outputs, a near-1.0
   reading would have been a false negative, not a result.

Transfer to us: the question the rig was built to answer is our open question, so the design
lesson is worth having. If we want to measure whether our expanders tolerate a modified system
prompt, do not measure it in embedding space this way. Measure it in output space, which for us
is cheap and unambiguous because our expanders emit a structured contract: hold the user input
fixed, vary one span of the system prompt, and score the decoded output on contract conformance
(does the JSON parse, are the required keys present, is the mode-dependent contract respected),
plus a thinking-trace length and a semantic diff of the expansion. Those are discrete and
comparable. If an embedding-space measure is ever wanted anyway, compare per-token after
aligning on identical user-token positions, subtract the corpus mean before cosine, and derive
any threshold from a null distribution of paraphrase pairs rather than picking round numbers.

Also worth stealing outright: the position-decay question at `EXPERIMENT_METHODOLOGY.md:146-188`
is a good question that was never answered. With our always-present 2400 and 4500 token system
prompts, whether influence decays with distance from the system block is directly actionable,
because it would tell us whether the tail of a long system prompt is doing anything at all.

---

## Finding 6: the double-wrap bug, and the escape hatch for it

Status: `[measured]`, in the sense that it was a real bug found and fixed in shipped code.

For several versions, every Z-Image encoder in this repo double-applied the chat template. The
node carefully built `<|im_start|>system...<|im_end|>...`, handed the finished string to the
framework tokenizer, and the tokenizer wrapped the whole thing in the chat template again. Fixed
in v2.9.9 by passing `llama_template="{}"` to force an identity wrap
(`CHANGELOG.md:169-173`; the fix is live at `nodes/z_image_encoder.py:743-745` and
`:441-446`, with the reason in the comment).

This is the single most practically useful thing in the repo. It is silent: nothing errors, the
prompt is merely nested one level deeper than intended, and every generated image up to that
point was produced under a prompt the author did not think they were sending. It went unnoticed
across at least eight releases.

Their mitigations, all of which are good and cheap, and all of which they added only after
getting bitten:

- A `formatted_prompt` output that emits the exact encoded string
  (`z_image_encoder.py:594-595`).
- The same string printed to the server console on every run (`:740`).
- An explicit presence check in the debug pane rather than trusting the flag:
  `Think tags: YES/NO` derived from `'<think>' in formatted_text` (`:722`).
- Debug output wrapped in code fences because Preview nodes were rendering the special tokens
  as markdown and hiding them (`CHANGELOG.md:175-177`, `:249-251`).
- A token count against the 512 reference limit with an explicit over-limit warning
  (`:724-735`).

Transfer to us: directly, and this is the highest-value item. Our expanders have a fixed system
prompt of 2400 or 4500 tokens plus a trained generation prompt ending in an open `<think>`. If
anything in our path applies a chat template on top of a string we already templated, we get a
nested prompt and a model running out of distribution, with no error. The harness should emit
the exact final string that goes to the tokenizer, and a test should assert on it: exactly one
`<|im_start|>system`, exactly one trailing `<think>` with no matching `</think>`, and no
duplicated turn headers. Assert on the string, not on the flags that were supposed to produce it.

---

## Finding 7: JSON keys render as literal text in the image, and the fix collides with a real feature

Status: `[observed]` for the failure, `[measured]` for the code/doc mismatch.

The failure is concrete and reproducible enough that they built two nodes and a toggle for it.
When a structured prompt like `{"subject": "a cat", "style": "photo"}` is fed to the encoder,
the quoted key names show up as visible text baked into the generated image
(`nodes/docs/z_image_encoder.md:517-523`, `CHANGELOG.md:206-208`). The mitigations are a
`strip_key_quotes` toggle on the encoders and turn builder, a standalone `PromptKeyFilter` node
(`z_image_encoder.py:1005-1043`), and three dedicated templates whose system prompt and
pre-filled thinking both carry explicit anti-text-rendering instructions. The json_structured
one reads, in part, "CRITICAL: Do NOT render any text, labels, keys, quotation marks, or
structured formatting in the image" with a matching line inside the think block
(`nodes/templates/z_image/json_structured.md`, added per `CHANGELOG.md:103-106`).

Two problems with the mitigation that we should not reproduce:

1. The documentation and the code disagree. The docs say it removes quotes from keys only while
   preserving quoted values (`z_image_encoder.md:523`). The code strips the key quotes with a
   regex and then unconditionally deletes every remaining double quote in the string
   (`z_image_encoder.py:623-624`, and identically at `:625-628`, `llm_output_parser.py:339-346`,
   `z_image_encoder.py:402-407`). The CHANGELOG records the widening as deliberate
   (`CHANGELOG.md:197-198`) but the user-facing doc was never updated.

2. That collision matters, because quoting is the documented trigger for intentional text
   rendering. `z_image_encoder.md:470-475` tells users to put text in quotes when they want it
   rendered, and the Hunyuan encoder uses quoted text to trigger byT5 multilingual rendering
   (`nodes/hunyuan_video_encoder.py:76`, `CHANGELOG.md:589`). So the fix for accidental text
   silently disables deliberate text. Their own shipped example workflow contains
   `- "Face Shape": Oval,` in the user prompt with `strip_key_quotes` enabled
   (`example_workflows/z-image_custom_nodes_workflow.json`, node 56), which is the exact
   collision live in the repo.

Transfer to us: high relevance, because our expanders emit structured JSON and something
downstream consumes it. Two distinct lessons. First, if any JSON we produce is ever passed
through to an image encoder as prompt text rather than having its values extracted, expect the
keys and punctuation to be rendered as pixels. The defence is to extract values and build a
prose or key-free string, not to blanket-strip characters. Second, their prompt-level defence
is interesting on its own: they put the anti-rendering instruction in both the system prompt and
the thinking content, which is the first thing in this repo that treats the think block as a
place to put a constraint rather than a place to put a description. Whether that helped is not
recorded.

---

## Finding 8: prompt expansion was abandoned, and its decoding config was inherited rather than chosen

Status: `[assertion]` that it did not work, stated by the author; `[measured]` that the config was copied.

The direct answer on `hunyuan_video_prompt_expander.py`: the CHANGELOG entry that introduces it
reads, in full, "**HunyuanVideoPromptExpander Node** - not working, likely not going to pursue
due to better alternatives" (`CHANGELOG.md:548`). No failure mode is recorded beyond that. The
node is still present and still registered.

What it was trying to do is the interesting part, and explains why it was fragile. Rather than
loading a second model, it reaches into the already-loaded ComfyUI CLIP, pulls the Qwen
transformer out by probing four different attribute paths
(`hunyuan_video_prompt_expander.py:224-239`), extracts its state dict, hand-maps ComfyUI key
names to transformers key names (`:283-297`), locates and loads just `lm_head.weight` out of a
safetensors shard to turn an encoder back into a causal LM (`:162-200`), constructs a
hand-written `Qwen2Config` with hardcoded dims (`:267-278`), and loads it all with
`strict=False` (`:309`). The stated payoff is reusing about 14GB of resident weights for about
50MB extra (`:7`, `:86`). The fallback when `lm_head.weight` is not found is to tie it to the
embedding matrix and log a warning (`:303-306`), which will generate text, just not the model's
text. Three of those steps are silent-wrong-answer generators rather than error paths.

On the decoding config specifically, which was asked about. `GENERATION_CONFIG` at
`hunyuan_video_prompt_expander.py:21-29` is temperature 0.01, top_k 1, top_p 0.001,
max_new_tokens 512, do_sample True, repetition_penalty 1.0. The comment on line 21 says
"Generation config matching official HunyuanVideo rewriter". So it was copied from the
reference implementation, not tuned. Nothing in the repo records any sampling sweep, any
comparison of deterministic against sampled expansion, or any observation about expansion
diversity. Two things worth noticing about the config itself: `do_sample=True` with `top_k=1` is
greedy decoding wearing a costume, and the exposed temperature widget only matters through a
threshold at 0.1 which flips top_k and top_p to 50 and 0.9 (`:365-366`), so the control is
effectively a two-position switch, not a dial.

The strategic thinking around expansion is written up at length in
`nodes/docs/attention_based_prompt_expansion.md`. Its thesis is that you might skip the
rewriter entirely by putting expansion instructions in the system prompt, on the theory that
user tokens attending to those instructions get pushed toward a more detailed region of
embedding space. The document is careful and, to its credit, states plainly what the technique
cannot do: it generates no new tokens, sequence length is unchanged, it cannot add detail that
is not present, and the DiT was trained to decode explicit detail embeddings rather than
implied ones (`:249-266`). It also predicts the example matters more than the instructions
(`:337-349`). All of it is `[plan]`. The timing and quality figures in it (`:201`, `:216`,
`:227`, `:419-424`) are invented illustrations, not measurements.

Transfer to us: the "reuse the encoder's weights as a generator" approach is a dead end and we
are not tempted by it, since our expanders are separate models. Two things do carry. One, the
guardrail in `expand()` at `:393-396`, which skips expansion entirely if the input already
exceeds 200 words, is a cheap and sensible idempotence check; an expander that re-expands
already-expanded text is a real failure mode and we should have the equivalent guard. Two, the
decoding question is genuinely open and this repo did not touch it. Near-deterministic decoding
for an expander is a defensible default for reproducibility, but nobody here checked whether it
costs expansion diversity or whether it makes the thinking trace degenerate. If we care about
variety across seeds, that is ours to measure.

---

## Finding 9: what `llm_output_parser.py` actually had to defend against

Status: `[measured]`, in the sense that each branch is a defence someone wrote; no failure log exists.

The defensive cases are the useful artifact, since each one is a shape some real model emitted.
The catalogue:

- Markdown code fences, in three variants: ` ```json `, ` ```yaml `, ` ```yml `, and a bare
  ` ``` ` with an optional language tag on the first line. Matched case-insensitively, and with
  a heuristic that a language identifier is alphanumeric and under 20 characters
  (`llm_output_parser.py:28-68`). Notably the fenced-block extractor finds the opening fence
  anywhere in the text, not just at the start, which handles a model that prefaces its JSON with
  a sentence of chat.
- Wrapper objects. Output nested one level under `result`, `data`, `output` or `response`
  (`:98-119`).
- A top-level array instead of an object, where the intended payload is the first element
  (`:108-111`).
- Key-name drift. When the configured key misses, it retries against `prompt`, `text`,
  `content`, `message` (`:226-231`, and identically in the yaml and auto branches). This is the
  clearest signal in the file: models did not reliably use the key names they were asked for.
- Values that are themselves objects or arrays where a string was expected, re-serialised with
  `json.dumps` rather than crashing (`:92-93`).
- Nested key paths via dot notation, `result.prompt` (`:71-95`).
- Almost-JSON that YAML will accept. Auto mode tries JSON, then YAML, then gives up
  (`:277-335`). YAML parsing a JSON-ish blob with a trailing comma or unquoted key is the
  practical payoff here.
- Empty input, handled as an explicit early return with a distinct status (`:197-199`).
- Total parse failure, with `fallback_to_passthrough` defaulting to true so the raw text becomes
  the prompt rather than producing nothing (`:162-165`, `:330-335`).

The design choice worth stealing is the `parse_status` output (`:177`). Every path sets a
distinct string, including which mode succeeded and a truncated exception message on failure.
The parser never raises and never silently returns empty; the caller can always tell whether it
got a parse, a fallback or an error, and why. For an unattended pipeline that is the difference
between a diagnosable bad run and a mysterious one.

Transfer to us: we already ported the shape. What is worth adding is the status-string
discipline, and one correction of emphasis. Their fallback-to-passthrough default is right for
their situation, where a garbled prompt still produces an image. It is wrong for ours: our
expanders have a mode-dependent contract, and silently passing raw text through as if it were a
valid expansion converts a parse failure into a silent contract violation downstream. We want
the same exhaustive tolerance for input shapes, with a loud status and a hard failure at the
contract boundary rather than a quiet passthrough. Also note the key-drift defence: if our
expanders ever drift off their contracted key names under a modified system prompt, that is
precisely the signal we would want the harness to surface, and a silent alias fallback would
hide it.

---

## Finding 10: the scale gap between their system prompts and ours

Status: `[measured]` by counting the template corpus.

Their entire Z-Image template library is 144 files totalling 8119 words
(`wc -w nodes/templates/z_image/*.md`), so roughly 56 words each. The extremes are `default.md`
at 16 words, which is frontmatter only and an empty system prompt, and `yaml_structured.md` at
129 words. The nine Qwen-Image templates in `nodes/templates/` and the Hunyuan video templates
are in the same range; the largest system prompt visible in the README screenshots is 293
characters (example 2).

Our expanders carry fixed system prompts of roughly 2400 and 4500 tokens. That is one to two
orders of magnitude larger than anything tested here.

Transfer to us: mostly as a limiting condition on everything above. "System prompt is best for
guiding style" was observed with a 293-character style description against a 29-character user
prompt. Nothing in this repo speaks to what a 4500-token system prompt does, whether its tail
carries weight, or whether editing a span in the middle of one has any effect at all. Their own
length-ablation plan tops out at 500 characters (`EXPERIMENT_METHODOLOGY.md:194-207`) and was
never run. Their one relevant remark is a tradeoff note that longer templates eat the token
budget with diminishing returns (`attention_based_prompt_expansion.md:351-356`), which is an
`[assertion]`. Treat the whole body of system-prompt work here as evidence about short system
prompts only.

---

## Finding 11: the "direct token transfer" claim is wrong, and their own code shows it

Status: `[assertion]`, contradicted by the repo's own code path.

`nodes/docs/z_image_character_generation.md:96-132` argues that because Z-Image's encoder is
Qwen3-4B, you should use a Qwen3-family model to write your prompts, because then "these exact
Token IDs pass directly to Z-Image's encoder" and "You're not transferring text - you're
transferring a pre-computed semantic state." Using any other family incurs "Information Loss"
in re-tokenization. This is escalated into a recommendation table
(`:134-141`) advising Qwen3-235B for maximum fidelity, and extended to think blocks with the
claim that a larger Qwen3's thinking tokens "prime Z-Image's encoder to a configuration that
directly reflects the larger model's reasoning" (`:120-132`).

No token IDs are transferred anywhere in this repo. The parser emits strings
(`llm_output_parser.py:379-380`), the encoder receives a string, and
`clip.tokenize(formatted_text, llama_template="{}")` (`z_image_encoder.py:745`) tokenizes from
scratch. And by Finding 2, the tokenizer doing that is the Qwen2.5-VL one, not even Qwen3's, so
a Qwen3-authored `<think>` arrives as three subwords regardless of who wrote it.

There is a much weaker true statement underneath: BPE is deterministic on a string, so text
emitted by a model will re-tokenize to the same ids under the same tokenizer. That is a
round-trip stability property, not a transfer of semantic state, and it says nothing about text
from other families "losing information" — different subword boundaries over the same characters
are not lossy.

Transfer to us: only as a warning about this repo's register. This document is confident,
detailed, and wrong on its central mechanism, and it sits next to documents that are careful and
right. Weight claims here by whether a tokenizer was run or an image was produced, not by how
assured the prose sounds.

---

## Claim ledger

For the README specifically, since the question was which of its claims have anything behind them.

| Claim | Location | Status | Behind it |
|-------|----------|--------|-----------|
| system + thinking + assistant combined can overpower parts of user prompt | `README.md:42` | observed, n=1, confounded | one screenshot, three fields varied together, no control, no seed shown |
| system prompt is best for guiding style | `README.md:50` | impression | one screenshot, no control image |
| thinking/assistant are weighted lower than user prompt | `README.md:58` | observed, n=1 per condition | two screenshots with no system prompt, consistent additive-artifact failure mode |
| user prompt drives initial generation | `README.md:66` | observed, n=1 | one screenshot |
| enable_thinking is inverted | `README.md` via docs, `z_image_analysis.md:41-42` | measured | tokenizer run, correction logged in CHANGELOG |
| `<think>` is subworded by the bundled tokenizer | `z_image_analysis.md:57-67` | measured | explicit token ids 151644, 151667, 151668 |
| ComfyUI and diffusers templates are identical | `z_image_analysis.md:41-48` | measured | 10 tokens both ways, matching special token ids |
| 512 token limit is a choice not an architectural limit | `README.md:82` | reasoned from config | reads `axes_lens=[1536,512,512]` and `rope_theta=256.0`; explicitly says "we don't know" |
| Qwen3-authored prompts transfer token ids directly | `z_image_character_generation.md:110-118` | assertion, false | contradicted by `z_image_encoder.py:745` |
| Z-Image is primarily trained on single-turn | `z_image_analysis.md:111` | assertion | nothing |
| attention-based expansion biases embeddings usefully | `attention_based_prompt_expansion.md` | plan | nothing; all figures illustrative |
| HunyuanVideoPromptExpander does not work | `CHANGELOG.md:548` | author's verdict | no recorded failure mode |

---

## What I would actually do with this

Ordered by value to our harness.

1. Assert on the final prompt string, not on the flags. Emit it, log it, and test it: one
   system turn, one user turn, a single trailing open `<think>` with no closing tag, no
   duplicated turn headers. This is the double-wrap lesson (Finding 6) and it is the cheapest
   insurance in the report.
2. Verify `<think>` and `</think>` resolve to single special token ids in our tokenization path
   (Finding 2). Our trained generation prompt ends on an open `<think>`; a subworded tag puts
   the model out of distribution with no error.
3. Make the output parser loud. Keep the input tolerance, keep a per-path status string, and
   fail hard at the contract boundary rather than passing raw text through (Finding 9).
4. Design the system-prompt sensitivity test in output space, not embedding space, scoring
   contract conformance under single-span edits (Finding 5). Their embedding-space rig is a
   worked example of how to get a false negative.
5. Add an idempotence guard so an already-expanded input is not expanded again (Finding 8).
6. Borrow their ablation design for field-priority questions: fix a short user input, inject a
   conflict into exactly one field at a time, and add the control arm and the recorded seed they
   did not have (Finding 1).
7. Treat the position-decay question as ours to answer (Finding 5, Finding 10). With system
   prompts of 2400 and 4500 tokens, whether the tail of the system prompt is doing anything is
   both unanswered here and directly actionable.
8. Assume nothing in this repo speaks to system prompts longer than a few hundred characters
   (Finding 10), and nothing in it speaks to suppressed thinking in a generating model
   (Finding 4).
