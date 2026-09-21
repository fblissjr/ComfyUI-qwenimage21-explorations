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

- **The expander answer is graded from the backend's own split, not from a
  rejoined one.** heylook returns thinking as its own content block; the client
  used to fold that back into ComfyUI's inline shape so one parser could grade
  either backend, which meant splitting a string we had just joined.
  `answer.grade_parts` takes the parts directly and shares the contract logic,
  so parity is kept and the round trip is gone. Prompted by the heylook side;
  the round trip was safe in practice, and a body containing `</think>` is the
  case it could not have promised — now a test.
- **Server presets are supported, by expanding them here rather than sending
  them.** The node takes a `preset` by name or id. **The server refuses a
  preset as a request field** — a deliberate 422, because named sampler bundles
  were removed in v2.0.30 — so the UI expands them client-side and so does
  this.

  **The spelling split is a boundary, not a defect**, which makes it stable
  enough to build on. Preset params speak the server's internal vocabulary;
  `/v1/messages` is Anthropic-conformant, so the rename sits at that boundary
  exactly as `stop_reason`'s does. That makes `expand_preset` a boundary
  adapter in the sense AGENTS.md means — normalise the upstream convention once,
  at the edge, so nothing inland has to know about it.

  *And a correction to how this was found, because the method is the part worth
  keeping.* The heylook side credits a schema description, and
  `Preset.params` does carry one saying `enable_thinking` is spelled `thinking`
  on that route. **It was not read.** The rename turned up from listing every
  param key across the stored presets and intersecting it with the wire's valid
  fields — the same compare-two-things move as everything else found today. The
  doc would have worked; the comparison did work, and it also finds what no doc
  says.

  **The translation is load-bearing, not tidiness.** A preset's `params` use
  the server's internal spellings, and `enable_thinking` appears in most of the
  stored ones while being a 422 on the wire. Verified both directions against
  the live server: the expanded field set answers 200 and the same preset
  forwarded verbatim answers 422. Forwarding would have failed on exactly the
  presets people use.

  Precedence, so it is predictable: raw text, then a local template, then the
  preset, then the checkpoint. *Corrected within the session:* the preset's
  system prompt was first placed **below** the checkpoint, which a survey then
  showed was wrong for nearly every real preset — **18 of the 19 stored ones
  carry a system prompt**, two are named `qwen_image-t2i` and
  `qwen_image-edit`, and the normal PE wiring supplies a checkpoint. Selecting
  those would have silently ignored exactly what was selected. A preset now
  beats the checkpoint, and `system_source` is an output so the winner is never
  a guess.

  **The cost of that order is real and belongs to whoever picks the preset.** A
  general-purpose preset's prompt replaces the trained contract and the answer
  will not conform; `contract_ok` is what reports it. Sampler values and the
  reasoning level layer over the profile the same way — `reasoning_effort`
  rides through as an explicit field and was verified to answer 200.

  **Not a dropdown, and that is a ComfyUI limit rather than a choice.** A
  combo's options are fixed when the schema is built at startup, while
  `base_url` is a runtime input, so a dropdown would have to guess the server
  address before the graph names it and would hang or empty out when the server
  is down. It is a string resolved at execute, and a miss raises naming every
  preset the server has.

- **The client's field spellings are verified against the live server, not
  assumed.** heylook added 422s naming the right spelling for three fields it
  used to drop in silence — `enable_thinking`, `max_new_tokens` and
  `system_prompt`. Our exact field set answers 200, so nothing here was on the
  wrong side of it. The guard was committed but not yet deployed when
  first probed: all three wrong spellings answered 200, and the `system_prompt`
  probe ran with **no system prompt at all** and said nothing — the silent
  drop, demonstrated rather than described. Committed and deployed are two more
  things that should agree, and briefly did not. **Re-checked after the server
  was restarted at 2.0.50: our field set still answers 200 and all three
  wrong spellings now answer 422 naming the right one**, so the guard is live
  and this client is verified against it rather than against a reading of it.

- **An alarm-shaped claim deserves more scrutiny than a reassuring one, and
  both sides of this exchange proved it the hard way.** The `max_tokens` cap
  was relayed, checked here against the live schema, and withdrawn; the heylook
  side then executed their own cascade and confirmed the withdrawal — a request
  value wins, and the floor is reached only when the request is silent. Two
  failure modes are worth keeping, because neither was a reasoning error:

  - **Their probe passed a dict where the cascade reads attributes**, so every
    field resolved to `None` and it printed the alarming answer. The
    instrument never reached the subject. The tell was an internal
    contradiction in the output — config appearing to beat the request, against
    the documented order — not the headline number.
  - **This side nearly repeated it.** The first verification here read a
    checkout on disk that was six versions behind the running server, and a
    conclusion was already forming when the owner said so. Same shape: an
    instrument aimed at the wrong build.

  **What held was redundancy, not care.** The cap was refuted by the live
  schema *and* corroborated by a smoke run that had recorded no truncated row.
  One line of evidence would have been a coin flip; two that agree are a
  finding. The general rule, worth pricing before acting: a probe that confirms
  the scarier reading has earned suspicion, not relief.

- **A cancelled or timed-out expansion used to keep running on the server.**
  heylook writes nothing for a non-streaming request until it finishes, so
  hanging up does not stop it: the run continues and blocks everything queued
  behind it. The client now sends `X-Request-ID` and issues
  `DELETE /v1/requests/{id}` whenever a request does not deliver, timeout or
  otherwise. Found from a wire reference relayed by the heylook side and
  checked against that server's own spec, which carries the route.
  `tests/test_heylook.py` covers both directions, including that a delivered
  request cancels nothing.

- **The expanders' sampling never shipped with the weights, rather than being
  lost in conversion.** Relayed from the heylook side as a conversion loss;
  checked here instead, and both HF checkpoints and both mlx-vlm conversions
  carry `_from_model_config: true` with **no sampling keys at all**, byte for
  byte the same shape. Nothing was dropped because nothing was there. The
  practical upshot is the same — send them explicitly or set them per model —
  but the diagnosis matters: re-converting recovers nothing, and no server can
  read them off the artifact. Recorded in `profiles.py`, next to the
  asymmetry it explains: the system prompt **does** travel with the weights,
  which is why `templates.py` prefers the checkpoint's copy.
- **Tuning `top_k` or `min_p` away from the reference would be an override, not
  a fix.** The heylook side makes a reasonable case that `top_k` 20 is
  aggressive for prose and that `min_p` is the better diversity lever. Both are
  arguments about output taste, and both differ from what `pe_core` specifies,
  so they belong in `profiles.OVERRIDES` with a reason if the owner wants them
  — the same treatment the t2i cap got. Worth one caveat their vantage does not
  include: these models emit a contract-bearing JSON envelope around the prose,
  so a constraint that costs diversity in a rewrite may be buying conformance
  in the envelope, and the vendor chose 20 knowing what the model emits.

- **The relayed heylook claims were checked against that server, and they do
  not all hold.** There is **no server-side `max_tokens` cap** — the schema
  bounds it below only, and the options endpoint says a request field wins over
  the per-model default — so nothing was being truncated, and the smoke run's
  zero truncated rows were evidence rather than a coincidence. That was also
  challenged fairly (zero is consistent with the sentinel never matching), so
  `tests/test_heylook.py` now pins `truncated` against the three `stop_reason`
  values the API declares. The `engines` tag **is** real and supersedes the
  provider-key read. Relayed claims get checked; this batch was right twice and
  wrong once, in the direction of alarm.
- **An inference about heylook's KV knobs is withdrawn, and the server's own
  text is why.** Told the owner that
  `max_kv_size` and `cache_type` bound the allocation and offered to time
  reloads; the heylook side says both are inert for this checkpoint family and
  `context_length` allocates nothing on MLX. Now confirmed from
  `/v1/admin/model-options`, which names the architectures affected —
  `qwen3_5` among them — and says `max_kv_size` is "NOT A PREALLOCATION and not
  a load-time lever". So the withdrawal stands on evidence rather than
  deference, and it was wrong on two counts: preallocation, and applicability.
  [`../quantization-strategy.md`](../quantization-strategy.md) section 17.

- **Step count matters much less for edit than for t2i, and the owner's read
  that 20 is enough holds where it was made.** Swept 16/20/25/30/40 twice, same
  prompt and seed in each sweep, one scene each. **Edit with one reference
  barely moves at all** across that whole range — the adjacent differences are
  a fraction of a level out of 255 and do not shrink, so there is nothing to
  converge to; 16 is already close to 40. **t2i moves several times more and is
  still moving at 20**, settling only around 30: its distance to the 40-step arm
  falls steadily where edit's is flat from the start. A reference constrains the
  trajectory hard enough that the schedule has little left to decide, which is
  the mechanism the split is consistent with.

  **What this does not establish.** A pixel difference is not a quality
  judgement, and looking at the t2i pair the gap is mostly the subject shifting
  scale rather than detail improving. One scene per mode, one seed, judged by
  eye.

  **Acted on 2026-09-20, by the owner: the templates carry per-mode steps**, in
  `scripts/build_example_workflows.py::STEPS`. **t2i keeps the official 25 and
  edit drops to 20.** t2i was briefly set to 30 — where the sweep says it
  settles — and the owner reverted it as not worth the extra steps. That is a
  fair reading of the evidence rather than a contradiction of it: "still
  moving" is not "better", and the sweep measured where a sample stops changing,
  not where it stops improving. So the asymmetry the sweep found is spent on
  edit, which has headroom to give up, and not on t2i, which would only cost
  more.

- **The shipped graph now runs end to end in its current form, preset-driven.**
  Until this point the example graphs had been validated against the live
  schemas but never executed since the rebuild around `SamplerCustomAdvanced`:
  the two edges the test harness could not cover were the expander's own, and
  the harness was otherwise a strict subgraph. Closed by running the file
  unmodified with only `base_url`, `preset` and the brief supplied at runtime —
  preset fetched and expanded, system prompt taken from it with no
  `checkpoint_dir` at all, through the encode node, the sigma schedule and the
  custom sampler to a saved image. **No address is stored in the file**; it
  ships with a placeholder.

- **The whole graph was exercised on a live server, and the pipeline is
  deterministic.** t2i at two canvases, edit at one reference, at two of the
  same size, and at two of different sizes and aspects, plus a step sweep — all
  through `SamplerCustomAdvanced` and the sigmas node. The control that makes
  the rest readable: the same arm run twice is **pixel-identical**, so any
  difference between arms is signal rather than noise.
- **The int8_convrot text encoder changes the sample without visibly degrading
  it** (one pair, t2i and edit, everything else held). Both members of the pair
  are equally good to look at; the composition, subject, palette and lighting
  survive and small details move. **This is one pair judged by eye, not a
  panel**, and the repo's own rule applies: a changed trajectory yields a
  different sample, not a worse one, so a pixel difference is not a quality
  verdict. Recorded as "no visible degradation on one pair", which is all it is.
- **Test prompts must be rewritten prompts, not briefs** (the owner). The
  encoder only ever sees the expander's output, and the two registers are
  nothing alike, so driving a render from a brief is off-distribution and
  quietly voids the comparison. `prompt_bank/` now holds real expander outputs
  as exemplars plus hand-written entries held to their shape.

- **A sigmas node, because the stock route rests on a coincidence** (the
  owner: we should not rely on an accident). `ModelSamplingFlux` does reproduce
  2.1's dynamic shift exactly, but only because Flux's VAE-8-plus-patch-2
  token count equals 2.1's VAE-16 unpatched one, and it still leaves the
  terminal stretch undone. `QwenImage21Sigmas` computes both from the
  checkpoint's own scheduler values and reads the canvas from the latent it is
  handed, so it cannot disagree with the sampler. [`sampling.md`](sampling.md).
- **Corrected the same day: `shift_terminal` was described as impossible in
  ComfyUI.** It is not. No *stock* node does it, and nothing prevents one from
  doing it — the transform is three lines on the sigma vector. *Corrected
  again the same day:* that entry described the wrong ordering as changing
  nothing, and it does not. Stretching after the trailing zero is appended
  stretches the zero too, so the schedule ends at the terminal and the sampler
  stops with that much noise left in the image. It is now a named mode
  (`stop_short`) beside `release` and `off`, and `stretch_to_terminal` raises
  on a curve that already ends at zero — so the accident is prevented at the
  one place it could be written, not just detected afterwards.

- **The structured encode node reproduces core's exactly, verified on a
  render rather than argued.** Same seed, same fixed prompt, same reference
  image, only the encode node swapped: the two outputs are **pixel-identical**,
  every pixel of the frame, differing only in the PNG's embedded workflow
  metadata. Checked against caching rather than trusted — the structured arm's
  history shows only the loaders cached, with the encode, sampler, decode and
  save all genuinely re-run. So its defaults are a real baseline to A/B
  against, which is the whole claim the node makes.

- **The t2i cap stays at 24000, as a declared override rather than a drift**
  (the owner: deliberately raised to avoid truncation). `profiles.py` now
  separates `REFERENCE`, which must mirror upstream exactly, from `OVERRIDES`,
  which carries the house value together with the reason it exists.
  `tests/test_profiles_match_upstream.py` fails on any difference that is not
  declared, and on any declared override that no longer differs from the
  reference. The earlier entry below recorded this as a bug; it was a bug in
  `REFERENCE`, and the value the harness sends was the owner's choice all along.

- **`profiles.py` had drifted from the reference it claims to mirror, and
  nothing would have caught it.** Its t2i cap read 24000 against upstream's
  16256 -- and against this repo's own
  [`../quantization-strategy.md`](../quantization-strategy.md) section 17,
  which had it right. Upstream also sets `min_p`, which we neither carried nor
  sent. Both fixed, and `tests/test_profiles_match_upstream.py` now reads
  `pe_core.py` with `ast` and compares every value; it fails on the exact drift
  when reintroduced. [`stages.md`](stages.md) recorded this stage's guard as
  **nothing**, which is how the gap was found rather than discovered by a bad
  render.

- **Two nodes built: the expander as one node, and the encoder opened up**
  (the owner, choosing both shapes). `QwenImage21PEExpand` folds system-prompt
  resolution, the heylook request, parsing and grading into a single
  chat-shaped node, because the four-piece version put four nodes in front of
  every sampler; the pure nodes stay registered for anyone who wants the
  pieces. `QwenImage21EncodeStructured` exposes the encoder's system turn,
  `keep_vision` and the canvas choice, **with defaults that reproduce the stock
  node** — it is a research surface for A/B, not a recommendation, and editing
  the system prompt is off-distribution against all five implementations
  ([`upstream.md`](upstream.md) section 1). Example graphs for both modes are
  in `example_workflows/`, generated by `scripts/build_example_workflows.py`.
  The encoder's chat string is assembled in `chat.py` rather than formatted
  from a template, because core's `template.format(text)` raises on a system
  prompt containing braces.

- **The predecessor's two-resolution idea does not port to 2.1, and that is a
  property of the architecture rather than a preference.**
  [`../bridge-encoder-findings.md`](../bridge-encoder-findings.md) Finding 3
  calls separate vision-tower and VAE resolution paths for one image the
  cleanest idea in that repo. In 2.1 each vision-language image slot stands for
  a fixed group of latent tokens, so the encoder's slot count and the VAE's
  latent grid are two views of one number: diffusers **raises** when they
  disagree, diffusers and sglang each say in place that one resize feeds both,
  and LightX2V suppresses the processor's own resize because a second one can
  change the slot count. Finding 3 stands as a description of the older stack
  and is withdrawn as something to carry forward.
  [`upstream.md`](upstream.md) section 2.

- **Hardcoding the system-turn drop index is settled against, five
  implementations to nothing.** All five derive it — four by tokenizing the
  system message, ComfyUI core by scanning the token stream for the second
  `<|im_start|>` and adjusting for images expanded ahead of it. This is the
  same hazard [`../bridge-encoder-findings.md`](../bridge-encoder-findings.md)
  Finding 1 records the predecessor paying for, and it is now settled by
  agreement rather than by our reading of one repo's mistake.
  [`upstream.md`](upstream.md) section 1.

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
