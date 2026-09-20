# Quantization strategy: Qwen-Image 2.1 prompt expanders and text encoder

> **PARKED 2026-09-20, by the owner.** The PE path stays on the verified-bf16
> mlx-vlm conversions served by heylook (section 25). No quantized PE artifact
> is being built.
>
> Why, in one line: the format ComfyUI can actually execute fast is produced by
> a data-free quantizer, so the calibration work this document spends most of
> its length on has nothing to feed, and no measurement has yet shown a quality
> problem worth solving.
>
> What would reopen it: wanting a PE artifact that runs in ComfyUI rather than
> over the LAN, or an MLX deliverable where AWQ is the right tool and
> calibration is how you use it. If that happens, start at section 20 (tool
> choice), then section 19 (the degenerate-corpus arithmetic, which is the
> non-obvious part and still stands).
>
> What is NOT parked: the harness in `src/`, the census in `scripts/`, and the
> capability gaps in section 23. Those are independent of any quant decision.

Last updated: 2026-09-20

Working notes. Every claim carries a pointer; re-derive rather than trust.
`<models>/` is wherever checkpoints live on your machine; `<qwen-image-2.1-repo>`
is a checkout of the upstream Qwen-Image-2.1 repo. `coderef/` entries are
reference checkouts, not dependencies -- port from them, never import.

Target hardware: RTX 4090 (sm_89, 24 GB), 128 GB RAM. ComfyUI at commit c194dd00.

## 1. What the models are

Both PE checkpoints are fine-tuned **Qwen3.5-VL 9B**, not Qwen3-VL:
`model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`, hybrid attention
(3x `linear_attention` GatedDeltaNet + 1x `full_attention`, `full_attention_interval: 4`),
32 layers, hidden 4096, intermediate 12288, vocab 248320, untied `lm_head`,
vision tower depth 27 / hidden 1152. 9.410 B params, 17.53 GiB bf16 each.

Local dirs (names are download-mangled):
- `Qwen-Image-2.1-PE-T21` = **t2i**  (system_prompt.txt ~10 KB)
- `Qwen-Image-2.1-PE-I21` = **edit / i2i** (system_prompt.txt ~18 KB)
- `Qwen-Image-2.1-PE-I2I` was an empty stray dir; removed.

Weight-set differs between the two (distinct shard hashes); index maps identical.
Shards are complete: 4 shards sum to 18,819,721,168 B vs `total_size` 18,819,627,488 B;
the 93,680 B delta is the four safetensors headers.

## 2. Weight budget (per model, bf16)

| group | params | GiB | % |
|---|---|---|---|
| lang mlp (gate/up/down) | 4.832 B | 9.00 | 51.3 |
| lang linear_attn in_proj_qkv / in_proj_z / out_proj | 1.611 B | 3.00 | 17.1 |
| embed_tokens | 1.017 B | 1.89 | 10.8 |
| lm_head | 1.017 B | 1.89 | 10.8 |
| lang self_attn q/k/v/o | 0.470 B | 0.88 | 5.0 |
| visual tower | 0.456 B | 0.85 | 4.8 |
| linear_attn small (a/b/conv1d/A_log/dt_bias/norm) + norms | 0.007 B | 0.01 | 0.1 |
| **total** | **9.410 B** | **17.53** | |

Big-matmul set eligible for 4-bit = mlp + linear_attn big + self_attn = 6.913 B (73.4%).
All `in_features` (4096, 12288) divide cleanly by `group_size=16` and
`convrot_groupsize=256`; K/2 int8 packing lands on 2048 / 6144. No padding paths.

## 3. Headline: the work is already done upstream

`Comfy-Org/Qwen-Image-2.1` already ships both PE models converted to ComfyUI
single-file format and quantized:

| file | GiB |
|---|---|
| `text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` | 8.82 |
| `text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors` | 8.82 |
| `text_encoders/qwen3vl_8b_w4a8.safetensors` | 5.88 |
| `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 8.71 |
| `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | 6.76 |
| `diffusion_models/qwen_image_2.1_bf16.safetensors` | 13.25 |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | 0.63 |

Download beats building. Render-stage residency with the official quants:
diffusion 6.76 + TE w4a8 5.88 + VAE 0.63 = 13.27 GiB, leaving ~10 GiB for
activations.

**Residency is an open question, but NOT because of `d39cdfdb`.**
CORRECTED 2026-09-20 (h3guy, from the call chain): that commit touches three
lines in `model_management.py::text_encoder_device` and only removes a CPU
fallback for the encode. It changes where the encoder computes, not whether it
stays resident. Residency is decided elsewhere and the commit did not touch it:
`ComfyUI/comfy/sd.py:255` constructs a CLIP on the **offload** device whenever
`aimdo_enabled` (`model_management.py:1232`); `ComfyUI/comfy/sd.py:299-300` only
force-loads when construction device == load device, which differs under aimdo;
and `CLIP.generate` calls `load_models_gpu(..., memory_required=...)` without
`force_full_load`. The patcher is a `CoreModelPatcher` unless
`te_disable_dynamic`, so `free_memory(..., for_dynamic=True)` can shrink it
rather than unload it whole. Dynamic vram is on by default
(`cli_args.py:318-321`).

So whether PE, the encoder and the DiT are ever co-resident is a question about
eviction and graph execution order. The observable is the server's own
`Unloading <class>` debug lines (`model_management.py:920`) on a real run with
recorded cache state. If residency is the case for w4a8, make it from those
lines. (Also: `ComfyUI/comfy/pinned_memory.py` is host-RAM pinning for transfer
bandwidth, not VRAM residency -- the words collide.)

## 4. Format choice on Ada (sm_89)

From `ComfyUI/comfy/ops.py::get_disabled_quant_formats` and `ComfyUI/comfy/model_management.py`:

- `nvfp4`, `mxfp8` require `props.major >= 10` (Blackwell). On a 4090 they are
  **disabled and emulated** -> slower than bf16. Avoid.
- `float8_e4m3fn` / `float8_e5m2`: native (sm 8.9 passes). ~9 GiB.
- `int8_tensorwise` (+ convrot): native. What Comfy-Org shipped for PE.
- `asym_w4a8_int8`: native int8 path, plus the batch-1 **gemv** kernel added in
  ComfyUI `9a77c1db` ("Qwen3/3.5/3.8 cudagraphs and w4a8 gemv support") --
  aimed exactly at autoregressive decode. ~6 GiB.

`CLIP.generate` (`ComfyUI/comfy/sd.py:483`) wraps the call in `comfy.ops.use_quantized_matmul`,
so the quantized kernels are live during PE generation -- not just a storage win.

## 5. What the official quant actually does (verified by range-reading the header)

`qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` has **no**
`__metadata__`; it uses the per-layer `.comfy_quant` uint8 tensor mechanism.
Per quantized linear: `weight` (I8), `weight_scale` (F32, per-channel `[N,1]`),
`comfy_quant` (U8 JSON).

Payloads:
- language model: `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`
- vision tower:   `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 64}`
- embed_tokens / lm_head: same as language model

**Everything is int8**, including `embed_tokens`, `lm_head`, and the whole
vision tower (108 quantized ViT linears; only norms and biases stay BF16):

| group | GiB in official file | dtypes |
|---|---|---|
| lang mlp | 4.50 | I8 96, F32 96, U8 96 |
| lang linear_attn | 1.51 | I8 72, F32 72, U8 72, BF16 144 |
| lm_head | 0.95 | I8 1, F32 1, U8 1 |
| embed_tokens | 0.95 | I8 1, F32 1, U8 1 |
| visual | 0.47 | I8 108, F32 108, U8 108, BF16 225 |
| lang self_attn | 0.44 | I8 32, F32 32, U8 32, BF16 16 |
| norms/misc | 0.00 | BF16 65 |
| **total** | **8.82** | |

Detector keys confirmed present: `model.language_model.layers.0.linear_attn.A_log`
and `input_layernorm.weight` shape `[4096]` -> loads as `qwen35_9b`.

## 6. The one gap worth building

Comfy-Org shipped w4a8 for the qwen3vl_8b *text encoder* but only int8_convrot
for the PE models. A local build wins on two axes at once, which is the actual
argument for doing it:

1. **Smaller.** ~6.4 GiB vs 8.82.
2. **Vision tower precision is an open experiment, not a known win.**
   CORRECTED 2026-09-20 (h3guy). The H3 "100% BF16 ViT" language describes a
   RECIPE, not a result: `awq_quantization_suite/config/recipe.yaml` ignores
   `re:.*visual.*`, so no arm ever quantized the tower and no A/B exists.
   RE-CORRECTED 2026-09-20 (h3guy, second pass, from the safetensors headers):
   **all four arms in that holdout keep the vision tower BF16** -- int8_convrot
   and nvfp4_awq carry 350/351 `comfy_quant` tensors and every one is a
   language-decoder linear. So that record compares decoder treatment and load
   path and says NOTHING about tower precision. If citing
   `bench/results/2026-08-25_four_encoders_holdout_layer50.json`, state in the
   citation that all four arms hold the tower bf16 -- the file itself records
   per-arm relative L2 and nothing about what each arm quantizes, which is what
   misled a careful reader once already.

   **The reframed picture, which is stronger:** count every encoder artifact
   anyone has built -- two W4A16 AWQ candidates, two community quants, and
   Comfy-Org's `qwen3vl_8b_w4a8`. Five artifacts, three independent parties,
   and every one leaves the conditioning encoder's vision tower bf16. Both PE
   checkpoints quantize theirs. That is not one org being inconsistent; it is a
   convergent split between *conditioning* encoders and *generation* models that
   three parties landed on independently.

   RE-CORRECTED again 2026-09-20 (owner, via h3guy): **the owner's position is
   that the vision tower needs to be bf16**, for reasons not enumerated here.
   Treat that as a strong prior from the person who has looked at far more
   output from these encoders than anyone else in this thread. I had flipped the
   burden of proof onto bf16 on the strength of Comfy-Org's PE files; flip it
   back. On an encoder, every artifact anyone ships agrees with the owner.

   So: **bf16 tower is the default arm**; quantized-tower is the alternate arm
   for the experiment. If a quantized-tower arm is built it must mirror
   Comfy-Org's graded recipe (section 15), not a uniform group size, or it is a
   strawman. The experiment still matters -- see "Why the encoder tower question
   cannot be settled by reading checkpoints" in section 15 -- and if it runs, its
   result outranks both the census and the prior. Hold load path constant, which
   is what the H3 holdout could not do and core's native path gives us both ways.

Mixed plan (per-layer mixing is the native design -- the reference
`minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` quantizes 200 of 474 weights):

| group | format | GiB |
|---|---|---|
| mlp.{gate,up,down}_proj, linear_attn.{in_proj_qkv,in_proj_z,out_proj}, self_attn.{q,k,v,o}_proj | `asym_w4a8_int8` g16, convrot 256 | ~3.62 |
| embed_tokens, lm_head | `float8_e4m3fn` | ~1.90 |
| visual tower | bf16 (or fp8 at ~0.43) | 0.85 |
| norms, conv1d, A_log, dt_bias, in_proj_a/b | bf16 | 0.01 |
| **total** | | **~6.38** |

Producer: `comfy_kitchen.tensor.AsymW4A8Int8Layout.quantize(tensor, group_size=16,
convrot_groupsize=256, symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True)`.
Run under the ComfyUI venv's python -- comfy_kitchen 0.2.35 exists only there
(`a local comfy-kitchen checkout` is 0.2.12 and has no w4a8 layout).

**Why the kernel win lands here and nowhere else.** `sd1_clip.py:114` builds
text-encoder ops with `full_precision_mm=True`, and the only `use_quantized_matmul`
call sites are `ComfyUI/comfy/sd.py:483` (`CLIP.generate`) and `ace15.py:165`. So for
qwen3vl_8b doing conditioning, w4a8 is **storage-only** -- weights dequantize and
the matmul runs bf16. PE generation is the single place in this stack where the
quantized kernels actually fire.

**Gate it on measurement.** comfy_kitchen's quantize is RTN + Lloyd-Max codebook
with no calibration. These models emit structured JSON behind a 10-18 KB system
prompt with thinking traces; the failure mode is `parse_ok: false`. Metric is
parse_ok rate against the int8_convrot baseline on a fixed prompt set.

## 7. On-disk format ComfyUI expects

Two equivalent mechanisms, both live:

1. safetensors `__metadata__["_quantization_metadata"]` =
   `{"layers": {"<module path>": {"format": "asym_w4a8_int8", "group_size": 16,
   "convrot": true, "convrot_groupsize": 256}, ...}}`.
   `ComfyUI/comfy/utils.py::convert_old_quants` expands each entry into a
   `<module path>.comfy_quant` uint8 JSON tensor at load.
2. Per-layer `.comfy_quant` uint8 tensors written directly.

`detect_layer_quantization` then returns `{"mixed_ops": True}` -> `mixed_precision_ops`.

Tensor names per quantized linear:
- fp8: `weight`, `weight_scale`
- asym_w4a8_int8: `weight` (int8, [N, K/2]), `weight_s_rel` (fp8 per-group),
  `weight_s_channel` (fp32 per-channel), `weight_codebook`

Key naming: keep HF names verbatim (`model.language_model.*`, `model.visual.*`,
`lm_head.*`). ComfyUI remaps at load (`ComfyUI/comfy/sd.py:1930`):
`model.language_model.` -> `model.`, `model.visual.` -> `visual.`, `lm_head.` -> `model.lm_head.`.
ComfyUI's module names match HF names 1:1 after that, so nothing else is needed.

Detection: `model.language_model.layers.0.linear_attn.A_log` present and
`...layers.0.input_layernorm.weight` shape[0] == 4096 -> `TEModel.QWEN35_9B`
-> `qwen35_9b` config (`ComfyUI/comfy/text_encoders/qwen35.py:98`), which matches this
checkpoint exactly. Tokenizer is bundled (`ComfyUI/comfy/text_encoders/qwen35_tokenizer`).
No MTP weights in these checkpoints, so speculative decoding is off.

## 8. llm-compressor does not apply here

ComfyUI core reads only the two mechanisms above. compressed-tensors output
(`weight_packed` / `weight_scale` / `weight_shape`, `quantization_config`) is not
in `QUANT_ALGOS` and needs a custom loader node.

That node was built once for the H3 encoder and the lane was **closed by the owner
on 2026-08-27** ("calibrating or quantising our own encoder (the llm-compressor
AWQ/GPTQ lane)", coderef/ComfyUI-h3-explorations `docs/roadmap.md` "Closed lanes");
the code was deleted 2026-09-13. Nothing about this model reopens it.

For the record, the lane is technically reopenable: llm-compressor does support
this architecture -- `Qwen3_5ForConditionalGeneration` maps to
`build_qwen3_5_dense_smoothquant_mappings` in
`src/llmcompressor/modifiers/transform/smoothquant/dynamic_mappings.py:133`.
comfy_kitchen 0.2.35 also carries `TensorCoreAWQW4A16Layout`, so the kernel
exists; only the core loader is missing. Owner's call.

## 9. The real risk is the prompt harness, not the quantization

- `Qwen35ImageTokenizer.llama_template` is
  `"<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"` -- **no system turn**,
  and the `TextGenerate` node has no system-prompt input. The PE system prompts
  must go into the prompt widget as a full chat string starting with `<|im_start|>`,
  which sets `skip_template=True`.
- With `skip_template=True` the `<think>\n</think>\n` suffix is **not** appended
  (it sits inside the `else`), leaving thinking on -- which matches the PE README's
  "thinking on by default". Using the default template with `thinking=False`
  force-closes the think block against how these were trained.
- For the edit model, place `<|vision_start|><|image_pad|><|vision_end|>` manually,
  in image order: the tokenizer scans for token 248056 and injects image data
  positionally, and the system prompt addresses images as `<image1>`, `<image2>`.
- Output is thinking trace + JSON. A parse step is needed to pull
  `positive_prompt` / `wh_ratio` / `ratio_follow`. Core has no such node, KJNodes'
  string nodes are join/constant/convert only, and shrug-prompter has cleanup and
  templating but no field pull. **ComfyUI-QwenImageWanBridge already solves this**
  -- see section 11.
- CLIPLoader `type` is not guarded for the qwen35 branch (`ComfyUI/comfy/sd.py:1929` has no
  clip_type check), so the default `stable_diffusion` reaches it; detection is by
  state dict.

**Gate:** one valid JSON out of t2i, and one out of i2i with an image, at the
official int8_convrot weights, before any A/B is interpretable.

## 10. hfutils

No change needed for the recommended path. If building bf16 or w4a8 locally,
`hfutils convert --to single` does the shard merge -- streams tensor bytes,
preserves each shard's insertion order, merges `__metadata__` last-write-wins
with a warning on conflict (`src/hfutils/formats/safetensors.py:34,103`).
The quantizer would be a separate script under the ComfyUI venv, not an hfutils
dependency (hfutils keeps torch behind the `[ml]` extra; comfy_kitchen is heavier still).

## 11. Port from ComfyUI-QwenImageWanBridge

That repo was built for an earlier Qwen-Image and targets exactly this problem
class: autoregressive LLM text encoders with custom system prompts, thinking
blocks and turn builders. Most of the harness gap in section 9 is already
solved there.

### Worth porting

| file | what it gives | adaptation needed |
|---|---|---|
| `nodes/llm_output_parser.py` | `LLMOutputParser`: strips markdown code fences, parses JSON/YAML, dot-notation field extraction, configurable key names, emits `parse_status` | map keys to `rewritten_prompt` / `wh_ratio` / `ratio_follow`; strip the leading `<think>...</think>` before parsing; add a `wh_ratio`/`ratio_follow` output pair |
| `nodes/z_image_encoder.py::format_conversation` | builds the `<\|im_start\|>` chat string from a messages dict with system/user/assistant + thinking | add a generation-prompt mode (see below); Z-Image's version renders *completed* assistant turns |
| `nodes/template_loader.py` + `nodes/templates/*.md` | markdown files with YAML frontmatter (`mode:`, `vision:`) + system prompt body, auto-discovered into a node dropdown | drop the two PE `system_prompt.txt` files in as templates |
| `nodes/qwen_multi_reference.py`, `nodes/qwen_image_batch.py` | multi-image ordering and aspect-preserving batching | relevant to the edit model's `<image1>`/`<image2>` ordering |

`parse_status` is directly the `parse_ok` metric section 6 wants for the A/B.

### Do NOT port

`nodes/hunyuan_video_prompt_expander.py` extracts weights from ComfyUI's CLIP and
bolts on an `lm_head` to generate. That was necessary before core had generation.
It is now obsolete: core ships `TextGenerate` + `CLIP.generate` for qwen35,
including the cudagraph and MTP paths. Porting it would re-implement core badly.

### What reading the bridge repo turned up (harness bugs waiting to happen)

Checked against the PE `chat_template.jinja` and the official runners
(`prompt_rewrite/run_transformers.py:63`, `run_vllm.py:138` -- both pass
`enable_thinking=True`):

1. **The generation prompt ends with an OPEN `<think>\n`.** The template's
   `add_generation_prompt` branch emits `<|im_start|>assistant\n` then
   `<think>\n` when thinking is on, and `<think>\n\n</think>\n\n` when off.
   ComfyUI's qwen35 tokenizer with `thinking=True` appends **nothing** after
   `<|im_start|>assistant\n`, so the model has to open the block itself against
   how it was trained. Its `thinking=False` suffix is `<think>\n</think>\n`,
   which also differs from the template's `<think>\n\n</think>\n\n`.
   Under `skip_template=True` the harness supplies all of it -- which is the
   argument for the ported builder rather than the node's own template.

2. **Images go first in the user turn, in order, then the text**
   (`pe_core.py::build_messages`). The system prompt addresses them as
   `<image1>`, `<image2>`; reordering silently re-points every reference.

3. **Sampling defaults do not match.** PE per-task values live in
   `pe_core.py::PROFILES`. ComfyUI's `TextGenerate` schema
   (`ComfyUI/comfy_extras/nodes_textgen.py`) defaults differ on `min_p` and
   `repetition_penalty`, and its `max_length` default is far below what a
   thinking trace plus JSON needs -- both PE `max_new_tokens` values fit under
   the node's ceiling, but neither is close to its default. Read both and set
   them explicitly; a silent truncation mid-thinking looks exactly like a
   quantization regression.

## 12. The qwen3vl_8b encoder: no custom node needed

Question raised: only the bf16 encoder was downloaded; is there value in wrapping
a custom encoder node to quantize only parts of it (as the h3 repo does)?

**Comfy-Org's `qwen3vl_8b_w4a8.safetensors` already is that selective build.**
Verified by range-reading its header:

| group | GiB | format |
|---|---|---|
| lang mlp | 2.85 | `asym_w4a8_int8`, group_size 16, convrot 256 |
| **visual (ViT)** | **1.07** | **BF16, all 351 tensors, untouched** |
| lang self_attn | 0.79 | `asym_w4a8_int8`, group_size 16, convrot 256 |
| lm_head | 0.58 | `int8_tensorwise` + convrot 256 |
| embed_tokens | 0.58 | `int8_tensorwise` + convrot 256 |
| norms/misc | 0.00 | BF16 |
| **total** | **5.88** | |

It keeps the vision tower entirely in bf16 -- the design the H3 AWQ recipe
adopted on reasoned grounds, though per section 6 that prior is NOT supported by
measurement and the later holdout went the other way. Contrast with the PE
`int8_convrot` files, which quantize their ViT (both t2i and i2i: 108 layers at
groupsize 64). Comfy-Org made opposite calls for the encoder and the PE models,
which is itself worth noting rather than assuming either is right.

The ViT is load-bearing here: `ComfyUI/comfy/text_encoders/qwen_image21.py` builds
`<image{n}><|vision_start|><|image_pad|><|vision_end|>` refs for edit workflows
and `QwenImage21TEModel.encode_token_weights` tracks image spans to swap vision
tokens for reference latents in the DiT. So an encoder with a quantized ViT
would degrade exactly the edit path.

Also note `QwenImage21Tokenizer.tokenize_with_weights` defaults `thinking=True`,
while the `TextGenerate` node defaults `thinking=False` -- opposite defaults for
the two Qwen paths in the same pipeline.

**Caveat on reading anyone's defaults as endorsements** (owner, via h3guy): the
owner primarily runs the **bf16** Qwen3-VL encoder themselves, even though
`h3_config.MODELS["clip"]` ships int8 on every graph. `ENCODER_INT8` won a
holdout against two admittedly-bad candidates, which is not the same as being
preferred. Do not infer a preference from a shipped default.

**Conclusion: download `qwen3vl_8b_w4a8.safetensors`, write no node.** Per
section 6 this is storage-only (conditioning never enters `use_quantized_matmul`),
but footprint is the binding constraint on a 24 GB card, and the bf16 file is
the largest single item in the stack.

## 13. Two rules for the build, from h3guy's review

### 13a. `full_precision_matrix_mult` silently turns kernels off

`use_quantized_matmul` (`ComfyUI/comfy/ops.py:1727`) only flips modules where
`can_use_quantized_matmul` is true, and that predicate (`ComfyUI/comfy/ops.py:1340`)
requires `not self._full_precision_mm_config`. That flag is read **from the
file**, not from the call: `ComfyUI/comfy/ops.py:1197` sets it from
`full_precision_matrix_mult` in each layer's `comfy_quant` JSON. A w4a8 build
that writes `full_precision_matrix_mult: true` is storage-only even inside
`CLIP.generate` -- no speedup, and a wrong conclusion about the format.

Verified on all three official files (2026-09-20) by **full scan** of every
`comfy_quant` blob. The blobs are clustered (span ~0.02 MB per file), so one
range request per file fetches all of them:

| file | blobs | distinct configs | `full_precision_matrix_mult` |
|---|---|---|---|
| `pe_t2i.int8_convrot` | 310 | 202 @ g256, 81 @ g64, 27 @ g16 | 0 layers |
| `pe_i2i.int8_convrot` | 310 | 202 @ g256, 81 @ g64, 27 @ g16 | 0 layers |
| `qwen3vl_8b_w4a8` | 254 | 252 `asym_w4a8_int8` g16/convrot256, 2 `int8_tensorwise` | 0 layers |

All three fire their kernels, subject to `get_disabled_quant_formats(device)`
(sm_89: int8 and w4a8 allowed, nvfp4/mxfp8 disabled). **Assert this key is
absent in anything we produce, as a build-time check, not a hope.**

**Method note -- do NOT audit this by payload length.** An earlier pass here
grouped `comfy_quant` tensors by their uint8 length and read one blob per length
class. h3guy flagged it as unsound in principle and it was wrong in practice on
these exact files: `convrot_groupsize: 64` and `convrot_groupsize: 16` are both
71 bytes, so one length class held two distinct configs and the spot-read
reported all 108 as g64 when it is 81 + 27. The general collision h3guy
constructed is worse, because a shorter format name buys back exactly the bytes
the flag costs:

    {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}   -> 72
    {"format": "nvfp4", "convrot": true, "full_precision_matrix_mult": true}   -> 72
    {"format": "mxfp8", "convrot": true, "full_precision_matrix_mult": true}   -> 72

So on a mixed-format file -- exactly where the audit matters -- length grouping
can pick an unflagged representative and report all clear. Read every blob; it
is ~22 KB. Keep length grouping only as a homogeneity diagnostic ("N layers,
K configs"), never as the flag assertion.

### 13ab. The residency case is architectural, not a VRAM reading

The bridge repo's `nodes/hunyuan_video_prompt_expander.py` states the play I was
reaching for: "Extracts weights from ComfyUI's CLIP and adds only the lm_head
for generation... reuses ~14GB of already-loaded weights, only loading ~50MB
extra for lm_head" (its numbers). Share the base with the conditioning encoder,
add a head, pay almost nothing for expansion.

**That is structurally impossible here, and this is the residency argument.**
Verified from both checkpoints' headers:

| | conditioning encoder | PE models |
|---|---|---|
| family | Qwen3-VL 8B (`deepstack` present) | Qwen3.5-VL 9B (`linear_attn` present) |
| vocab | 151936 | 248320 |
| `linear_attn` | absent | present |
| `deepstack` | present | absent |

Different architecture families, different vocabularies, different tokenizers.
No head-sharing is possible; a second full model is paid for no matter what.
That is a cleaner statement of the residency case than anything derivable from
`d39cdfdb` or from a single run, and it can be made from architecture alone.

### 13b. Decide the acceptance measurement and baseline arm BEFORE building

The H3 program built two W4A16 candidates and both lost to a community quant on
their own holdout. The only reason that decision was clean is that the held-out
row set, the BF16 reference arm and the aggregation script all existed before
the candidates did.

CORRECTED 2026-09-20 (owner, via h3guy): **do not read that result as evidence
that building your own quant is a bad bet.** Those artifacts were done poorly.
The owner's words, near-verbatim: we control the calibration, the group size and
everything else; we just did it badly that time, and then priorities shifted.
The lane closed because a weak first attempt coincided with attention moving
elsewhere, not because the approach was shown to be unpromising. Two artifacts
from one program with one calibration corpus is a fact about that program.

What survives, and was the owner's point independently: decide the acceptance
measurement and the baseline arm before you build the candidate. That is what
made the decision clean even though the candidate was bad.

For a generation model that means: a fixed prompt set, a grading method, and the
baseline arm (official `int8_convrot`) all chosen and running **first**. Not a
VRAM figure after the fact. This is convenient here -- the PE harness (section
11) IS the measurement instrument, and it is needed regardless of the quant
decision, so it is the correct first build either way.

**Pin the decode, or there is no measurement.** You cannot A/B a quant on
sampled text. The bridge repo's `hunyuan_video_prompt_expander.py::GENERATION_CONFIG`
does this with temperature 0.01 / top_k 1 / top_p 0.001 / repetition_penalty 1.0
/ do_sample True, commented as matching the official HunyuanVideo rewriter.
On ComfyUI's `TextGenerate` there is a cleaner route: `sampling_mode: "off"`
sets `do_sample=False` (`ComfyUI/comfy_extras/nodes_textgen.py:58`), i.e. true greedy.
In fairness to the node being substituted away from, its emulation is not
behaviourally wrong -- `hunyuan_video_prompt_expander.py:365-367` sets `top_k=1`
below temperature 0.1, and top_k=1 is argmax, so it is already deterministic.
What the epsilon costs is numerical: temperature 0.01 scales logits by 100x
before a softmax that top_k=1 has made irrelevant, a live overflow hazard in
half precision that buys nothing. Prefer real greedy to remove a pointless
hazard, not because the original drifts.

With greedy decode the comparison becomes a diff, and the natural metrics are
first-divergent-token index plus final `parse_ok`. Note this is a MEASUREMENT
configuration only -- production uses the sampled `pe_core.py::PROFILES` values,
so the harness must expose both and record which was used.

### 13c. Where the compressed-tensors recipe lives, if ever needed

The deleted `MiniMaxH3AWQEncoderLoader` has a surviving standalone copy at
coderef/ComfyUI-h3-explorations `docs/research/awq_quantization_suite/workflows/comfyui_minimax_h3_awq_loader.py`.
It rewrites compressed-tensors into per-layer `.comfy_quant` tensors at load
(writes ~line 225, reads back at 355 and 595-607). History, not a supported
path, but it is the recipe if compressed-tensors ever needs bridging.

## 14. Borrowing map across the owner's packs

House rules (h3guy, and they are the owner's conventions):
- These are all the owner's own repos. Refer to them as such, not as third-party
  prior art.
- **Port, never reuse.** Re-implement against this model rather than importing
  across packs: model semantics differ even where the ComfyUI abstraction is shared.
- Read each repo's own `CLAUDE.md` before borrowing. They differ per repo and
  encode conventions not inferable from the code.

| repo | take | note |
|---|---|---|
| ComfyUI-QwenImageWanBridge | `llm_output_parser.py` shape, `format_conversation`, `template_loader` + `templates/*.md`, `GENERATION_CONFIG` decode pinning, `inspect_qwen_models.py`, `qwen_state_dict_mapping.py`, `qwen_token_debugger.py`, `qwen_token_analyzer_standalone.py`, `MODEL_FORMAT_EXPLAINED.md`, `nodes/docs/attention_based_prompt_expansion.md` | same model family, one generation earlier; legacy V1 node API so shells get rewritten regardless |
| ComfyUI-H3-Quant | **the template for shipping a quantization claim**: one finding, one node acting on it, every claim linked to its measurement record, README bounding its own scope in the first paragraph, and `check_h3_loud_blocks.py` (41 lines, no GPU, no ComfyUI) letting a reader disconfirm on their own checkpoint. Defines what "error" means before quoting one. | 279 lines total. If we ship any claim about the PE quants, ship the equivalent disconfirmation script |
| krea2-explorations | research-repo structure: `src/` (importable nodes) vs `scripts/` (probes run directly); measure-first, adversarial README leading with falsifications; `CLAUDE.md` "Cross-repo contracts" rule -- **ship the data contract, not the node** (single-sourced tracked JSON generated from the Python, with a sync-guard test) | relevant if any recommendation must be consumed by a non-ComfyUI caller |
| shrug-prompter | small-pack discipline: ten nodes, ~1500 lines, real `tests/` + `conftest.py` + `scripts/smoke.py`, and an explicit README statement that there is **no multi-provider abstraction** because it is tuned to one server's API surface | the owner wants BOTH backends here. Resolution: two concrete backends, not a provider abstraction. Do not build the generic quantized-LLM-loader layer before it is earned |

## 15. Comfy-Org's PE recipe, and what governs a quantizer's choices

Full config census of `pe_t2i.int8_convrot` (identical in `pe_i2i`), every
`comfy_quant` blob mapped to its module role:

| convrot_groupsize | layers | roles |
|---|---|---|
| 256 | 202 | the ENTIRE language model: `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (24 ea), `mlp.{gate,up,down}_proj` (32 ea), `self_attn.{q,k,v,o}_proj` (8 ea), plus `lm_head` and `embed_tokens` |
| 64 | 81 | vision tower, three roles: `visual.blocks.N.{attn.proj, attn.qkv, mlp.linear_fc1}` (27 each) |
| 16 | 27 | vision tower, ONE role: `visual.blocks.N.mlp.linear_fc2`, all 27 blocks |

**WITHDRAWN 2026-09-20. The grading is arithmetic, not judgement.**

An earlier draft read this table as "the tower is treated as MORE sensitive than
the language model" and called it the finding that drives the build. That was
wrong, and it was wrong in exactly the way flagged two sections earlier: a
"role" that is really a shape.

ConvRot's rotation is a block-diagonal **regular Hadamard of order 4^k** applied
along the input-feature axis (`comfy_kitchen/tensor/int8_utils.py:11-74`; the
method is arXiv:2512.03673, named in `tensor/convrot_w4a4.py:3-9`). So
`convrot_groupsize` must be a power of 4 that **divides `in_features`**. Run the
arithmetic on this model:

| module | in_features | factorisation | 256? | 64? | observed |
|---|---|---|---|---|---|
| language model (all) | 4096 / 12288 | 2^12 / 2^12*3 | yes | -- | **256** |
| ViT attn.qkv / attn.proj / mlp.linear_fc1 | 1152 | 2^7 * 3^2 | no (4.5) | yes (18) | **64** |
| ViT mlp.linear_fc2 | 4304 | 2^4 * 269 | no | no (67.25) | **16** |

`linear_fc2` gets 16 because **16 is the largest legal value for 4304**, not
because anyone judged it sensitive. The ViT's 64 is forced the same way.
Verified here: 1152 and 4304 admit nothing larger.

What is NOT forced is the language model's 256. `AsymW4A8Int8Layout.quantize`
accepts 1024 and 4096 on a 4096-wide input (tested directly), and 4096/12288
both admit 1024. So 256 there is a choice, and it matches the ConvRot paper's
own ablated default across {16, 64, 256, 1024}.

**Net: there is no per-module sensitivity judgement anywhere in this file.** One
global default of 256, plus the largest feasible fallback wherever 256 does not
divide. The three-class census output was real; the story read into it was not.

**Consequence for the build, inverted.** If the paper's claim holds that larger
N0 suppresses outliers better, the vision tower is getting the model's *weakest*
rotation purely because of its dimensions -- a forced compromise, not
protection. That makes the bf16-tower prior stronger, not weaker, and it removes
the "mirror the graded recipe" instruction: there is no graded recipe to mirror.
A quantized-tower arm should use the largest legal group per layer and be
reported as dimension-constrained.

**Unverified:** the research subagent reported, from compiled-kernel strings I
could not reproduce, that the int8 *fused* path accepts only 256 and the int4
path only {16, 64, 256}. `quantize()` accepting a value does not mean the fused
gemm/gemv accepts it. If true, no ViT layer here qualifies for the fused int8
path at all, which would be a second reason -- independent of quality -- that
shipped conditioning encoders leave their towers in bf16. Worth settling before
relying on either number.

### A quantizer makes three decisions, and they have different evidence

Separating them is what resolved a long back-and-forth in which three tidy
cross-model stories were built and dissolved. Evidence below is h3guy's, from a
census across every quantized checkpoint on this machine, printing `in_features`
and `out_features` beside the role column so confounds are visible in the output.

**1. Inclusion (quantize this module at all?) -- ROLE, established, matched-shape
control, unanimous.** Every H3 checkpoint quantizes `blocks.N.*` and leaves
`token_refiner.blocks.N.*` unquantized at four *identical* weight shapes:

    in=7168  out=5376    blocks.N.attn.out_proj   quantized | token_refiner...  not
    in=5376  out=21504   blocks.N.attn.qkv_proj   quantized | token_refiner...  not
    in=5376  out=28672   blocks.N.mlp.fc1         quantized | token_refiner...  not
    in=14336 out=5376    blocks.N.mlp.fc2         quantized | token_refiner...  not

Holds across int8_convrot, fp8_scaled and w4a8_mixed, pruned and unpruned, the
hybrids, the PDD bake and the fastvideo VSA build, with no exception; the music3
DiT and text encoder repeat it at 2048x2048 and 4096x4096, krea2 at 6144x6144.
Shape held constant, treatment varies with position. **Shape-only inclusion
rules are dead.**

**2. The `full_precision_matrix_mult` flag -- ROLE, established, matched-shape
control.** In `krea2_raw_fp8_scaled`, at in=6144 out=6144 within the same blocks,
`attn.wq` is plain while `attn.gate` and `attn.wo` are flagged; again at
in=2560 out=2560, `attn.wk/wq/wv` plain against `attn.gate/wo` flagged. The
flagged set across that file is exactly `attn.gate`, `attn.wo`, `mlp.down` --
the output projections, each consuming an activation-multiplied or
attention-weighted signal.

**3. Group size -- STILL CONFOUNDED.** adaLN's g64 in H3 and our `linear_fc2`'s
g16 are both uniquely shape-identified (in_features 2688 and 4304 respectively,
each belonging to that module and nothing else). No control exists for this
decision in any file examined. Treat group-size-by-role as a prior, not a
finding.

Also established, as a negative, on both sides: convrot group size does **not**
track `in_features`. In the PE file `mlp.down_proj` (12288) and
`gate_proj`/`up_proj` (4096) all sit at g256; in H3's `fl2va_int8_convrot`,
in_features 5376, 7168 and 14336 all sit at g256. Any "finer groups for bigger
layers" heuristic is refuted in both families.

### The MLP down-projection: a finding, with a stated limit

H3's fp8 builds flag exactly `blocks.N.mlp.fc2` -- H3's MLP down-projection.
krea2 flags `mlp.down`, the same role in a different family, and krea2's
attention pairs supply the matched-shape control that H3's fc2 lacks. So the
down-projection is singled out by two quantizers in two model families, resting
on a control rather than on uniqueness.

The limit, stated because the sets are not identical: H3 flags only the MLP
down-projection; krea2 also flags `attn.gate` and `attn.wo`. **The intersection
is the MLP down-projection**, and that is what has two families behind it. The
mechanism differs from our g16 observation (full-precision flag vs finer group),
so this supports the *prior* about post-gate down-projections without making our
group-size observation a second instance of it.

### Why the encoder tower question cannot be settled by reading checkpoints

Our own PE file offers no matched-shape control: every 2-D linear of meaningful
size is quantized, and the unquantized `linear_attn.in_proj_a`/`in_proj_b` are
uniquely shaped at [32, 4096].

That is not a quirk of our file. h3guy censused every quantized checkpoint on
this machine for the presence of a matched-shape control, and **exactly two
files have none -- and both are text encoders**
(`qwen3vl_32b_minimax_h3_int8_convrot`, `qwen3vl_32b_minimax_h3_nvfp4_awq`).
Every DiT, the music3 pair and krea2 all have at least one shape whose treatment
varies.

So the question we have spent the most effort on -- why is the vision tower
excluded in a conditioning encoder -- is being asked of the only file class
structurally incapable of answering it. In those headers every treatment class
is uniquely shape-identified, so nothing distinguishes "the tower is excluded
because of what it is" from "because of what shape it is."

**Consequences:**

1. The five-artifacts-three-parties convergence (section 12) remains real as
   *observed practice*. The inclusion control (decision 1 above) establishes
   that inclusion is role-governed *in general*. Neither establishes why the
   tower specifically is out in an encoder. That stays open.
2. It stays open for a structural reason, not for want of effort: a census
   across every encoder anyone ships would still come back silent.
3. **This elevates the tower arm from optional to necessary.** It is not a
   nice-to-have confirmation of a convention -- it is the only instrument that
   reaches the question. Running it on a 9B through core's native path, with
   load path held constant both ways, produces evidence that does not exist
   anywhere today.

General lesson worth keeping: a single-file read needs no cross-model story,
which makes it robust for what it *can* answer (the graded recipe above), but a
single file can be silent on the question being asked in a way a census across
files is not. Check whether the file class can answer the question before
treating its silence as evidence.

### Method notes for the census

- Print `in_features`/`out_features` beside the role column. Three cross-model
  stories died here because a "role" turned out to be uniquely shape-identified,
  and every one was checkable from a header already open. The output should hand
  the reader the confound rather than wait to be asked.
- Run it cheap and broad rather than carefully. Each dissolution came from
  adding one more file, not from harder thinking about files already in hand.
- Role normalisation: `\.layers\.\d+\.` -> `.layers.N.`, same for `.blocks.N.`.


## 16. Calibration data is the priority surface, not the recipe

Owner's direct guidance (via h3guy, 2026-09-20), and it redirects the build:

> Calibration data is probably a large untapped area on its own.

It is the input with the most control and the least prior art, and it is where
the previous H3 effort was weakest. That effort spent its time on the recipe,
and **the recipe is not where it lost**. A strong recipe with a weak corpus is
the known failure mode here.

For these models specifically that means asking what calibration data actually
looks like for:

- **The PE models:** a prompt expander that emits a thinking trace and then
  structured JSON, behind a 10-18 KB system prompt, at the sampling settings in
  `pe_core.py::PROFILES`. Calibration rows that do not carry the system prompt
  and the thinking format are calibrating a different distribution from the one
  the model runs in.
- **The conditioning encoder:** a model that sees reference images in edit
  workflows, whose vision tower the owner wants left bf16 anyway -- so the
  corpus question is about the language path conditioned on image tokens.

The llm-compressor library is the tool to work from, and researching current
practice for these specific architectures is worth dedicated effort rather than
being folded into recipe work.

## 17. heylook backend: live, with three sampling-parity gaps

Server reachable on the LAN; 41 endpoints under `/v1`. Per the heylook-provider
skill, the Anthropic-conformant `/v1/messages` is the inference route (the
OpenAI-compatible `/v1/chat/completions` was removed in 1.79.66). Both PE models
are already served as MLX:

- `Qwen-Image-2.1-PE-T21-mlx` — caps `chat, vision, thinking`, ctx 262144
- `Qwen-Image-2.1-PE-I21-mlx` — same

Both report `thinking_default: true` / `enable_thinking: true`, matching the
checkpoints' trained format.

**Parity gaps against `pe_core.py::PROFILES`** — the server's `sampler_defaults`
for both PE models do not match the reference implementation. Left unset per
request, a heylook row and a ComfyUI row are not the same experiment:

| knob | heylook default | `pe_core.py` reference |
|---|---|---|
| `top_k` | 0 | 20 |
| `presence_penalty` | 0.0 | 1.5 for t2i, 0.0 for edit |
| `max_tokens` | 4096 | 16256 (t2i), 24000 (edit) |

The `max_tokens` gap is the dangerous one: 4096 will truncate a long thinking
trace mid-stream, and a truncated trace produces unparseable JSON that looks
exactly like a quantization regression. Both backends must send explicit
sampling params on every request rather than relying on either side's defaults,
and the harness must record what it sent.

Note MLX quantization is a THIRD artifact class, unrelated to comfy_kitchen
formats and to compressed-tensors. The heylook path consumes MLX-quantized
weights and DOES read the checkpoint's own `chat_template.jinja` and
`tokenizer.json` — unlike the ComfyUI path, which uses ComfyUI's bundled
`qwen35_tokenizer` and ignores the checkpoint's template entirely. So template
control lives in different places per backend, and the harness is what gives
parity across them.

## 18. Diffusers-pipeline metadata (config only, no weights)

`<models>/Qwen-Image-2.1-diffusers/` is the upstream diffusers pipeline metadata:
`QwenImage21Pipeline` with `processor` (Qwen3VLProcessor), `scheduler`
(FlowMatchEulerDiscreteScheduler), `text_encoder`
(`Qwen3VLForConditionalGeneration`), `transformer`
(`QwenImage21Transformer2DModel`), `vae` (`AutoencoderKLQwenImage21`).

Confirms the conditioning encoder: `qwen3_vl`, 36 layers, hidden 4096, vocab
151936, `deepstack_visual_indexes` [8, 16, 24], vision depth 27 / hidden 1152 /
out_hidden_size 4096. Distinct from the PE models' `qwen3_5` in every one of
those except vision depth.

Transformer: 32 layers, 32 heads, attention_head_dim 128, context_in_dim 4096,
in/out channels 64, mlp_ratio 3, `causal_condition: true`.

`processor/chat_template.jinja` here belongs to the ENCODER, not to the PE
models — do not cross them.

## 19. The ordinary calibration row is degenerate for these checkpoints

Research subagent finding, independently re-verified here 2026-09-20.

**The arithmetic.** Measured with each checkpoint's own `tokenizer.json`
(`add_special_tokens=False`):

| | tokens |
|---|---|
| T21 `system_prompt.txt` | 2427 |
| I21 `system_prompt.txt` | 4510 |
| t2i briefs (`prompt_rewrite/data/t2i_example.jsonl`) | median 8, range 7-11 |
| edit briefs (`edit_example.jsonl`) | median 15, range 4-53 |

So a t2i row built the conventional way (system + brief + `add_generation_prompt`)
is ~99% a byte-identical prefix shared with every other row. AWQ's smoothing
statistic is `sum(|x|, dim=0) / count` pooled over tokens
(`modifiers/transform/awq/base.py:415-447`), so a 512-row corpus contributes on
the order of 1.25M system-prompt token positions against ~5.6k brief positions.
**The pooled channel mean is the system prompt's channel mean.** A 512-row
corpus carries roughly the signal of one example, and every ordinary sanity
check passes: 512 distinct rows, distinct briefs, no duplicates.

**The fix, as two separate decisions.** Full rows (system + images + brief +
thinking trace + JSON answer), masked to the assistant turn with
`use_loss_mask=True`. Conflating these is the mistake:

- The system prompt **must be in the row**, because it conditions every
  generation-position activation.
- The system prompt **must not be in the statistic**, by the arithmetic above.
- The mask filters the statistic, not the forward pass.

**Hard constraint on algorithm choice, verified here.** `loss_mask` exists only
in AWQ, the pipelines, the collator, the args and state -- grep across
`src/llmcompressor/` returns hits in `datasets/utils.py:306`,
`modifiers/transform/awq/base.py`, `pipelines/basic/pipeline.py:49-67`,
`pipelines/sequential/pipeline.py:123`, and nowhere in GPTQ. Reading
`modifiers/quantization/gptq/helpers.py::accumulate_hessian`, it reshapes the
input and does `hessian += inp.matmul(inp.t())` over every token with no mask
parameter at all. **If we want the assistant-turn mask, the algorithm must be
AWQ. GPTQ cannot express it.**

**Two implementation traps the subagent verified empirically:**

1. llm-compressor's own `examples/awq/qwen3_next_thinking_example.py:60-64`
   warns that `add_generation_prompt=True` over-counts `prompt_len` and works
   around it. **That caveat is wrong for these checkpoints** -- verified at token
   level through the real `Qwen3VLProcessor` with an image (gen 790 tokens, full
   815, `torch.equal(full[:790], gen)` True). Copying the workaround would
   under-count by the `<think>\n` primer.
2. **Build the mask from processor output, not the text render.** A 1024x768
   image is one `<|image_pad|>` in text but 768 tokens in `input_ids`. A
   text-derived mask would be ~22 entries against an 815-token sequence, and
   `DataCollatorWithTruncation` truncates every key to the minimum
   (`datasets/utils.py:306-314`), silently chopping `input_ids` to 22. The run
   completes and the corpus is fragments.

**Sourcing: the owner's existing collections do not work as PE briefs.** Both
are output-side artifacts -- `coderef/shrug-prompter/templates/styles/` bodies are
system prompts (its README says so), and `prompt_bank/` is video prompts with
soundscape fields, wrong modality and register. Useful as coverage-axis
vocabulary only. The authoritative shape is `prompt_rewrite/data/*_example.jsonl`:
very short, mixed Chinese/English, typos preserved, with a `task_type` field.
Plan: hand-write a few hundred stratified briefs against axes the system prompts
themselves name, generate the thinking+JSON halves with the fp16 model at
`pe_core.py::PROFILES` sampling, filter on `parse_ok` plus mechanical contract
checks. This is self-distillation and `parse_ok` is syntax-only -- both flagged.

**Near-duplicate trap, re-derived from the arithmetic.** Dedup on the brief
text, never the rendered row: every row shares thousands of identical tokens, so
any whole-row similarity metric reports ~99% for every pair and is saturated. A
whole-row check would pass a corpus of 512 copies of one brief. Holdout splits
go by provenance key and whole strata, image-grouped for edit.

**Truncation is the headline failure mode.** Copying `MAX_SEQUENCE_LENGTH = 2048`
from the VLM examples cuts below *both* system prompts, the mask goes all-zero,
and nothing errors.

**Falsifier, to run BEFORE the corpus effort.** Run AWQ twice on two different
corpora and compare per-layer scale vectors by cosine similarity. No eval
harness needed. If the scales come back near-identical, corpus design is not the
lever and the effort is misplaced. This is the cheapest possible test of the
whole premise and it belongs first.

**Genuine gap confirmed (report section 4.4):** no published work on long fixed
system prompts in calibration rows, nor on calibration sequence length for
hybrid linear-attention models. The owner's read that this area is untapped is
correct.

Full report: `docs/calibration-research.md`.

## 20. Verdict: tool choice splits by destination, not by model

Recipe subagent report, with the loader question re-verified here.

**The transformers risk I flagged is not real.** transformers 5.17.0 parses
`model_type: qwen3_5` and instantiates both PE checkpoints with exact key match
against their indexes, zero missing / zero unexpected. The 5.4.0-saved
checkpoints load clean on the version `setup.py:122` demands.

**Verdict:**
- **ComfyUI target -> use comfy_kitchen, not llm-compressor.** Core has no
  int4-weight/16-bit-activation format at all; `comfy_kitchen`'s
  `AsymW4A8Int8Layout` produces a format core executes, reports fast matmul
  available on this 4090, and compresses better than W4A16. llm-compressor's
  W4A16 gives less compression, no kernel, and a custom node to maintain. This
  is a statement about format availability, not about quantization difficulty or
  any prior attempt.
- **MLX target -> llm-compressor is the right tool and the only one.**
  comfy_kitchen's layout is bespoke to ComfyUI; compressed-tensors/AWQ is the
  portable artifact.
- **One artifact for both -> llm-compressor plus a custom-node bridge**, paying a
  node and the loss of kernel acceleration for portability, knowingly.

**The calibration corpus work transfers to either tool.** The corpus, the
acceptance measurement and the held-out split are not llm-compressor-specific.
That makes section 19's falsifier-then-corpus sequencing correct regardless of
which tool wins.

### Loader mechanism: both prior accounts were partly right

I relayed h3guy's description to the subagent, which disputed it. Reading
`awq_quantization_suite/workflows/comfyui_minimax_h3_awq_loader.py` settles it:

- h3guy correct: it **does** write `{prefix}.comfy_quant` tensors (line 224-225)
  and read them back (355, 595-607) -- but in memory at load, not into the file.
- Subagent correct: the shipped `-comfy.safetensors` has **zero** comfy_quant
  tensors. Independently confirmed earlier in this session: its suffixes are
  `weight_packed` / `weight_scale` / `weight_shape` with metadata keys
  `scheme, config, format, quantization`. And it is storage-only -- an eager
  dequantization backend (comment at 316), not an accelerated kernel.
- **Subagent wrong on one point that matters:** it reported `h3_awq_w4a16` is
  "deliberately not in `QUANT_ALGOS`". The loader **injects** it at runtime --
  lines 286-292 assign into both `comfy.quant_ops.QUANT_ALGOS` and
  `comfy.ops.QUANT_ALGOS`, with a guard against clobbering a different contract.

That last point is good news for the bridge: a custom node can register a new
format into `QUANT_ALGOS` at runtime. The `KeyError` at `ComfyUI/comfy/ops.py:1205` only
bites if you emit a format nobody registered. The pattern is proven, and cheaper
than "core cannot be extended" implied.

### Findings that change the experiment design

1. **AWQ cannot reach either vision tower.** 0 of 110 tower linears match the
   resolved balance patterns -- the towers are named `attn.qkv` /
   `mlp.linear_fc1`, which the mappings do not cover. So any tower-quantized arm
   is **AWQ on the language stack plus plain round-to-nearest on the tower**.
   That asymmetry must be stated whenever that arm is reported; it is not a
   clean bf16-vs-AWQ tower comparison.
2. **`visual.blocks.N.mlp.linear_fc2` cannot be quantized at group 128** -- its
   column count is not divisible, and llm-compressor raises before saving
   (`group_size_validation.py:113-127`). Note this is the *same* module
   Comfy-Org singled out for 16x finer convrot grouping (section 15). Two tools
   hitting the same module from different directions.
3. **Model 3 has no config** -- the single file needed an HF directory rebuilt
   around it. Done and verified: one prefix rename is an exact bijection over
   all 750 tensors, `from_pretrained` loads it, weights bitwise identical
   including DeepStack mergers. Geometry taken from what ComfyUI actually runs
   (`ComfyUI/comfy/text_encoders/qwen3vl.py:16`, `llama.py:342-348`) -- `rope_theta`
   differs from the transformers default and would have been silently wrong.
4. **The encoder's calibration corpus can be generated perfectly on-distribution**
   by running T21/I21 over real briefs and keeping `rewritten_prompt`: the
   encoder's input distribution *is* the PE models' output distribution.
5. **ComfyUI does not feed the encoder bare prompt text.**
   `ComfyUI/comfy/text_encoders/qwen_image21.py:9-11` wraps every prompt in a template
   and prefixes reference blocks per image at `:20-22`. Corpus text must be
   built through that exact template.
6. **Encoder acceptance needs the un-normed final hidden state.**
   `qwen_image21.py:37-38` sets `layer_norm_hidden_state = False`, while
   transformers 5.x returns the post-norm state -- capture the norm's input via
   forward hook. And score only the kept slice, since `:53-80` drops the system
   turn and the vision spans.

### Open unknown

The sequential pipeline has never been run on **dense** Qwen3.5 here --
`tests/.../tracing/test_models.py:166` lists only
`Qwen3_5MoeForConditionalGeneration`. Tracing GatedDeltaNet is the one real
unknown, and only a run settles it. Both quantize scripts' `--dry-run` stops
just short of it.

### Housekeeping

The rebuilt encoder HF directory in the the rebuilt-encoder output directory is ~16
GiB of scratchpad disk. Delete if not proceeding to model 3.

## 21. CORRECTION to section 20: a wrapper CAN reach an accelerated W4A16 kernel

Section 20's verdict rested on "llm-compressor's W4A16 gives you no kernel."
**That is wrong**, and the question the user raised -- if we wrap it like the h3
repo did, why does core's native support matter? -- is the right one.

Verified on this machine (2026-09-20):

    comfy_kitchen.gemv_awq_w4a16(x, qweight, wscales, wzeros, bias, group_size)
    comfy_kitchen.tensor.TensorCoreAWQW4A16Layout
        MIN_SM_VERSION      = None          (no architecture floor)
        QUANTIZES_INPUT     = False         (activations stay 16-bit)
        supports_fast_matmul() -> True      on this RTX 4090 (sm 8.9)
        Params fields: scale, zeros, group_size, orig_dtype, orig_shape, transposed
        also: quantize, dequantize, get_plain_tensors, requantize_kwargs,
              state_dict_tensors

So a real accelerated W4A16 path exists in the installed comfy_kitchen. It is
simply **not registered in core's `QUANT_ALGOS`**. And per section 20, a custom
node can register a format into `QUANT_ALGOS` at runtime -- the h3 AWQ loader
does exactly that at lines 286-292, assigning into both `comfy.quant_ops` and
`comfy.ops`.

**What core natively supports therefore does not bound what a wrapper can do.**
It bounds only what works with zero custom code. The h3 loader's
dequantize-then-`F.linear` was a property of that implementation, not a
necessity of the format.

### Revised tradeoff

| | W4A16 via llm-compressor + wrapper | W4A8 via comfy_kitchen, core-native |
|---|---|---|
| artifact | compressed-tensors, portable (also feeds MLX) | bespoke to ComfyUI |
| kernel on 4090 | `gemv_awq_w4a16`, fast matmul True | fast matmul True |
| activations | 16-bit | int8 |
| compression | less | more |
| custom node | yes (proven pattern) | no |
| calibration | AWQ with assistant-turn mask (section 19) | RTN + Lloyd-Max codebook, no calibration |

The portability argument **does not cost kernel acceleration**, which is what
section 20 claimed. The real axes are compression, activation precision, and
whether AWQ calibration (section 19) beats comfy_kitchen's uncalibrated RTN --
which is exactly the open question the corpus work exists to answer.

### The genuine remaining unknown

Whether a compressed-tensors AWQ artifact repacks cleanly into
`TensorCoreAWQW4A16Layout`'s expected layout: nibble packing order, zero-point
convention, group axis. `requantize_kwargs` and `get_plain_tensors` suggest a
conversion path is intended, but this is unverified and is the real work item.

Also note `gemv` means batch-1 decode -- the PE generation case exactly. Whether
a batched gemm path exists for the encoder's single prefill is a separate
question and was not checked.

**Net: section 20's split-by-destination verdict is withdrawn.** A single
portable W4A16 artifact serving both ComfyUI and MLX, with a wrapper node and a
real kernel, is viable. The decision is now a genuine engineering tradeoff
rather than a format-availability constraint.

## 22. Edit-mode taxonomy, read out of the I21 system prompt

The official blog is JS-rendered and returns no content to a fetch. The I21
`system_prompt.txt` is the authoritative source anyway -- it is what the weights
were trained against. It names a far richer mode taxonomy than "edit an image",
and these are the **stratification axes for the calibration corpus** (section 19
called for axes "the system prompts themselves name"; here they are).

### Modes, with their distinguishing contract

| mode | trigger | `<imageN>` tags | ratio contract |
|---|---|---|---|
| single-image edit | N=1, "change this picture" | **forbidden** -- refer naturally (图像/图片中/the image) | `ratio_follow` = `<image1>` when no orientation hint |
| single-image scene generation | N=1 but input is identity reference only (拍一套写真, cosplay成X, 穿越到古代) | forbidden | do **NOT** follow input ratio; pick `wh_ratio` by scene semantics |
| multi-image edit | N>=2 | **mandatory and non-negotiable** | `ratio_follow` = the canvas image |
| style transfer | 画成X的风格 / 风格迁移 | per N | follows the **content** image, not the style reference |
| 合影 / 合照 (group photo) | multi-image, no canvas | mandatory | all images are identity sources |
| outpainting | 扩图 / 延伸画面 | per N | do **NOT** follow input ratio -- infer from extension direction |
| panoramic | 全景 / panorama | per N | its own rule |
| three-view / multi-grid | 三视图 / 多宫格 | per N | derived from subject orientation x panel layout (1x3, 3x1, 2x2) |

### Orthogonal axes the corpus must also span

- **Two independent language decisions**, which the prompt opens by warning must
  not be conflated. (A) the language of descriptive prose outside quotes --
  final, non-negotiable. (B) the language of text rendered *into* the image,
  resolved in strict priority: explicit target language > dominant language of
  text already in the image > language of the instruction. Chinese, English,
  Japanese, Korean, Thai, Arabic, French are all named.
- **Text-bearing vs text-free inputs** -- decision (B) branches on whether the
  input image already contains text.
- **Intent branch**: "change this picture" (constrain, minimal) vs "a new
  picture of this subject" (construct actively, build scene/lighting/layout).
  The prompt calls elaboration scale intent-branched.
- **Identity-preserving edits** -- faces, personal accessories, product design
  and markings, and rendering medium (photo / anime / illustration / sketch /
  3D / painting) are invariants unless targeted.

### Masking: where it lives, and a plausible gap

CORRECTED. An earlier draft said "there is no masking concept", which
overstated a narrow finding. Accurate scoping:

- **True and still relevant:** the I21 system prompt has no mask vocabulary --
  grep for mask / inpaint / region / crop returns nothing. So the *prompt
  expander* never sees or reasons about a mask, and calibration rows for it
  should not carry one.
- **Also true:** masking works with Qwen-Image 2.1 in ComfyUI today, through the
  model-agnostic latent path -- `SetLatentNoiseMask` (`nodes.py:1558`),
  `VAEEncodeForInpaint` (`nodes.py:412`), `InpaintModelConditioning`
  (`nodes.py:453`). Masking happens at the latent/sampler stage, downstream of
  both the PE model and the text encoder.
- The `mask` identifiers inside `ComfyUI/comfy/ldm/qwen_image21/model.py:167-173` are
  **attention** masks (segment causal masking, text segments causal, image
  blocks attending to everything before their end), not image masks. Easy to
  misread.

**Node surface:** `TextEncodeQwenImage21` (`ComfyUI/comfy_extras/nodes_qwen.py:111`)
takes clip, positive and negative prompts, an optional VAE, a `resolution`, and
up to 16 reference images via `io.Autogrow`. **No mask input.** Neither did the
2.0-era `TextEncodeQwenImageEdit` or `TextEncodeQwenImageEditPlus`.

**The plausible gap, worth a PR.** QI2.1's edit path conditions on reference
latents spliced into the sequence (`ref_latents`, and `image_slots` from
`qwen_image21.py:53-80`). A generic `SetLatentNoiseMask` constrains only the
*output* latent -- nothing tells the model which region of a *reference* image
is the subject. Region-aware reference conditioning is not expressible with the
current node surface. Check upstream for an open PR before building one.

Note also `TextEncodeQwenImage21` returns an empty latent sized to the first
reference image, with a tooltip warning that any other size shifts the edit --
so a mask used with it must match that latent's size.

### Consequence for the acceptance measurement (section 13b / G1)

The `wh_ratio` / `ratio_follow` contract is **mode-dependent and mechanically
checkable**, which makes G1 much stronger than "the JSON parses". Per mode:

- outpainting: `ratio_follow` must be `""` and `wh_ratio` set
- single-image edit, no orientation hint: `ratio_follow` = `<image1>`, `wh_ratio` = `""`
- single-image scene generation: the inverse -- `wh_ratio` set, `ratio_follow` = `""`
- multi-image: `ratio_follow` names the canvas image specifically
- N=1: presence of any `<imageN>` tag in `rewritten_prompt` is a **violation**
- N>=2: absence of `<imageN>` tags is a violation
- `rewritten_prompt` must never contain resolution or ratio strings ("2:3",
  "16:9", "1920x1080", "2K") -- those go only in the two ratio fields

These are per-mode contract assertions, not syntax checks, and a quantized
candidate that starts violating them mode-selectively is exactly the degradation
a parse-only gate would miss.

## 23. Region selection is native, in-band, and the PE models have no vocabulary for it

Source: the release blog (local markdown copy in the new repo under
`internal/reference/`). This supersedes section 22's masking paragraph again.

### QI2.1 supports three region-selection mechanisms, none of them a latent mask

Quoting the blog: "Qwen-Image-2.1 supports **circles, painted annotations, and
separate masks** to specify where an edit should take place."

1. **Circles** drawn onto the input image, referenced by colour in the
   instruction. The blog's example edits three regions at once: "Remove the
   metal watch in the blue circle, change the hair in the red circle to black,
   and replace the area in the green circle with gray short-sleeved linen
   pajamas."
2. **Painted annotations** -- a region painted onto the image ("adds a diver to
   the area marked in white").
3. **A separate mask as a second input image** -- "Qwen-Image-2.1 also accepts
   the original image and a separate mask as two inputs", used because circles
   and painted annotations obscure original content.

**All three are in-band.** The region marker is either painted into the
reference image's pixels or passed as another *image*. None of them is a
ComfyUI `MASK` type or a latent-space mask. That is why
`TextEncodeQwenImage21` has no mask input and needs none: it accepts up to 16
reference images (the blog claims up to 10 supported), and a mask image is
simply one of them.

So my earlier claim that "masking happens downstream at the latent stage" was
wrong for QI2.1's native mechanism. Generic `SetLatentNoiseMask` remains
available and orthogonal, but it is not how this model does local editing.

### The gap: the PE system prompts have no vocabulary for any of it

Grepped both `system_prompt.txt` files for circle / annotation / painted-region
/ colour-marker / mask vocabulary, and separately for transparency / RGBA /
alpha / layer. **Zero hits in either model, for either family.** The only
"paint" matches are incidental ("repainting", "text painted into the image").

Three concrete capability gaps between the image model and its own expanders:

1. **Circle / painted annotation.** The PE model *sees* the annotated image
   through its vision tower and the brief says "the blue circle". Its rules say
   to keep "the user's own action verb, spatial relations and described state
   intact", which may carry it -- but its stronger rule is "Resolve ambiguity,
   then commit... turn imprecise spatial reference into something concrete and
   observable." That rule points the wrong way here: resolving "the blue circle"
   into a description of its contents **destroys the annotation reference** the
   downstream model needs. Untested, and a specific predicted failure.
2. **Separate mask image.** The Image Reference Rules require every image at
   N>=2 to be tagged and its role stated -- "which one is the canvas... and
   which supply material to transfer". A mask is neither. The prompt has no
   category for it, so the expander will likely describe a mask image as
   content.
3. **Native transparency / RGBA.** QI2.1 unifies Qwen-Image-Layered: the prompt
   decides whether output carries an alpha channel, transparent images can be
   edited directly, and an RGBA layer can be extracted from an RGB photo.
   Neither expander mentions transparency at all.

These belong in the eval corpus as their own strata, both to characterise
baseline behaviour and because they are where a quantized candidate would most
plausibly degrade first.

### Architecture notes from the blog, confirming local reads

- Visual generation component: **32 Single-Stream DiT layers, 7B params** --
  matches `transformer/config.json` (32 layers, 32 heads, head dim 128).
- **Mixed-granularity attention**: text (including the system prefix and editing
  instructions) uses a **token-level causal mask**, image generation uses a
  **chunk-level mask**. This is exactly the `segments` code at
  `ComfyUI/comfy/ldm/qwen_image21/model.py:167-173`, and it is what the `mask`
  identifiers there mean -- architecture, not inpainting.
- **KV cache reuse**: input images and editing instructions are static context,
  computed and cached on the first step. That is what the `QwenImage21Cache`
  node (`ComfyUI/comfy_extras/nodes_qwen.py:185`) exposes.
- Other named tasks: panorama from a selfie, infographic expansion, three-view
  character reference to storyboard, and sequential local edits assembled into
  animation.

## 24. What ConvRot is, and what the literature says about our two model classes

Research pass, 2026-09-20. Full report: `docs/convrot-research.md`.

### ConvRot is published, and the local code names its source

arXiv:2512.03673, "ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for
Diffusion Transformers". `comfy_kitchen/tensor/convrot_w4a4.py:3-9` says
outright that it is "the plain ConvRot path from the paper". The transform is
readable Python, not compiled: `tensor/int8_utils.py:11-74` builds a
block-diagonal **regular** Hadamard of order 4^k as Kronecker powers of
`[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`, applied along the input-feature
axis in contiguous groups. It is symmetric and **involutory** (H@H = I), which
is why dequantization reuses the same rotate call.

Regular rather than Sylvester/Walsh for a stated reason: every row and column of
the normalised regular matrix sums to 1, where Sylvester at order 256 maps the
all-ones vector to a single spike. The paper's framing is that LLMs have
column-wise outliers, which plain Hadamard handles, while DiTs also have
row-wise outliers that plain Hadamard *amplifies*.

**Structurally unlike QuaRot/SpinQuant.** ConvRot rotates inside a single
linear, `(xH)(WH)^T = xW^T` exactly. There is no residual-stream invariance and
no matched projection pairs.

Comfy's `asym_w4a8_int8` extends the paper with a QuIP#/QAM-W-style fixed
Lloyd-Max codebook, a kurtosis gate, and ALS group-scale refinement. None of
that is published, and no Comfy-Org writeup of ConvRot exists.

### The GatedDeltaNet invariance worry does not apply

Because ConvRot self-inverts inside each linear, block structure is irrelevant
to its correctness. `conv1d`, `A_log`, `dt_bias`, the gated RMSNorm and the fp32
recurrent state are not `nn.Linear` and never enter the path (Comfy keeps the
state fp32, `comfy_kitchen/gated_delta.py:59-80`). This was the risk flagged as
"could silently break a PE quant"; it is closed.

### There IS literature on quantizing hybrid linear-attention models

- **arXiv:2609.04098** (2026-09-03) is directly on point: NVFP4 W4A4 on a hybrid
  27B with 48 GatedDeltaNet and 16 full-attention layers. It quantizes
  `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` -- all 496
  linears -- excluding only lm_head, embeddings, convolutions and norms. Two
  results that cut against intuition: recurrent-state error is **flat, not
  accumulating** (each delta-rule write overwrites along the current key
  direction), and the gate/decay projections `a` and `b` are the **least**
  sensitive, because the log-space parameterisation compresses error. It uses no
  rotation at all.
- **DAMP** (arXiv:2608.27513) quantizes the recurrent state itself and reports
  that after Hadamard, residual INT8 error stays concentrated in a few key
  channels.
- **MambaQuant** (ICLR 2025, arXiv:2501.13484) is the rotation-specific
  reference and argues Hadamard cannot equalise channel variance in SSMs, hence
  KLT. It also applies rotations per-linear, and never covers gated delta or
  hybrids.

**The intersection "ConvRot x linear attention" is empty.** Nobody has published
it.

### And none on graded per-module group sizes

arXiv:2601.22347 (ICML 2026) studies block sizes 16-2048, treats it as a
**global** hyperparameter, and names per-layer assignment as future work. So the
per-role story section 15 withdrew was not only wrong about this file -- it
would have been a novel claim with no support.

On ViT sensitivity: **RegCache** (arXiv:2510.04547) tests CLIP/SigLIP/SigLIP2/
DINOv2 at W4A4-W8A8 and localises sensitivity to "the MLP projection layers in
one or two middle layers", instrumenting FC2 input norms -- but it does **not**
rank module types, and an earlier leading-prompt reading of it overstated that.
**QuantVGGT** (arXiv:2509.21302) applies global random Hadamard to a ViT and
states QuaRot-style methods "do not generalize well" beyond 2D-visual/language
models; its hard case is register/special tokens, as RegCache's is.

### Hardware

All three ConvRot formats are native on this 4090: `MIN_SM_VERSION` is (7,5) for
`convrot_w4a4` and `int8_tensorwise`, (8,0) for `asym_w4a8_int8`, and
`ComfyUI/comfy/ops.py:1710-1723` disables them only when the device lacks int8
compute, which sm_89 has. `asym_w4a8_int8` has real fused CUDA kernels
(`launch_quantize_w4a8_convrot`, `launch_w4a8_codebook_gemm_chunked`,
`launch_w4a8_codebook_gemv`), so it is the best quality-per-byte reachable path
here, not an eager fallback. The ConvRot paper's own benchmarks were run on a
4090.

### Claims the report marks unverified

The INT8-vs-NVFP4 comparison in 2609.04098 (taken from a search summary, not the
body); the Hopper int4-lowering claim; whether the fused kernel uses a fast
butterfly or a dense N0xN0 matmul; and the QAM-W/HARP/QuIP# comparison rows
(built from abstracts). Plus the fused-path group-size restriction noted in
section 15.

## 25. First end-to-end run: harness works, bf16 reference arm established

`scripts/smoke_heylook.py`, upstream example briefs, heylook/MLX backend,
reference sampling from `PROFILES`. Raw records: `data/smoke_heylook.jsonl` (gitignored; regenerate with the script).

**Result: 7/7 completed, 7/7 parse_ok, 6/7 contract_ok, 0 truncated.**

The harness works end to end. This was the gate everything downstream waited on.

**Precision: VERIFIED bf16.** heylook's admin API exposes no quantization field,
so this was checked against the converted models' own configs
(`coderef/heylook-models/`, converted with mlx-vlm):

| check | pe-t2i | pe-i2i |
|---|---|---|
| `quantization` block in config.json | absent | absent |
| declared dtype | bfloat16 | bfloat16 (text_config) |
| index total_size | 18,819,627,488 B | 18,819,627,488 B |
| tensor count | 760 | 760 |
| chat_template.jinja vs source | byte-identical | byte-identical |

The total size and tensor count match the HF source exactly -- a 4-bit
conversion would be roughly a quarter of it. So the conversion is unquantized,
complete, and carries the trained template unmodified.

**This run therefore IS a bf16 reference arm.** (Top-level `dtype` is absent on
pe-i2i, but that asymmetry is inherited: the source I21 config lacks it too, and
it is not a conversion artifact.)

### The bf16 baseline is NOT 100%, and that is the most useful number here

Row `1412128` (`single_scene_complex`, 1 image) fails `tags:present_at_single_image`.
It is a genuine model error, not a false positive -- the rewrite contains
"Keeping the symbol in `<image1>` completely unchanged" and "Remove the plain
black backdrop of `<image1>` entirely", while the system prompt says "For
single-image input (N = 1), do NOT use tags -- refer to the image naturally".

Its ratio fields are correct (`ratio_follow` = `<image1>`, `wh_ratio` = ""), so
only the tag rule broke. Note the prompt itself creates the tension: tags are
forbidden in `rewritten_prompt` but required in `ratio_follow` at N=1.

**Consequence for the quant comparison: the bf16 reference is 6/7, not 7/7.**
Precision is verified above, so this is the model's own behaviour and not
conversion damage. A candidate scoring 6/7 has not regressed. Any acceptance
threshold set against an assumed-perfect reference would have been wrong from
the first run.

### Instrument limit 1: heylook reports `input_tokens: 1` for image requests

Isolated by a two-case probe:

| case | input_tokens |
|---|---|
| I21 model, **no** image | 4527 (correct: 4510-token system prompt + brief + wrapper) |
| T21 model, **with** image | **1** (wrong) |

So it is **image-specific, not model-specific** -- any request carrying an image
loses input-token accounting. Both PE models report correctly on text-only
requests.

This costs us the ability to verify prompt construction and input-side
truncation on every edit row, and it matters for calibration planning, where
image token cost is roughly pixels/1024 and a large photo can outweigh the
system prompt. Mitigation: predict input tokens client-side from the
checkpoint's tokenizer plus the pixel formula, and compare where the server
does report. Worth reporting upstream as a server-side accounting bug.

### Instrument limit 2 (avoided): the default token cap would have truncated

The `text_edit` row generated 9130 output tokens against the reference cap of
24000. heylook's own default for these models is 4096. That row would have
truncated mid-trace, parsed as invalid JSON, and -- run against a quantized
candidate -- looked exactly like a quantization regression. Pinning `max_tokens`
from `PROFILES` is not hygiene; it is what makes the measurement mean anything.

### Cost, for planning

Output length varies more than 4x across rows (1480 to 9130 tokens) and edit
rows with images run minutes each on this backend. A full stratified corpus
across the eight edit modes will not be a quick loop; budget accordingly, and
prefer the greedy configuration for comparison runs so two runs of one arm agree.
