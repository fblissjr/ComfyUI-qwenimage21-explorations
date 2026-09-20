# ConvRot and rotation-based quantization: research report

Date of research: 2026-09-20. Read-only pass over `ComfyUI`, its venv copy of
`comfy_kitchen` 0.2.35+sol.8176242, plus web sources. Nothing was modified.

## Evidence key

Every claim below is tagged:

- **(a) verified** — read in code on this machine, with `path:line`.
- **(b) published** — stated in a primary source (paper, repo, issue) with a URL.
- **(c) claimed** — asserted somewhere, no evidence I could check.
- **(d) unknown** — searched, found nothing.

A note on method: several external claims below were first pulled by a fetch
tool whose summarizer answers leading questions agreeably. Where a claim is
load-bearing I re-fetched with a neutral prompt and kept the neutral answer.
Two claims changed materially under that re-check and are marked.

---

## Q1. What ConvRot is, precisely

### It is a published method, and the implementation is a faithful port

**(b)** ConvRot is arXiv:2512.03673, *"ConvRot: Rotation-Based Plug-and-Play
4-bit Quantization for Diffusion Transformers"*, Feice Huang, Zuliang Han, Xing
Zhou, Yihuang Chen, Lifei Zhu, Haoqian Wang. Submitted 2025-12-03, v1 only, no
venue listed. https://arxiv.org/abs/2512.03673

**(a)** `comfy_kitchen` says so itself. The layout module's docstring reads
"This is the plain ConvRot path from the paper adapted to kitchen's existing
int4 tensor-core GEMM"
(`comfy_kitchen/tensor/convrot_w4a4.py:3-9`).
So it is neither bespoke nor a vendor rename: it is a named PTQ method with a
paper, ported by Comfy-Org, and the port names its source.

**(b)** Third-party uptake exists beyond Comfy. There is an open torchao
enhancement request, https://github.com/pytorch/ao/issues/4695, opened
2026-08-05, and an SGLang PR for a DiT INT8 W8A8 ConvRot path,
https://github.com/sgl-project/sglang/pull/38040. The reference implementation
is https://github.com/feice-huang/ConvRot.

### What transform it actually applies

**(a)** The transform is fully readable in Python; only the fused CUDA kernel is
compiled. The eager path is the definition:

- `comfy_kitchen/tensor/int8_utils.py:11-38` `_build_hadamard`. Docstring:
  "Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot)." It rejects
  any size that is not a power of 4 (`int8_utils.py:21-22`), then builds
  `H = h4 ⊗ h4 ⊗ ...` from the fixed base
  `[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]` and divides by `sqrt(size)`.
- `int8_utils.py:41-55` `_rotate_weight`: reshape `W` to
  `[out_f, n_groups, group_size]` along the input axis and right-multiply each
  group by `H^T`. Block-diagonal, contiguous groups, input-feature axis only.
- `int8_utils.py:58-74` `_rotate_activation`: the same reshape on `x`, right
  multiply by `H`, applied online at inference.
- The identical pair is duplicated in the W4A4 eager backend at
  `comfy_kitchen/backends/eager/convrot_w4a4.py:15-74`.

**(a)** I verified the matrix's properties numerically in pure Python for sizes
4, 16, 64, 256:

- `H` is symmetric, and `H @ H == I`. It is an **involution**, which is why
  dequantize can call the same `_rotate_weight` to invert
  (`backends/eager/convrot_w4a4.py:140-153`) rather than transposing anything.
- Every row sum and every column sum of the normalized `H` equals exactly 1.
  That is the "regular" property.
- Contrast with the Sylvester/Walsh matrix of the same order built from
  `[[1,1],[1,-1]]`: its column-sum infinity norm at order 256 is 16, and
  `H_sylvester @ ones` is a single spike of magnitude 16 with 255 zeros. The
  regular matrix maps `ones` to `ones`, spread evenly over all 256 coordinates.

That last contrast is the whole point of the method. **(b)** The paper calls
`||H^T · 1||_inf` the *column discrepancy* and proves (its Theorem 3.3) that
matrices with every row and column summing to `±sqrt(n)` exist only at orders
`n = 4^k`, constructed as `H_{4^(k+1)} = H_{4^k} ⊗ H_4`. The practical
consequence: a Sylvester Hadamard, whose first column is all ones, takes a
token's DC component and concentrates it into one output coordinate, creating a
fresh row-wise outlier. A regular Hadamard cannot, because it maps a constant
vector to a constant vector. The paper's framing is that LLMs show column-wise
(per-channel) outliers, which plain Hadamard handles, whereas diffusion
transformers also show **row-wise** (per-token) outliers, which plain Hadamard
*amplifies*. ConvRot is the fix for the row-wise case.

**(a)** A local comment states the row-wise magnitude directly for one model:
`ComfyUI/comfy/ldm/wan/model_animate2.py:177` says "convrot is what lets
low-bit survive the ~125x per-channel outliers here". That is a code comment,
so treat it as (c) for the magnitude, (a) for the fact that someone at Comfy
wrote down this rationale.

### How `convrot_groupsize` parameterizes it

**(a)** It is the block size `N0` of the block-diagonal rotation, in input
features. Constraints enforced in code:

- Must be a power of 4 (`int8_utils.py:21`).
- Must divide `in_features` (`int8_utils.py:48-49`,
  `backends/eager/convrot_w4a4.py:81-82`, `tensor/w4a8_int8.py:152-163`).
- Default 256 everywhere (`tensor/convrot_w4a4.py:31`, `tensor/int8.py:76`,
  `tensor/w4a8_int8.py:33`, `comfy/ops.py:1030`).

**(b)** The paper ablated `N0` in {16, 64, 256, 1024} and chose 256 as a single
global default. Larger `N0` suppresses outliers better and costs more. The
grouping is what drops rotation cost from `O(K^2)` to `O(K)` in channel count
`K`. So group size is a precision/cost dial, not a per-module knob in the paper.

### Where ConvRot sits in the rotation family

The critical structural difference, and the one that drives the answers to Q2
and Q3:

**(a)** ConvRot is a **per-linear-layer, self-inverting** transform. The
forward is `(x H)(W H)^T`. Because `H = H^T` and `H H = I`, that equals `x W^T`
exactly, with no assumptions about anything outside the layer. Verified in the
eager forward: rotate the activation inline
(`backends/eager/convrot_w4a4.py:195-198`), against a weight rotated offline
(`:135-137`). The int8 path does the same, gated on a flag
(`backends/eager/quantization.py:1036-1042`).

That is *not* how QuaRot and SpinQuant work. Those rotate the residual stream
once and rely on computational invariance through RMSNorm and matched
projection pairs, so the rotation cancels across a block rather than within a
layer. ConvRot buys locality at the cost of only mixing within a 256-wide
window, so it cannot move outlier energy across window boundaries.

Positioning against the family the task named:

| Method | Rotation | Scope | Calibration | Relation to ConvRot |
|---|---|---|---|---|
| QuaRot (2404.00456) | randomized Sylvester Hadamard | global, residual stream | none for rotation | ConvRot replaces global with block, and Sylvester with regular |
| SpinQuant (2405.16406, ICLR 2025) | learned on Stiefel manifold | global | yes, optimizes rotation | ConvRot is fixed and data-free |
| QAM-W (2605.26339) | deterministic block-Hadamard, 2D pairing, Lloyd-Max codebook on the unit circular Gaussian | block | none for the rotation | closest published cousin to Comfy's `asym_w4a8_int8`; see below |
| HARP (2605.29843) | learnable orthogonal blocks over a fixed Sylvester base mixer | block | yes | ConvRot is the fixed, un-learned point in that design space |
| QuIP# / QTIP | randomized Hadamard for incoherence, then vector/trellis codebooks | global | none for rotation | same "rotate until Gaussian, then use a fixed codebook" logic |
| ConvRot (2512.03673) | regular Hadamard, order `4^k`, symmetric/involutory | block, per linear | none | — |

Provenance note on that table: the ConvRot, QuaRot and SpinQuant rows rest on
papers I read or on well-established prior knowledge. The QAM-W, HARP and
QuIP#/QTIP rows are **(c)**, assembled from abstracts and search-result prose,
not from reading those papers. Verify before quoting them.

**(a)** Comfy's `asym_w4a8_int8` is a hybrid of ConvRot and the QuIP#/QAM-W
codebook idea, and the code says so. `backends/eager/w4a8_int8.py:11-16` ships a
hard-coded 16-level Lloyd-Max table with the comment "ConvRot makes every
layer's rotated groups Gaussian, so this one table matches a per-tensor fit and
skips the k-means." There is a kurtosis gate that falls back to fitting a
codebook if the rotated groups stay heavy-tailed
(`w4a8_int8.py:17-37`), plus two ALS passes refining group scales
(`w4a8_int8.py:202-208`). None of that is in the ConvRot paper. It is Comfy-Org
engineering on top.

### What is undocumented publicly

- **(d)** No Comfy-Org blog post or release note explaining ConvRot was found.
  The closest public documentation is a third-party toolkit's format docs at
  https://github.com/Comfy-Org/comfy-quants (`docs/formats/int8_tensorwise.md`,
  `docs/quantization/int8_w8a8.md`), which describe the flags
  (`--convrot/--no-convrot`, `--convrot-groupsize`, power of four, default 256)
  and the skip rule (rotation is omitted for any layer whose `in_features` is
  not divisible by the group size) but cite no paper.
- **(d)** The `asym_w4a8_int8` codebook/ALS design, the graded per-module group
  sizes, and the embedding-table rotation are undocumented publicly. They exist
  only as code and docstrings.
- **(a)** The embedding path is real: `TensorWiseINT8Layout.dequantize_embedding`
  gathers rows first and un-rotates only those rows
  (`comfy_kitchen/tensor/int8.py:179-191`), passing `group_size=0` when the
  table was not rotated. So the embedding table is rotated like any weight, and
  the lookup un-rotates per row rather than materializing `[vocab, dim]`.

---

## Q2. Rotation-based quantization of vision encoders and ViTs

### Does rotation transfer to ViTs? Partly, and the literature says so explicitly

**(b)** The strongest primary source is *Activation Quantization of Vision
Encoders Needs Prefixing Registers*, arXiv:2510.04547 (v5, October 2025),
https://arxiv.org/html/2510.04547v5. Tested on CLIP-B/16, OpenCLIP-B/16,
**SigLIP-B/16, SigLIP2-B/16**, DINOv2-B/14 at W8A8, W6A6 and W4A4. That is
directly the family of the Qwen3-VL vision tower.

Two findings matter here.

**(b) Sensitivity is localized to MLP projections in one or two middle blocks.**
Quoted from Section 3.1: "quantization-sensitive layers — i.e., layers that
incur substantial accuracy drops when quantized — are localized to the MLP
projection layers in one or two middle layers." Their procedure quantizes each
layer alone to W8A8 and measures ImageNet-1k accuracy drop, and they instrument
"Maximum norm of the FC2 layer input tokens for each layer" (Figure 3 caption).

Be precise about what this does and does not say. The paper selects a sensitive
**block index** by ablation and instruments FC2 inputs. A neutral re-fetch
confirmed it "does not rank individual modules like fc1, fc2, qkv, or proj." So
this is support for *MLP projections in middle blocks* being the sensitive
region, and for FC2 inputs being where the massive-activation norms are
measured. It is **not** a published ranking that says fc2 beats qkv. Do not
cite it as one.

Caveat on provenance: my first fetch of this paper, with a leading prompt,
returned a stronger and partly wrong summary (it claimed the paper shows
Hadamard rotation is insufficient, in the main body, and that it ranks modules).
A neutral re-fetch corrected both: rotation-based methods appear **only in the
appendix**, where RegCache is *combined* with Hadamard rotation, and no
standalone comparative claim is made. Treat "rotation alone does not fix vision
encoders" as **(c)**, not (b), on the strength of this paper.

**(b) The mechanism is register tokens / massive activations.** The proposed
method, RegCache, discovers middle-layer registers and prefixes them as a
precomputed KV cache, then deletes the remaining outlier tokens. The problem
being solved is token-wise (row-wise), not channel-wise, which is exactly the
regime ConvRot's regular Hadamard was designed for. That is a coherent story,
but nobody has published the combination.

**(b)** A second ViT data point: *Quantized Visual Geometry Grounded
Transformer* (QuantVGGT), arXiv:2509.21302 v4, dated 2026-03-07,
https://arxiv.org/html/2509.21302. It quantizes a 1.2B vision transformer to
W8A8 and W4A4 using a **global** random Hadamard preserving computational
invariance, `XW^T = (XH)(WH)^T`, plus post-local channel-wise smoothing. Its
explicit statement is that rotation methods like QuaRot "perform well on
existing 2D-visual and language models" but "do not generalize well to
large-scale 3D models like VGGT". Its identified hard case is again
data-independent special tokens (camera and register tokens) producing extreme
token variance. So: rotation does transfer to ViTs, and the published failure
mode is special tokens, the same one as 2510.04547.

**(b)** Also in this space, Ω-QVLA (arXiv:2605.28803) applies composite rotation
to vision-language-action models, and DiRotQ (arXiv:2605.16732) does
rotation-aware W4A4 for diffusion transformers. Neither was read in depth.

### What people actually ship

**(a)/(b)** The observation that conditioning encoders keep their vision tower
in bf16 is corroborated. On this machine,
`models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (27.14 GB)
has 1603 tensors and exactly 350 `comfy_quant` markers, all
`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`, and
all on `model.layers.{0..49}.{self_attn.q,k,v,o_proj, mlp.gate,up,down_proj}`.
Zero markers contain "visual" or "vision". The public README for the same family
of build (https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot)
states the conversion ran with `--exclude-layers '^visual\.'` and that 551
tensors, "including the complete vision tower", stay BF16. That README also
describes the language matrices as *learned* row-wise INT8 ConvRot produced with
"AdamW AdaRound optimization with plateau early stopping", which is AdaRound
layered on top of ConvRot and is not in the paper.

I could not read the 256 / 64 / 16 ladder from a checkpoint directly: no
vision-tower ConvRot checkpoint exists on this disk, and the two local ConvRot
text encoders are LM-only and uniformly 256. But the ladder is fully explained
arithmetically from the model's dimensions and the kernels' constraints, which
is the next section and the strongest finding in this part of the report.

### The 256 / 64 / 16 ladder is arithmetic, not a sensitivity judgment

**This is the most important correction in the report.** The graded group sizes
are fully determined by divisibility and by what the kernels accept. No
per-module sensitivity decision was made, so there is no design rationale to
look for in the literature.

**(a)** The rule is: `convrot_groupsize` must be a power of 4
(`comfy_kitchen/tensor/int8_utils.py:21`) and must divide `in_features`
(`int8_utils.py:48-49`). Comfy-quants' docs add that a layer whose
`in_features` is not divisible gets rotation skipped entirely. Take the largest
power of 4 that divides each `in_features` of Qwen3-VL-8B (vision
`hidden_size` 1152, vision `intermediate_size` 4304, text `hidden_size` 4096,
text `intermediate_size` 12288, from
https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct config.json):

| Module | `in_features` | Largest power-of-4 divisor |
|---|---|---|
| LM q/k/v/o_proj, gate/up_proj | 4096 | 4096 |
| LM down_proj | 12288 | 4096 |
| vision `attn.qkv` | 1152 | **64** |
| vision `attn.proj` | 1152 | **64** |
| vision `mlp.linear_fc1` | 1152 | **64** |
| vision `mlp.linear_fc2` | 4304 | **16** |
| vision `merger.linear_fc1` | 4608 | 256 |

1152 = 2^7 · 3^2, so 256 = 2^8 cannot divide it; 64 is the ceiling. 4304 =
2^4 · 269, so even 64 fails (4304 mod 64 = 16) and 16 is the ceiling. The
observed 256 / 64 / 16 ladder is reproduced exactly by one mechanical rule,
with 256 being the ordinary default wherever it is legal.

**(a)** The kernel side closes it. Strings in the compiled extension
(`comfy_kitchen/backends/cuda/_C.abi3.so`) show the int4 ConvRot path accepts
exactly `{16, 64, 256}` — "INT4 ConvRot quantization requires group_size 16,
64, or 256" — which is the observed set, and the int8 fused path accepts only
256 ("convrot fused kernel only supports group_size 256", "convrot rotate
kernel only supports group_size 256", "convrot staged quantize only supports
group_size 256", "INT8 M=1 ConvRot linear requires group_size 256 and K
divisible by 256").

Two consequences:

- **The premise "the finest rotation group is reserved for the post-gate
  down-projection" is false as a design claim.** `mlp.linear_fc2` gets 16
  because 16 is the only legal value for an input of 4304, not because anyone
  judged it sensitive. The fact that it coincides with the module
  arXiv:2510.04547 instruments is a coincidence of SigLIP's dimension choices.
- **A Qwen3-VL-8B vision tower cannot use the fused int8 ConvRot CUDA kernel at
  all.** Every 1152-input and 4304-input layer is off the 256-only fast path.
  That is a plausible second reason, beyond quality, why shipped conditioning
  encoders leave the vision tower in bf16: rotating it buys no fused kernel.
  This is my inference from the kernel constraints, so **(c)**, but it is
  checkable by timing a rotated 1152-wide layer.

Note the local Qwen3-VL-**4B** checkpoint does not exhibit the ladder, because
its vision tower is 1024 hidden / 4096 intermediate, both clean powers of 4
(verified from tensor shapes in
`models/text_encoders/qwen3vl_4b_bf16.safetensors`). The ladder is specific to
the 8B tower's dimensions.

### Is there published support for graded per-module-role group sizes?

**No. (d), and the closest paper says it is future work.** With the above, this
question is largely moot for the observed checkpoints, but it still matters if
anyone wants to grade group sizes *deliberately*.

**(b)** *Pushing the Limits of Block Rotations in Post-Training Quantization*,
arXiv:2601.22347 v2, 2026-05-27, ICML, https://arxiv.org/html/2601.22347v2, is
the paper that would contain this if anyone had it. It studies block sizes 16,
32, 64, 128, 256, 512, 1024, 2048; it treats block size as a **global**
hyperparameter; it applies block rotation only at the `R3` position (the
down-projection input) and leaves `R1`/`R2` full-vector; and it names per-layer
block-size assignment as "an exciting direction for future work" without
recommendations. Its Figure 1 is an activation histogram "at the third down
projection layer in Llama3 1B", and Table 3 reports rotation compute for the
down-projection input specifically. Models: Llama3 1B/3B/8B, Qwen3 1.7B/4B/8B.
No vision transformers.

**(b)** ConvRot's own paper likewise ablated `N0` globally and picked one value.

Deliberate grading is cheap to try, since group size is a per-layer field in the
checkpoint marker (`comfy/ops.py:1231-1235`) and needs no kernel change, but on
this card it is constrained: int4 accepts `{16, 64, 256}` and fused int8 accepts
only 256.

Direction matters here and cuts against fine groups. The ConvRot paper's
ablation says **larger** `N0` suppresses outliers better. So at 1152 hidden the
vision tower is already getting a *weaker* rotation than the LM does, purely
because of its dimension, and at 4304 the `fc2` input gets the weakest rotation
in the whole model on the module the ViT literature flags as the sensitive
region. That is the opposite of protection. If the vision tower is ever
quantized, this is the thing to measure first.

One counter-consideration, **(c)**, my inference: at `N0 = 16` the rotation
block exactly matches the int4 quantization group size (`group_size: int = 16`,
`tensor/w4a8_int8.py:134`; the CUDA extension likewise expects `s_rel` of shape
`[N, K/16]`), so every quantization group is exactly one rotation block and no
outlier can straddle a scale boundary. Whether that compensates for the smaller
mixing window is unmeasured and unpublished.

---

## Q3. Rotation-based quantization of hybrid linear-attention models

This turned out to have more literature than expected, and the framing in the
task is inverted for ConvRot specifically. Two separate answers follow.

### 3a. For ConvRot, computational invariance is not required at all

**(a)** ConvRot never relies on invariance through the residual stream. It
rotates `x` and `W` inside one `nn.Linear` call and the two rotations cancel
because `H` is symmetric and involutory. The question "does invariance hold
through a GatedDeltaNet block" is therefore **moot for ConvRot**. Correctness is
unconditional on any architecture; only accuracy is at stake. Concretely, a
`in_proj_qkv` in a GatedDeltaNet layer is handled exactly like a `q_proj` in an
attention layer: same code path, same flag, same group size
(`comfy/ops.py:1026-1036`).

That is a genuine advantage over QuaRot/SpinQuant for this architecture, and it
is the single most important finding for this question.

**(a)** What is *not* covered, and would need separate handling:

- `conv1d` is a depthwise convolution, not `nn.Linear`. It never enters the
  ConvRot path. Neither do `A_log`, `dt_bias`, the gated RMSNorm weights, or the
  recurrent state.
- The recurrent state in Comfy's own kernel is fp32 and stays fp32:
  `comfy_kitchen/gated_delta.py:59-80`, "The fp32 state `[B, Hv, DK, DV]` is
  updated in place."

So nothing silently breaks structurally. The silent-failure risk is elsewhere:
quantizing `in_proj_a` and `in_proj_b` at the same aggression as `in_proj_qkv`,
since their outputs feed exponential/sigmoid parameterizations where relative
error behaves differently. The published evidence on that turns out to be
reassuring, see below.

### 3b. For global-rotation methods, the invariance concern is real and published

**(b)** MambaQuant, arXiv:2501.13484 v2, February 2025, ICLR 2025,
https://arxiv.org/html/2501.13484v2, is the reference. Its central claim is that
Hadamard rotation is **insufficient** for Mamba, quoted: "the Hadamard
transformation fails to achieve variance alignment across channels ... The
inconsistent variances inevitably result in an uneven numerical distribution."
The argument: after rotating centered data, the per-channel variance depends on
input-specific eigenvectors, and "H is a fixed matrix while both K and λ are
input-dependent, it is not feasible for the Hadamard transformation to uniformly
adjust the channel variances." Their fixes are KLT-enhanced rotation (multiply
the Hadamard by a calibration-derived KLT matrix, offline) and smooth-fused
rotation (smoothing absorbed into weights via a modified SiLU and the parallel
scan, online). Notably, MambaQuant applies rotations **locally per linear
layer**, not globally through the residual stream, which is the same structural
choice ConvRot makes. Models: Vim-T/S/B, Mamba-2D, Mamba-3D, Mamba-370m through
2.8b. Bit widths W8A8 and W4A8. **No mention of gated delta networks,
Qwen3-Next, or hybrid attention models.**

### 3c. GatedDeltaNet specifically: there IS 2026 literature, and it is directly on point

**(b)** *Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the
Recurrent Half of a Hybrid 27B LLM*, arXiv:2609.04098 v1, posted **2026-09-03**,
not peer reviewed, https://arxiv.org/html/2609.04098. Model: a hybrid 27B with
"48 of 64 layers are Gated DeltaNet and only 16 are full attention", hidden
5120. This is structurally the same family as the Qwen3.5-VL 9B target (3:1
GatedDeltaNet:full_attention).

Its module-by-module answer is exactly the list the task asked about:

- Quantized to NVFP4 W4A4, all 496 linear layers, including **`in_proj_qkv`,
  `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`**.
- Kept in BF16: "lm_head, embeddings, convolutions, and norms" — so `conv1d`,
  all norms including the gated RMSNorm, plus `A_log` and `dt_bias`.
- **Recurrent state error does not accumulate.** Quoted: "The full-Minima state
  error is flat: relS=12.96% at token 256 and 12.31% at token 32,768", with the
  mechanism "each write overwrites the state along the current key direction, so
  old errors are deleted key by key as new tokens arrive." This directly
  contradicts the standard worry that a recurrence integrates quantization
  noise.
- **The gate and decay projections are the least sensitive, not the most.**
  Quoted: "fully quantizing the gate projections a and b ... moves y by only
  2.1% and 2.6%, the two smallest effects", because the log-space
  parameterization compresses error: "a ~11% pre-activation error becomes a 7.5%
  error on 1−α and a 5.2% error on β."
- **No rotation or Hadamard is used anywhere in that paper.** It relies on
  NVFP4's 16-element block scaling to localize outliers instead.
- Hardware: one RTX PRO 6000, 96 GB, SM120, native NVFP4.
- On INT8: the paper's position (as summarized in search results, not verified
  in the body) is that uniform INT8/FP8 gives a worse accuracy/storage tradeoff
  than NVFP4 on reasoning tasks. Treat as **(c)**.

**(b)** A second, complementary source: *DAMP: Decay-Aware Mixed-Precision
Recurrent-State Quantization*, arXiv:2608.27513 v1, 2026-08-27,
https://arxiv.org/html/2608.27513. It quantizes the **recurrent state matrices**
of Gated DeltaNet and Kimi Delta Attention on hybrid models (Qwen3.6-35B-A3B
with 30 GDN + 10 attention layers; Kimi-Linear-48B). Its Finding 1 is directly
relevant to anyone planning to lean on Hadamard here: "After applying the
Hadamard transform, the remaining INT8 reconstruction error is still
concentrated in a small subset of key channels." Its fix is a static
mixed-precision layout keeping a small set of key channels in FP16, ranked by
the product of quantization-error energy and decay-based persistence.

**(b)** Also found, not read: *When Good Enough Is Optimal:
Multiplication-Only Matrix Inversion Approximation for Quantized Gated DeltaNet*
(arXiv:2606.06034).

### What this means for the PE target

- Applying ConvRot to every `nn.Linear` of a GatedDeltaNet layer is
  mathematically safe and has published company: 2609.04098 quantizes exactly
  that set and reports it works at 4 bits.
- Leave `conv1d`, `A_log`, `dt_bias`, the gated RMSNorm, and the fp32 state
  alone. Both 2609.04098 (by exclusion list) and Comfy's own kernel (by
  construction) do this.
- Do not pre-emptively protect `in_proj_a`/`in_proj_b`. Published evidence says
  they are the *least* sensitive, for a specific and believable reason.
- The state itself is the one place where a Hadamard is documented to leave
  residual structure (DAMP Finding 1). ConvRot does not touch the state, so this
  is only a concern if state quantization is added later.
- **(d)** Nobody has published ConvRot, or any regular-Hadamard block rotation,
  applied to a GatedDeltaNet or hybrid model. The intersection "ConvRot ×
  linear attention" is empty. The nearest neighbours are MambaQuant (rotation,
  older SSMs, no gated delta) and 2609.04098 (gated delta, no rotation).

---

## Q4. Practical verdict for an RTX 4090 (sm_89, 24 GB)

### All three ConvRot formats are native on Ada. That is verified.

**(a)** Minimum architectures declared in code:

- `TensorCoreConvRotW4A4Layout.MIN_SM_VERSION = (7, 5)` —
  `comfy_kitchen/tensor/convrot_w4a4.py:118`
- `AsymW4A8Int8Layout.MIN_SM_VERSION = (8, 0)` —
  `comfy_kitchen/tensor/w4a8_int8.py:125`
- `TensorWiseINT8Layout.MIN_SM_VERSION = (7, 5)` —
  `comfy_kitchen/tensor/int8.py:65`

sm_89 clears all three. Nothing about ConvRot requires Blackwell.

**(b)** The paper's own evaluation was on an RTX 4090 24 GB, reporting a speedup
and a memory reduction on FLUX.1-dev at 50 steps (values and conditions in
arXiv:2512.03673; do not copy them without the hardware and step count
attached). So the method was designed and measured on this exact card.

**(b)** One discrepancy worth knowing: the reference repo
https://github.com/feice-huang/ConvRot now states it requires an
"NVFP4-capable GPU (Blackwell / sm_120)" and depends on torchao. That is the
upstream repo's *current* packaging, which moved to NVFP4 storage after the
paper. It is not a contradiction of the paper's 4090 result, and it has no
bearing on Comfy's port, which uses int4/int8 MMA and its own kernels.

### Why the 4090 is unusually well suited to this specific method

**(c)/(b)** INT4 tensor cores are an Ampere/Ada-era feature. A Hopper
microbenchmarking study (arXiv:2501.12084) reports that on Ada, INT4 `mma`
compiles to `IMMA.16832.S4.S4` tensor-core instructions, whereas on Hopper the
same instructions lower to `IMAD` sequences on the CUDA cores. If that holds,
the 4090 is one of the last cards with a genuinely native int4 MMA path, which
is precisely what `convrot_w4a4` targets. I did not verify this against NVIDIA's
PTX ISA docs; treat the Hopper half as (c).

### Reachable ranking on this card

1. **`int8_tensorwise` + convrot (W8A8)** — the safe default. Native IMMA,
   supported since Turing, and the format Comfy actually ships most widely
   (every local `*_int8_convrot.safetensors` uses it). Weights rotated offline,
   activations rotated online and quantized per row inside the fused kernel
   (`backends/eager/quantization.py:1036-1042`). The activation rotation is the
   only extra cost, and it is `O(K · N0)` per token rather than `O(K^2)`.
2. **`asym_w4a8_int8` (W4A8)** — the interesting one, and the best
   quality-per-byte on this card. Int4 weights with fp8 per-group scales, a
   16-level Lloyd-Max codebook, ALS-refined group scales, run through the int8
   GEMM. sm_80+, so native on Ada. **(a)** It has real CUDA kernels, not just an
   eager fallback: `launch_quantize_w4a8_convrot`,
   `launch_w4a8_codebook_gemm_chunked` and `launch_w4a8_codebook_gemv` are
   present in `backends/cuda/_C.abi3.so`, described there as "Fused W4A8
   inference orchestration: ConvRot activation quantization followed by chunked
   decode/GEMM" and a "Fused W4A8 decode GEMV (M<=8): in-register int4+codebook
   dequant, no workspace". It has no published paper of its own; its nearest
   published relative is QAM-W (2605.26339), which independently arrived at
   block-Hadamard plus a single Lloyd-Max codebook fitted to a Gaussian.
3. **`convrot_w4a4`** — the paper's headline path, native on Ada. **(a)** The
   only gate is device capability: `get_disabled_quant_formats`
   (`comfy/ops.py:1710-1723`) disables `int8_tensorwise`, `convrot_w4a4` and
   `asym_w4a8_int8` together, and only when
   `comfy.model_management.supports_int8_compute(device)` is false. sm_89 has
   int8 compute, so all three are enabled on a 4090. W4A4 activations are the
   aggressive end; for a conditioning encoder whose vision tower has register
   outliers, this is where quality risk concentrates.
4. **NVFP4 / MXFP8** — emulated on Ada, as the task states. Also, the most
   relevant GatedDeltaNet result (2609.04098) is NVFP4-only on SM120 hardware,
   so its recipe is not directly portable to a 4090. The nearest reachable
   analogue is `asym_w4a8_int8`, which shares the "int4 weights, small groups,
   fp8 scales" shape.

### The hard constraint to plan around: group size 256 or no fused int8 kernel

**(a)** From `backends/cuda/_C.abi3.so`: every int8 ConvRot CUDA kernel is
256-only, and K must be divisible by 256. The int4 ConvRot path is more
permissive, accepting `{16, 64, 256}`. That means:

- Any layer whose `in_features` is not a multiple of 256 either loses the
  rotation entirely (int8 path, rotation skipped per the comfy-quants rule) or
  drops to eager/Triton.
- For the Qwen3-VL-8B vision tower (1152 and 4304), **no layer qualifies for
  the fused int8 ConvRot kernel**. If that tower is to be quantized with
  rotation on a 4090, the int4 paths (`asym_w4a8_int8` at group 64/16, or
  `convrot_w4a4`) are the only ones with native kernel support.
- For the Qwen3.5-VL 9B linear-attention layers, check each projection's
  `in_features` against 256 before assuming the fast path. The GatedDeltaNet
  projections have unusual widths (`in_proj_a`, `in_proj_b` are typically
  num-heads-wide, far below 256), so several will fall out of the rotated path
  entirely by divisibility. That is not a correctness problem, it is a silent
  loss of the rotation on exactly the small projections.

### Is ConvRot competitive with the alternatives on this card?

For a 4090, yes, and the comparison is not close for the DiT/vision use case:

- Versus **AWQ W4A16** (also in `comfy_kitchen/tensor/awq_w4a16.py`, and present
  locally as `qwen3vl_32b_minimax_h3_w4a16_awq_v*.safetensors`): AWQ needs
  calibration data and leaves activations at 16 bits, so it saves memory but not
  activation bandwidth. ConvRot W8A8/W4A8 is data-free for the rotation and
  quantizes activations.
- Versus **fp8_e4m3 scaled**: native on Ada and simpler, but it has no mechanism
  for row-wise (per-token) outliers, which is the failure mode that matters for
  both DiTs and vision towers.
- Versus **QuaRot/SpinQuant ports**: SpinQuant needs a rotation-training run;
  QuaRot needs a global residual-stream transform that must be re-derived per
  architecture and is the thing MambaQuant showed breaks down on SSMs. ConvRot
  needs neither.

The honest weakness: ConvRot mixes only within a 256-wide (or narrower) window.
An outlier channel whose energy needs to move further than that window cannot be
spread. That is the price of `O(K)`, and it is why the group size is exposed per
layer.

---

## Corrections to the premises in the task

Two of the task's framing assumptions did not survive checking. Both are
recorded here because they change what should be done next.

1. **"Progressively finer rotation groups (256 → 64 → 16, the finest reserved
   for `mlp.linear_fc2`)" is not a design choice.** It is `largest power of 4
   dividing in_features`, intersected with the kernel's accepted set. See Q2.
   There is no graded-sensitivity scheme to study or reproduce.
2. **"Rotation methods rely on computational invariance through the residual
   stream and on rotating matched pairs of projections"** is true of QuaRot and
   SpinQuant but **not** of ConvRot, which is self-inverting inside a single
   linear. The GatedDeltaNet invariance question does not apply to it. See Q3a.

## Summary of what is genuinely unpublished

- **(d)** ConvRot applied to a vision encoder or ViT of any kind. The paper is
  DiT-only (FLUX.1-dev, FLUX.1-schnell; the repo adds Qwen-Image and
  Z-Image-Turbo, all diffusion).
- **(d)** ConvRot, or any regular-Hadamard block rotation, applied to
  Mamba/GatedDeltaNet/hybrid linear attention.
- **(d)** Graded per-module-role rotation group sizes, in any architecture. The
  one paper that studies block size across the full range explicitly defers this
  to future work. Note the Comfy checkpoints do not constitute prior art here,
  since their grading is forced by divisibility.
- **(d)** What happens to rotation quality when the group size is capped by an
  awkward dimension. SigLIP-style towers at 1152 and 4304 can never use the
  paper's recommended 256, and nobody has measured the cost of that.
- **(d)** The `asym_w4a8_int8` design (ConvRot + Lloyd-Max codebook + kurtosis
  gate + ALS group scales). It has published relatives but no paper of its own.
- **(d)** Any Comfy-Org writeup of ConvRot. The method's provenance survives
  only as a docstring line in
  `comfy_kitchen/tensor/convrot_w4a4.py:5`.

## Things to verify before relying on them

1. The arithmetic explanation of the ladder, against an actual prompt-expander
   checkpoint. Parse the safetensors header and read the per-layer
   `comfy_quant` byte ranges; the script pattern is in this session's history
   and takes seconds even on a 27 GB file. Prediction to test: every
   `convrot_groupsize` equals the largest power of 4 dividing that layer's
   `in_features`, capped at 256. Any layer that violates it is a deliberate
   choice worth asking about.
2. The INT8-versus-NVFP4 claim from 2609.04098, which came from a search
   summary rather than the paper body.
3. The Hopper INT4 lowering claim, against NVIDIA's PTX ISA documentation.
4. Whether the fused CUDA kernel (`backends/cuda/_C.abi3.so`) implements the
   rotation as a fast `log4(N0)`-stage butterfly or as a dense `N0 x N0`
   matmul. The eager reference uses a dense matmul
   (`tensor/int8_utils.py:72`), and a dense `N0=256` block costs `N0/N` of the
   GEMM, which is material at the Qwen3-VL vision hidden size of 1152.

## Sources

Primary, read directly:
- https://arxiv.org/abs/2512.03673 and https://arxiv.org/html/2512.03673v1 — ConvRot
- https://arxiv.org/html/2510.04547v5 — RegCache, vision encoder activation quantization
- https://arxiv.org/html/2501.13484v2 — MambaQuant
- https://arxiv.org/html/2609.04098 — NVFP4 W4A4 for Gated DeltaNet hybrids
- https://arxiv.org/html/2608.27513 — DAMP, recurrent-state quantization
- https://arxiv.org/html/2601.22347v2 — block rotations, block-size study
- https://arxiv.org/html/2509.21302 — QuantVGGT
- https://github.com/pytorch/ao/issues/4695 — torchao ConvRot request
- https://github.com/feice-huang/ConvRot — reference implementation
- https://github.com/Comfy-Org/comfy-quants — format docs
- https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot — shipped build description

Named but not read in depth:
- https://arxiv.org/abs/2404.00456 QuaRot, https://arxiv.org/abs/2405.16406 SpinQuant
- https://arxiv.org/abs/2605.26339 QAM-W, https://arxiv.org/abs/2605.29843 HARP
- https://arxiv.org/abs/2605.16732 DiRotQ, https://arxiv.org/abs/2605.28803 Ω-QVLA
- https://arxiv.org/abs/2606.06034 quantized Gated DeltaNet matrix inversion
- https://github.com/sgl-project/sglang/pull/38040 SGLang ConvRot INT8 DiT path
