"""SageAttention for Qwen-Image 2.1's attention, via ComfyUI's override hook.

Read out of `comfy/ldm/qwen_image21/model.py` on 2026-09-20, at ComfyUI
`c194dd00`. Re-read before trusting any line number here; this model landed
recently and moved twice in the week it was read.

## What the model hands an attention override

2.1 is 32 single-stream blocks, 32 heads, head_dim 128, MHA (no GQA), bf16.
head_dim 128 is SageAttention's native size, so nothing is padded. Each block
has exactly one attention call site (`Attention.forward`), and it reaches
`optimized_attention`, so `transformer_options["optimized_attention_override"]`
is the supported way in.

There are two call shapes, and they are not equally worth taking:

- `block_causal_attention` splits Q into segments and calls attention once per
  segment against the prefix `k[:, :end]`. **Text segments carry a real bool
  mask** (a `tril`); **image segments -- the references and the target -- carry
  `mask=None`**. The target is the last and largest segment and its `end` is
  the full sequence, so it is a plain unmasked attention over everything.
- `prefix_cached_attention` runs instead once the prefix K/V cache is warm:
  one call per block, `mask=None`, Q the target rows only, K/V a fresh
  `cat(prefix, target)`. The cache is on by default (`comfy/model_base.py`,
  `QwenImage21.current_patcher`, enabled whenever the patcher carries no hook
  patches), so on a normal graph this is what most steps run.

Installing an override does **not** turn the prefix cache off. The model's
`hooked` guard lists `blocks_replace`, `post_input`, `single_block` and
`attn1_patch`; an attention override is not among them.

## The policy, and why it is this one

Sage the unmasked large-KV calls, decline everything else to whatever was
already handling it. That is the target image segment on the block-causal path
and the single call per block on the cached path -- the two that carry the
work. The masked text segments are declined by default, and when `sage_masked` opts
them in they go to `sageattn_qk_int8_pv_fp16_triton` rather than to whatever
`sage_mode` names. That started as a correctness guard: the sm89 fp8++ kernel
applied `attn_mask` to its last two K blocks only, which a causal mask -- what
2.1's text segments carry -- falls outside of. **That kernel was fixed on
2026-09-20** (the sage fork's `tests/repros/repro_fp8_mask_window.py` is the
gate), so the routing now stands on a different reason and a weaker one:
Triton skips a K block that is entirely masked and the CUDA kernel has no
equivalent, so at every masked shape measured Triton is both faster and more
accurate. The other CUDA kernels -- `fp8_cuda` (fp32+fp32), `fp16_cuda`, sm80
-- still drop a mask whole; only the fp8++ variant ever implemented one. The
naming trap survives the fix: fp16 *triton* serves a mask, fp16 *cuda* does
not.

**`sage_masked` still buys coverage, not speed.** At the text segments' shape
torch SDPA beats both, and those segments are short either way; leaving them on
ComfyUI's default is correct and the quicker arm. The opt-in exists to put
every call on one kernel when that is what you are testing. Matrix and timings:
the sage fork's `tests/bench/masked_kernel_survey/`.

## Two things this deliberately does not do

**It does not claim the memory saving the H3 pack's forward patch gets.**
`sageattn_consume` frees the float q/k/v as soon as they are quantized, but
only the caller can make that reachable: here `Attention.forward` binds q, k
and v as locals across the whole call, so they stay alive no matter what this
override drops. What consume can free is what this override itself allocated --
the contiguous copies below, when they happen. The originals are out of reach
from an override, which is the difference between this and a forward patch.
The `clone_v` trick from the H3 pack does not apply either: 2.1 projects q, k
and v through separate `Linear`s rather than one fused buffer, so there is no
shared allocation for a clone to break up.

**It does not carry H3's rotated-quantizer default.** `qk_rotate` was graded on
captured H3 activations, not on this model, so `auto` here means what sage's
own dispatcher picks. The rotated mode is exposed as an explicit choice and is
ungraded on 2.1.

## Unmeasured

Nothing here has been run on a GPU. There is no A/B and no speed claim for this
model. What would produce one: the same graph at a fixed seed and canvas with
`sage_mode` at `auto` and at `off`, plus a `get_dispatch_counts()` snapshot
before and after each render, so the arms are distinguishable by something
other than the wall clock. A second pair with `QwenImage21Cache` set to
`device=off` measures the block-causal path instead of the cached one, which is
a different shape wearing the same node's name.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

# mode -> extra kwargs for the kernel call. "auto" passes nothing and lets
# sage's dispatcher choose, which on Ada resolves to fp8++ (fp32+fp16). The
# explicit entries exist so a suspected accuracy problem can be bisected
# without editing code, not because any of them is known better here.
#
# "fp8++" against "fp8" is NOT an isolation of the accumulator: upstream ties
# `fp32+fp16` to a different V quantization scale_max as well, so a delta
# between them is the pair, not the accumulator. Do not report one as the cost
# of fp16 accumulation.
MODES = {
    "auto": {},
    "fp8++": {"pv_accum_dtype": "fp32+fp16"},
    "fp8": {"pv_accum_dtype": "fp32+fp32"},
    "fp8++ rotated (ungraded here)": {"pv_accum_dtype": "fp32+fp16", "qk_rotate": True},
    "fp16": {"pv_accum_dtype": "fp32", "_kernel": "sageattn_qk_int8_pv_fp16_cuda"},
}

# Keywords this fork gained late. A stock SageAttention, or an older build of
# the fork, takes **kwargs and ignores what it does not know -- so naming one
# of these would run as a plain call, with no error and a render that looks
# like every other. Checked against the installed signature instead.
_KEYWORD_FLOOR = {"qk_balance": "v0.7.19", "qk_rotate": "v0.7.20"}

MODE_NAMES = list(MODES)

_FALLBACK_LOGGED = False
_SEEN_FIRE: set = set()


def reset_telemetry():
    """Let the next run report a fallback and re-announce its shapes."""
    global _FALLBACK_LOGGED
    _FALLBACK_LOGGED = False
    _SEEN_FIRE.clear()


def _log_fallback_once(exc):
    global _FALLBACK_LOGGED
    if not _FALLBACK_LOGGED:
        _FALLBACK_LOGGED = True
        logger.warning(
            "[qwen21-sage] sage declined or raised (%s: %s); this call and any "
            "later failure use the previous attention for the rest of the run. "
            "The render continues, just without sage.",
            type(exc).__name__, exc,
        )


def should_sage(mask, kv_len, min_kv_len, sage_masked, scale=None):
    """Does this call go to sage, or back to whatever was handling it?

    `kv_len` is the K sequence length, not Q's -- see the module docstring.
    Kept free of torch so the policy is testable without a GPU or a runtime.
    """
    # A custom softmax scale is not plumbed through to the kernel here. Rare,
    # and wrong silently if assumed.
    if scale is not None:
        return False
    if mask is not None and not sage_masked:
        return False
    return kv_len >= min_kv_len


def attention_configs(diffusion_model):
    """Attention methods the checkpoint asked for, as (module name, method).

    2.1 checkpoints can carry a per-block attention choice, which ComfyUI loads
    into each block's `ComfyAttention.config` and applies as `preferred_attention`.
    An override outranks it: `wrap_attn` consults `optimized_attention_override`
    before `preferred_attention`, and pops the latter before the override is
    ever called -- so the override cannot see what it is displacing and has to
    be asked here, at patch time, instead.

    Empty on a plain bf16 checkpoint, which carries no such config.
    """
    found = []
    for name, module in getattr(diffusion_model, "named_modules", lambda: [])():
        config = getattr(module, "config", None)
        if isinstance(config, dict) and config.get("attention"):
            found.append((name, config["attention"]))
    return found


def build_kernel(mode):
    """Resolve `mode` to `(callable, kwargs)`; raise if the install cannot serve it.

    The callable takes a `[q, k, v]` list it is free to empty. Everything but
    the fp16 diagnostic goes through `sageattn_consume`, which does that; the
    fp16 kernel has no consuming entry point, so its wrapper clears the list
    without the early release and is heavier by exactly that.
    """
    import inspect

    try:
        import sageattention as sa
    except ImportError as exc:
        raise RuntimeError(
            "sageattention is not installed in this ComfyUI's environment. "
            "This node needs the Ada fork built from source."
        ) from exc

    if not hasattr(sa, "sageattn_consume"):
        raise RuntimeError(
            "The installed sageattention has no sageattn_consume(). This node "
            "needs the Ada fork at a version that provides it; a stock "
            "SageAttention install will not work."
        )

    extra = dict(MODES[mode])
    attr = extra.pop("_kernel", None)

    params = inspect.signature(sa.sageattn_qk_int8_pv_fp8_cuda).parameters
    for keyword, since in _KEYWORD_FLOOR.items():
        if keyword in extra and keyword not in params:
            raise RuntimeError(
                f"mode {mode!r} needs a sageattention with {keyword} on "
                f"sageattn_qk_int8_pv_fp8_cuda ({since} or later); the "
                "installed one has no such keyword and would ignore it silently."
            )

    base_kwargs = {
        "is_causal": False,
        # 2.1's image segments are unmasked and ComfyUI passes smooth_k=False on
        # its own sage path. Off also keeps the K-mean pass from allocating a
        # full copy of K on top of the quantized tensors, which is what would
        # eat the release `sageattn_consume` is here for.
        "smooth_k": False,
        **extra,
    }

    if attr is None:
        return sa.sageattn_consume, base_kwargs

    kernel = getattr(sa, attr)

    def call_without_release(qkv, **kw):
        q, k, v = qkv
        qkv.clear()
        return kernel(q, k, v, **kw)

    return call_without_release, base_kwargs


def _last_kernel():
    """The kernel name of the call that just ran, or None.

    Imported at call time rather than when the override is built: importing
    sageattention initializes a CUDA context, and this module is meant to stay
    importable -- and its reshape contract testable -- on a machine with the
    card busy or absent.
    """
    try:
        from sageattention import get_last_dispatched_kernel
    except ImportError:
        return None
    return get_last_dispatched_kernel()


def build_masked_kernel():
    """The kernel masked calls go to, whatever `sage_mode` asks for.

    This used to be a safety rail: until 2026-09-20 the fp8 path's general-mask
    handling was wrong, so letting the mode widget reach it would have been
    letting someone select a broken kernel. That kernel is fixed, so this is
    now a preference rather than a rail -- but the preference is backed both
    ways. Triton skips a K block that is entirely masked and the CUDA kernel
    has no equivalent, so it is faster at every masked shape measured, and it
    quantizes PV to fp16 rather than fp8, so it is also the more accurate arm.
    Measurements: the sage fork's `tests/bench/masked_kernel_survey/`.

    Still not exposed on the mode widget, and that is the judgement call to
    revisit if anyone wants to A/B the two masked kernels: the node list is
    append-only, so a widget added here is permanent, and nothing measured so
    far argues for the other side.
    """
    import sageattention as sa

    def call(qkv, **kw):
        q, k, v = qkv
        qkv.clear()
        kw.pop("pv_accum_dtype", None)      # an fp8 knob; meaningless here
        kw.pop("qk_rotate", None)
        kw.pop("qk_balance", None)
        return sa.sageattn_qk_int8_pv_fp16_triton(q, k, v, **kw)

    return call


def make_override(kernel_fn, kernel_kwargs, min_kv_len=1024, sage_masked=False,
                  verbose=True, previous=None):
    """Build the `optimized_attention_override` closure.

    `previous` preserves an override already on the model, so this composes
    rather than clobbers, and a declined call lands there rather than skipping
    straight to ComfyUI's default.
    """
    import torch

    masked_kernel = build_masked_kernel() if sage_masked else None

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def fallback():
            target = func if previous is None else functools.partial(previous, func)
            return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape, **kwargs)

        want_ndim = 4 if skip_reshape else 3
        if q.ndim != want_ndim or k.ndim != want_ndim or v.ndim != want_ndim:
            _log_fallback_once(ValueError(
                f"expected {want_ndim}D q/k/v for skip_reshape={skip_reshape}, "
                f"got {q.ndim}D/{k.ndim}D/{v.ndim}D"))
            return fallback()

        if skip_reshape:                     # [B, H, L, D]
            b, _, _, dim_head = q.shape
            kv_len = k.shape[2]
            layout = "HND"
        else:                                # [B, L, H*D]
            b, _, dim = q.shape
            dim_head = dim // heads
            kv_len = k.shape[1]
            layout = "NHD"

        if not should_sage(mask, kv_len, min_kv_len, sage_masked, kwargs.get("scale")):
            return fallback()
        if q.dtype not in (torch.bfloat16, torch.float16) or dim_head > 128:
            return fallback()

        if skip_reshape:
            qq, kk, vv = q, k, v
        else:
            qq, kk, vv = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        # The block-causal path hands over `q[:, start:end]`, whose batch stride
        # is the unsliced sequence. The fork's kernels take real strides
        # (csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh) and its quantizers pass them
        # through, so this is a belt rather than a fix -- verified by source
        # read 2026-09-20, not by a run. A batch of one reports contiguous, so
        # the common graph pays nothing for it.
        qq, kk, vv = (t if t.is_contiguous() else t.contiguous() for t in (qq, kk, vv))

        # The list is what `sageattn_consume` empties. `fallback` closes over
        # the ORIGINALS, not these, so the release can never turn a kernel
        # failure into a NameError. What it actually frees is whatever the
        # contiguous copies above allocated; when they were no-ops, nothing --
        # the caller's q/k/v are alive in its own frame either way.
        qkv = [qq, kk, vv]
        del qq, kk, vv
        run = kernel_fn if mask is None else masked_kernel
        try:
            out = run(qkv, attn_mask=mask,
                      **dict(kernel_kwargs, tensor_layout=layout))
        except Exception as exc:
            _log_fallback_once(exc)
            return fallback()

        if verbose:
            key = (_last_kernel(), layout, b, kv_len, heads, dim_head,
                   mask is not None)
            if key not in _SEEN_FIRE:
                _SEEN_FIRE.add(key)
                logger.info(
                    "[qwen21-sage] fired kernel=%s layout=%s b=%d kv_len=%d "
                    "heads=%d head_dim=%d masked=%s",
                    *key)

        if layout == "HND":
            return out if skip_output_reshape else \
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out.transpose(1, 2) if skip_output_reshape else \
            out.reshape(b, -1, heads * dim_head)

    # Marked so anyone inspecting a patched model can tell this override from
    # another pack's, and reach what it wrapped.
    override.qwen21_kernel = "sage"
    override.qwen21_previous = previous
    return override
