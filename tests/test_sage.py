"""The sage routing policy.

The policy is one function and two of its three decisions are easy to get
backwards, so they are pinned rather than reasoned about at each reading:
which segments carry a mask, and which tensor the length comes from. The
second is the one that matters -- a Q-derived length reads small for the
single most expensive call in the model, and a sensible threshold would skip
exactly the call the node exists to take.

Shapes here come from reading `comfy/ldm/qwen_image21/model.py` at ComfyUI
c194dd00 on 2026-09-20. They are what the model hands an override, not
measurements of anything.
"""

import pytest

from qwenimage21_explorations import sage


# A 2048x2048 canvas at the release's 16x latent downscale: a 128x128 grid
# consumed unpatched, so 16384 target rows. TEXT is a plausible prompt length;
# the policy does not depend on its exact value.
TARGET_ROWS = (2048 // 16) ** 2
TEXT_ROWS = 300


def test_target_segment_is_taken():
    # block-causal, last segment: unmasked, K is the whole sequence.
    assert sage.should_sage(None, TEXT_ROWS + TARGET_ROWS, 1024, False)


def test_cached_path_is_taken():
    # prefix_cached_attention: Q is the target rows only, K is prefix + target,
    # no mask. The call the node mainly exists for.
    assert sage.should_sage(None, TEXT_ROWS + TARGET_ROWS, 1024, False)


def test_text_segment_is_declined_by_default():
    # block-causal, text segment: carries a tril mask and is short. Either
    # reason alone is enough to decline it.
    assert not sage.should_sage(object(), TEXT_ROWS, 1024, False)


def test_a_mask_declines_a_call_the_length_gate_would_have_taken():
    # The one test where the mask is the ONLY reason to decline. Without it the
    # mask branch is dead: every masked case above is also short, so deleting
    # the branch leaves the suite green. Found by mutating it on 2026-09-20.
    assert not sage.should_sage(object(), TEXT_ROWS + TARGET_ROWS, 1024, False)


def test_sage_masked_opts_in_but_the_length_gate_still_applies():
    assert not sage.should_sage(object(), TEXT_ROWS, 1024, True)
    assert sage.should_sage(object(), TEXT_ROWS + TARGET_ROWS, 1024, True)


def test_a_short_unmasked_call_is_declined():
    # A reference image small enough to sit under the gate is still unmasked;
    # the length is what declines it, so mask=None must not be a free pass.
    assert not sage.should_sage(None, 512, 1024, False)


def test_a_custom_softmax_scale_is_declined_whatever_else_holds():
    # Not plumbed to the kernel. Silently wrong if assumed, so it outranks
    # every other reason to accept.
    assert not sage.should_sage(None, TEXT_ROWS + TARGET_ROWS, 1024, True, scale=0.5)


def test_the_gate_is_inclusive_at_its_own_value():
    assert sage.should_sage(None, 1024, 1024, False)
    assert not sage.should_sage(None, 1023, 1024, False)


def test_a_zero_gate_takes_every_unmasked_call():
    assert sage.should_sage(None, 1, 0, False)


# ---------------------------------------------------------------------------
# The checkpoint's own attention choice


class _Module:
    def __init__(self, config=None):
        self.config = config


class _Model:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        return list(self._modules)


def test_a_plain_checkpoint_asks_for_nothing():
    model = _Model([("blocks.0.attn.comfy_attention", _Module(None))])
    assert sage.attention_configs(model) == []


def test_an_embedded_attention_method_is_reported_with_its_module():
    model = _Model([
        ("blocks.0.attn.comfy_attention", _Module({"attention": "comfy_kitchen_int8"})),
        ("blocks.1.attn.comfy_attention", _Module(None)),
    ])
    assert sage.attention_configs(model) == [
        ("blocks.0.attn.comfy_attention", "comfy_kitchen_int8")]


def test_a_config_without_an_attention_key_is_not_a_claim_on_attention():
    # Other modules carry configs too; only an `attention` entry displaces.
    model = _Model([("blocks.0.mlp", _Module({"quantization": "int8"}))])
    assert sage.attention_configs(model) == []


def test_a_model_that_cannot_be_walked_is_not_an_error():
    # The node passes whatever `diffusion_model` resolved to, including None.
    assert sage.attention_configs(None) == []


@pytest.mark.parametrize("mode", sage.MODE_NAMES)
def test_every_mode_names_kwargs_the_kernel_layer_can_consume(mode):
    # `_kernel` is the one key build_kernel pops rather than forwards; anything
    # else here reaches a kernel as a keyword, and the fork ignores keywords it
    # does not know rather than raising.
    extra = dict(sage.MODES[mode])
    extra.pop("_kernel", None)
    assert all(isinstance(key, str) for key in extra)


def test_auto_passes_no_kernel_kwargs():
    # `auto` means the dispatcher's pick. H3's rotated quantizer was graded on
    # captured H3 activations, not on this model, so it is not folded in here.
    assert sage.MODES["auto"] == {}


# ---------------------------------------------------------------------------
# The override's contract with ComfyUI
#
# Four combinations of layout and skip_output_reshape, each silently wrong if
# transposed the other way -- a wrong one does not raise, it renders noise. The
# kernel is stubbed, so these run on the CPU and touch no CUDA context.

torch = pytest.importorskip("torch")

HEADS, DIM_HEAD = 4, 8


def _identity_kernel(qkv, tensor_layout="NHD", **kwargs):
    """Return V unchanged, so the reshape is the only thing under test."""
    q, k, v = qkv
    qkv.clear()
    return v


def _spy_fallback():
    calls = []

    def func(q, k, v, heads, **kwargs):
        calls.append(kwargs)
        return torch.zeros(1)

    return func, calls


def _override(**kw):
    kw.setdefault("min_kv_len", 0)
    return sage.make_override(_identity_kernel, {}, verbose=False, **kw)


def test_nhd_in_reshapes_back_to_the_packed_layout():
    # [B, L, H*D] in, same out: what a block_causal or cached call passes.
    b, length = 1, 64
    x = torch.randn(b, length, HEADS * DIM_HEAD, dtype=torch.bfloat16)
    func, _ = _spy_fallback()
    out = _override()(func, x, x, x, HEADS)
    assert out.shape == (b, length, HEADS * DIM_HEAD)
    torch.testing.assert_close(out, x)


def test_nhd_in_with_skip_output_reshape_returns_head_major():
    b, length = 1, 64
    x = torch.randn(b, length, HEADS * DIM_HEAD, dtype=torch.bfloat16)
    func, _ = _spy_fallback()
    out = _override()(func, x, x, x, HEADS, skip_output_reshape=True)
    assert out.shape == (b, HEADS, length, DIM_HEAD)
    torch.testing.assert_close(out, x.view(b, length, HEADS, DIM_HEAD).transpose(1, 2))


def test_hnd_in_reshapes_back_to_the_packed_layout():
    b, length = 1, 64
    x = torch.randn(b, HEADS, length, DIM_HEAD, dtype=torch.bfloat16)
    func, _ = _spy_fallback()
    out = _override()(func, x, x, x, HEADS, skip_reshape=True)
    assert out.shape == (b, length, HEADS * DIM_HEAD)
    torch.testing.assert_close(out, x.transpose(1, 2).reshape(b, length, HEADS * DIM_HEAD))


def test_hnd_in_with_skip_output_reshape_passes_through():
    b, length = 1, 64
    x = torch.randn(b, HEADS, length, DIM_HEAD, dtype=torch.bfloat16)
    func, _ = _spy_fallback()
    out = _override()(func, x, x, x, HEADS, skip_reshape=True, skip_output_reshape=True)
    assert out.shape == (b, HEADS, length, DIM_HEAD)
    torch.testing.assert_close(out, x)


def test_the_kv_length_comes_from_k_not_q():
    # The cached path's shape: a short Q against the whole sequence as K. Read
    # from Q this is 16 rows and the gate skips the model's costliest call.
    q = torch.randn(1, 16, HEADS * DIM_HEAD, dtype=torch.bfloat16)
    kv = torch.randn(1, 4096, HEADS * DIM_HEAD, dtype=torch.bfloat16)
    func, calls = _spy_fallback()
    out = sage.make_override(_identity_kernel, {}, min_kv_len=1024, verbose=False)(
        func, q, kv, kv, HEADS)
    assert calls == [], "declined a call whose K is well over the gate"
    assert out.shape == (1, 4096, HEADS * DIM_HEAD)


def test_a_declined_call_reaches_the_previous_override_not_the_default():
    seen = []

    def previous(func, q, k, v, heads, **kwargs):
        seen.append("previous")
        return torch.zeros(1)

    x = torch.randn(1, 16, HEADS * DIM_HEAD, dtype=torch.bfloat16)
    func, calls = _spy_fallback()
    sage.make_override(_identity_kernel, {}, min_kv_len=1024, verbose=False,
                       previous=previous)(func, x, x, x, HEADS)
    assert seen == ["previous"]
    assert calls == [], "went to ComfyUI's default instead of the chained override"


def test_a_kernel_that_raises_degrades_rather_than_killing_the_render():
    def exploding(qkv, **kwargs):
        qkv.clear()
        raise RuntimeError("kernel rejected the shape")

    x = torch.randn(1, 64, HEADS * DIM_HEAD, dtype=torch.bfloat16)
    func, calls = _spy_fallback()
    sage.reset_telemetry()
    out = sage.make_override(exploding, {}, min_kv_len=0, verbose=False)(func, x, x, x, HEADS)
    assert len(calls) == 1, "a kernel failure did not fall back"
    assert out.shape == (1,)


def test_fallback_survives_the_list_being_emptied():
    # sageattn_consume empties the list it is handed. `fallback` must close over
    # the originals, or the release turns a kernel failure into a NameError --
    # the render dies on what should have been a graceful degrade.
    def consume_then_raise(qkv, **kwargs):
        qkv.clear()
        raise RuntimeError("after release")

    x = torch.randn(1, 64, HEADS * DIM_HEAD, dtype=torch.bfloat16)
    func, calls = _spy_fallback()
    sage.reset_telemetry()
    sage.make_override(consume_then_raise, {}, min_kv_len=0, verbose=False)(
        func, x, x, x, HEADS)
    assert len(calls) == 1


def test_an_unexpected_rank_declines_instead_of_raising():
    x = torch.randn(1, 64, HEADS, DIM_HEAD, dtype=torch.bfloat16)  # 4D where skip_reshape=False says 3D
    func, calls = _spy_fallback()
    sage.reset_telemetry()
    sage.make_override(_identity_kernel, {}, min_kv_len=0, verbose=False)(
        func, x, x, x, HEADS)
    assert len(calls) == 1


def test_fp32_is_declined_rather_than_silently_quantized():
    x = torch.randn(1, 64, HEADS * DIM_HEAD, dtype=torch.float32)
    func, calls = _spy_fallback()
    sage.make_override(_identity_kernel, {}, min_kv_len=0, verbose=False)(
        func, x, x, x, HEADS)
    assert len(calls) == 1


def test_a_sliced_q_is_made_contiguous_before_the_kernel():
    # What block_causal_attention hands over: a narrow of the full sequence,
    # whose batch stride is still the unsliced length.
    captured = {}

    def capture(qkv, **kwargs):
        captured["q_contiguous"] = qkv[0].is_contiguous()
        q, k, v = qkv
        qkv.clear()
        return v

    full = torch.randn(2, 128, HEADS, DIM_HEAD, dtype=torch.bfloat16)
    sliced = full[:, 64:].flatten(2)
    assert not sliced.view(2, -1, HEADS, DIM_HEAD).is_contiguous()
    func, _ = _spy_fallback()
    sage.make_override(capture, {}, min_kv_len=0, verbose=False)(
        func, sliced, sliced, sliced, HEADS)
    assert captured["q_contiguous"]


def test_the_kv_length_comes_from_k_in_the_head_major_branch_too():
    # 2.1 always calls with skip_reshape=False, so this branch only runs when
    # another patch hands a call over. Pinned anyway: the HND tests above all
    # use equal q and k lengths, which cannot tell the two tensors apart, and
    # mutating this line left the suite green on 2026-09-20.
    q = torch.randn(1, HEADS, 16, DIM_HEAD, dtype=torch.bfloat16)
    kv = torch.randn(1, HEADS, 4096, DIM_HEAD, dtype=torch.bfloat16)
    func, calls = _spy_fallback()
    out = sage.make_override(_identity_kernel, {}, min_kv_len=1024, verbose=False)(
        func, q, kv, kv, HEADS, skip_reshape=True, skip_output_reshape=True)
    assert calls == [], "declined a call whose K is well over the gate"
    assert out.shape == (1, HEADS, 4096, DIM_HEAD)
