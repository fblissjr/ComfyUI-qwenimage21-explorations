"""Exercise the sage override at Qwen-Image 2.1's real call shapes.

The unit tests stub the kernel, so they prove the policy and the reshape
contract and nothing about the kernel. This runs the real one, on a GPU, at
the two shapes the model actually produces -- the prefix-cached call and the
block-causal target segment -- and prints what it finds beside torch SDPA.

It is a smoke test, not a benchmark and not an accuracy grade: one shape class,
one process, no warmup discipline, and SDPA in the same dtype rather than a
high-precision reference. A number it prints says the kernel ran and produced
something of the right shape and magnitude. It does not say the kernel is
accurate enough to ship a render on, which is a question for an in-pipeline
A/B at a fixed seed.

    python scripts/smoke_sage.py
    python scripts/smoke_sage.py --seq 16384      # a 2048x2048 canvas
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch                                                  # noqa: E402
import torch.nn.functional as F                               # noqa: E402

from qwenimage21_explorations import sage                     # noqa: E402

# 2.1's DiT: 32 heads, head_dim 128, one attention per block, bf16.
HEADS, HEAD_DIM = 32, 128


def reference(q, k, v, mask=None):
    """SDPA over the same tensors, head-major, in the input dtype."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2)


def run(name, q_len, kv_len, mask=None, mode="auto", device="cuda"):
    torch.manual_seed(0)
    shape = lambda n: (1, n, HEADS * HEAD_DIM)  # noqa: E731
    q = torch.randn(shape(q_len), dtype=torch.bfloat16, device=device)
    kv = torch.randn(shape(kv_len), dtype=torch.bfloat16, device=device)

    kernel_fn, kernel_kwargs = sage.build_kernel(mode)
    override = sage.make_override(kernel_fn, kernel_kwargs, min_kv_len=0,
                                  sage_masked=mask is not None, verbose=False)

    def default_attention(q, k, v, heads, **kwargs):
        raise AssertionError(f"{name}: the override declined a call it should have taken")

    got = override(default_attention, q, kv, kv, HEADS, mask=mask)

    want = reference(q.view(1, q_len, HEADS, HEAD_DIM),
                     kv.view(1, kv_len, HEADS, HEAD_DIM),
                     kv.view(1, kv_len, HEADS, HEAD_DIM),
                     mask=mask).reshape(1, q_len, HEADS * HEAD_DIM)

    assert got.shape == want.shape, f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    err = (got.float() - want.float()).abs()
    denom = want.float().abs().mean().clamp_min(1e-6)
    print(f"  {name:<34} q={q_len:<6} kv={kv_len:<6} "
          f"max_abs={err.max():.4f} mean_abs/mean_|ref|={err.mean() / denom:.4f}")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096,
                    help="target image rows; 16384 is a 2048x2048 canvas")
    ap.add_argument("--prefix", type=int, default=300, help="text rows ahead of the target")
    ap.add_argument("--mode", default="auto", choices=sage.MODE_NAMES)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device; this script needs the card the kernel targets")

    import sageattention

    print(f"sage {getattr(sageattention, '__version__', 'unknown')}, "
          f"mode {args.mode!r}, {torch.cuda.get_device_name(0)}")

    before = sageattention.get_dispatch_counts()
    total = args.prefix + args.seq

    print("\nshapes the model produces:")
    # prefix_cached_attention: target rows query the whole sequence, no mask.
    run("cached: target vs whole sequence", args.seq, total, mode=args.mode)
    # block_causal_attention, last segment: same, and how it runs cold.
    run("block-causal: target segment", args.seq, total, mode=args.mode)
    # block_causal_attention, text segment: short, and the one that carries a mask.
    text_mask = torch.ones((args.prefix, args.prefix), dtype=torch.bool,
                           device="cuda").tril()
    run("block-causal: text segment (masked)", args.prefix, args.prefix,
        mask=text_mask, mode=args.mode)

    after = sageattention.get_dispatch_counts()
    delta = {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)}
    print(f"\ndispatch counts this run: {delta}")
    if sum(delta.values()) != 3:
        sys.exit(f"expected 3 sage dispatches, saw {sum(delta.values())} -- "
                 "something fell back silently")
    print("every call reached sage.")


if __name__ == "__main__":
    main()
