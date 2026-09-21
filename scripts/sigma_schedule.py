#!/usr/bin/env python3
"""Qwen-Image 2.1's sigma schedule: what the reference computes, what ComfyUI uses.

The release ships a flow-match schedule with **dynamic shifting**: mu is an
affine function of the target's latent token count, read from the checkpoint's
`scheduler/scheduler_config.json`. diffusers, sglang, DiffSynth-Studio and
LightX2V all compute it that way and all feed it the TARGET grid only --
reference images never enter it.

ComfyUI instead bakes one constant, `comfy/supported_models.py::QwenImage21`'s
`sampling_settings["shift"]`, which equals the computed mu at 1024x1024 and
drifts either way from there. It also has no `shift_terminal`, which the
config sets, so its schedule runs to zero instead of the stretched terminal.

Neither is reachable from a widget on the stock graph, but the stock
`ModelSamplingFlux` node gets the dynamic half exactly right if you give it the
values `--widgets` prints: its hardcoded token count happens to match 2.1's
latent grid, and only its shift bounds are Flux's.

Usage:
  python scripts/sigma_schedule.py            # mu by canvas
  python scripts/sigma_schedule.py --widgets  # values for ModelSamplingFlux
  python scripts/sigma_schedule.py --sigmas   # full schedules, needs ComfyUI importable
"""

from __future__ import annotations

import argparse
import math
import pathlib

# <models>/Qwen-Image-2.1/scheduler/scheduler_config.json
BASE_LEN, MAX_LEN = 256, 8192
BASE_SHIFT, MAX_SHIFT = 0.5, 0.9
SHIFT_TERMINAL = 0.02
# comfy/supported_models.py::QwenImage21.sampling_settings["shift"]
COMFY_SHIFT = 0.69
# comfy_extras/nodes_model_advanced.py::ModelSamplingFlux.patch, hardcoded
NODE_X1, NODE_X2 = 256, 4096

CANVASES = [(512, 512), (768, 768), (1024, 768), (1024, 1024),
            (1328, 1328), (1536, 1536), (1664, 928), (2048, 2048)]


def seq_len(w: int, h: int) -> int:
    """Target latent tokens. The VAE is 16x and 2.1 consumes latents unpatched."""
    return (w // 16) * (h // 16)


def reference_mu(w: int, h: int) -> float:
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_LEN - BASE_LEN)
    return seq_len(w, h) * slope + BASE_SHIFT - slope * BASE_LEN


def node_widgets() -> tuple[float, float]:
    """base_shift / max_shift making the stock node reproduce the reference mu."""
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_LEN - BASE_LEN)
    base = BASE_SHIFT - slope * BASE_LEN + slope * NODE_X1
    return base, base + slope * (NODE_X2 - NODE_X1)


def reference_sigmas(steps: int, w: int, h: int, terminal: bool = True) -> list[float]:
    mu = reference_mu(w, h)
    s = [1.0 - i * (1.0 - 1.0 / steps) / (steps - 1) for i in range(steps)]
    s = [math.exp(mu) / (math.exp(mu) + (1 / x - 1)) for x in s]
    if terminal:
        scale = (1 - s[-1]) / (1 - SHIFT_TERMINAL)
        s = [1 - (1 - x) / scale for x in s]
    return s + [0.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widgets", action="store_true")
    ap.add_argument("--sigmas", action="store_true")
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()

    if args.widgets:
        base, mx = node_widgets()
        print("Stock ModelSamplingFlux node, to reproduce 2.1's dynamic mu:")
        print(f"  base_shift = {base:.4f}")
        print(f"  max_shift  = {mx:.4f}")
        print("  width / height = the canvas you are sampling at")
        print("\nIt cannot supply shift_terminal; nothing in ComfyUI implements that.")
        return 0

    if args.sigmas:
        # This repo sits at ComfyUI/custom_nodes/<repo>/, so the root is three up.
        import sys
        root = str(pathlib.Path(__file__).resolve().parents[3])
        if root not in sys.path:
            sys.path.insert(0, root)
        import comfy.model_sampling as ms
        import comfy.samplers

        class Cfg:
            sampling_settings = {"shift": COMFY_SHIFT}

        m = ms.ModelSamplingFlux(Cfg())
        comfy_sig = [float(x) for x in comfy.samplers.simple_scheduler(m, args.steps)]
        for w, h in ((1024, 1024), (1328, 1328)):
            ref = reference_sigmas(args.steps, w, h)
            worst = max(abs(a - b) for a, b in zip(comfy_sig, ref))
            no_term = reference_sigmas(args.steps, w, h, terminal=False)
            worst_nt = max(abs(a - b) for a, b in zip(comfy_sig, no_term))
            print(f"{w}x{h} at {args.steps} steps: max|comfy - reference| = {worst:.4f}"
                  f"   (ignoring shift_terminal: {worst_nt:.4f})")
            print(f"   tail  comfy {[round(x,4) for x in comfy_sig[-3:]]}"
                  f"  reference {[round(x,4) for x in ref[-3:]]}")
        return 0

    print(f"{'canvas':>12}{'latent':>10}{'tokens':>8}{'reference mu':>14}{'comfy':>8}{'delta':>9}")
    for w, h in CANVASES:
        mu = reference_mu(w, h)
        print(f"{f'{w}x{h}':>12}{f'{w//16}x{h//16}':>10}{seq_len(w,h):>8}"
              f"{mu:>14.4f}{COMFY_SHIFT:>8.2f}{mu - COMFY_SHIFT:>+9.4f}")
    print("\nmu is the exponential shift's exponent, so the sampler sees exp(mu).")
    print("The reference ties it to the TARGET grid only; reference images never enter it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
