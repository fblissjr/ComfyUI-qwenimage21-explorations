"""The sigma schedule Qwen-Image 2.1 ships, built the way the release builds it.

The checkpoint's `scheduler/scheduler_config.json` asks for a flow-match Euler
schedule with **dynamic shifting** and a **terminal stretch**. diffusers,
sglang, DiffSynth-Studio and LightX2V all honour both. ComfyUI honours neither:
its `supported_models.py::QwenImage21` carries one constant shift, correct at
1024x1024 and drifting either way from there, and the string `shift_terminal`
appears nowhere in it.

Both halves are ordinary arithmetic on the sigma vector, which is why this
module is twenty lines and needs no model. `scripts/sigma_schedule.py` prints
the difference against what ComfyUI would otherwise produce.

Two things worth stating because they are easy to get wrong:

* **mu is derived from the TARGET's latent grid and nothing else.** Not the
  reference images, not their count or shapes, not the encoder, not the step
  count. All four implementations were read on this and all four agree. The
  reference latents share the joint sequence but are static context -- prefilled
  once and never denoised -- so they are not on the trajectory this schedule
  governs.
* **The stretch happens before the trailing zero, not after.** diffusers
  stretches the computed curve so its last sigma is the terminal, then appends
  a zero (`scheduling_flow_match_euler_discrete.py::stretch_shift_to_terminal`,
  then its `torch.cat`). Doing it the other way round is **not** a harmless
  no-op: the trailing zero itself becomes the terminal, so the schedule never
  reaches zero and the sampler stops short of full denoise. `TERMINAL_MODES`
  makes that a named choice instead of an ordering accident, and
  `stretch_to_terminal` refuses a curve that already ends at zero so the
  accident cannot be made silently.

The constants below are the checkpoint's. They are defaults rather than
literals at the call site so a checkpoint that declares different ones can be
served by the same code.
"""

from __future__ import annotations

import math

#: `<models>/Qwen-Image-2.1/scheduler/scheduler_config.json`.
BASE_SEQ_LEN = 256
MAX_SEQ_LEN = 8192
BASE_SHIFT = 0.5
MAX_SHIFT = 0.9
SHIFT_TERMINAL = 0.02
#: The VAE is 16x and 2.1 consumes latents unpatched, so one latent cell is one token.
VAE_SCALE = 16


def latent_tokens(width: int, height: int) -> int:
    """Target tokens for a canvas in pixels."""
    return (width // VAE_SCALE) * (height // VAE_SCALE)


def dynamic_mu(
    seq_len: int,
    *,
    base_seq_len: int = BASE_SEQ_LEN,
    max_seq_len: int = MAX_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
) -> float:
    """The shift exponent, affine in the target's token count."""
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return seq_len * slope + base_shift - slope * base_seq_len


#: How the schedule ends. `release` is what the checkpoint asks for.
TERMINAL_MODES = ("release", "off", "stop_short")


def stretch_to_terminal(curve: list[float], terminal: float) -> list[float]:
    """Stretch a CURVE so its last value is `terminal`. No trailing zero yet.

    Raises on a curve that already ends at zero, which is the ordering mistake
    this function exists to make impossible: stretching after the zero is
    appended turns the zero itself into `terminal`, and the sampler then stops
    short of full denoise. `schedule(terminal_mode="stop_short")` is how to ask
    for that on purpose.
    """
    if not curve:
        return curve
    if curve[-1] <= 0.0:
        raise ValueError(
            "stretch_to_terminal takes the computed curve, before the trailing zero. "
            "A curve ending at zero stretches to one ending at the terminal, which "
            'stops the sampler short of full denoise; ask for terminal_mode="stop_short" '
            "if that is what you want."
        )
    if curve[-1] >= 1.0:
        return list(curve)  # single step: no span to stretch
    scale = (1.0 - curve[-1]) / (1.0 - terminal)
    return [1.0 - (1.0 - s) / scale for s in curve]


def schedule(
    steps: int,
    seq_len: int,
    *,
    denoise: float = 1.0,
    shift_terminal: float | None = SHIFT_TERMINAL,
    terminal_mode: str = "release",
    **mu_kwargs,
) -> list[float]:
    """The full sigma vector, trailing zero included.

    `denoise` follows core's `BasicScheduler`: build a longer schedule and keep
    its tail, so a partial denoise starts part-way down the same curve rather
    than on a differently shaped one.

    `terminal_mode` selects how it ends: `release` stretches the curve and then
    appends zero, as the checkpoint asks; `off` appends zero to the unstretched
    curve, which is what ComfyUI does; `stop_short` compresses the whole
    schedule so its FINAL value is the terminal and it never reaches zero.
    """
    if terminal_mode not in TERMINAL_MODES:
        raise ValueError(f"terminal_mode must be one of {TERMINAL_MODES}, got {terminal_mode!r}")
    if steps < 1 or denoise <= 0.0:
        return []
    total = steps if denoise >= 1.0 else int(steps / denoise)
    mu = math.exp(dynamic_mu(seq_len, **mu_kwargs))

    if total == 1:
        curve = [1.0]
    else:
        step = (1.0 - 1.0 / total) / (total - 1)
        curve = [1.0 - i * step for i in range(total)]
    curve = [mu / (mu + (1.0 / s - 1.0)) for s in curve]

    if not shift_terminal or terminal_mode == "off":
        out = curve + [0.0]
    elif terminal_mode == "release":
        out = stretch_to_terminal(curve, shift_terminal) + [0.0]
    else:  # stop_short: closed form of stretching the curve WITH its zero
        out = [1.0 - (1.0 - s) * (1.0 - shift_terminal) for s in curve + [0.0]]

    return out[-(steps + 1):]
