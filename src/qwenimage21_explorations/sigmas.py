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
  stretches the schedule so its last *computed* sigma is the terminal, then
  appends a zero
  (`scheduling_flow_match_euler_discrete.py::stretch_shift_to_terminal`, then
  its `torch.cat`). Stretching a schedule that already ends in zero is a no-op
  and silently changes nothing.

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


def schedule(
    steps: int,
    seq_len: int,
    *,
    denoise: float = 1.0,
    shift_terminal: float | None = SHIFT_TERMINAL,
    **mu_kwargs,
) -> list[float]:
    """The full sigma vector, trailing zero included.

    `denoise` follows core's `BasicScheduler`: build a longer schedule and keep
    its tail, so a partial denoise starts part-way down the same curve rather
    than on a differently shaped one.
    """
    if steps < 1 or denoise <= 0.0:
        return []
    total = steps if denoise >= 1.0 else int(steps / denoise)
    mu = math.exp(dynamic_mu(seq_len, **mu_kwargs))

    if total == 1:
        sigmas = [1.0]
    else:
        step = (1.0 - 1.0 / total) / (total - 1)
        sigmas = [1.0 - i * step for i in range(total)]
    sigmas = [mu / (mu + (1.0 / s - 1.0)) for s in sigmas]

    # Stretch first, then append the zero. The other order is a no-op.
    # A single-step schedule ends at 1.0, so there is no span to stretch and
    # the reference's scale factor would be zero; leave it alone.
    if shift_terminal and sigmas[-1] < 1.0:
        scale = (1.0 - sigmas[-1]) / (1.0 - shift_terminal)
        sigmas = [1.0 - (1.0 - s) / scale for s in sigmas]

    return (sigmas + [0.0])[-(steps + 1):]
