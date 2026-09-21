"""ComfyUI nodes for the Qwen-Image 2.1 prompt expanders.

These wrap the pure functions in `qwenimage21_explorations.{chat,answer,templates}`
and add nothing of their own, so the harness behaves identically here, on the
heylook/MLX backend, and in the offline scripts.
"""

from __future__ import annotations

from pathlib import Path

import math

import comfy.model_management
import comfy.utils
import node_helpers
import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .. import answer as answer_mod
from .. import chat, sigmas as sigmas_mod, templates
from ..backends import heylook
from ..profiles import GREEDY, PROFILES

CATEGORY = "QwenImage21/PE"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATES_DIR = _REPO_ROOT / "templates"


class PESystemPrompt(io.ComfyNode):
    """Resolve a task's system prompt and its reference sampling settings."""

    @classmethod
    def define_schema(cls):
        local = ["(none)"] + sorted(templates.local_templates(_TEMPLATES_DIR))
        return io.Schema(
            node_id="QwenImage21PESystemPrompt",
            display_name="Qwen-Image 2.1 PE System Prompt",
            category=CATEGORY,
            description=(
                "Resolves the expander's system prompt. Prefers the checkpoint's own "
                "system_prompt.txt, which is what the weights were trained against."
            ),
            inputs=[
                io.Combo.Input("task", options=list(templates.TASKS), default="t2i"),
                io.String.Input(
                    "checkpoint_dir", default="",
                    tooltip="Directory holding the checkpoint's system_prompt.txt. Preferred source.",
                ),
                io.Combo.Input(
                    "local_template", options=local, default="(none)", optional=True,
                    tooltip="A variant from templates/. Overrides the checkpoint copy.",
                ),
                io.String.Input(
                    "override", multiline=True, default="", optional=True,
                    tooltip="Raw system prompt text. Wins over everything else.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="system_prompt"),
                io.String.Output(display_name="task"),
                io.String.Output(display_name="source"),
                io.Float.Output(display_name="temperature"),
                io.Float.Output(display_name="top_p"),
                io.Int.Output(display_name="top_k"),
                io.Float.Output(display_name="presence_penalty"),
                io.Int.Output(display_name="max_tokens"),
            ],
        )

    @classmethod
    def execute(cls, task, checkpoint_dir, local_template="(none)", override="") -> io.NodeOutput:
        tpl = None
        if local_template and local_template != "(none)":
            tpl = _TEMPLATES_DIR / f"{local_template}.md"
        sp = templates.resolve(
            explicit_text=override,
            template_path=tpl,
            ckpt_dir=checkpoint_dir.strip() or None,
        )
        p = PROFILES[task]
        return io.NodeOutput(
            sp.text, task, sp.source,
            p["temperature"], p["top_p"], p["top_k"], p["presence_penalty"], p["max_tokens"],
        )


class PEPrompt(io.ComfyNode):
    """Build the chat string the expander was trained on."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21PEPrompt",
            display_name="Qwen-Image 2.1 PE Prompt",
            category=CATEGORY,
            description=(
                "Renders system + images + brief + the generation prompt. Starts with "
                "<|im_start|> so ComfyUI passes it through untouched, and ends with an "
                "OPEN <think> block when thinking is on, matching the trained format."
            ),
            inputs=[
                io.String.Input("system_prompt", multiline=True, default=""),
                io.String.Input("brief", multiline=True, dynamic_prompts=False, default="",
                                tooltip="The user's raw request, in any language. Kept verbatim."),
                io.Image.Input("images", optional=True,
                               tooltip="Wire the SAME batch to Generate Text. Images go first, in order."),
                io.Boolean.Input("thinking", default=True,
                                 tooltip="On is the trained default; both official runners pass enable_thinking=True."),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Int.Output(display_name="n_images"),
            ],
        )

    @classmethod
    def execute(cls, system_prompt, brief, images=None, thinking=True) -> io.NodeOutput:
        n = 0 if images is None else int(images.shape[0])
        conv = chat.Conversation(system=system_prompt, thinking=thinking).with_brief(brief, n_images=n)
        return io.NodeOutput(chat.render_generation_prompt(conv), n)


class PEParse(io.ComfyNode):
    """Parse the answer and grade it against the task's contract."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21PEParse",
            display_name="Qwen-Image 2.1 PE Parse",
            category=CATEGORY,
            description=(
                "Splits the thinking trace, parses the JSON, and checks the mode-dependent "
                "contract. parse_ok is syntax only; contract_ok is the gate that catches a "
                "model still emitting valid JSON while breaking the rules."
            ),
            inputs=[
                io.String.Input("generated_text", multiline=True, default=""),
                io.Combo.Input("task", options=list(templates.TASKS), default="t2i"),
                io.Int.Input("n_images", default=0, min=0, max=16),
            ],
            outputs=[
                io.String.Output(display_name="rewritten_prompt"),
                io.String.Output(display_name="wh_ratio"),
                io.String.Output(display_name="ratio_follow"),
                io.String.Output(display_name="thinking"),
                io.Boolean.Output(display_name="parse_ok"),
                io.Boolean.Output(display_name="contract_ok"),
                io.String.Output(display_name="violations"),
            ],
        )

    @classmethod
    def execute(cls, generated_text, task, n_images) -> io.NodeOutput:
        a = answer_mod.parse_and_grade(generated_text, task=task, n_images=n_images)
        return io.NodeOutput(
            a.rewritten_prompt, a.wh_ratio, a.ratio_follow, a.thinking,
            a.parse_ok, a.contract_ok, ", ".join(a.violations),
        )


# ---------------------------------------------------------------------------
# The conditioning encoder, opened up


CANONICAL_SYSTEM = chat.ENCODER_SYSTEM


class EncodeStructured(io.ComfyNode):
    """`TextEncodeQwenImage21` with the parts core fixes exposed. A research surface."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21EncodeStructured",
            display_name="Qwen-Image 2.1 Encode (structured)",
            category=CATEGORY,
            description=(
                "The stock encode node with its fixed parts opened up: the system turn, whether "
                "vision tokens stay in the conditioning, and which reference sets the canvas. "
                "The defaults reproduce the stock node. Changing the system prompt is "
                "OFF-DISTRIBUTION -- every implementation of 2.1 sends the same one."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Vae.Input("vae", optional=True,
                             tooltip="Without it the references condition through the text encoder alone and no reference latents are produced."),
                io.Int.Input("resolution", default=1024, min=0, max=4096, step=32,
                             tooltip="Target area for every reference, at multiples of 32, aspect preserved. 0 keeps each at its own size. Set it to round(sqrt(canvas_w*canvas_h)) to size references the way sglang does."),
                io.String.Input("system", multiline=True, default=CANONICAL_SYSTEM, optional=True,
                                tooltip="OFF-DISTRIBUTION if changed. Blank falls back to the canonical text, because an absent system turn breaks core's drop."),
                io.Combo.Input("canvas_from", options=["first", "last"], default="first", optional=True,
                               tooltip="Which reference sizes the emitted latent. Core takes the first; diffusers and LightX2V take the last."),
                io.Boolean.Input("keep_vision", default=False, optional=True,
                                 tooltip="Keep vision tokens in the conditioning. Core forces this on only when no VAE is wired; forcing it on WITH reference latents is off-distribution."),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    optional=True,
                    tooltip="Reference images, seen by the text encoder and spliced in as VAE latents.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, vae=None, resolution=1024,
                system=CANONICAL_SYSTEM, canvas_from="first", keep_vision=False,
                images: io.Autogrow.Type = None) -> io.NodeOutput:
        ref_latents, images_vl, sizes = [], [], []
        for image in _autogrow_images(images):
            # One resize for both readers: a vision slot covers a fixed group of latents.
            samples = image[:1].movedim(-1, 1)
            if resolution > 0:
                ratio = samples.shape[3] / samples.shape[2]
                width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
                height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
            else:
                width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
            width, height = max(32, width), max(32, height)
            if (width, height) == (samples.shape[3], samples.shape[2]):
                s = image[:1]
            else:
                s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
            sizes.append((width, height))
            rgb = s[:, :, :, :3]
            if s.shape[-1] > 3:
                rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])  # vision sees alpha over white, the vae keeps it
            images_vl.append(rgb)
            if vae is not None:
                ref_latents.append(vae.encode(s))

        keep = keep_vision or not ref_latents
        out = []
        for text in (prompt, negative_prompt):
            tokens = clip.tokenize(chat.render_encoder_prompt(text, len(images_vl), system),
                                   images=images_vl, keep_vision=keep)
            cond = clip.encode_from_tokens_scheduled(tokens)
            if ref_latents:
                cond = node_helpers.conditioning_set_values(cond, {"reference_latents": ref_latents}, append=True)
            out.append(cond)

        latent_w, latent_h = sizes[-1 if canvas_from == "last" else 0] if sizes else (resolution or 1024,) * 2
        latent = torch.zeros([1, 64, latent_h // 16, latent_w // 16],
                             device=comfy.model_management.intermediate_device())
        return io.NodeOutput(out[0], out[1], {"samples": latent})


# ---------------------------------------------------------------------------
# The release's sigma schedule


class Sigmas(io.ComfyNode):
    """The schedule the checkpoint asks for: dynamic shift, and the terminal stretch."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21Sigmas",
            display_name="Qwen-Image 2.1 Sigmas",
            category=CATEGORY,
            description=(
                "Builds the flow-match schedule the checkpoint's scheduler config specifies. "
                "Core carries one constant shift, right at 1024x1024 and drifting either way, "
                "and implements no shift_terminal at all. Wire this into SamplerCustomAdvanced."
            ),
            inputs=[
                io.Latent.Input("latent",
                                tooltip="The canvas being sampled. The shift is read from its shape, so it cannot disagree with what the sampler gets."),
                io.Int.Input("steps", default=25, min=1, max=10000),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Follows core's BasicScheduler: keeps the tail of a longer schedule."),
                io.Float.Input("shift_terminal", default=sigmas_mod.SHIFT_TERMINAL,
                               min=0.0, max=1.0, step=0.001, optional=True,
                               tooltip="The checkpoint's value. 0 disables the stretch, which is what core does."),
                io.Int.Input("base_seq_len", default=sigmas_mod.BASE_SEQ_LEN, min=1, max=1 << 20, optional=True),
                io.Int.Input("max_seq_len", default=sigmas_mod.MAX_SEQ_LEN, min=1, max=1 << 20, optional=True),
                io.Float.Input("base_shift", default=sigmas_mod.BASE_SHIFT, min=0.0, max=100.0, step=0.01, optional=True),
                io.Float.Input("max_shift", default=sigmas_mod.MAX_SHIFT, min=0.0, max=100.0, step=0.01, optional=True),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, latent, steps, denoise=1.0, shift_terminal=sigmas_mod.SHIFT_TERMINAL,
                base_seq_len=sigmas_mod.BASE_SEQ_LEN, max_seq_len=sigmas_mod.MAX_SEQ_LEN,
                base_shift=sigmas_mod.BASE_SHIFT, max_shift=sigmas_mod.MAX_SHIFT) -> io.NodeOutput:
        h, w = latent["samples"].shape[-2:]
        values = sigmas_mod.schedule(
            steps, int(h) * int(w), denoise=denoise,
            shift_terminal=shift_terminal or None,
            base_seq_len=base_seq_len, max_seq_len=max_seq_len,
            base_shift=base_shift, max_shift=max_shift,
        )
        return io.NodeOutput(torch.FloatTensor(values))


class QwenImage21Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        # Append only: saved graphs match widget values by index.
        return [PESystemPrompt, PEPrompt, PEParse, PEExpand, EncodeStructured, Sigmas]


async def comfy_entrypoint() -> QwenImage21Extension:
    return QwenImage21Extension()


# ---------------------------------------------------------------------------
# The expander, as one node


#: heylook serves the two expanders under these names; see the smoke script.
HEYLOOK_MODELS = {"t2i": "Qwen-Image-2.1-PE-T21-mlx", "edit": "Qwen-Image-2.1-PE-I21-mlx"}


def _to_pil(image):
    """One ComfyUI IMAGE frame to PIL, for the heylook request."""
    from PIL import Image

    arr = (image[0] if image.ndim == 4 else image).clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(arr[:, :, :3])


def _autogrow_images(images):
    """Autogrow dict to a list in socket order, first frame of each."""
    images = images or {}
    return [images[n] for n in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])) if images[n] is not None]


class PEExpand(io.ComfyNode):
    """Expand a brief through a prompt expander served by heylook, and grade the answer."""

    @classmethod
    def define_schema(cls):
        local = ["(none)"] + sorted(templates.local_templates(_TEMPLATES_DIR))
        return io.Schema(
            node_id="QwenImage21PEExpand",
            display_name="Qwen-Image 2.1 PE (heylook)",
            category=CATEGORY,
            description=(
                "Sends the brief to a Qwen-Image 2.1 prompt expander on a heylook server and "
                "returns the rewritten prompt. Sampling is sent explicitly from the reference "
                "profile on every request, because neither the server's defaults nor ComfyUI's "
                "reproduce it, and the server's default token cap truncates a long thinking trace."
            ),
            inputs=[
                io.Combo.Input("task", options=list(templates.TASKS), default="t2i",
                               tooltip="t2i for text-to-image, edit when reference images are wired."),
                io.String.Input("base_url", default="http://localhost:8080",
                                tooltip="heylook server. The Anthropic-conformant /v1/messages route is used."),
                io.String.Input("model", default="",
                                tooltip="Blank derives the served name from task. Set it to override."),
                io.Combo.Input("sampling", options=["reference", "greedy"], default="reference",
                               tooltip="reference is the upstream profile. greedy makes two runs of one arm agree, which is what a comparison needs."),
                io.String.Input("brief", multiline=True, dynamic_prompts=False, default="",
                                tooltip="The user's request, in any language. Sent verbatim."),
                io.String.Input("checkpoint_dir", default="", optional=True,
                                tooltip="Directory holding the expander's system_prompt.txt. Preferred source."),
                io.Combo.Input("local_template", options=local, default="(none)", optional=True,
                               tooltip="A variant from templates/. Overrides the checkpoint copy."),
                io.String.Input("system_override", multiline=True, default="", optional=True,
                                tooltip="Raw system prompt. Wins over everything else."),
                io.Boolean.Input("thinking", default=True, optional=True,
                                 tooltip="On is the trained default. heylook takes this as a plain bool."),
                io.Int.Input("timeout", default=900, min=30, max=7200, optional=True,
                             tooltip="Seconds. An edit row with images can run minutes on this backend."),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    optional=True,
                    tooltip="Reference images for edit mode. Sent first in the user turn, in order.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="rewritten_prompt"),
                io.String.Output(display_name="wh_ratio"),
                io.String.Output(display_name="ratio_follow"),
                io.String.Output(display_name="thinking"),
                io.Boolean.Output(display_name="contract_ok"),
                io.String.Output(display_name="violations"),
            ],
        )

    @classmethod
    def execute(cls, task, base_url, model, sampling, brief, checkpoint_dir="", local_template="(none)",
                system_override="", thinking=True, timeout=900,
                images: io.Autogrow.Type = None) -> io.NodeOutput:
        tpl = _TEMPLATES_DIR / f"{local_template}.md" if local_template not in ("", "(none)") else None
        system = templates.resolve(
            explicit_text=system_override,
            template_path=tpl,
            ckpt_dir=checkpoint_dir.strip() or None,
        ).text

        frames = _autogrow_images(images)
        profile = (GREEDY if sampling == "greedy" else PROFILES)[task]
        resp = heylook.generate(
            base_url=base_url,
            model=model.strip() or HEYLOOK_MODELS[task],
            system=system,
            brief=brief,
            images=[_to_pil(f) for f in frames],
            thinking=thinking,
            timeout=timeout,
            **profile,
        )
        graded = answer_mod.parse_and_grade(resp.as_inline(), task=task, n_images=len(frames))
        violations = list(graded.violations)
        if resp.truncated:
            # Reads downstream as unparseable JSON, which looks like a model fault; name it.
            violations.insert(0, "response:truncated")
        return io.NodeOutput(
            graded.rewritten_prompt, graded.wh_ratio, graded.ratio_follow, graded.thinking,
            graded.contract_ok, ", ".join(violations),
        )
