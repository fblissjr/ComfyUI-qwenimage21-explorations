"""ComfyUI nodes for the Qwen-Image 2.1 prompt expanders.

These wrap the pure functions in `qwenimage21_explorations.{chat,answer,templates}`
and add nothing of their own, so the harness behaves identically here, on the
heylook/MLX backend, and in the offline scripts.
"""

from __future__ import annotations

from pathlib import Path

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .. import answer as answer_mod
from .. import chat, templates
from ..profiles import PROFILES

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


class QwenImage21Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        # Append only: saved graphs match widget values by index.
        return [PESystemPrompt, PEPrompt, PEParse]


async def comfy_entrypoint() -> QwenImage21Extension:
    return QwenImage21Extension()
