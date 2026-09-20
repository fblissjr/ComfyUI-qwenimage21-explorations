"""Chat-string construction for the Qwen-Image 2.1 prompt expanders.

The expanders are fine-tuned Qwen3.5-VL 9B. ComfyUI ships its own qwen35
tokenizer and template and never reads the checkpoint's `chat_template.jinja`,
so on the ComfyUI path the string built here IS the contract. On the heylook /
MLX path the checkpoint's own jinja applies instead; this module exists so both
backends can be driven from one construction and compared.

Two properties of the trained format that ComfyUI's default template does not
reproduce, both read from the checkpoints' `chat_template.jinja`:

1. With thinking ON (the trained default -- both official runners pass
   `enable_thinking=True`), the generation prompt ends with an OPEN ``<think>``
   block. The model continues inside it. ComfyUI's tokenizer appends nothing
   after ``<|im_start|>assistant\\n`` in that case, leaving the model to open
   the block itself.
2. With thinking OFF the template emits ``<think>\\n\\n</think>\\n\\n`` -- a
   blank line inside and two newlines after. ComfyUI emits
   ``<think>\\n</think>\\n``.

Because the rendered string starts with ``<|im_start|>``, ComfyUI's
``Qwen35ImageTokenizer.tokenize_with_weights`` takes its ``skip_template``
branch and passes the text through untouched, which is what makes this
construction authoritative rather than additive.

Images go FIRST in the user turn, in order: the system prompts address them as
``<image1>``, ``<image2>``, ... and reordering silently re-points every
reference in the rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    n_images: int = 0


@dataclass(frozen=True)
class Conversation:
    """A conversation to render. `thinking` selects the generation-prompt form."""

    system: str
    turns: list[Turn] = field(default_factory=list)
    thinking: bool = True

    def with_brief(self, brief: str, n_images: int = 0) -> "Conversation":
        return Conversation(
            system=self.system,
            turns=[*self.turns, Turn("user", brief, n_images)],
            thinking=self.thinking,
        )


def render_generation_prompt(conv: Conversation) -> str:
    """Render a conversation up to (and including) the generation prompt.

    The result is what gets fed to `CLIP.generate`. It always ends in an
    assistant turn that the model is expected to continue.
    """
    parts: list[str] = []
    if conv.system:
        parts.append(f"{IM_START}system\n{conv.system.strip()}{IM_END}\n")

    for turn in conv.turns:
        if turn.role == "user":
            body = VISION_BLOCK * turn.n_images + turn.content
            parts.append(f"{IM_START}user\n{body}{IM_END}\n")
        elif turn.role == "assistant":
            parts.append(f"{IM_START}assistant\n{turn.content}{IM_END}\n")
        else:
            raise ValueError(f"unexpected role {turn.role!r}")

    # The generation prompt. Mirrors the checkpoints' chat_template.jinja
    # `add_generation_prompt` branch exactly.
    parts.append(f"{IM_START}assistant\n")
    parts.append("<think>\n" if conv.thinking else "<think>\n\n</think>\n\n")
    return "".join(parts)


def render_training_row(conv: Conversation, thinking_trace: str, answer: str) -> str:
    """Render a COMPLETE row: prompt plus a finished assistant turn.

    Used to build calibration corpora, where the row must cover the decode
    regime and not just the prefill. The assistant turn carries the thinking
    trace and the JSON answer as the model actually emits them.
    """
    prefix = render_generation_prompt(conv)
    if conv.thinking:
        # prefix already ends with the open "<think>\n"
        return f"{prefix}{thinking_trace.strip()}\n</think>\n\n{answer.strip()}{IM_END}\n"
    return f"{prefix}{answer.strip()}{IM_END}\n"


def assistant_span(conv: Conversation, thinking_trace: str, answer: str) -> tuple[int, int]:
    """Character span of the assistant turn inside `render_training_row`.

    A calibration statistic pooled over all tokens is dominated by the system
    prompt, which is thousands of tokens and byte-identical across every row.
    Masking the statistic to this span is what makes a corpus carry more signal
    than one repeated document. Returned in characters; callers convert to token
    indices through the real processor, never from the text render (an image is
    one `<|image_pad|>` in text but many tokens in `input_ids`).
    """
    prefix = render_generation_prompt(conv)
    full = render_training_row(conv, thinking_trace, answer)
    return len(prefix), len(full)
