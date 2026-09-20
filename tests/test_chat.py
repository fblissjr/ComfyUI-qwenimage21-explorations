"""The chat string is the contract on the ComfyUI path, so its exact bytes matter.

ComfyUI never reads the checkpoint's chat_template.jinja; it uses its own
bundled tokenizer. These assertions pin the two places ComfyUI's default
template diverges from the trained format.
"""

from qwenimage21_explorations.chat import (
    Conversation,
    render_generation_prompt,
    render_training_row,
    assistant_span,
)


def conv(thinking=True, n=0, brief="a corgi"):
    return Conversation(system="SYS", thinking=thinking).with_brief(brief, n_images=n)


def test_generation_prompt_ends_with_open_think():
    # The trained format leaves <think> OPEN and the model continues inside it.
    # ComfyUI's tokenizer appends nothing here, which is the divergence.
    out = render_generation_prompt(conv(thinking=True))
    assert out.endswith("<|im_start|>assistant\n<think>\n")
    assert "</think>" not in out


def test_thinking_off_matches_template_whitespace():
    # The template emits a blank line inside and two newlines after;
    # ComfyUI emits "<think>\n</think>\n".
    out = render_generation_prompt(conv(thinking=False))
    assert out.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_starts_with_im_start_so_comfy_skips_its_template():
    assert render_generation_prompt(conv()).startswith("<|im_start|>")


def test_images_come_first_and_in_order():
    out = render_generation_prompt(conv(n=3, brief="BRIEF"))
    user = out.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    assert user.count("<|image_pad|>") == 3
    # every vision block precedes the brief; reordering re-points <imageN> refs
    assert user.index("BRIEF") > user.rindex("<|vision_end|>")


def test_no_vision_block_without_images():
    assert "<|vision_start|>" not in render_generation_prompt(conv(n=0))


def test_training_row_closes_the_think_block():
    row = render_training_row(conv(thinking=True), "reasoning", '{"a": 1}')
    assert "<think>\nreasoning\n</think>\n\n" in row
    assert row.endswith('{"a": 1}<|im_end|>\n')


def test_assistant_span_excludes_the_system_prompt():
    # A statistic pooled over all tokens is dominated by the system prompt,
    # which is identical in every row. The span is what a mask filters to.
    c = Conversation(system="S" * 5000, thinking=True).with_brief("b")
    row = render_training_row(c, "t", "{}")
    start, end = assistant_span(c, "t", "{}")
    assert "S" * 5000 not in row[start:end]
    assert end == len(row)
