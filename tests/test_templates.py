"""System-prompt resolution, including where a server preset sits in the order.

Nothing covered this before today, which `docs/wiki/stages.md` recorded as
"nothing" in its guard column. The preset's position is the part with a real
cost behind it: it beats the checkpoint, so a general-purpose preset silently
replaces the trained contract unless the source is reported.
"""

import pytest

from qwenimage21_explorations import templates


@pytest.fixture
def ckpt(tmp_path):
    (tmp_path / "system_prompt.txt").write_text("FROM CHECKPOINT", encoding="utf-8")
    return tmp_path


@pytest.fixture
def tpl(tmp_path):
    p = tmp_path / "v.md"
    p.write_text("---\ntask: t2i\n---\n\nFROM TEMPLATE", encoding="utf-8")
    return p


def test_raw_text_wins_over_everything(ckpt, tpl):
    r = templates.resolve_with_preset(explicit_text="RAW", template_path=tpl,
                                      preset_text="FROM PRESET", ckpt_dir=ckpt)
    assert (r.text, r.source) == ("RAW", "explicit")


def test_a_local_template_beats_a_preset(ckpt, tpl):
    r = templates.resolve_with_preset(template_path=tpl, preset_text="FROM PRESET", ckpt_dir=ckpt)
    assert (r.text, r.source) == ("FROM TEMPLATE", "template")


def test_a_preset_beats_the_checkpoint(ckpt):
    """The deliberate one: picking a preset is the point of picking it."""
    r = templates.resolve_with_preset(preset_text="FROM PRESET", ckpt_dir=ckpt)
    assert (r.text, r.source) == ("FROM PRESET", "preset")


def test_the_checkpoint_is_the_floor(ckpt):
    r = templates.resolve_with_preset(ckpt_dir=ckpt)
    assert (r.text, r.source) == ("FROM CHECKPOINT", "checkpoint")


def test_a_blank_preset_does_not_displace_the_checkpoint(ckpt):
    r = templates.resolve_with_preset(preset_text="   ", ckpt_dir=ckpt)
    assert r.source == "checkpoint"


def test_nothing_at_all_sends_no_system_prompt():
    """Chosen by the owner: run with the server's own template rather than refuse.

    The app never sends a checkpoint, so a preset without a system prompt used
    to fail the whole run here.
    """
    r = templates.resolve_with_preset()
    assert (r.text, r.source) == ("", "none")


def test_a_preset_without_a_system_prompt_overrides_nothing(ckpt):
    assert templates.resolve_with_preset(preset_text="", ckpt_dir=ckpt).source == "checkpoint"
    assert templates.resolve_with_preset(preset_text="").source == "none"


def test_the_checkpoint_copy_is_preferred_over_a_sibling_file(ckpt):
    """resolve's own contract, unchanged: the prompt that ships with the weights."""
    assert templates.from_checkpoint(ckpt).source == "checkpoint"
    with pytest.raises(FileNotFoundError, match="not interchangeable"):
        templates.from_checkpoint(ckpt / "nope")
