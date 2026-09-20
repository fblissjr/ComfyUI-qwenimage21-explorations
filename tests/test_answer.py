"""parse_ok is syntax only. contract_ok is the gate that catches a model still
emitting valid JSON while breaking the mode-dependent rules."""

import pytest

from qwenimage21_explorations.answer import parse, parse_and_grade


def wrap(body, think="reasoning"):
    return f"<think>\n{think}\n</think>\n\n{body}"


def test_clean_t2i_passes():
    a = parse_and_grade(wrap('{"rewritten_prompt":"a corgi","wh_ratio":"16:9"}'), task="t2i")
    assert a.parse_ok and a.contract_ok
    assert a.rewritten_prompt == "a corgi"
    assert a.thinking == "reasoning"


def test_code_fences_are_stripped():
    a = parse_and_grade(
        wrap('```json\n{"rewritten_prompt":"x","wh_ratio":"1:1"}\n```'), task="t2i"
    )
    assert a.parse_ok


def test_truncated_thinking_trace_is_unparseable_not_a_crash():
    # A max_tokens below the trace length truncates mid-thinking. This must read
    # as a parse failure, which is why max_tokens is pinned from PROFILES.
    a = parse_and_grade("<think>\nreasoning that never ends", task="t2i")
    assert not a.parse_ok
    assert "json:unparseable" in a.violations


def test_positive_prompt_alias_accepted():
    a = parse_and_grade(wrap('{"positive_prompt":"x","wh_ratio":"1:1"}'), task="t2i")
    assert a.rewritten_prompt == "x"


@pytest.mark.parametrize(
    "body,task,n,expect",
    [
        # Image Reference Rules: tags mandatory at N>=2, forbidden at N==1.
        ('{"rewritten_prompt":"edit <image1>","wh_ratio":"","ratio_follow":"<image1>"}',
         "edit", 1, "tags:present_at_single_image"),
        ('{"rewritten_prompt":"blend them","wh_ratio":"","ratio_follow":"<image1>"}',
         "edit", 2, "tags:missing_at_multi_image"),
        # wh_ratio and ratio_follow are mutually exclusive on edit.
        ('{"rewritten_prompt":"<image1> x","wh_ratio":"1:1","ratio_follow":"<image1>"}',
         "edit", 2, "ratio:both_set"),
        ('{"rewritten_prompt":"<image1> x","wh_ratio":"","ratio_follow":""}',
         "edit", 2, "ratio:neither_set"),
        # Resolution strings belong in the ratio fields, never in the prompt.
        ('{"rewritten_prompt":"poster 1920x1080","wh_ratio":"16:9"}',
         "t2i", 0, "prompt:carries_resolution"),
        ('{"rewritten_prompt":"x","wh_ratio":"widescreen"}',
         "t2i", 0, "wh_ratio:malformed"),
        ('{"rewritten_prompt":"x","wh_ratio":"1:1","ratio_follow":"<image1>"}',
         "t2i", 0, "ratio_follow:set_on_t2i"),
    ],
)
def test_contract_violations(body, task, n, expect):
    a = parse_and_grade(wrap(body), task=task, n_images=n)
    assert a.parse_ok, "these cases are valid JSON; the point is the contract"
    assert any(v.startswith(expect) for v in a.violations), a.violations


def test_prose_around_the_object_is_tolerated():
    a = parse(wrap('Here you go:\n{"rewritten_prompt":"x","wh_ratio":"1:1"}\nHope that helps.'))
    assert a.parse_ok
