"""`PROFILES` claims to be the reference configuration; this checks that it is.

`stages.md` records that nothing read this constant back against upstream, and
on 2026-09-20 the t2i cap had drifted from the reference while the docstring
still claimed to mirror it. A drift here is silent: it reads downstream as the
model behaving differently, not as the harness asking for something else.

Reads `pe_core.py` with `ast` rather than importing it, so the check needs none
of the reference runner's dependencies. Skips where `coderef/` is absent, which
is every machine but the one that made the checkout.
"""

import ast
from pathlib import Path

import pytest

from qwenimage21_explorations.profiles import OVERRIDES, PROFILES, REFERENCE

PE_CORE = Path(__file__).resolve().parents[1] / "coderef/Qwen-Image-2.1/prompt_rewrite/pe_core.py"
#: pe_core names the cap `max_new_tokens`; ours is the wire name.
ALIASES = {"max_new_tokens": "max_tokens"}


def _literal(node):
    return ast.literal_eval(node)


@pytest.fixture(scope="module")
def upstream():
    if not PE_CORE.is_file():
        pytest.skip("needs coderef/Qwen-Image-2.1")
    tree = ast.parse(PE_CORE.read_text())

    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Profile":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    try:
                        defaults[stmt.target.id] = _literal(stmt.value)
                    except ValueError:
                        pass

    profiles = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "PROFILES":
            for key, call in zip(node.value.keys, node.value.values):
                over = {kw.arg: _literal(kw.value) for kw in call.keywords
                        if isinstance(kw.value, ast.Constant)}
                profiles[_literal(key)] = {**defaults, **over}
    assert profiles, "could not read PROFILES out of pe_core.py"
    return profiles


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_reference_matches_upstream_exactly(upstream, task):
    ours, theirs = REFERENCE[task], upstream[task]
    for name, value in theirs.items():
        ours_name = ALIASES.get(name, name)
        if ours_name not in ours:
            continue  # not a sampling knob we send; image_max_pixels lives in the client
        assert ours[ours_name] == value, f"{task}.{ours_name}: ours {ours[ours_name]!r}, upstream {value!r}"


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_we_send_no_knob_upstream_does_not_set(upstream, task):
    theirs = {ALIASES.get(k, k) for k in upstream[task]}
    assert set(PROFILES[task]) <= theirs


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_what_we_send_differs_from_the_reference_only_where_declared(task):
    """An undeclared divergence is drift; a declared one is a decision."""
    declared = OVERRIDES.get(task, {})
    differs = {k for k, v in PROFILES[task].items() if REFERENCE[task].get(k) != v}
    assert differs == set(declared), f"{task}: undeclared {differs - set(declared)}"


def test_every_override_actually_changes_something():
    """A stale override is a comment claiming a difference that is not there."""
    for task, over in OVERRIDES.items():
        for k, v in over.items():
            assert REFERENCE[task][k] != v, f"{task}.{k} override equals the reference"
