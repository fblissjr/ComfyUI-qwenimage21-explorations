"""The generator's widget lists must match the nodes they describe.

Widget values in a graph are positional, so the generator hand-lists the widget
order per node type. For our own nodes that duplicates the schema, and on
2026-09-20 adding an input to one node left the generator a value short --
caught only by running the workflow check against a live server, which is not
the check anyone runs by reflex.

Needs ComfyUI importable, because a node schema is defined in its terms. Skips
where it is not, like the other checks here that need something external.
"""

import importlib
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
#: Only these render as widgets; a combo does too. Sockets are link-only.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


@pytest.fixture(scope="module")
def nodes():
    root = REPO.parents[1]          # this repo sits at ComfyUI/custom_nodes/<repo>/
    if not (root / "comfy").is_dir():
        pytest.skip("needs ComfyUI alongside this checkout")
    for p in (str(root), str(REPO / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        return importlib.import_module("qwenimage21_explorations.nodes")
    except Exception as exc:                      # comfy present but not importable here
        pytest.skip(f"ComfyUI not importable: {type(exc).__name__}")


def generator_widgets():
    ns: dict = {}
    src = (REPO / "scripts/build_example_workflows.py").read_text()
    body = src[src.index("WIDGETS = {"):]
    exec(body[:body.index("\n}\n") + 3], ns)      # noqa: S102 -- our own file, a literal dict
    return ns["WIDGETS"]


def schema_widgets(node_cls):
    """Widget ids in schema order.

    Every input's class is named `Input`; `io_type` is what distinguishes a
    widget from a socket, and it is the same rule the workflow generator's
    server check applies.
    """
    return [i.id for i in node_cls.define_schema().inputs if i.io_type in WIDGET_TYPES]


@pytest.mark.parametrize("attr", ["PEExpand", "EncodeStructured", "Sigmas"])
def test_generator_widget_list_matches_the_node(nodes, attr):
    cls = getattr(nodes, attr)
    node_id = cls.define_schema().node_id
    declared = generator_widgets().get(node_id)
    if declared is None:
        pytest.skip(f"{node_id} is not used by the example graphs")
    assert declared == schema_widgets(cls), (
        f"{node_id}: the generator lists {declared}, the node declares {schema_widgets(cls)}"
    )
