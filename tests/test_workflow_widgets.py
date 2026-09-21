"""The generator's widget lists must match the nodes they describe.

Widget values in a graph are positional, so the generator hand-lists the widget
order per node type. For our own nodes that duplicates the schema, and on
2026-09-20 adding an input to one node left the generator a value short --
caught only by running the workflow check against a live server, which is not
the check anyone runs by reflex.

Needs ComfyUI importable, because a node schema is defined in its terms. Skips
where it is not, like the other checks here that need something external.
"""

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
#: Only these render as widgets; a combo does too. Sockets are link-only.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


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


def graph_node(kind):
    """The node of that type in a generated graph, or None."""
    import json
    doc = json.loads((REPO / "example_workflows/qwen_image_2.1_edit_heylook_pe.json").read_text())
    return next((n for n in doc["nodes"] if n["type"] == kind), None)


@pytest.mark.parametrize("attr", ["PEExpand", "EncodeStructured", "Sigmas", "Canvas"])
def test_generated_graph_carries_every_output_the_node_declares(nodes, attr):
    """A short output list still validates slot by slot, so it needs its own check.

    On 2026-09-20 the node gained `system_source` and the graph kept six
    outputs; the widget check could not see it and the per-slot type check had
    nothing to compare the seventh against.
    """
    cls = getattr(nodes, attr)
    node = graph_node(cls.define_schema().node_id)
    if node is None:
        pytest.skip("not used by the example graphs")
    # A socket with no display name shows its type, which is what the graph carries.
    declared = [o.display_name or o.io_type for o in cls.define_schema().outputs]
    assert [o["name"] for o in node["outputs"]] == declared


@pytest.mark.parametrize("attr", ["PEExpand", "EncodeStructured", "Sigmas", "Canvas"])
def test_generator_widget_list_matches_the_node(nodes, attr):
    cls = getattr(nodes, attr)
    node_id = cls.define_schema().node_id
    declared = generator_widgets().get(node_id)
    if declared is None:
        pytest.skip(f"{node_id} is not used by the example graphs")
    assert declared == schema_widgets(cls), (
        f"{node_id}: the generator lists {declared}, the node declares {schema_widgets(cls)}"
    )
