"""The canvas the expander chose, at the area the user chose.

Upstream renders `wh_ratio` from a fixed table of ~4 MP sizes (the model
README's WH_RATIO_TO_SIZE). The owner chose the ratio at the graph's own area
instead, so an expanded run costs what an unexpanded one does.
"""

from qwenimage21_explorations.canvas import choose, encoded_size


def test_no_ratio_leaves_the_size_exactly_as_given():
    # Not snapped: without a ratio the node must not change what the latent got before it.
    assert choose("", "", 1120, 896, [], 1024) == (1120, 896)


def test_a_ratio_keeps_the_area_and_takes_the_shape():
    w, h = choose("16:9", "", 1024, 1024, [], 1024)
    assert (w, h) == (1376, 768)
    assert w % 32 == 0 and h % 32 == 0


def test_a_square_ratio_at_a_square_size_changes_nothing():
    assert choose("1:1", "", 1024, 1024, [], 1024) == (1024, 1024)


def test_a_portrait_ratio_flips_the_shape():
    w, h = choose("9:16", "", 1024, 1024, [], 1024)
    assert h > w


def test_an_unreadable_ratio_falls_back_to_the_size():
    for bad in ("wide", "16/9", "0:9", "16:0", " ", "16:9:1"):
        assert choose(bad, "", 1024, 768, [], 1024) == (1024, 768), bad


def test_edit_without_a_ratio_is_the_first_reference_as_the_encoder_sized_it():
    """Core's own edit latent, which its tooltip says avoids shifting the edit."""
    refs = [(1600, 1200), (800, 1600)]
    assert choose("", "", 1024, 1024, refs, 1024) == encoded_size(1600, 1200, 1024)


def test_the_encoded_size_follows_core_encode_node():
    # comfy_extras/nodes_qwen.py::TextEncodeQwenImage21: area resolution**2,
    # the reference's aspect, each side rounded to 32.
    assert encoded_size(1600, 1200, 1024) == (1184, 896)
    assert encoded_size(1000, 1000, 1024) == (1024, 1024)
    # resolution 0 keeps each reference at its own size, rounded to 32.
    assert encoded_size(1000, 700, 0) == (992, 704)


def test_ratio_follow_takes_that_reference():
    refs = [(1600, 1200), (800, 1600)]
    assert choose("", "<image2>", 1024, 1024, refs, 1024) == encoded_size(800, 1600, 1024)


def test_ratio_follow_wins_when_the_answer_sets_both():
    # The contract makes them exclusive; a broken answer that sets both still
    # names the canvas being edited, which is the stronger claim.
    refs = [(1600, 1200), (800, 1600)]
    assert choose("16:9", "<image2>", 1024, 1024, refs, 1024) == encoded_size(800, 1600, 1024)


def test_ratio_follow_past_the_last_reference_is_ignored():
    refs = [(1600, 1200)]
    assert choose("", "<image3>", 1024, 1024, refs, 1024) == encoded_size(1600, 1200, 1024)


def test_an_edit_ratio_keeps_the_first_reference_area():
    refs = [(1000, 1000)]
    w, h = choose("16:9", "", 640, 640, refs, 1024)
    assert (w, h) == (1376, 768)  # the reference's 1024x1024 area, not the unused 640x640
