#!/usr/bin/env python3
"""Generate the example workflows under `example_workflows/`.

Hand-written graph JSON rots quietly: a slot index moves and the file still
looks fine until someone drags it in. Building them here keeps the two graphs
consistent with each other and lets `--check` fail on a structural break.

The graph shape follows the official Qwen-Image 2.1 workflows, with the prompt
expander added ahead of the encode node. Upstream ships the expanders as a
separate codebase and no engine wires one in, so this is the part that has no
reference graph to copy.

Usage:  python scripts/build_example_workflows.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "example_workflows"

# The Comfy-Org release filenames. The quantized variants are named
# ..._int8_convrot.safetensors and drop in wherever these appear.
UNET = "qwen_image_2.1_bf16.safetensors"
CLIP = "qwen3vl_8b_bf16.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"
HEYLOOK = "http://localhost:8080"

# Widget order per node type, so widgets_values lines up with the schema.
WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "QwenImage21Cache": ["device", "dtype"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "TextEncodeQwenImage21": ["prompt", "negative_prompt", "resolution"],
    "LoadImage": ["image", "upload"],
    "SaveImage": ["filename_prefix"],
    "VAEDecode": [],
    "QwenImage21PEExpand": ["task", "base_url", "model", "sampling", "brief", "checkpoint_dir",
                            "local_template", "system_override", "thinking", "timeout"],
    "MarkdownNote": ["text"],
}


class Graph:
    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self._nid = 0
        self._lid = 0

    def add(self, type_: str, pos, widgets=None, size=(380, 120), title=None) -> dict:
        self._nid += 1
        node = {
            "id": self._nid, "type": type_, "pos": list(pos), "size": list(size),
            "flags": {}, "order": self._nid - 1, "mode": 0,
            "inputs": [], "outputs": [], "properties": {"Node name for S&R": type_},
            "widgets_values": list(widgets or []),
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node

    def out(self, node: dict, name: str, type_: str) -> tuple[dict, int]:
        node["outputs"].append({"name": name, "type": type_, "links": []})
        return node, len(node["outputs"]) - 1

    def sock(self, node: dict, name: str, type_: str, *, widget=False, optional=False):
        entry = {"localized_name": name.split(".")[-1], "name": name, "type": type_, "link": None}
        if optional:
            entry["shape"] = 7
        if widget:
            entry["widget"] = {"name": name}
        node["inputs"].append(entry)
        return node, len(node["inputs"]) - 1

    def link(self, src: tuple[dict, int], dst: tuple[dict, int], type_: str):
        (snode, sslot), (dnode, dslot) = src, dst
        self._lid += 1
        self.links.append([self._lid, snode["id"], sslot, dnode["id"], dslot, type_])
        snode["outputs"][sslot]["links"].append(self._lid)
        dnode["inputs"][dslot]["link"] = self._lid

    def json(self) -> dict:
        return {
            "id": "00000000-0000-0000-0000-000000000000", "revision": 0,
            "last_node_id": self._nid, "last_link_id": self._lid,
            "nodes": self.nodes, "links": self.links, "groups": [],
            "config": {}, "extra": {}, "version": 0.4,
        }


NOTE = """## Prompt expansion through heylook

`Qwen-Image 2.1 PE (heylook)` sends the brief to a prompt expander served by
heylook and wires the rewritten prompt into the encode node.

**Set before running**
- `base_url` -- your heylook server.
- `checkpoint_dir` -- the expander checkpoint's folder, for its `system_prompt.txt`.
  The prompt ships inside the checkpoint on purpose; a second copy drifts.
- `model` -- blank derives the served name from `task`.

**Worth knowing**
- `sampling: reference` is the upstream profile. Use `greedy` when comparing two
  arms, or two runs of the same arm disagree.
- `contract_ok` false with `violations` set means the expander broke its own
  answer rules. The rewrite may still be usable; read `violations`.
- `response:truncated` means the token cap cut the thinking trace. That reads
  downstream as bad JSON, so it is named rather than left to look like a model fault.
- Swap `TextEncodeQwenImage21` for `Qwen-Image 2.1 Encode (structured)` to reach
  the system turn, `keep_vision`, and which reference sets the canvas.
"""


def common(g: Graph, *, edit: bool):
    unet = g.add("UNETLoader", (40, 40), [UNET, "default"])
    g.out(unet, "MODEL", "MODEL")
    cache = g.add("QwenImage21Cache", (40, 180), ["auto", "default"])
    g.sock(cache, "model", "MODEL")
    g.out(cache, "MODEL", "MODEL")
    g.link((unet, 0), (cache, 0), "MODEL")

    clip = g.add("CLIPLoader", (40, 320), [CLIP, "qwen_image", "default"])
    g.out(clip, "CLIP", "CLIP")
    vae = g.add("VAELoader", (40, 460), [VAE])
    g.out(vae, "VAE", "VAE")
    return unet, cache, clip, vae


def build(edit: bool) -> dict:
    g = Graph()
    _, cache, clip, vae = common(g, edit=edit)
    g.add("MarkdownNote", (40, 600), [NOTE], size=(460, 420), title="Note: prompt expansion")

    loader = None
    if edit:
        loader = g.add("LoadImage", (480, 40), ["example.png", "image"], size=(380, 320))
        g.out(loader, "IMAGE", "IMAGE")
        g.out(loader, "MASK", "MASK")

    brief = ("put the cat on a small wooden boat at dawn" if edit
             else "a capybara wearing a wizard hat, oil painting")
    pe = g.add("QwenImage21PEExpand", (900, 40),
               ["edit" if edit else "t2i", HEYLOOK, "", "reference", brief, "",
                "(none)", "", True, 900], size=(420, 460))
    if edit:
        g.sock(pe, "images.image_1", "IMAGE", optional=True)
        g.link((loader, 0), (pe, 0), "IMAGE")
    for name in ("rewritten_prompt", "wh_ratio", "ratio_follow", "thinking"):
        g.out(pe, name, "STRING")
    g.out(pe, "contract_ok", "BOOLEAN")
    g.out(pe, "violations", "STRING")

    enc = g.add("TextEncodeQwenImage21", (1380, 40), ["", "", 1024], size=(420, 300))
    g.sock(enc, "clip", "CLIP")
    if edit:
        g.sock(enc, "images.image_1", "IMAGE", optional=True)
        g.sock(enc, "vae", "VAE", optional=True)
    g.sock(enc, "prompt", "STRING", widget=True)
    g.sock(enc, "negative_prompt", "STRING", widget=True)
    g.out(enc, "positive", "CONDITIONING")
    g.out(enc, "negative", "CONDITIONING")
    g.out(enc, "latent", "LATENT")
    g.link((clip, 0), (enc, 0), "CLIP")
    if edit:
        g.link((loader, 0), (enc, 1), "IMAGE")
        g.link((vae, 0), (enc, 2), "VAE")
    g.link((pe, 0), (enc, enc["inputs"].index(next(i for i in enc["inputs"] if i["name"] == "prompt"))), "STRING")

    ks = g.add("KSampler", (1860, 40), [0, "randomize", 25, 1, "euler", "simple", 1], size=(320, 280))
    for name, type_ in (("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("latent_image", "LATENT")):
        g.sock(ks, name, type_)
    g.out(ks, "LATENT", "LATENT")
    g.link((cache, 0), (ks, 0), "MODEL")
    g.link((enc, 0), (ks, 1), "CONDITIONING")
    g.link((enc, 1), (ks, 2), "CONDITIONING")

    if edit:
        # The node's own latent matches the reference, and any other size shifts the edit.
        g.link((enc, 2), (ks, 3), "LATENT")
    else:
        empty = g.add("EmptyLatentImage", (1380, 400), [1024, 1024, 1])
        g.out(empty, "LATENT", "LATENT")
        g.link((empty, 0), (ks, 3), "LATENT")

    dec = g.add("VAEDecode", (2220, 40), [])
    g.sock(dec, "samples", "LATENT")
    g.sock(dec, "vae", "VAE")
    g.out(dec, "IMAGE", "IMAGE")
    g.link((ks, 0), (dec, 0), "LATENT")
    g.link((vae, 0), (dec, 1), "VAE")

    save = g.add("SaveImage", (2480, 40), ["qwen_image_2.1_pe"], size=(420, 460))
    g.sock(save, "images", "IMAGE")
    g.link((dec, 0), (save, 0), "IMAGE")
    return g.json()


def validate(doc: dict) -> list[str]:
    """Structural checks: a slot that moved is the failure this catches."""
    errs = []
    by_id = {n["id"]: n for n in doc["nodes"]}
    for n in doc["nodes"]:
        want = WIDGETS.get(n["type"])
        if want is not None and len(n["widgets_values"]) != len(want):
            errs.append(f"{n['type']} has {len(n['widgets_values'])} widget values, expected {len(want)}")
    seen = set()
    for lid, sid, sslot, did, dslot, type_ in doc["links"]:
        if lid in seen:
            errs.append(f"duplicate link id {lid}")
        seen.add(lid)
        for nid, slot, key in ((sid, sslot, "outputs"), (did, dslot, "inputs")):
            node = by_id.get(nid)
            if node is None:
                errs.append(f"link {lid} references missing node {nid}")
            elif slot >= len(node[key]):
                errs.append(f"link {lid}: {node['type']} has no {key}[{slot}]")
            elif node[key][slot]["type"] != type_:
                errs.append(f"link {lid}: {node['type']}.{key}[{slot}] is "
                            f"{node[key][slot]['type']}, link says {type_}")
    for n in doc["nodes"]:
        for i in n["inputs"]:
            if i["link"] is None and i.get("shape") != 7 and "widget" not in i:
                errs.append(f"{n['type']}.{i['name']} is required and unconnected")
    return errs


SOCKETS = {"MODEL", "CLIP", "VAE", "IMAGE", "MASK", "LATENT", "CONDITIONING"}


def check_against_server(doc: dict, base_url: str) -> list[str]:
    """Validate node types, input names and widget counts against a live ComfyUI.

    The structural pass above cannot see a schema change upstream; this can.
    A node type missing from `/object_info` is frontend-only (MarkdownNote) and
    is skipped rather than reported.
    """
    import json as _json
    import urllib.request

    with urllib.request.urlopen(f"{base_url.rstrip('/')}/object_info", timeout=60) as r:
        oi = _json.load(r)

    errs = []
    for n in doc["nodes"]:
        spec = oi.get(n["type"], {}).get("input")
        if spec is None:
            continue
        widgets, allowed = [], set()
        for section in ("required", "optional"):
            for name, val in spec.get(section, {}).items():
                allowed.add(name)
                typ = val[0] if isinstance(val, (list, tuple)) and val else val
                opts = val[1] if isinstance(val, (list, tuple)) and len(val) > 1 else {}
                if typ == "COMFY_AUTOGROW_V3" or (isinstance(typ, str) and typ in SOCKETS):
                    continue
                widgets.append(name)
                # Some inputs render a second widget beside themselves: a seed's
                # control, and an image combo's upload button. The official
                # graphs carry both, so the counts have to allow for them.
                for flag, extra in (("control_after_generate", "control"), ("image_upload", "upload")):
                    if isinstance(opts, dict) and opts.get(flag):
                        widgets.append(f"{name}.{extra}")
        if len(n["widgets_values"]) != len(widgets):
            errs.append(f"{n['type']}: {len(n['widgets_values'])} widget values, "
                        f"server implies {len(widgets)} -> {widgets}")
        for i in n["inputs"]:
            if i["name"].split(".")[0] not in allowed:
                errs.append(f"{n['type']}: input {i['name']!r} is not in the server's schema")
        outs = oi[n["type"]]["output"]
        for k, o in enumerate(n["outputs"]):
            if k >= len(outs):
                errs.append(f"{n['type']}: output[{k}] beyond the server's {len(outs)}")
            elif outs[k] != o["type"]:
                errs.append(f"{n['type']}: output[{k}] is {o['type']}, server says {outs[k]}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the files on disk differ")
    ap.add_argument("--server", metavar="URL",
                    help="also validate against a running ComfyUI, e.g. http://127.0.0.1:8188")
    args = ap.parse_args()

    rc = 0
    for name, edit in (("qwen_image_2.1_t2i_heylook_pe.json", False),
                       ("qwen_image_2.1_edit_heylook_pe.json", True)):
        doc = build(edit)
        errs = validate(doc)
        if args.server:
            errs += check_against_server(doc, args.server)
        for e in errs:
            print(f"{name}: {e}", file=sys.stderr)
        rc |= bool(errs)
        text = json.dumps(doc, indent=2) + "\n"
        path = OUT / name
        if args.check:
            current = path.read_text() if path.is_file() else ""
            if current != text:
                print(f"{name}: on disk differs from the generator", file=sys.stderr)
                rc |= 1
            else:
                print(f"{name}: up to date, {len(doc['nodes'])} nodes")
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path.relative_to(OUT.parent)}  ({len(doc['nodes'])} nodes)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
