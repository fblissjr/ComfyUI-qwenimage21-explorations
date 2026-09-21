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
#: Mirrors the reference runner's per-image cap, which is also the node's default.
#: Set it to 0 in a graph that sizes its references upstream -- docs/wiki/sizing.md.
PE_MAX_PIXELS = 1024 * 1024

# Widget order per node type, so widgets_values lines up with the schema.
WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "QwenImage21Cache": ["device", "dtype"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "RandomNoise": ["noise_seed", "control_after_generate"],
    "CFGGuider": ["cfg"],
    "KSamplerSelect": ["sampler_name"],
    "SamplerCustomAdvanced": [],
    "QwenImage21Sigmas": ["steps", "denoise", "terminal_mode", "shift_terminal",
                          "base_seq_len", "max_seq_len", "base_shift", "max_shift"],
    "TextEncodeQwenImage21": ["prompt", "negative_prompt", "resolution"],
    "LoadImage": ["image", "upload"],
    "SaveImage": ["filename_prefix"],
    "VAEDecode": [],
    "QwenImage21PEExpand": ["task", "base_url", "model", "sampling", "brief", "preset",
                            "checkpoint_dir", "local_template", "system_override", "thinking",
                            "timeout", "max_pixels"],
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
- `base_url` -- your heylook server. The value shipped here is a placeholder;
  no address is stored in this file.
- **Either** `checkpoint_dir` -- the expander checkpoint's folder, for its
  `system_prompt.txt`, which ships inside the checkpoint on purpose --
  **or** `preset`, a preset stored on the server, by name or id.
- `model` -- blank derives the served name from `task`.

**Presets**
A preset carries sampler values, a reasoning level and usually a system prompt.
It is expanded here into explicit fields: the server refuses a preset as a
request field, and its stored spellings are not the wire's. Precedence is raw
text, then a local template, then the preset, then the checkpoint -- and the
`system_source` output says which one won, so it is never a guess. A
general-purpose preset's prompt replaces the trained contract and the answer
will not conform; `contract_ok` is what reports that.

**Worth knowing**
- `sampling: reference` is the upstream profile. Use `greedy` when comparing two
  arms, or two runs of the same arm disagree.
- `contract_ok` false with `violations` set means the expander broke its own
  answer rules. The rewrite may still be usable; read `violations`.
- `response:truncated` means the token cap cut the thinking trace. That reads
  downstream as bad JSON, so it is named rather than left to look like a model fault.
- Swap `TextEncodeQwenImage21` for `Qwen-Image 2.1 Encode (structured)` to reach
  the system turn, `keep_vision`, and which reference sets the canvas.
- heylook does no server-side resizing. If you size references upstream and wire
  the same image to both nodes, set the expander's `max_pixels` to 0 so it sends
  them untouched -- otherwise an image already on the 32-pixel grid can sit just
  over the area cap and earn a resample worth nothing.
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


def build(edit: bool, expander: bool = True) -> dict:
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
    pe = None
    if expander:
        pe = g.add("QwenImage21PEExpand", (900, 40),
                   ["edit" if edit else "t2i", HEYLOOK, "", "reference", brief, "", "",
                    "(none)", "", True, 900, PE_MAX_PIXELS], size=(420, 500))
        if edit:
            g.sock(pe, "images.image_1", "IMAGE", optional=True)
            g.link((loader, 0), (pe, 0), "IMAGE")
        for name in ("rewritten_prompt", "wh_ratio", "ratio_follow", "thinking"):
            g.out(pe, name, "STRING")
        g.out(pe, "contract_ok", "BOOLEAN")
        g.out(pe, "violations", "STRING")
        g.out(pe, "system_source", "STRING")

    enc = g.add("TextEncodeQwenImage21", (1380, 40),
                ["" if expander else brief, "", 1024], size=(420, 300))
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
    if pe is not None:
        slot = enc["inputs"].index(next(i for i in enc["inputs"] if i["name"] == "prompt"))
        g.link((pe, 0), (enc, slot), "STRING")

    # SamplerCustomAdvanced rather than KSampler, because KSampler builds its own
    # sigmas and core's schedule for 2.1 is wrong in two ways -- docs/wiki/sampling.md.
    noise = g.add("RandomNoise", (1860, 40), [0, "randomize"], size=(300, 80))
    g.out(noise, "NOISE", "NOISE")
    guider = g.add("CFGGuider", (1860, 170), [1.0], size=(300, 120))
    for name, type_ in (("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")):
        g.sock(guider, name, type_)
    g.out(guider, "GUIDER", "GUIDER")
    g.link((cache, 0), (guider, 0), "MODEL")
    g.link((enc, 0), (guider, 1), "CONDITIONING")
    g.link((enc, 1), (guider, 2), "CONDITIONING")

    sampler = g.add("KSamplerSelect", (1860, 330), ["euler"], size=(300, 60))
    g.out(sampler, "SAMPLER", "SAMPLER")

    sig = g.add("QwenImage21Sigmas", (1860, 430),
                [25, 1.0, "release", 0.02, 256, 8192, 0.5, 0.9], size=(320, 240))
    g.sock(sig, "latent", "LATENT")
    g.out(sig, "SIGMAS", "SIGMAS")

    ks = g.add("SamplerCustomAdvanced", (2220, 40), [], size=(320, 160))
    for name, type_ in (("noise", "NOISE"), ("guider", "GUIDER"),
                        ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"), ("latent_image", "LATENT")):
        g.sock(ks, name, type_)
    g.out(ks, "output", "LATENT")
    g.out(ks, "denoised_output", "LATENT")
    g.link((noise, 0), (ks, 0), "NOISE")
    g.link((guider, 0), (ks, 1), "GUIDER")
    g.link((sampler, 0), (ks, 2), "SAMPLER")
    g.link((sig, 0), (ks, 3), "SIGMAS")

    if edit:
        # The node's own latent matches the reference; any other size shifts the edit.
        src = (enc, 2)
    else:
        empty = g.add("EmptyLatentImage", (1380, 400), [1024, 1024, 1])
        g.out(empty, "LATENT", "LATENT")
        src = (empty, 0)
    # One latent feeds both: the schedule's shift cannot disagree with what is sampled.
    g.link(src, (ks, 4), "LATENT")
    g.link(src, (sig, 0), "LATENT")

    dec = g.add("VAEDecode", (2220, 40), [])
    g.sock(dec, "samples", "LATENT")
    g.sock(dec, "vae", "VAE")
    g.out(dec, "IMAGE", "IMAGE")
    g.link((ks, 1), (dec, 0), "LATENT")
    g.link((vae, 0), (dec, 1), "VAE")

    save = g.add("SaveImage", (2480, 40), ["qwen_image_2.1_pe"], size=(420, 460))
    g.sock(save, "images", "IMAGE")
    g.out(save, "images", "IMAGE")     # an output node still declares one; the official graphs carry it
    g.link((dec, 0), (save, 0), "IMAGE")
    return g.json()


#: Widgets the frontend renders beside another one. They occupy a slot in
#: widgets_values and are not inputs the API format carries.
COMPANION_WIDGETS = {"control_after_generate", "upload"}
#: Node types the backend does not know; they exist only in the editor.
FRONTEND_ONLY = {"MarkdownNote"}


def to_api(doc: dict) -> dict:
    """The UI graph as an API prompt, offline.

    ComfyUI's own /object_info could supply the widget order, but a template
    has to be buildable without a running server, so WIDGETS is the source and
    `tests/test_workflow_widgets.py` is what keeps it honest against the nodes.
    """
    by_link = {l[0]: (str(l[1]), l[2]) for l in doc["links"]}
    out: dict = {}
    for n in doc["nodes"]:
        if n["type"] in FRONTEND_ONLY:
            continue
        names = WIDGETS.get(n["type"])
        if names is None:
            raise KeyError(f"{n['type']} has no WIDGETS entry; add one before emitting API format")
        inputs = {k: v for k, v in zip(names, n["widgets_values"])
                  if k not in COMPANION_WIDGETS}
        for i in n["inputs"]:
            if i.get("link") is not None:
                inputs[i["name"]] = list(by_link[i["link"]])
        out[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
    return out


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


#: Only these render as widgets; a combo (a list of options) does too. Everything
#: else is a link-only socket. A whitelist, because the socket types are open-ended
#: -- NOISE, GUIDER, SAMPLER and SIGMAS all arrived this way.
#: "COMBO" is the V3 spelling; a bare list of options is the legacy one, and
#: both are live in the same /object_info.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


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
                if not (isinstance(typ, list) or typ in WIDGET_TYPES):
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
        # A SHORT list is the drift that hides: a node gains an output, the
        # graph keeps the old count, and every present slot still validates.
        if len(n["outputs"]) != len(outs):
            names = oi[n["type"]].get("output_name") or outs
            errs.append(f"{n['type']}: graph has {len(n['outputs'])} outputs, "
                        f"server declares {len(outs)} -> {list(names)}")
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
    ap.add_argument("--api-out", metavar="DIR", type=Path,
                    help="also write API-format templates there, for a front end to patch")
    args = ap.parse_args()

    if args.api_out:
        # Four, rather than one the app edits: a front end that adds or removes
        # nodes is re-authoring the graph, and these are snapshots.
        for name, edit, pe in (("qi21_t2i", False, False), ("qi21_t2i_pe", False, True),
                               ("qi21_edit", True, False), ("qi21_edit_pe", True, True)):
            doc = build(edit, pe)
            errs = validate(doc)
            for e in errs:
                print(f"{name}: {e}", file=sys.stderr)
            if errs:
                return 1
            api = to_api(doc)
            args.api_out.mkdir(parents=True, exist_ok=True)
            (args.api_out / f"{name}.json").write_text(json.dumps(api, indent=2) + "\n")
            print(f"wrote {args.api_out / f'{name}.json'}  ({len(api)} nodes)")

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
