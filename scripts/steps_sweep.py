#!/usr/bin/env python3
"""Sweep t2i step counts on a running ComfyUI, as the owner renders.

Answers one question: has t2i converged at the official 25 steps, or is it
still moving toward the 40-50 every other implementation defaults to
(docs/wiki/sampling.md section 2)? It measures convergence, not quality. The
contact sheets are the quality evidence, and choosing `STEPS` stays the
owner's call.

The graph is `build_example_workflows.py`'s plain t2i graph, with
`QwenImage21SageAttention` (`sage_mode=auto`) inserted between the cache and
the guider, because that is how the owner renders: sage comes from this
repo's node, never from `--use-sage-attention`. Prompts are the generated t2i
entries in `prompt_bank/`, body verbatim, each sized through
`canvas.choose` from its own `wh_ratio`. So the two prompts run on two
canvases and two values of mu.

Per (prompt, seed), every arm is compared against a high-step reference
render at two scales. Full resolution sees detail. A 1/16 downsample sees
layout and subject scale, which is where the withdrawn 2026-09-20 sweep put
most of its t2i gap.

`--sage-modes` turns the same grid into the A/B `sage.py` names as missing.
Each mode is one of `sage.MODES`, `off` (the node left out, so ComfyUI's own
attention), or a mode with `+masked` appended to set `sage_masked`. When `off`
is in the list, every other mode is also compared with `off` at the same step
count. That distance is sage's own error, to be read against the gap between
step counts. `--reference 0` skips the high-step render.

`--task edit` runs the plain edit graph instead. Each prompt file names its
reference image in a `reference:` frontmatter field, resolved against
`--ref-dir` and uploaded to the server's input folder under
`qi21_steps_sweep/`. The canvas follows the reference, as the graph does
when no shape is given.

Gates, recorded with the results rather than assumed:
- determinism: the first arm is rendered twice, with ComfyUI's execution cache
  freed in between, and the two PNGs must be byte-identical.
- sage fired: the server log must gain a `[qwen21-sage] fired` line, and no
  `declined or raised` line, during the sweep.

Writes `<out>/results.jsonl` (one conditions row, then one row per render),
the PNGs, and one contact sheet per (prompt, seed), one row per mode.

Usage:
  uv run --no-project --with numpy --with pillow python scripts/steps_sweep.py \\
      --out data/steps_sweep/<date> [--server http://127.0.0.1:8188] \\
      [--steps 20,25,30,40,50] [--reference 100] [--seeds 1,2,3] [--limit N] \\
      [--sage-modes auto]    # e.g. off,auto,fp8,fp16,auto+masked
      [--task edit --prompts 'prompt_bank/edit_sweep_*.md' --ref-dir DIR]
  python scripts/steps_sweep.py --report data/steps_sweep/<date>   # summarise a run
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import build_example_workflows as bew  # noqa: E402
from qwenimage21_explorations import canvas  # noqa: E402

SAGE_ID = "90"
FIRED = "[qwen21-sage] fired"
DECLINED = "[qwen21-sage] sage declined or raised"


def load_prompt(path: Path) -> dict:
    _, front, body = path.read_text().split("---", 2)
    meta = {}
    for line in front.strip().splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip().strip('"')
    return {"name": path.stem, "wh_ratio": meta.get("wh_ratio", ""),
            "reference": meta.get("reference", ""), "prompt": body.strip()}


def graph(prompt: str, width: int, height: int, steps: int, seed: int, prefix: str,
          mode: str = "auto", reference: str = "") -> dict:
    api = bew.to_api(bew.build(bool(reference), False, save_prefix=prefix))
    by_type = {v["class_type"]: k for k, v in api.items()}
    api[by_type["TextEncodeQwenImage21"]]["inputs"]["prompt"] = prompt
    if reference:
        api[by_type["LoadImage"]]["inputs"]["image"] = reference
    else:
        api[by_type["EmptyLatentImage"]]["inputs"].update(width=width, height=height)
    api[by_type["QwenImage21Sigmas"]]["inputs"]["steps"] = steps
    api[by_type["RandomNoise"]]["inputs"]["noise_seed"] = seed
    if mode == "off":
        return api
    name, masked = mode.removesuffix("+masked"), mode.endswith("+masked")
    cache = by_type["QwenImage21Cache"]
    api[SAGE_ID] = {"class_type": "QwenImage21SageAttention",
                    "inputs": {"model": [cache, 0], "sage_mode": name, "sage_masked": masked}}
    api[by_type["CFGGuider"]]["inputs"]["model"] = [SAGE_ID, 0]
    return api


def slug(mode: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in mode.replace("++", "pp")).strip("-")


def call(server: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(server + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def upload(server: str, path: Path) -> str:
    """Put a reference in the server's input folder; returns the LoadImage name."""
    boundary = uuid.uuid4().hex
    fields = [("subfolder", "qi21_steps_sweep"), ("type", "input"), ("overwrite", "true")]
    body = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                    for k, v in fields)
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
             f"Content-Type: image/png\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(server + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]


def render(server: str, api: dict) -> tuple[bytes, float]:
    pid = call(server, "/prompt", {"prompt": api, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        h = call(server, f"/history/{pid}").get(pid)
        if h and h.get("status", {}).get("completed"):
            break
        if h and h.get("status", {}).get("status_str") == "error":
            raise RuntimeError(json.dumps(h["status"]["messages"])[-2000:])
        time.sleep(1)
    msgs = {m[0]: m[1]["timestamp"] for m in h["status"]["messages"]}
    seconds = (msgs["execution_success"] - msgs["execution_start"]) / 1000
    img = next(o["images"][0] for o in h["outputs"].values() if o.get("images"))
    q = urllib.parse.urlencode({k: img[k] for k in ("filename", "subfolder", "type")})
    with urllib.request.urlopen(f"{server}/view?{q}", timeout=60) as r:
        return r.read(), seconds


def new_log(server: str, since: str) -> tuple[list[str], str]:
    """Server log lines stamped after `since`. The buffer is a bounded deque, so
    this is polled after every render rather than diffed once at the end."""
    entries = call(server, "/internal/logs/raw")["entries"]
    fresh = [e for e in entries if e["t"] > since]
    return [e["m"] for e in fresh], (fresh[-1]["t"] if fresh else since)


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def coarse(png: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(png)).convert("RGB")
    return np.asarray(im.resize((im.width // 16, im.height // 16), Image.Resampling.BOX))


def sheet(rows: list[list[bytes]], path: Path, tile: int = 384):
    first = Image.open(io.BytesIO(rows[0][0]))
    h = round(tile * first.height / first.width)
    out = Image.new("RGB", (tile * max(map(len, rows)), h * len(rows)), "white")
    for r, row in enumerate(rows):
        for c, png in enumerate(row):
            im = Image.open(io.BytesIO(png)).convert("RGB")
            out.paste(im.resize((tile, h), Image.Resampling.LANCZOS), (c * tile, r * h))
    out.save(path)


def report(out: Path) -> int:
    """Summarise a results file: gates, then per mode and step count the
    median seconds and the mean distances over prompts and seeds."""
    rows = [json.loads(l) for l in (out / "results.jsonl").open()]
    for r in rows:
        if r["kind"] == "determinism":
            print(f"determinism: identical={r['identical']}")
        if r["kind"] == "sage":
            kernels = sorted({l.split("kernel=")[1].split()[0] for l in r["fired_lines"]})
            print(f"sage: fired kernels {kernels}, declined {r['declined']}")
    renders = [r for r in rows if r["kind"] == "render"]
    dist = [r for r in rows if r["kind"] == "distance"]
    keys = ["mad_ref", "mad_ref_coarse", "mad_40", "mad_off", "mad_off_coarse"]
    keys = [k for k in keys if any(r.get(k) is not None for r in dist)]
    print(f"{'mode':32} {'steps':>5} {'median_s':>8} " + " ".join(f"{k:>14}" for k in keys))
    for mode in dict.fromkeys(r.get("mode", "auto") for r in renders):
        for n in sorted({r["steps"] for r in renders}):
            secs = sorted(r["seconds"] for r in renders if r.get("mode", "auto") == mode and r["steps"] == n)
            d = [r for r in dist if r.get("mode", "auto") == mode and r["steps"] == n]
            cells = []
            for k in keys:
                v = [r[k] for r in d if r.get(k) is not None]
                cells.append(f"{sum(v) / len(v):14.2f}" if v else f"{'-':>14}")
            print(f"{mode:32} {n:5} {secs[len(secs) // 2]:8.2f} " + " ".join(cells))
    return 0


def main() -> int:
    if sys.argv[1:2] == ["--report"]:
        return report(Path(sys.argv[2]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", default="20,25,30,40,50")
    ap.add_argument("--reference", type=int, default=100)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--limit", type=int, default=0, help="stop after N renders; for timing one")
    ap.add_argument("--sage-modes", default="auto")
    ap.add_argument("--task", choices=["t2i", "edit"], default="t2i")
    ap.add_argument("--prompts", help="glob of prompt files; default prompt_bank/<task>_*.md")
    ap.add_argument("--ref-dir", type=Path, default=Path("."), help="where edit references are")
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    files = sorted(Path(f) for f in glob.glob(args.prompts)) if args.prompts else \
        sorted(REPO.glob(f"prompt_bank/{args.task}_*.md"))
    prompts = [load_prompt(p) for p in files]
    if not prompts:
        raise SystemExit("no prompt files matched")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = open(args.out / "results.jsonl", "a")

    stats = call(args.server, "/system_stats")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    rows.write(json.dumps({"kind": "conditions", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "repo_commit": commit, "system": stats["system"],
                           "devices": [d["name"] for d in stats["devices"]],
                           "task": args.task, "prompts": [p["name"] for p in prompts],
                           "sage_modes": args.sage_modes.split(","), "steps": steps,
                           "reference": args.reference,
                           "seeds": seeds, "sampler": "euler", "cfg": 1.0,
                           "terminal_mode": "release"}) + "\n")
    _, since = new_log(args.server, "")
    seen: list[str] = []
    try:
        sweep(args, prompts, steps, seeds, rows, seen, since)
    finally:
        rows.write(json.dumps({"kind": "sage", "fired_lines": [m.strip() for m in seen if FIRED in m],
                               "declined": sum(DECLINED in m for m in seen)}) + "\n")
        rows.close()
    return 0


def sweep(args, prompts, steps, seeds, rows, seen, since):
    modes = args.sage_modes.split(",")
    counts = ([args.reference] if args.reference else []) + steps
    done = 0
    for p in prompts:
        w, h = canvas.choose(p["wh_ratio"], "", 1024, 1024, [], 1024)
        ref_name = upload(args.server, args.ref_dir / p["reference"]) if args.task == "edit" else ""
        for seed in seeds:
            arms = {}
            for mode in modes:
                for n in counts:
                    tag = f"{p['name']}_s{seed}_{slug(mode)}_{n:03d}"
                    api = graph(p["prompt"], w, h, n, seed, f"qi21_steps_sweep/{tag}", mode, ref_name)
                    png, seconds = render(args.server, api)
                    if done == 0:
                        call(args.server, "/free", {"free_memory": True})
                        again, _ = render(args.server, api)
                        rows.write(json.dumps({"kind": "determinism", "prompt": p["name"], "seed": seed,
                                               "mode": mode, "steps": n, "identical": again == png}) + "\n")
                    lines, since = new_log(args.server, since)
                    seen += lines
                    (args.out / f"{tag}.png").write_bytes(png)
                    arms[mode, n] = png
                    rows.write(json.dumps({"kind": "render", "prompt": p["name"], "seed": seed,
                                           "mode": mode, "reference": p["reference"] or None,
                                           "size": list(Image.open(io.BytesIO(png)).size), "steps": n,
                                           "seconds": seconds}) + "\n")
                    rows.flush()
                    done += 1
                    print(f"{tag}: {seconds:.1f}s", flush=True)
                    if args.limit and done >= args.limit:
                        return
            px = {k: np.asarray(Image.open(io.BytesIO(v)).convert("RGB")) for k, v in arms.items()}
            for mode in modes:
                for n in steps:
                    row = {"kind": "distance", "prompt": p["name"], "seed": seed, "mode": mode, "steps": n}
                    if args.reference:
                        ref = (mode, args.reference)
                        row.update(mad_ref=mad(px[mode, n], px[ref]),
                                   mad_ref_coarse=mad(coarse(arms[mode, n]), coarse(arms[ref])),
                                   mad_40=mad(px[mode, n], px[mode, 40]) if 40 in steps else None)
                    if "off" in modes and mode != "off":
                        row.update(mad_off=mad(px[mode, n], px["off", n]),
                                   mad_off_coarse=mad(coarse(arms[mode, n]), coarse(arms["off", n])))
                    rows.write(json.dumps(row) + "\n")
            sheet([[arms[mode, n] for n in steps + counts[:1 if args.reference else 0]] for mode in modes],
                  args.out / f"sheet_{p['name']}_s{seed}.png")


if __name__ == "__main__":
    sys.exit(main())
