#!/usr/bin/env python3
"""End-to-end smoke test of the PE harness against a heylook server.

This is the gate the whole plan rests on: until one valid, contract-clean answer
comes out of each model, nothing downstream is interpretable. It runs the
upstream example briefs -- the authoritative shape, typos and mixed languages
preserved -- and grades every answer with the same module the ComfyUI nodes use.

Sampling is sent explicitly from PROFILES on every request. The server's own
per-model defaults are NOT the reference settings, and its default max_tokens is
far below a full thinking trace.

Usage:
  python scripts/smoke_heylook.py --base-url http://HOST:PORT --data <qwen-image-2.1-repo>/prompt_rewrite/data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwenimage21_explorations import answer as answer_mod  # noqa: E402
from qwenimage21_explorations.backends import heylook  # noqa: E402
from qwenimage21_explorations.profiles import PROFILES  # noqa: E402

TASKS = {
    "t2i": dict(jsonl="t2i_example.jsonl", model="Qwen-Image-2.1-PE-T21-mlx", ckpt_env="PE_T2I_CKPT"),
    "edit": dict(jsonl="edit_example.jsonl", model="Qwen-Image-2.1-PE-I21-mlx", ckpt_env="PE_EDIT_CKPT"),
}


def run_task(task: str, *, base_url: str, data_dir: Path, ckpt: Path, limit: int) -> list[dict]:
    spec = TASKS[task]
    profile = PROFILES[task]
    system = (ckpt / "system_prompt.txt").read_text(encoding="utf-8").strip()
    rows = [json.loads(l) for l in (data_dir / spec["jsonl"]).read_text().splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]

    out = []
    for row in rows:
        images = [data_dir / p for p in row.get("input_images", [])]
        t0 = time.time()
        try:
            resp = heylook.generate(
                base_url=base_url, model=spec["model"], system=system,
                brief=row["prompt"], images=images, **profile,
            )
        except Exception as e:  # network or server-side failure is a result, not a crash
            out.append(dict(id=row["id"], task=task, error=f"{type(e).__name__}: {e}"))
            print(f"  {row['id']:<12} ERROR {type(e).__name__}: {str(e)[:80]}")
            continue

        graded = answer_mod.parse_and_grade(
            resp.as_inline(), task=task, n_images=len(images)
        )
        rec = dict(
            id=row["id"], task=task, task_type=row.get("task_type", ""),
            n_images=len(images), brief=row["prompt"],
            stop_reason=resp.stop_reason, truncated=resp.truncated,
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            seconds=round(time.time() - t0, 1),
            parse_ok=graded.parse_ok, contract_ok=graded.contract_ok,
            violations=graded.violations,
            wh_ratio=graded.wh_ratio, ratio_follow=graded.ratio_follow,
            thinking_chars=len(graded.thinking),
            rewritten_prompt=graded.rewritten_prompt,
        )
        out.append(rec)
        flag = "ok " if rec["contract_ok"] else "FAIL"
        print(
            f"  {row['id']:<12} {flag} parse={rec['parse_ok']!s:<5} "
            f"in={rec['input_tokens']:<6} out={rec['output_tokens']:<6} "
            f"{rec['seconds']:>5.1f}s  {row.get('task_type','')[:20]:<20} "
            f"{','.join(rec['violations']) if rec['violations'] else ''}"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--data", required=True, type=Path, help="prompt_rewrite/data directory")
    ap.add_argument("--t2i-ckpt", required=True, type=Path)
    ap.add_argument("--edit-ckpt", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/smoke_heylook.jsonl"))
    ap.add_argument("--tasks", nargs="+", default=["t2i", "edit"], choices=list(TASKS))
    args = ap.parse_args()

    ckpts = {"t2i": args.t2i_ckpt, "edit": args.edit_ckpt}
    results: list[dict] = []
    for task in args.tasks:
        print(f"\n=== {task} ===")
        results += run_task(task, base_url=args.base_url, data_dir=args.data,
                            ckpt=ckpts[task], limit=args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")

    graded = [r for r in results if "error" not in r]
    n_parse = sum(r["parse_ok"] for r in graded)
    n_contract = sum(r["contract_ok"] for r in graded)
    n_trunc = sum(r["truncated"] for r in graded)
    print(f"\n{len(graded)}/{len(results)} completed | parse_ok {n_parse}/{len(graded)} "
          f"| contract_ok {n_contract}/{len(graded)} | truncated {n_trunc}")
    print(f"wrote {args.out}")
    return 0 if graded and n_contract == len(graded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
