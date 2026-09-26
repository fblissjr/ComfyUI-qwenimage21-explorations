#!/usr/bin/env python3
"""Send every stored heylook preset through the prompt expander's request path.

The node's tests stub heylook, so they hold its contract as it was when they
were written; a server that tightens a rule passes them and fails the node.
This sends the real request -- planned by `expander.plan`, as the node plans
it -- for each preset, task and model, capped at one token, and fails on any
the server refuses, printing the server's reason.

heylook has no validate-only call: a request it accepts loads the model and
produces that one token. The address is taken at run time and never written
anywhere.

Usage:
  python scripts/check_presets_live.py --base-url http://HOST:PORT [--task t2i edit] [--model ID ...]

`--model` defaults to the trained expander for each task; pass "" among the
ids to include it alongside others.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwenimage21_explorations import expander  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True, help="the heylook server, e.g. http://HOST:PORT")
    ap.add_argument("--task", nargs="+", default=["t2i", "edit"], choices=["t2i", "edit"])
    ap.add_argument("--model", nargs="+", default=[""], help='model ids; "" is the task\'s trained expander')
    ap.add_argument("--timeout", type=int, default=300, help="per request, seconds (a first load is slow)")
    args = ap.parse_args()

    rows = expander.check_presets(args.base_url, tasks=args.task, models=args.model, timeout=args.timeout)
    failed = [r for r in rows if r.error]
    for r in rows:
        mark = "FAIL" if r.error else "ok  "
        depth = r.depth or "-"
        print(f"{mark} {r.task:4} {r.preset:32} {r.model:40} thinking={r.thinking!s:5} depth={depth}")
        if r.error:
            print(f"     {r.error}")
    print(f"\n{len(rows) - len(failed)}/{len(rows)} accepted")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
