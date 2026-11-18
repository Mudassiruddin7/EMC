#!/usr/bin/env python3
"""Keep the better draft: the new run's draft replaces the previous best only when its
same-GPU score (kit formula against the baseline measured on this machine) is higher.

Writes <results>/choice.json and <state>/final_sub (the folder of the draft that is kept).

    choose.py --new TAG --new-sub DIR [--prev TAG --prev-sub DIR] --results DIR --state DIR
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--new-sub", required=True)
    ap.add_argument("--prev", default="")
    ap.add_argument("--prev-sub", default="")
    ap.add_argument("--results", required=True)
    ap.add_argument("--state", required=True)
    a = ap.parse_args()
    res = Path(a.results)
    kit = Path(os.environ["EMC2_HOME"])
    cfg = json.loads((kit / "contract.json").read_text())["score"]
    base = json.loads((res / "eval_baseline.json").read_text())
    rows = {}
    for name, tag, sub in (("new", a.new, a.new_sub), ("previous", a.prev, a.prev_sub)):
        p = res / f"{tag}.json"
        if tag and p.exists() and sub and Path(sub).is_dir():
            m = json.loads(p.read_text())
            rows[name] = {"tag": tag, "folder": sub, "acceptance_length": m["acceptance_length"],
                          "tpot_median_s": m["tpot_median_s"], "memory_bytes": m["memory_bytes"],
                          "same_gpu": score(m, base, cfg)}
    if not rows:
        raise SystemExit("no evaluated draft to choose from")
    keep = max(rows, key=lambda n: rows[n]["same_gpu"]["score"])
    out = {"kept": keep, "rows": rows,
           "rule": "the new draft replaces the previous one only with a higher same-GPU score"}
    (res / "choice.json").write_text(json.dumps(out, indent=2))
    (Path(a.state) / "final_sub").write_text(rows[keep]["folder"] + "\n")
    for n, r in rows.items():
        print(f"{n:9s} {r['tag']}: tau {r['acceptance_length']:.4f}, TPOT {r['tpot_median_s'] * 1000:.2f} ms, "
              f"same-GPU score {r['same_gpu']['score']}")
    print(f"kept: {keep} ({rows[keep]['folder']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
