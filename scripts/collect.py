#!/usr/bin/env python3
"""Collect kit results into results/summary.json and results/summary.md.

Two scores are reported for every evaluated draft:
  * official  - the kit's own score against the organizers' published baseline numbers;
  * same-GPU  - the same formula against the baseline as measured in this run on this GPU
                (only when a full baseline evaluation is available).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EMC_ROOT", Path.home() / "emc"))
KIT = Path(os.environ.get("EMC2_HOME", ROOT / "kit"))
RES = ROOT / "results"


def score(m, base, cfg):
    w, cap = cfg["weights"], cfg["cap"]
    s_acc = (m["acceptance_length"] - 1) / (base["acceptance_length"] - 1)
    s_mem = min(cap, base["memory_bytes"] / m["memory_bytes"])
    s_tpot = min(cap, base["tpot_median_s"] / m["tpot_median_s"])
    return {"score": round(100 * (w["acceptance"] * s_acc + w["memory"] * s_mem + w["tpot"] * s_tpot), 2),
            "s_acceptance": round(s_acc, 4), "s_memory": round(s_mem, 4), "s_tpot": round(s_tpot, 4),
            "ranked": m["acceptance_length"] >= cfg["acceptance_gate"] * base["acceptance_length"]}


def main():
    cfg = json.loads((KIT / "contract.json").read_text())["score"]
    runs = {}
    for p in sorted(RES.glob("*.json")):
        d = json.loads(p.read_text())
        if "acceptance_length" in d and "per_prompt" in d:
            runs[p.stem] = d
    local_base = runs.get("eval_baseline")
    rows, out = [], {"runs": {}}
    for tag, m in runs.items():
        e = {k: m.get(k) for k in ("method", "prompts", "acceptance_length", "acceptance_by_domain",
                                   "tpot_median_s", "ttft_median_s", "memory_bytes", "weight_bytes",
                                   "kv_cache_bytes", "decode_total_s", "engine")}
        e["official_score"] = m.get("score")
        if local_base and m["prompts"] == local_base["prompts"]:
            e["same_gpu_score"] = score(m, local_base, cfg)
        out["runs"][tag] = e
        off = (m.get("score") or {}).get("score")
        same = (e.get("same_gpu_score") or {}).get("score")
        rows.append(f"| {tag} | {m['prompts']} | {m['acceptance_length']:.3f} | "
                    + " / ".join(f"{m['acceptance_by_domain'].get(d, 0) or 0:.2f}" for d in ("chat", "code", "math", "longform"))
                    + f" | {m['tpot_median_s'] * 1000:.2f} | {m['memory_bytes'] / 1e9:.3f} | "
                    + (f"{off:.1f}" if off is not None else "-") + " | " + (f"{same:.1f}" if same is not None else "-") + " |")
    for name in ("datagen_checks", ):
        p = RES / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    for p in sorted(ROOT.glob("runs/*/train_summary.json")):
        t = json.loads(p.read_text())
        out.setdefault("training", {})[p.parent.name] = {k: t.get(k) for k in ("data", "sim", "best_step", "train_seconds", "equalize", "export_fp8", "export_bf16")}
    (RES / "summary.json").write_text(json.dumps(out, indent=2))
    md = ["| run | prompts | acceptance | chat / code / math / longform | TPOT ms | memory GB | score (official) | score (same GPU) |",
          "|---|---|---|---|---|---|---|---|", *rows]
    (RES / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
