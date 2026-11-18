"""Short training runs that compare training recipes before the full run.

Every run starts from the same weights, sees the same data in the same order for the same
number of steps, and is scored by the offline simulator on the held-out validation set
(through FP8 rounding when --qat is given). The recipe with the highest validation
acceptance length is written to <state>/train_args for the full training run.

Recipes (each adds one published idea to the one before it):
  control         the recipe of the previous run (fixed step weights 1 / 0.7 / 0.5,
                  full-distribution distillation)
  accw_topk       acceptance-aware step weights - each step weighted by the derivative of
                  the acceptance length with respect to its accuracy, which favours early
                  steps (cf. Gumiho; FastMTP's decaying step weights) - plus distillation of
                  the target's top-10 tokens only (HASS, harmonized objective distillation)
  accw_topk_mask  + later steps trained almost only where the engine reaches them
                  (token-alignable training, GRIFFIN): floor 0.3 -> 0.1
  accw_topk_lr2   accw_topk with twice the learning rate

    python -m emcdraft.ablate --target DIR --init DIR --data DIR --out runs/ablate --results results --state state
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RECIPES = [
    ("control", "previous recipe", ["--step-w", "1", "0.7", "0.5", "--topk", "0", "--floor", "0.3", "--rel-lr", "1e-3"]),
    ("accw_topk", "acceptance-aware step weights + top-10 distillation",
     ["--step-w", "auto", "--topk", "10", "--floor", "0.3", "--rel-lr", "1e-3"]),
    ("accw_topk_mask", "accw_topk + stricter masking of unreachable later steps",
     ["--step-w", "auto", "--topk", "10", "--floor", "0.1", "--rel-lr", "1e-3"]),
    ("accw_topk_lr2", "accw_topk + twice the learning rate",
     ["--step-w", "auto", "--topk", "10", "--floor", "0.3", "--rel-lr", "2e-3"]),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--target", "--init", "--data", "--out", "--results", "--state"):
        ap.add_argument(k, required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--qat", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out)
    report = {"steps": a.steps, "recipes": {}}
    for name, desc, flags in RECIPES:
        run = out / name
        summary = run / "train_summary.json"
        if not summary.exists():          # finished runs are reused after an interruption
            cmd = [sys.executable, "-m", "emcdraft.train", "--target", a.target, "--init", a.init,
                   "--data", a.data, "--out", str(run), "--max-steps", str(a.steps), "--eval-every", "100",
                   "--no-export", "--quick", "--max-minutes", "600", *flags]
            if a.dev:
                cmd += ["--dev", a.dev]
            if a.qat:
                cmd.append("--qat")
            log(f"recipe {name}: {desc}")
            subprocess.run(cmd, check=True)
        s = json.loads(summary.read_text())
        start = s["sim"]["fp8_untrained" if a.qat else "init"]["val"]["acceptance_length"]
        report["recipes"][name] = {"description": desc, "flags": flags, "val_tau_start": start,
                                   "val_tau_best": s["best_val_tau"],
                                   "history": s["history"], "seconds": s.get("total_seconds")}
        log(f"recipe {name}: validation tau {start:.4f} -> {s['best_val_tau']:.4f}")
    best = max(report["recipes"], key=lambda n: report["recipes"][n]["val_tau_best"])
    report["chosen"] = best
    Path(a.results).mkdir(parents=True, exist_ok=True)
    (Path(a.results) / "ablation.json").write_text(json.dumps(report, indent=2))
    (Path(a.state) / "train_args").write_text(" ".join(report["recipes"][best]["flags"]) + "\n")
    log(f"chosen recipe: {best} ({report['recipes'][best]['description']})")


if __name__ == "__main__":
    main()
