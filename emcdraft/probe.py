"""Engine probe: the unmodified baseline head, exactly re-scaled and stored in FP8 (no training).

Evaluating this folder with the kit answers two questions before any training time is spent:
does the stock engine load FP8-typed MTP tensors, and how much acceptance does plain rounding
cost on its own.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .head import HeadConfig, build_head, equalize_, export_submission, load_mtp_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = HeadConfig.from_target(json.loads((Path(a.target) / "config.json").read_text()))
    head = build_head(load_mtp_state(a.baseline), cfg)
    factors = equalize_(head)
    info = export_submission(head, a.out, Path(a.target) / "config.json", True,
                             {"team_id": "probe", "method_description": "baseline head, FP8 storage, untrained",
                              "training_data": "none"})
    print(json.dumps({"equalize": factors, **info}, indent=2))


if __name__ == "__main__":
    main()
