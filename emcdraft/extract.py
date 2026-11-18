"""Hidden states of the target along token sequences, plus the checks the training relies on.

1. Dev references (the kit's target-only outputs): features for the hold-out evaluation,
   teacher-forced agreement of the reference implementation with the engine's greedy choices,
   and the offline simulator replaying the *baseline* head - its acceptance must match what the
   kit measured, otherwise the simulator (and the training objective) is not trustworthy.
2. Generated training sequences (emcdraft.gen_vllm): feature shards for training, under a
   wall-clock budget, domains interleaved so a cut-off keeps the mix.

    python -m emcdraft.extract --target DIR --kit DIR --baseline DIR --gen data/gen --out data --results results
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .head import HeadConfig, build_head, load_mtp_state
from .sim import DOMAINS, argmax_logits, save_shard, simulate
from .target import PAD_ID, Target


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_batches(seqs, max_tokens):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["ids"]))
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * len(seqs[i]["ids"]) > max_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def featurize(tgt, seqs, max_tokens, agree):
    """Teacher-forced hidden states for every sequence (kept on the GPU)."""
    feats = []
    queue = make_batches(seqs, max_tokens)
    while queue:
        b = queue.pop(0)
        L = max(len(seqs[i]["ids"]) for i in b)
        x = torch.full((len(b), L), PAD_ID, dtype=torch.long, device=tgt.device)
        for r, i in enumerate(b):
            x[r, :len(seqs[i]["ids"])] = torch.tensor(seqs[i]["ids"], device=tgt.device)
        try:
            h = tgt.hidden(x)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(b) == 1:
                raise
            log(f"  out of memory on {len(b)} x {L}; splitting the batch")
            queue[:0] = [b[: len(b) // 2], b[len(b) // 2:]]
            continue
        for r, i in enumerate(b):
            s = seqs[i]
            n, P = len(s["ids"]), s["P"]
            hid = h[r, : n - 1].clone()
            pred = argmax_logits(hid[P - 1:], tgt.W)
            agree[s["domain"]][0] += int((pred == x[r, P:n]).sum())
            agree[s["domain"]][1] += n - P
            feats.append({"tokens": x[r, :n].clone(), "hidden": hid, "P": P, "T": n,
                          "domain": s["domain"], "id": s.get("id", "")})
        del h, x
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--batch-tokens", type=int, default=int(os.environ.get("EXTRACT_BATCH_TOKENS", 16384)))
    ap.add_argument("--minutes", type=float, default=float(os.environ.get("EXTRACT_MINUTES", 22)))
    ap.add_argument("--group", type=int, default=256, help="sequences per training shard")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out, res = Path(a.out), Path(a.results)
    for sub in ("dev", "train"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for p in (out / sub).glob("*"):
            p.unlink()
    res.mkdir(parents=True, exist_ok=True)
    report = {}

    t0 = time.time()
    tgt = Target(a.target)
    report["load_seconds"] = round(time.time() - t0, 1)
    report["hidden_state_call"] = tgt.mode
    log(f"target loaded in {report['load_seconds']}s (hidden states via '{tgt.mode}')")

    # 1. dev references
    kit = Path(a.kit) / "data/dev"
    ref = {r["id"]: r["token_ids"] for r in map(json.loads, open(kit / "reference.jsonl")) if r}
    dev = [json.loads(line) for line in open(kit / "prompts.jsonl") if line.strip()]
    seqs = [{"ids": p["prompt_token_ids"] + ref[p["id"]], "P": len(p["prompt_token_ids"]),
             "domain": p["domain"], "id": p["id"]} for p in dev]
    agree = defaultdict(lambda: [0, 0])
    t0 = time.time()
    feats = featurize(tgt, seqs, a.batch_tokens, agree)
    for d in DOMAINS:
        save_shard(str(out / "dev" / d), [f for f in feats if f["domain"] == d])
    report["dev_teacher_forced_agreement"] = {d: round(v[0] / max(1, v[1]), 4) for d, v in agree.items()}
    log(f"dev features: {len(feats)} sequences in {time.time() - t0:.0f}s; teacher-forced agreement with "
        f"the engine's greedy outputs {report['dev_teacher_forced_agreement']}")

    cfg = HeadConfig.from_target(json.loads((Path(a.target) / "config.json").read_text()))
    head = build_head(load_mtp_state(a.baseline), cfg, tgt.device)
    E = tgt.E
    sim = simulate(head, lambda i: F.embedding(i, E), tgt.W, feats)
    base = json.loads((kit / "baseline.json").read_text())
    report["sim_baseline_dev"] = sim
    report["kit_baseline_dev"] = {k: base[k] for k in ("acceptance_length", "acceptance_by_domain")}
    log(f"simulated baseline acceptance {sim['acceptance_length']:.3f} vs kit-measured "
        f"{base['acceptance_length']:.3f}; per domain "
        + ", ".join(f"{d} {sim['acceptance_by_domain'][d]:.3f}/{base['acceptance_by_domain'][d]:.3f}" for d in DOMAINS)
        + f"; step accuracies {[round(x, 3) for x in sim['step_accuracy']]}")
    (res / "extract_checks.json").write_text(json.dumps(report, indent=2))
    gap = abs(sim["acceptance_length"] - base["acceptance_length"])
    if gap > 0.25:
        raise SystemExit(f"the simulator disagrees with the kit by {gap:.3f}: refusing to train on it")
    del head, feats
    torch.cuda.empty_cache()

    # 2. generated training sequences
    rng = random.Random(a.seed)
    gen = []
    for d in DOMAINS:
        p = Path(a.gen) / f"{d}.jsonl"
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                if len(r["o"]) >= 8:
                    gen.append({"ids": r["p"] + r["o"], "P": len(r["p"]), "domain": d})
    rng.shuffle(gen)
    log(f"{len(gen)} generated sequences to featurize")
    agree = defaultdict(lambda: [0, 0])
    done = defaultdict(lambda: [0, 0])
    t0 = time.time()
    for g in range(0, len(gen), a.group):
        if time.time() - t0 > a.minutes * 60:
            log("time budget reached")
            break
        feats = featurize(tgt, gen[g:g + a.group], a.batch_tokens, agree)
        save_shard(str(out / "train" / f"g{g // a.group:04d}"), feats)
        for f in feats:
            done[f["domain"]][0] += 1
            done[f["domain"]][1] += f["T"] - f["P"]
        del feats
        n = sum(v[1] for v in done.values())
        log(f"  group {g // a.group}: {n} generated tokens featurized, {n / (time.time() - t0):.0f} tok/s")
    report["train_sequences"] = {d: {"sequences": v[0], "generated_tokens": v[1]} for d, v in done.items()}
    report["train_teacher_forced_agreement"] = {d: round(v[0] / max(1, v[1]), 4) for d, v in agree.items()}
    report["extract_seconds"] = round(time.time() - t0, 1)
    (res / "extract_checks.json").write_text(json.dumps(report, indent=2))
    log(f"done: {report['train_sequences']}")


if __name__ == "__main__":
    main()
