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


def featurize(tgt, seqs, max_tokens, agree, on_batch=None):
    """Teacher-forced hidden states for every sequence (kept on the GPU). With on_batch, it is
    called as on_batch(batch_indices, ids, final_hidden, aux) after every forward pass, aux being
    the captured auxiliary layers (Target.capture_aux) or None."""
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
            aux = tgt.take_aux() if getattr(tgt, "_aux_ids", None) else None
        except torch.OutOfMemoryError:
            if getattr(tgt, "_aux_ids", None):
                tgt._aux = {}
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
        if on_batch is not None:
            on_batch(b, x, h, aux)
        del h, x, aux
    return feats


def finish_eagle(trainer, dev_eagle, embed, W, out_dir, res):
    """Score the online-trained EAGLE-3 draft on the dev set and export it for the engine."""
    from .eagle import export_eagle
    trainer.head.eval()
    summary = trainer.summary()
    if dev_eagle:
        summary["dev"] = simulate(trainer.head, embed, W, dev_eagle)
        log(f"EAGLE-3 after {trainer.n} steps: simulated dev acceptance {summary['dev']['acceptance_length']:.4f} "
            f"{ {d: round(v, 3) for d, v in summary['dev']['acceptance_by_domain'].items()} } "
            f"steps {[round(x, 3) for x in summary['dev']['step_accuracy']]}")
    meta = {"team_id": os.environ.get("TEAM_ID", "mudassiruddin"),
            "method_description": ("EAGLE-3 draft (one decoder layer over three fused intermediate states of the "
                                   "target, shared embeddings and LM head) trained with a 3-step recursive "
                                   "drafting objective on the target's own outputs"),
            "training_data": ("Greedy answers of the target itself to public instruction prompts: GSM8K/Orca-Math, "
                              "WritingPrompts, UltraChat/Alpaca/Dolly, Python instruction sets; dev prompts excluded")}
    out_dir.mkdir(parents=True, exist_ok=True)
    summary["export"] = export_eagle(trainer.head, out_dir / "sub_bf16", meta)
    torch.save(trainer.head.state_dict(), out_dir / "eagle_state.pt")
    (out_dir / "eagle_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (res / "eagle_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"EAGLE-3 exported to {out_dir / 'sub_bf16'}: {summary['export']}")


def shard_part(name: str) -> int:
    """Training shards of the first extraction are g0000...; later appended parts p<k>g0000..."""
    return int(name[1:name.index("g")]) if name.startswith("p") else 1


def plan_train(out: Path, gen: Path, append: bool):
    """Which generated files to featurize and how to name the new shards.

    Returns (files, shard prefix, consumed file names, number of kept shards). Without append
    every earlier shard is deleted. With append the shards are kept and only files not listed in
    <out>/consumed_gen.json are featurized (a first extraction without that list consumed the
    plain <domain>.jsonl files)."""
    train = out / "train"
    train.mkdir(parents=True, exist_ok=True)
    kept = sorted(train.glob("*.json"))
    if not append or not kept:
        for p in train.glob("*"):
            p.unlink()
        kept, consumed, prefix = [], set(), "g"
    else:
        listed = out / "consumed_gen.json"
        consumed = (set(json.loads(listed.read_text())) if listed.exists()
                    else {f"{d}.jsonl" for d in DOMAINS})
        prefix = f"p{max(shard_part(p.stem) for p in kept) + 1}g"
    files = [p for p in sorted(Path(gen).glob("*.jsonl"))
             if p.name not in consumed and p.name.split(".")[0].split("-")[0] in DOMAINS]
    return files, prefix, consumed, len(kept)


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
    ap.add_argument("--fp8", action="store_true", default=os.environ.get("EXTRACT_FP8", "0") == "1",
                    help="store training hidden states as FP8 with per-position scales (half the size)")
    ap.add_argument("--eagle-out", default=os.environ.get("EAGLE_OUT", ""),
                    help="also train an EAGLE-3 draft online from the same forward passes; export it here")
    a = ap.parse_args()
    out, res = Path(a.out), Path(a.results)
    (out / "dev").mkdir(parents=True, exist_ok=True)
    for p in (out / "dev").glob("*"):
        p.unlink()
    # EXTRACT_APPEND=1: keep the training shards of earlier extractions, featurize new files only
    files, prefix, consumed, kept = plan_train(out, Path(a.gen), os.environ.get("EXTRACT_APPEND", "0") == "1")
    res.mkdir(parents=True, exist_ok=True)
    report = {"kept_train_shards": kept, "files": [p.name for p in files]}
    log(f"training files to featurize: {report['files']} ({kept} earlier shards kept)")

    t0 = time.time()
    tgt = Target(a.target)
    report["load_seconds"] = round(time.time() - t0, 1)
    report["hidden_state_call"] = tgt.mode
    log(f"target loaded in {report['load_seconds']}s (hidden states via '{tgt.mode}')")
    eagle_cfg = None
    if a.eagle_out:
        from .eagle import EagleConfig
        eagle_cfg = EagleConfig(hidden=tgt.W.shape[1], vocab=tgt.W.shape[0])
        tgt.capture_aux(eagle_cfg.aux_layers)
        log(f"EAGLE-3: capturing target layers {eagle_cfg.aux_layers} for online training")

    # 1. dev references
    kit = Path(a.kit) / "data/dev"
    ref = {r["id"]: r["token_ids"] for r in map(json.loads, open(kit / "reference.jsonl")) if r}
    dev = [json.loads(line) for line in open(kit / "prompts.jsonl") if line.strip()]
    seqs = [{"ids": p["prompt_token_ids"] + ref[p["id"]], "P": len(p["prompt_token_ids"]),
             "domain": p["domain"], "id": p["id"]} for p in dev]
    agree = defaultdict(lambda: [0, 0])
    t0 = time.time()
    dev_eagle = []

    def keep_dev_aux(b, x, h, aux):
        if aux is not None:
            for r, i in enumerate(b):
                n = len(seqs[i]["ids"])
                dev_eagle.append({"tokens": x[r, :n].cpu(), "hidden": aux[r, :n - 1].cpu(), "P": seqs[i]["P"],
                                  "T": n, "domain": seqs[i]["domain"], "id": seqs[i].get("id", "")})

    feats = featurize(tgt, seqs, a.batch_tokens, agree, on_batch=keep_dev_aux)
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
    if a.fp8:   # what FP8 storage of the inputs costs the baseline head, on the dev set
        from .sim import fp8_dequantize, fp8_quantize
        q = [dict(f, hidden=fp8_dequantize(*fp8_quantize(f["hidden"]))) for f in feats]
        sim8 = simulate(head, lambda i: F.embedding(i, E), tgt.W, q)
        report["sim_baseline_dev_fp8_features"] = sim8
        log(f"FP8 feature storage: simulated baseline acceptance {sim8['acceptance_length']:.4f} "
            f"(bf16 features {sim['acceptance_length']:.4f})")
        del q
    (res / "extract_checks.json").write_text(json.dumps(report, indent=2))
    gap = abs(sim["acceptance_length"] - base["acceptance_length"])
    if gap > 0.25:
        raise SystemExit(f"the simulator disagrees with the kit by {gap:.3f}: refusing to train on it")
    del head, feats
    torch.cuda.empty_cache()

    # 2. generated training sequences
    rng = random.Random(a.seed)
    gen = []
    for p in files:
        d = p.name.split(".")[0].split("-")[0]
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if len(r["o"]) >= 8:
                    gen.append({"ids": r["p"] + r["o"], "P": len(r["p"]), "domain": d})
    rng.shuffle(gen)
    log(f"{len(gen)} generated sequences to featurize")
    agree = defaultdict(lambda: [0, 0])
    done = defaultdict(lambda: [0, 0])
    t0 = time.time()
    complete = True
    trainer, on_batch, embed = None, None, (lambda ids: F.embedding(ids, E))
    if eagle_cfg is not None:
        from .eagle import OnlineTrainer
        steps = sum(len(make_batches(gen[g:g + a.group], a.batch_tokens)) for g in range(0, len(gen), a.group))
        trainer = OnlineTrainer(eagle_cfg, tgt.device, steps, lr=float(os.environ.get("EAGLE_LR", 3e-4)), log=log)
        log(f"EAGLE-3: {sum(p.numel() for p in trainer.head.parameters()) / 1e6:.1f} M parameters, "
            f"{steps} training steps planned (one pass over the data)")

        def on_batch(b, x, h, aux, group):
            if aux is None:
                return
            n = x.shape[1]
            lens = torch.tensor([len(group[i]["ids"]) for i in b], device=x.device)
            P = torch.tensor([group[i]["P"] for i in b], device=x.device)
            trainer.step(embed, tgt.W, x, aux[:, :n - 1], h[:, :n - 1], P, lens)
            if trainer.n and trainer.n % 300 == 0 and dev_eagle and (not trainer.hist or trainer.hist[-1]["step"] != trainer.n):
                r = simulate(trainer.head, embed, tgt.W, dev_eagle[::4])
                trainer.hist.append({"step": trainer.n, "dev_quarter_tau": r["acceptance_length"]})
                log(f"  eagle step {trainer.n}: dev(1/4) tau {r['acceptance_length']:.4f}")
    for g in range(0, len(gen), a.group):
        if time.time() - t0 > a.minutes * 60:
            log("time budget reached")
            complete = False
            break
        group = gen[g:g + a.group]
        cb = (lambda b, x, h, aux, group=group: on_batch(b, x, h, aux, group)) if on_batch else None
        feats = featurize(tgt, group, a.batch_tokens, agree, on_batch=cb)
        save_shard(str(out / "train" / f"{prefix}{g // a.group:04d}"), feats, fp8=a.fp8)
        for f in feats:
            done[f["domain"]][0] += 1
            done[f["domain"]][1] += f["T"] - f["P"]
        del feats
        n = sum(v[1] for v in done.values())
        log(f"  group {g // a.group}: {n} generated tokens featurized, {n / (time.time() - t0):.0f} tok/s")
    report["train_sequences"] = {d: {"sequences": v[0], "generated_tokens": v[1]} for d, v in done.items()}
    report["train_teacher_forced_agreement"] = {d: round(v[0] / max(1, v[1]), 4) for d, v in agree.items()}
    report["extract_seconds"] = round(time.time() - t0, 1)
    if trainer is not None:
        finish_eagle(trainer, dev_eagle, embed, tgt.W, Path(a.eagle_out), res)
    if complete:      # a file cut off by the time budget is not marked, so it is not skipped later
        (out / "consumed_gen.json").write_text(json.dumps(sorted(consumed | {p.name for p in files})))
    (res / "extract_checks.json").write_text(json.dumps(report, indent=2))
    log(f"done: {report['train_sequences']}")


if __name__ == "__main__":
    main()
