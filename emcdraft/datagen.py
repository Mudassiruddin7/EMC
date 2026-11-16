"""Self-distillation data: the target's own greedy answers with its final hidden states.

1. sanity checks against the public dev set (chat template, greedy agreement with the kit's
   reference outputs, and the offline simulator reproducing the baseline head's acceptance);
2. hidden states for the dev references (hold-out features, never trained on);
3. greedy generation for fresh prompts in the four domains under a wall-clock budget.

Prompts in a batch all have the same token length, so no padding is ever fed to the model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import prompts as PR
from .head import HeadConfig, build_head, load_mtp_state
from .sim import DOMAINS, save_shard, simulate
from .target import CHAT_PREFIX, CHAT_SUFFIX, Target

SHARE = {"longform": 0.45, "chat": 0.22, "code": 0.15, "math": 0.18}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_jsonl(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def equal_length_batches(ids_list, B, rng):
    by = {}
    for ids in ids_list:
        by.setdefault(len(ids), []).append(ids)
    for size in (B, B // 2, B // 4, 8):
        pool = []
        for _, v in by.items():
            rng.shuffle(v)
            pool += [v[s:s + size] for s in range(0, len(v) - size + 1, size)]
        if len(pool) >= 4 or size == 8:
            rng.shuffle(pool)
            return pool
    return []


def gen_batch(tgt, ids, max_new, deadline):
    try:
        return [(ids, *tgt.generate(ids, max_new, deadline=deadline))]
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if ids.shape[0] < 8:
            raise
        log(f"  out of GPU memory at batch {ids.shape[0]}, splitting")
        h = ids.shape[0] // 2
        return gen_batch(tgt, ids[:h], max_new, deadline) + gen_batch(tgt, ids[h:], max_new, deadline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--minutes", type=float, default=float(os.environ.get("GEN_MINUTES", 22)))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("GEN_BATCH", 96)))
    ap.add_argument("--pool", type=int, default=int(os.environ.get("GEN_POOL", 12000)))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, res = Path(args.out), Path(args.results)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "dev").mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    report = {}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.target)
    dev = load_jsonl(Path(args.kit) / "data/dev/prompts.jsonl")
    ref = {r["id"]: r["token_ids"] for r in load_jsonl(Path(args.kit) / "data/dev/reference.jsonl")}
    pre, suf = list(CHAT_PREFIX), list(CHAT_SUFFIX)
    dev_texts = [tok.decode(p["prompt_token_ids"][len(pre):-len(suf)]) for p in dev]

    # -- chat template check: our fast path must reproduce the kit's prompt token ids
    ok = 0
    for p, text in list(zip(dev, dev_texts))[::16]:
        ok += (pre + tok(text, add_special_tokens=False)["input_ids"] + suf) == p["prompt_token_ids"]
    report["template_match"] = f"{ok}/{len(dev[::16])}"
    log(f"chat template check: {report['template_match']} dev prompts reproduced exactly")

    # -- prompt pools (CPU only, before the model is loaded)
    pools = {}
    for d in DOMAINS:
        log(f"building prompt pool: {d}")
        texts = PR.drop_dev_overlap(PR.build(d, args.pool, args.seed, log), dev_texts, log)
        ids = []
        for i in range(0, len(texts), 2000):
            enc = tok([t.strip() for t in texts[i:i + 2000]], add_special_tokens=False)["input_ids"]
            ids += [pre + e + suf for e in enc if 8 <= len(e) <= 440]
        pools[d] = ids
        log(f"  {d}: {len(ids)} prompts")
    report["pool_sizes"] = {d: len(v) for d, v in pools.items()}

    log("loading the target model")
    t0 = time.time()
    tgt = Target(args.target)
    report["load_s"] = round(time.time() - t0, 1)
    log(f"target loaded in {report['load_s']}s (hidden states via "
        f"{'base model' if tgt.use_base else 'LM-head hook'})")

    # -- greedy agreement with the kit's reference outputs (engine vs. this implementation)
    agree = []
    for p in [x for x in dev if x["domain"] == "math"][:6]:
        ids = torch.tensor([p["prompt_token_ids"]], device=tgt.device)
        toks, lens, _, _ = tgt.generate(ids, 48)
        r = ref[p["id"]][:48]
        g = toks[0, :len(r)].tolist()
        same = next((i for i, (a, b) in enumerate(zip(g, r)) if a != b), min(len(g), len(r)))
        agree.append(same / max(1, len(r)))
    report["greedy_prefix_agreement"] = round(sum(agree) / len(agree), 3)
    log(f"greedy agreement with reference outputs (first 48 tokens): {report['greedy_prefix_agreement']:.3f}")

    # -- hold-out features: hidden states along the dev references
    dev_seqs = {d: [] for d in DOMAINS}
    t0 = time.time()
    for p in dev:
        ids = p["prompt_token_ids"] + ref[p["id"]]
        x = torch.tensor([ids], device=tgt.device)
        h, _ = tgt.run(x, use_cache=False)
        dev_seqs[p["domain"]].append({"tokens": x[0], "hidden": h[0, :-1].clone(), "T": len(ids),
                                      "P": len(p["prompt_token_ids"]), "domain": p["domain"], "id": p["id"]})
    for d in DOMAINS:
        save_shard(str(out / "dev" / d), dev_seqs[d])
    log(f"dev features: {sum(len(v) for v in dev_seqs.values())} sequences in {time.time() - t0:.0f}s")

    # -- the simulator must reproduce the baseline head's measured acceptance
    cfg = HeadConfig.from_target(json.loads((Path(args.target) / "config.json").read_text()))
    head = build_head(load_mtp_state(args.baseline), cfg, tgt.device)
    E = tgt.embed.weight
    sim = simulate(head, lambda i: F.embedding(i, E), tgt.W, [s for d in DOMAINS for s in dev_seqs[d]])
    kit_base = json.loads((Path(args.kit) / "data/dev/baseline.json").read_text())
    report["sim_baseline_dev"] = sim
    report["kit_baseline_dev"] = {k: kit_base[k] for k in ("acceptance_length", "acceptance_by_domain")}
    log(f"simulated baseline acceptance {sim['acceptance_length']:.3f} "
        f"(kit measured {kit_base['acceptance_length']:.3f}); per domain "
        + ", ".join(f"{d} {sim['acceptance_by_domain'][d]:.3f}/{kit_base['acceptance_by_domain'][d]:.3f}"
                    for d in DOMAINS))
    log(f"step accuracies {[round(a, 3) for a in sim['step_accuracy']]}")
    (res / "datagen_checks.json").write_text(json.dumps(report, indent=2))
    del head, dev_seqs
    torch.cuda.empty_cache()

    # -- generation under a wall-clock budget
    stats = {}
    budget = args.minutes * 60
    for d in ("longform", "chat", "code", "math"):
        groups = equal_length_batches(pools[d], args.batch, rng)
        end = time.time() + budget * SHARE[d]
        n_tok = n_seq = 0
        t0 = time.time()
        for bi, batch in enumerate(groups):
            if time.time() > end - 20:
                break
            ids = torch.tensor(batch, device=tgt.device)
            tb = time.time()
            seqs = []
            for pid, toks, lens, hp, hs in gen_batch(tgt, ids, PR.MAX_NEW[d], end):
                P = pid.shape[1]
                for b in range(pid.shape[0]):
                    n = int(lens[b])
                    if n < 8:
                        continue
                    seqs.append({"tokens": torch.cat([pid[b], toks[b, :n]]),
                                 "hidden": torch.cat([hp[b], hs[b, :n - 1]]), "P": P, "domain": d})
                del toks, hp, hs
            if seqs:
                save_shard(str(out / "train" / f"{d}_{bi:04d}"), seqs)
            got = sum(int(s["tokens"].numel()) - s["P"] for s in seqs)
            n_tok += got
            n_seq += len(seqs)
            log(f"  {d} batch {bi}: {len(seqs)} seqs, {got} new tokens, "
                f"{got / max(1e-6, time.time() - tb):.0f} tok/s")
            del seqs
            torch.cuda.empty_cache()
        stats[d] = {"sequences": n_seq, "generated_tokens": n_tok, "seconds": round(time.time() - t0, 1)}
        log(f"{d}: {n_seq} sequences, {n_tok} generated tokens")
    report["generation"] = stats
    (res / "datagen_checks.json").write_text(json.dumps(report, indent=2))
    log("data generation finished")


if __name__ == "__main__":
    main()
