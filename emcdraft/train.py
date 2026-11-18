"""Fine-tune the MTP head on the target's own outputs.

Objective: for every position of a greedy answer, run the same K-step recursive drafting
the engine runs (step k re-uses the head's own hidden state from step k-1) and train every
step to produce the token the target actually emitted, plus (weight beta) the target's full
next-token distribution at that position. Later steps are weighted by whether the earlier
steps were right, because the engine only ever reaches them in that case.

With --qat the large matrices are trained through FP8 (E4M3) rounding (straight-through),
after an exact re-scaling that puts them in E4M3's relative-precision range, so the compact
on-disk variant is the model that was actually optimised.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .head import (FP8_TENSORS, HeadConfig, build_head, e4m3_round, equalize_, export_submission,
                   load_mtp_state, load_target_tensors, unroll)
from .sim import DOMAINS, batches, collate, load_shards, shards_bytes, simulate


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def head_loss(o, W, y, w, teach, beta, chunk=2048, topk=0):
    """Weighted CE (+ beta * soft-target CE) through the frozen LM head, in memory-bounded
    chunks. Returns a surrogate whose gradient w.r.t. `o` equals the loss gradient.

    topk > 0 restricts the soft target to the target's `topk` most likely tokens (the
    "harmonized objective distillation" of HASS): -sum_{t in top-k(p)} p(t) log q(t)."""
    n = o.shape[0]
    denom = w.sum().clamp(min=1.0)
    surrogate = torch.zeros((), device=o.device)
    ce_sum = 0.0
    correct = torch.zeros(n, dtype=torch.bool, device=o.device)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        z = o[s:e].detach().requires_grad_(True)
        logits = F.linear(z, W.to(z.dtype)).float()
        logp = F.log_softmax(logits, -1)
        ce = F.nll_loss(logp, y[s:e], reduction="none")
        loss = (ce * w[s:e]).sum()
        if teach is not None and beta > 0:
            with torch.no_grad():
                tl = F.linear(teach[s:e], W.to(teach.dtype)).float()
                if topk:
                    ix = tl.topk(topk, -1).indices
                    tp = F.softmax(tl, -1).gather(-1, ix)
                else:
                    tp = F.softmax(tl, -1)
            lq = logp.gather(-1, ix) if topk else logp
            loss = loss + beta * (-(tp * lq).sum(-1) * w[s:e]).sum()
        (g,) = torch.autograd.grad(loss / denom, z)
        surrogate = surrogate + (o[s:e].float() * g.float()).sum()
        ce_sum += float((ce.detach() * w[s:e]).sum())
        correct[s:e] = logits.argmax(-1) == y[s:e]
    return surrogate, ce_sum / float(denom), correct


def train_step(head, embed, W, batch, a, K=3):
    tokens, hidden, P, T = batch
    B, Tm = tokens.shape
    N = Tm - 1
    base = torch.arange(N, device=tokens.device)[None]
    total = torch.zeros((), device=tokens.device)
    info, prev = {}, None
    cuda = tokens.is_cuda
    with torch.autocast("cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda):
        for k, out in unroll(head, embed, tokens, hidden, K):
            valid = (base >= (P[:, None] - 1)) & ((base + 1 + k) <= (T[:, None] - 1))
            labels = tokens[:, (base[0] + 1 + k).clamp(max=Tm - 1)]
            w = valid.float() if prev is None else valid.float() * (a.floor + (1 - a.floor) * prev.float())
            bi, ni = valid.nonzero(as_tuple=True)
            # the target's own distribution for the same token: h_{i+k} predicts x_{i+k+1}
            teach = hidden[bi, (ni + k).clamp(max=N - 1)] if a.beta > 0 else None
            s_k, ce_k, corr = head_loss(out[bi, ni], W, labels[bi, ni], w[bi, ni], teach, a.beta,
                                        topk=getattr(a, "topk", 0))
            total = total + getattr(a, "step_w_cur", a.step_w)[k - 1] * s_k
            full = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
            full[bi, ni] = corr
            reach = valid if prev is None else valid & prev
            info[f"ce{k}"] = ce_k
            info[f"acc{k}"] = float((full & reach).sum()) / max(1.0, float(reach.sum()))
            prev = full if prev is None else prev & full
    return total, info


def split_val(seqs, frac, rng):
    by = {}
    for s in seqs:
        by.setdefault(s["domain"], []).append(s)
    train, val = [], []
    for d, v in by.items():
        rng.shuffle(v)
        n = max(4, min(40, int(len(v) * frac))) if len(v) >= 16 else 0
        val += v[:n]
        train += v[n:]
    if not val:                       # tiny smoke-test datasets: validate on a training slice
        val = train[:8]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--init", required=True, help="folder with the starting mtp.* weights")
    ap.add_argument("--data", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qat", action="store_true")
    ap.add_argument("--epochs", type=float, default=float(os.environ.get("TRAIN_EPOCHS", 2)))
    ap.add_argument("--max-minutes", type=float, default=float(os.environ.get("TRAIN_MINUTES", 22)))
    ap.add_argument("--batch-tokens", type=int, default=int(os.environ.get("TRAIN_BATCH_TOKENS", 8192)))
    ap.add_argument("--rel-lr", type=float, default=float(os.environ.get("TRAIN_REL_LR", 1e-3)),
                    help="peak Adam step as a fraction of each tensor's RMS")
    ap.add_argument("--beta", type=float, default=float(os.environ.get("TRAIN_BETA", 0.5)),
                    help="weight of the soft-target (distillation) term")
    ap.add_argument("--topk", type=int, default=int(os.environ.get("TRAIN_TOPK", 0)),
                    help="distil only the target's top-k tokens (0: full distribution)")
    ap.add_argument("--floor", type=float, default=float(os.environ.get("TRAIN_FLOOR", 0.3)),
                    help="weight of later-step positions the engine would not reach")
    ap.add_argument("--step-w", nargs="+", default=os.environ.get("TRAIN_STEP_W", "1 0.7 0.5").split(),
                    help="three step weights, or 'auto': each step weighted by the derivative of the "
                         "acceptance length with respect to its accuracy")
    ap.add_argument("--max-steps", type=int, default=0, help="cap on optimizer steps (short ablation runs)")
    ap.add_argument("--no-export", action="store_true", help="ablation run: no submission folders")
    ap.add_argument("--quick", action="store_true", help="ablation run: simulate the validation set only")
    ap.add_argument("--eval-every", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--team", default=os.environ.get("TEAM_ID", "mudassiruddin"))
    a = ap.parse_args()
    auto_w = [x.lower() for x in a.step_w] == ["auto"]
    if not auto_w:
        a.step_w = [float(x) for x in a.step_w]
        assert len(a.step_w) == 3, "--step-w takes three numbers or 'auto'"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()

    cfg = HeadConfig.from_target(json.loads((Path(a.target) / "config.json").read_text()))
    names = ["lm_head.weight", "model.language_model.embed_tokens.weight"]
    tt = load_target_tensors(a.target, names)
    W = tt[names[0]].to(dev)
    E = tt[names[1]].to(dev)
    embed = lambda ids: F.embedding(ids, E)
    head = build_head(load_mtp_state(a.init), cfg, dev)

    # hidden states stay on the GPU while they fit comfortably, otherwise in host memory and
    # each batch is copied over (TRAIN_DATA_DEVICE=cpu|cuda overrides)
    data_dev = os.environ.get("TRAIN_DATA_DEVICE") or ("cpu" if shards_bytes(a.data) > 24e9 else dev)
    seqs = load_shards(a.data, data_dev)
    # validation comes from the first extraction only, so with appended data it is the same
    # hold-out set as in the earlier run (whose head may be the starting point here)
    train, val = split_val([s for s in seqs if s["part"] == 1], 0.05, rng)
    train += [s for s in seqs if s["part"] != 1]
    dev_seqs = load_shards(a.dev, dev) if a.dev and Path(a.dev).exists() else []
    tok_by = {d: sum(s["T"] - s["P"] for s in seqs if s["domain"] == d) for d in DOMAINS}
    log(f"data: {len(train)} train / {len(val)} val sequences (held on {data_dev}), "
        f"generated tokens by domain {tok_by}")
    summary = {"args": vars(a), "data": {"train_sequences": len(train), "val_sequences": len(val),
                                         "generated_tokens": tok_by, "data_device": data_dev}, "sim": {}}

    def sims(tag):
        r = {"val": simulate(head, embed, W, val)}
        if dev_seqs and not a.quick:
            r["dev"] = simulate(head, embed, W, dev_seqs)
        summary["sim"][tag] = r
        log(f"[{tag}] val tau {r['val']['acceptance_length']:.4f}"
            + (f" | dev tau {r['dev']['acceptance_length']:.4f} "
               f"{ {d: round(v, 3) for d, v in r['dev']['acceptance_by_domain'].items()} } "
               f"steps {[round(x, 3) for x in r['dev']['step_accuracy']]}" if "dev" in r else ""))
        return r

    sims("init")
    if a.qat:
        probe = collate(val[:2], device=dev)
        with torch.no_grad():
            ref, _ = head.step(embed(probe[0][:, 1:]).float(), probe[1].float(),
                               torch.arange(probe[0].shape[1] - 1, device=dev)[None].expand(2, -1))
            summary["equalize"] = equalize_(head)
            new, _ = head.step(embed(probe[0][:, 1:]).float(), probe[1].float(),
                               torch.arange(probe[0].shape[1] - 1, device=dev)[None].expand(2, -1))
        drift = float((new - ref).abs().max() / ref.abs().max())
        summary["equalize"]["max_rel_drift"] = drift
        log(f"equalisation factors {summary['equalize']} (must be function-preserving)")
        if drift > 1e-2:
            raise SystemExit(f"equalisation changed the head's output (drift {drift:.3g})")
        if not a.quick:
            sims("equalized")
        head.set_fake_quant(True)
        sims("fp8_untrained")

    groups = []
    for n, p in head.named_parameters():
        eff = (1 + p.detach()) if p.ndim == 1 else p.detach()
        groups.append({"params": [p], "base": a.rel_lr * float(eff.float().pow(2).mean().sqrt()), "lr": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0)

    per_epoch = len(batches(train, a.batch_tokens))
    total_steps = max(1, int(per_epoch * a.epochs))
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    warm = max(5, total_steps // 30)
    key = "val"
    start = summary["sim"]["fp8_untrained" if a.qat else "init"][key]
    best = {"tau": start["acceptance_length"],
            "step": 0, "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
    # running per-step conditional accuracies, for acceptance-aware step weights
    acc_ema = list(start["step_accuracy"])
    log(f"training: {total_steps} steps planned ({per_epoch}/epoch), qat={a.qat}, "
        f"step weights {'auto' if auto_w else a.step_w}, top-k {a.topk}, floor {a.floor}, rel-lr {a.rel_lr}")
    step, t0, hist = 0, time.time(), []
    head.train()
    done = False
    while not done:
        for group in batches(train, a.batch_tokens, rng):
            if step == 12:
                per = (time.time() - t0) / 12
                fit = int((a.max_minutes * 60 - (time.time() - t_start) - 240) / max(per, 1e-3))
                if fit < total_steps - step:
                    total_steps = max(step + 20, step + fit)
                    log(f"{per:.2f}s/step: shortening the schedule to {total_steps} steps to fit the budget")
            f = (step + 1) / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base"] * f
            if auto_w:
                # d tau / d a_k for tau = 1 + a1 + a1 a2 + a1 a2 a3, normalised to step 1
                a1, a2, a3 = acc_ema
                d = [1 + a2 + a2 * a3, a1 * (1 + a3), a1 * a2]
                a.step_w_cur = [x / d[0] for x in d]
            loss, info = train_step(head, embed, W, collate(group, device=dev), a)
            acc_ema = [0.98 * e + 0.02 * info[f"acc{k + 1}"] for k, e in enumerate(acc_ema)]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 10 == 0 or step == 1:
                est = 1 + info["acc1"] * (1 + info["acc2"] * (1 + info["acc3"]))
                log(f"step {step}/{total_steps} ce {info['ce1']:.3f}/{info['ce2']:.3f}/{info['ce3']:.3f} "
                    f"acc {info['acc1']:.3f}/{info['acc2']:.3f}/{info['acc3']:.3f} batch-tau~{est:.3f} "
                    f"lr x{f:.2f} {(time.time() - t0) / step:.2f}s/step"
                    + (f" w {[round(x, 2) for x in a.step_w_cur]}" if auto_w else ""))
            if step % a.eval_every == 0 or step >= total_steps:
                r = simulate(head, embed, W, val)
                d = simulate(head, embed, W, dev_seqs[::4]) if dev_seqs else None
                hist.append({"step": step, "val": r["acceptance_length"],
                             "dev_quarter": d["acceptance_length"] if d else None})
                log(f"eval step {step}: val tau {r['acceptance_length']:.4f}"
                    + (f" | dev(1/4) tau {d['acceptance_length']:.4f}" if d else "")
                    + f" (best {best['tau']:.4f} @ {best['step']})")
                if r["acceptance_length"] > best["tau"]:
                    best = {"tau": r["acceptance_length"], "step": step,
                            "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
                head.train()
            if step >= total_steps:
                done = True
                break
    head.load_state_dict(best["state"])
    head.eval()
    summary["history"] = hist
    summary["best_step"] = best["step"]
    summary["train_seconds"] = round(time.time() - t0, 1)
    log(f"best checkpoint: step {best['step']} (val tau {best['tau']:.4f})")
    summary["best_val_tau"] = best["tau"]
    if a.no_export:
        summary["total_seconds"] = round(time.time() - t_start, 1)
        (out / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log("ablation run finished (no export)")
        return

    meta = {"team_id": a.team,
            "method_description": ("Replacement MTP head in the baseline layout, initialised from the target's "
                                   "built-in head and fine-tuned with a 3-step recursive drafting objective"
                                   + (", acceptance-aware step weights" if auto_w else "")
                                   + (f", top-{a.topk} distillation" if a.topk else "")
                                   + (" and FP8 (E4M3) quantisation-aware training" if a.qat else "")),
            "training_data": ("Greedy answers of the target itself (with its hidden states) to public "
                              "instruction prompts: GSM8K/Orca-Math, WritingPrompts, UltraChat/Alpaca/Dolly, "
                              "Python instruction sets; dev prompts excluded")}
    tcfg = Path(a.target) / "config.json"
    if a.qat:
        sims("final_fp8")
        summary["export_fp8"] = export_submission(head, out / "sub_fp8", tcfg, True, meta)
        back = load_mtp_state(out / "sub_fp8")
        for n in FP8_TENSORS:
            want = e4m3_round(head.state_dict()[n].float().cpu())
            assert torch.equal(back[n], want), f"FP8 round trip mismatch in {n}"
        head.set_fake_quant(False)
    sims("final_bf16")
    summary["export_bf16"] = export_submission(head, out / "sub_bf16", tcfg, False, meta)
    summary["total_seconds"] = round(time.time() - t_start, 1)
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("training finished")


if __name__ == "__main__":
    main()
