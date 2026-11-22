"""Offline acceptance simulator and feature-shard helpers.

Given target hidden states for a token sequence, `simulate` replays greedy speculative
decoding with K draft tokens exactly as the engine's proposer does (recursive use of the
head's own hidden state, verified keys/values for the prefix) and returns the acceptance
length  tau = 1 + accepted / rounds  - the same quantity the kit reads from the engine's
counters. It needs no server start, so it is used for validation during training.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .head import unroll, walk

DOMAINS = ("chat", "code", "math", "longform")


def fp8_quantize(h: torch.Tensor):
    """Per-row FP8 (E4M3) storage of hidden states: (uint8 codes, float32 row scales)."""
    hf = h.float()
    scale = (hf.abs().amax(-1, keepdim=True) / 448.0).clamp(min=1e-12)
    q = (hf / scale).to(torch.float8_e4m3fn)
    return q.view(torch.uint8), scale.squeeze(-1)


def fp8_dequantize(codes: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    return (codes.view(torch.float8_e4m3fn).float() * scale.float()[..., None]).to(dtype)


def save_shard(stem: str, seqs: list[dict], fp8: bool = False):
    """seqs: dicts with tokens [T] (long), hidden [T-1, H] (bf16), P, domain, optional id.
    fp8=True stores the hidden states as E4M3 codes with one scale per position (half the
    bytes of bf16; about 2-3% relative error per value)."""
    toks = torch.cat([s["tokens"] for s in seqs]).to(torch.int32).cpu().numpy()
    hid = torch.cat([s["hidden"] for s in seqs]).to(torch.bfloat16)
    np.save(stem + ".tok.npy", toks)
    if fp8:
        codes, scale = fp8_quantize(hid)
        np.save(stem + ".hid8.npy", codes.cpu().numpy())
        np.save(stem + ".scale.npy", scale.cpu().numpy().astype(np.float32))
    else:
        np.save(stem + ".hid.npy", hid.cpu().view(torch.int16).numpy())
    meta = {"T": [int(s["tokens"].numel()) for s in seqs], "P": [int(s["P"]) for s in seqs],
            "domain": [s["domain"] for s in seqs], "id": [s.get("id", "") for s in seqs]}
    Path(stem + ".json").write_text(json.dumps(meta))


def shards_bytes(data_dir) -> int:
    return sum(p.stat().st_size for pat in ("*.hid.npy", "*.hid8.npy") for p in Path(data_dir).glob(pat))


def load_shards(data_dir, device, max_tokens: int | None = None) -> list[dict]:
    """Every sequence of the shards in `data_dir` (sorted by name), with the extraction part it
    came from (shards g... are part 1, appended shards p<k>g... part k)."""
    seqs, total = [], 0
    for meta_path in sorted(Path(data_dir).glob("*.json")):
        stem = str(meta_path)[:-5]
        name = meta_path.stem
        part = int(name[1:name.index("g")]) if name.startswith("p") else 1
        meta = json.loads(meta_path.read_text())
        tok = torch.from_numpy(np.load(stem + ".tok.npy").astype(np.int64)).to(device)
        scale = None
        if Path(stem + ".hid8.npy").exists():       # FP8 codes + per-position scales
            hid = torch.from_numpy(np.load(stem + ".hid8.npy")).to(device)
            scale = torch.from_numpy(np.load(stem + ".scale.npy")).to(device)
        else:
            hid = torch.from_numpy(np.load(stem + ".hid.npy")).view(torch.bfloat16).to(device)
        to = ho = 0
        for T, P, dom, sid in zip(meta["T"], meta["P"], meta["domain"], meta["id"]):
            s = {"tokens": tok[to:to + T], "hidden": hid[ho:ho + T - 1], "P": P, "T": T,
                 "domain": dom, "id": sid, "part": part}
            if scale is not None:
                s["scale"] = scale[ho:ho + T - 1]
            seqs.append(s)
            to, ho, total = to + T, ho + T - 1, total + T
        if max_tokens and total >= max_tokens:
            break
    return seqs


def collate(seqs: list[dict], pad: int = 0, device=None):
    """Pad a group into (tokens, hidden, P, T); built where the data lives, then moved to `device`."""
    B, Tm = len(seqs), max(s["T"] for s in seqs)
    dev = seqs[0]["tokens"].device
    H = seqs[0]["hidden"].shape[-1]
    fp8 = "scale" in seqs[0]
    tokens = torch.full((B, Tm), pad, dtype=torch.long, device=dev)
    hidden = torch.zeros((B, Tm - 1, H), dtype=seqs[0]["hidden"].dtype, device=dev)
    scale = torch.zeros((B, Tm - 1), dtype=torch.float32, device=dev) if fp8 else None
    for b, s in enumerate(seqs):
        tokens[b, :s["T"]] = s["tokens"]
        hidden[b, :s["T"] - 1] = s["hidden"]
        if fp8:
            scale[b, :s["T"] - 1] = s["scale"]
    P = torch.tensor([s["P"] for s in seqs], device=dev)
    T = torch.tensor([s["T"] for s in seqs], device=dev)
    if device is not None and torch.device(device) != torch.device(dev):
        tokens, hidden, P, T = (x.to(device) for x in (tokens, hidden, P, T))
        scale = scale.to(device) if fp8 else None
    if fp8:                                      # move the compact codes, expand on arrival
        hidden = fp8_dequantize(hidden, scale)
    return tokens, hidden, P, T


def batches(seqs: list[dict], max_tokens: int, shuffle_rng=None) -> list[list[dict]]:
    """Length-bucketed batches with at most `max_tokens` padded positions each."""
    order = sorted(seqs, key=lambda s: s["T"])
    out, cur, cur_max = [], [], 0
    for s in order:
        if cur and max(cur_max, s["T"]) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, cur_max = [], 0
        cur.append(s)
        cur_max = max(cur_max, s["T"])
    if cur:
        out.append(cur)
    if shuffle_rng is not None:
        shuffle_rng.shuffle(out)
    return out


def argmax_logits(h: torch.Tensor, W: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    return torch.cat([F.linear(h[i:i + chunk], W).float().argmax(-1) for i in range(0, h.shape[0], chunk)])


@torch.no_grad()
def simulate(head, embed, W, seqs: list[dict], K: int = 3, max_tokens: int = 12000) -> dict:
    """Acceptance length per domain and overall, plus per-step conditional accuracies."""
    was_training = head.training
    head.eval()
    acc, drafts = defaultdict(int), defaultdict(int)
    hit, tot = [0] * K, [0] * K
    cuda = W.is_cuda
    for group in batches(seqs, max_tokens):
        tokens, hidden, P, T = collate(group, device=W.device)
        B, Tm = tokens.shape
        N = Tm - 1
        base = torch.arange(N, device=tokens.device)
        flags = []
        with torch.autocast("cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda):
            for k, out in unroll(head, embed, tokens, hidden, K):
                lab = tokens[:, (base + 1 + k).clamp(max=Tm - 1)]
                pred = argmax_logits(out.reshape(B * N, -1), W).view(B, N)
                flags.append(pred == lab)
        prev = None
        for k in range(K):
            valid = (base[None] >= (P[:, None] - 1)) & ((base[None] + 2 + k) <= (T[:, None] - 1))
            reach = valid if prev is None else valid & prev
            hit[k] += int((flags[k] & reach).sum())
            tot[k] += int(reach.sum())
            prev = flags[k] if prev is None else prev & flags[k]
        cpu_flags = [f.cpu() for f in flags]
        for b, s in enumerate(group):
            a, d = walk([f[b].tolist() for f in cpu_flags], s["P"], s["T"], K)
            acc[s["domain"]] += a
            drafts[s["domain"]] += d
    head.train(was_training)
    res = {"acceptance_by_domain": {d: 1 + acc[d] / drafts[d] for d in acc if drafts[d]},
           "acceptance_length": 1 + sum(acc.values()) / max(1, sum(drafts.values())),
           "step_accuracy": [hit[k] / max(1, tot[k]) for k in range(K)],
           "rounds": sum(drafts.values()), "sequences": len(seqs)}
    return res
