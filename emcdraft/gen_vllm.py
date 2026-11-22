"""Greedy answers of the target to the training prompts, generated with the serving engine.

Using vLLM (the same engine and settings family the competition scores with) makes the
training text exactly the kind of text the draft will be asked to predict, and is an order of
magnitude faster than generating with the reference implementation. Hidden states are added
afterwards by `emcdraft.extract` in one teacher-forced pass per sequence.

    python -m emcdraft.gen_vllm --target DIR --prompts data/prompts --out data/gen --results results
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from .prompts import DOMAINS, MAX_NEW

SHARE = {"longform": 0.45, "chat": 0.22, "code": 0.15, "math": 0.18}   # of generated tokens


def share_from_env(text: str | None) -> dict:
    """GEN_SHARE="longform=0.6,chat=0.2,code=0.1,math=0.1" overrides the domain mix."""
    share = dict(SHARE)
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() in share:
                share[k.strip()] = float(v)
    total = sum(share.values())
    return {k: v / total for k, v in share.items()}
MEAN_LEN = {"chat": 383, "code": 270, "math": 309, "longform": 938}     # dev reference means
STOP_IDS = [248046, 248044]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_llm(target: str):
    from vllm import LLM
    kw = dict(model=target, tokenizer=target, dtype="bfloat16", seed=0,
              max_model_len=int(os.environ.get("GEN_MAX_LEN", 2048)),
              gpu_memory_utilization=float(os.environ.get("GEN_GPU_UTIL", 0.90)),
              max_num_seqs=int(os.environ.get("GEN_MAX_SEQS", 128)),
              enable_prefix_caching=False, generation_config="vllm", language_model_only=True)
    for drop in ((), ("language_model_only",), ("language_model_only", "generation_config")):
        try:
            return LLM(**{k: v for k, v in kw.items() if k not in drop})
        except TypeError as e:
            log(f"engine rejected an option ({e}); retrying without {drop or 'nothing'}")
    raise RuntimeError("could not start the engine")


def previous_outputs(out: Path):
    """Answers already generated in `out` (any <domain>*.jsonl file): the prompts per domain,
    per-domain statistics, the token total, and the next free file suffix for new answers."""
    seen = {d: set() for d in DOMAINS}
    stats = {d: {"sequences": 0, "generated_tokens": 0, "stopped": 0} for d in DOMAINS}
    total = 0
    for d in DOMAINS:
        for p in sorted(out.glob(f"{d}*.jsonl")):
            for line in open(p):
                if not line.strip():
                    continue
                r = json.loads(line)
                seen[d].add(tuple(r["p"]))
                stats[d]["sequences"] += 1
                stats[d]["generated_tokens"] += len(r["o"])
                stats[d]["stopped"] += int(bool(r["o"]) and r["o"][-1] in STOP_IDS)
                total += len(r["o"])
    part = 2
    while any((out / f"{d}-{part}.jsonl").exists() for d in DOMAINS):
        part += 1
    return seen, stats, total, f"-{part}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tokens", type=float, default=float(os.environ.get("GEN_TOKENS", 2.0e6)))
    ap.add_argument("--minutes", type=float, default=float(os.environ.get("GEN_MINUTES", 30)))
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("GEN_CHUNK", 2000)))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # GEN_APPEND=1: keep earlier answers, count them towards --tokens and write new ones to
    # <domain>-<n>.jsonl (prompt selection is a superset of the earlier one: same seed)
    if os.environ.get("GEN_APPEND", "0") == "1":
        seen, stats, total, suffix = previous_outputs(out)
        log(f"append mode: {total} tokens already generated; new answers go to *{suffix}.jsonl")
    else:
        for p in out.glob("*.jsonl"):
            p.unlink()
        seen = {d: set() for d in DOMAINS}
        stats = {d: {"sequences": 0, "generated_tokens": 0, "stopped": 0} for d in DOMAINS}
        total, suffix = 0, ""
    previous = total

    share = share_from_env(os.environ.get("GEN_SHARE"))
    log(f"domain mix of the token budget: { {d: round(v, 3) for d, v in share.items()} }")
    items = []
    for d in DOMAINS:
        pool = [json.loads(line)["ids"] for line in open(Path(a.prompts) / f"{d}.jsonl") if line.strip()]
        rng.shuffle(pool)
        want = math.ceil(1.1 * a.tokens * share[d] / MEAN_LEN[d])
        new = [(d, ids) for ids in pool[:want] if tuple(ids) not in seen[d]]
        items += new
        log(f"{d}: {len(new)} new prompts (pool {len(pool)}, {len(seen[d])} answered before)")
    rng.shuffle(items)
    if not items or total >= a.tokens:
        log("nothing to generate")
        return

    from vllm import SamplingParams
    sp = {d: SamplingParams(temperature=0.0, max_tokens=MAX_NEW[d], stop_token_ids=STOP_IDS,
                            detokenize=False) for d in DOMAINS}
    t0 = time.time()
    llm = make_llm(a.target)
    log(f"engine ready in {time.time() - t0:.0f}s")
    t0 = time.time()
    files = {d: open(out / f"{d}{suffix}.jsonl", "a") for d in DOMAINS}
    for s in range(0, len(items), a.chunk):
        if time.time() - t0 > a.minutes * 60 or total >= a.tokens:
            log("budget reached")
            break
        chunk = items[s:s + a.chunk]
        tc = time.time()
        res = llm.generate([{"prompt_token_ids": ids} for _, ids in chunk], [sp[d] for d, _ in chunk],
                           use_tqdm=False)
        got = 0
        for (d, ids), r in zip(chunk, res):
            o = list(r.outputs[0].token_ids)
            files[d].write(json.dumps({"p": ids, "o": o}) + "\n")
            stats[d]["sequences"] += 1
            stats[d]["generated_tokens"] += len(o)
            stats[d]["stopped"] += int(bool(o) and o[-1] in STOP_IDS)
            got += len(o)
        for f in files.values():
            f.flush()
        total += got
        log(f"chunk {s // a.chunk}: {len(chunk)} prompts, {got} tokens, {got / (time.time() - tc):.0f} tok/s "
            f"(total {total})")
    for f in files.values():
        f.close()
    report = {"seconds": round(time.time() - t0, 1), "generated_tokens": total, "previous_tokens": previous,
              "by_domain": stats}
    Path(a.results).mkdir(parents=True, exist_ok=True)
    (Path(a.results) / "generation.json").write_text(json.dumps(report, indent=2))
    log(f"generated {total} tokens in {report['seconds']}s")


if __name__ == "__main__":
    main()
