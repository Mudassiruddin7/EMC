#!/usr/bin/env python3
"""Measurements beyond the official score, with the kit's own engine command line and client.

  batch  concurrent requests 1, 2, 4 on the official engine settings (which allow 4
         sequences), then 8 and 16 on an engine with room for 16: aggregate throughput,
         per-request TPOT and TTFT, acceptance length
  long   prompts of about 1K to 16K tokens (a document assembled from the kit's reference
         outputs, then an instruction), one request at a time: TTFT, TPOT, acceptance length
  k      draft lengths 8 and 12 instead of the competition's fixed 3 (information only)

Environment as for scripts/kit.py (EMC2_HOME, EMC_TARGET, EMC_SUB, EMC2_WORK, EMC_RESULTS);
--target-only measures the target without a draft. Writes $EMC_RESULTS/bench_<tag>.json.

    python scripts/bench.py --tag draft --what batch,long,k
"""
import argparse
import json
import os
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

kit = Path(os.environ["EMC2_HOME"]).resolve()
work = Path(os.environ.get("EMC2_WORK", "emc2-work")).resolve()
cache = work / "cache"
os.environ.update({
    "EMC2_HOME": str(kit), "EMC2_WORK": str(work), "PYTHONDONTWRITEBYTECODE": "1",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
    "XDG_CACHE_HOME": str(cache), "HF_HOME": str(cache / "hf"), "VLLM_CACHE_ROOT": str(cache / "vllm"),
    "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "TRITON_CACHE_DIR": str(cache / "triton"),
    "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
})
sys.path.insert(0, str(kit))
from emc2 import client, common, server  # noqa: E402
from emc2.common import load_json, load_jsonl  # noqa: E402

TARGET = Path(os.environ["EMC_TARGET"]).resolve()
for mod in (common, server):
    for name, value in (("TARGET_DIR", TARGET), ("WORK", work)):
        if hasattr(mod, name):
            setattr(mod, name, value)
DATA = kit / "data" / "dev"
CHAT_PREFIX = [248045, 846, 198]                                      # <|im_start|>user\n
CHAT_SUFFIX = [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]   # ... assistant, no thinking
LENGTHS = (1000, 2000, 4000, 8000, 12000, 16000)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ms(xs, q=0.5):
    xs = sorted(x for x in xs if x is not None)
    return round(1000 * xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None


def subset(prompts, per_domain):
    seen, out = {}, []
    for p in prompts:
        if seen.get(p["domain"], 0) < per_domain:
            seen[p["domain"]] = seen.get(p["domain"], 0) + 1
            out.append(p)
    return out


class Bench:
    def __init__(self, sub, contract, target, stop):
        self.spec = {"method": "mtp", "model": str(sub)} if sub else None
        self.contract, self.target, self.stop = contract, target, stop
        self.timeout = contract["request_timeout_s"]

    def server(self, name, **override):
        return server.VllmServer(dict(self.contract, **override), self.target, spec=self.spec, name=name)

    def counters(self, srv):
        return client.spec_counters(srv.url) if self.spec else (0.0, 0.0)

    def warm(self, srv):
        for w in load_jsonl(DATA / "warmup.jsonl"):
            client.complete(srv.url, w["prompt_token_ids"], w["max_tokens"], self.stop, self.timeout)

    def run(self, srv, prompts, concurrency=1):
        """Send the prompts with `concurrency` requests in flight; one row of measurements."""
        a0, d0 = self.counters(srv)
        t0 = time.monotonic()
        call = lambda p: client.complete(srv.url, p["prompt_token_ids"], p["max_tokens"], self.stop, self.timeout)
        with ThreadPoolExecutor(concurrency) as ex:
            res = list(ex.map(call, prompts))
        wall = time.monotonic() - t0
        a1, d1 = self.counters(srv)
        toks = sum(len(r["token_ids"]) for r in res)
        return {"requests": len(res), "concurrency": concurrency, "output_tokens": toks,
                "wall_s": round(wall, 2), "throughput_tok_s": round(toks / wall, 1),
                "tpot_median_ms": ms([r["tpot_s"] for r in res]), "tpot_p90_ms": ms([r["tpot_s"] for r in res], 0.9),
                "ttft_median_ms": ms([r["ttft_s"] for r in res]), "ttft_max_ms": ms([r["ttft_s"] for r in res], 1.0),
                "acceptance_length": round(1 + (a1 - a0) / (d1 - d0), 4) if self.spec and d1 > d0 else None}


def long_prompts(per_length=2):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TARGET / "tokenizer.json"))
    sep = tok.encode("\n\n", add_special_tokens=False).ids
    instr = tok.encode("\n\nSummarize the text above in detail, section by section.", add_special_tokens=False).ids
    stream = []
    for r in load_jsonl(DATA / "reference.jsonl"):
        stream += [t for t in r["token_ids"] if t < 248044] + sep        # plain text only
    while len(stream) < 2 * LENGTHS[-1]:
        stream += stream
    out = []
    for L in LENGTHS:
        body = L - len(CHAT_PREFIX) - len(instr) - len(CHAT_SUFFIX)
        for j in range(per_length):
            off = (j * len(stream) // per_length) % (len(stream) - body)
            ids = CHAT_PREFIX + stream[off:off + body] + instr + CHAT_SUFFIX
            out.append({"length": len(ids), "prompt_token_ids": ids, "max_tokens": 256})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--what", default="batch,long")
    ap.add_argument("--target-only", action="store_true")
    ap.add_argument("--per-domain", type=int, default=8)
    a = ap.parse_args()
    sub = None if a.target_only or not os.environ.get("EMC_SUB") else Path(os.environ["EMC_SUB"]).resolve()
    contract, target = load_json(kit / "contract.json"), load_json(kit / "target.json")
    b = Bench(sub, contract, target, load_json(DATA / "meta.json")["stop_token_ids"])
    prompts = subset(load_jsonl(DATA / "prompts.jsonl"), a.per_domain)
    out = {"tag": a.tag, "draft": str(sub) if sub else None, "prompts": len(prompts),
           "official_max_num_seqs": contract["max_num_seqs"], "batch": [], "long": [], "k": [], "errors": []}
    dest = Path(os.environ.get("EMC_RESULTS", work / "results")) / f"bench_{a.tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    save = lambda: dest.write_text(json.dumps(out, indent=2))
    what = set(a.what.split(","))

    def section(name, fn):
        if name not in what:
            return
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - keep the other measurements
            out["errors"].append(f"{name}: {type(e).__name__}: {e}")
            traceback.print_exc()
        save()

    def batch():
        for max_seqs, levels in ((contract["max_num_seqs"], (1, 2, 4)), (16, (8, 16))):
            with b.server(f"bench-batch{max_seqs}", max_num_seqs=max_seqs) as srv:
                b.warm(srv)
                for n in levels:
                    row = dict(b.run(srv, prompts, n), engine_max_num_seqs=max_seqs)
                    out["batch"].append(row)
                    log(f"batch {n}: {row}")
                    save()

    def long():
        rows = long_prompts()
        with b.server("bench-long") as srv:
            b.warm(srv)
            for L in LENGTHS:
                group = [r for r in rows if abs(r["length"] - L) < 64]
                row = dict(b.run(srv, group, 1), prompt_tokens=L)
                out["long"].append(row)
                log(f"context {L}: {row}")
                save()

    def k():
        if not sub:
            return
        for kk in (8, 12):
            with b.server(f"bench-k{kk}", num_speculative_tokens=kk) as srv:
                b.warm(srv)
                row = dict(b.run(srv, prompts, 1), num_speculative_tokens=kk)
                out["k"].append(row)
                log(f"K={kk}: {row}")
                save()

    section("batch", batch)
    section("long", long)
    section("k", k)
    log(f"wrote {dest}")
    return 0 if not out["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
