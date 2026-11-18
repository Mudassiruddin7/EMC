"""Training prompts for the four evaluated domains (chat, code, math, long-form writing).

Prompts come from public instruction datasets and are phrased like the public dev tasks.
Anything that overlaps the dev prompts is dropped, so the dev set stays a clean hold-out.

    python -m emcdraft.prompts --target DIR --kit DIR --out data/prompts --results results

writes data/prompts/<domain>.jsonl ({"ids": chat-templated prompt token ids}) and
results/prompts_checks.json (template round-trip check against the kit's dev prompts).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

CODE_T = ("Complete the following Python function. Reply with the full function in a single "
          "Python code block.\n\n```python\n{stub}```")
MATH_T = ("{q}\n\nSolve this step by step, then give the final answer on its own line as "
          "'Answer: <number>'.")
STORY_T = "Write a short story of about {n} words based on this prompt:\n\n{p}"

DOMAINS = ("chat", "code", "math", "longform")
MAX_NEW = {"chat": 512, "code": 512, "math": 384, "longform": 1024}   # as in the dev set
CHAT_PREFIX = [248045, 846, 198]                                       # <|im_start|>user\n
CHAT_SUFFIX = [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
# <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n   (enable_thinking = False)

DEF_RE = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(.*?\)\s*(->\s*[^:\n]+)?:\s*$", re.M)
TAG_RE = re.compile(r"^\s*\[\s*([A-Za-z]{2})\s*\]\s*")
DETOK = ((" n't", "n't"), (" 's", "'s"), (" 're", "'re"), (" 've", "'ve"), (" 'm", "'m"),
         (" 'll", "'ll"), (" 'd", "'d"), (" ,", ","), (" .", "."), (" ?", "?"), (" !", "!"),
         (" ;", ";"), (" :", ":"), ("`` ", '"'), (" ''", '"'), ("( ", "("), (" )", ")"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def take(name, config, split, n, seed):
    """Up to n rows of a shuffled stream; an unavailable source just yields nothing."""
    try:
        from datasets import load_dataset
        args = (name, config) if config else (name,)
        ds = load_dataset(*args, split=split, streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=min(10000, max(500, 2 * n)))
        out = []
        for row in ds:
            out.append(row)
            if len(out) >= n:
                break
        log(f"  source {name}: {len(out)} rows")
        return out
    except Exception as e:  # noqa: BLE001
        log(f"  source {name}: unavailable ({type(e).__name__}: {str(e)[:160]})")
        return []


def make_stub(instruction: str, code: str):
    """A HumanEval-style stub (signature + docstring) from an instruction/solution pair."""
    code = code.replace("\r\n", "\n").replace("\t", "    ")
    m = DEF_RE.search(code)
    instruction = instruction.strip()
    if not m or not (15 <= len(instruction) <= 500):
        return None
    doc = "\n    ".join(line.rstrip() for line in instruction.replace('"""', "'''").splitlines())
    return f'{m.group(0).rstrip()}\n    """{doc}\n    """\n'


def clean_story_prompt(p: str):
    m = TAG_RE.match(p)
    if not m or m.group(1).upper() != "WP":
        return None
    p = p[m.end():]
    for a, b in DETOK:
        p = p.replace(a, b)
    p = re.sub(r"\s+", " ", p).strip()
    return p if 15 <= len(p) <= 400 else None


def build(domain: str, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    if domain == "math":
        for r in take("openai/gsm8k", "main", "train", 8000, seed):
            out.append(MATH_T.format(q=r["question"].strip()))
        for r in take("microsoft/orca-math-word-problems-200k", None, "train", n, seed):
            q = r["question"].strip()
            if 20 <= len(q) <= 800:
                out.append(MATH_T.format(q=q))
    elif domain == "code":
        pairs = [(r["instruction"], r["output"]) for r in
                 take("iamtarun/python_code_instructions_18k_alpaca", None, "train", 2 * n, seed)]
        pairs += [(r["instruction"], r["output"]) for r in
                  take("sahil2801/CodeAlpaca-20k", None, "train", 2 * n, seed)]
        pairs += [(r["text"], r["code"]) for r in take("google-research-datasets/mbpp", "full", "train", 1000, seed)]
        free = []
        for instr, code in pairs:
            stub = make_stub(instr, code)
            if stub:
                out.append(CODE_T.format(stub=stub))
            elif 20 <= len(instr.strip()) <= 600 and "python" in instr.lower():
                free.append(instr.strip())
        out += free[: max(1, len(out) // 3)]
    elif domain == "longform":
        for r in take("euclaise/writingprompts", None, "train", 4 * n, seed):
            p = clean_story_prompt(r["prompt"])
            if p:
                words = 600 if rng.random() < 0.8 else rng.choice([300, 400, 500, 800])
                out.append(STORY_T.format(n=words, p=p))
    elif domain == "chat":
        for r in take("HuggingFaceH4/ultrachat_200k", None, "train_sft", n, seed):
            out.append(r["prompt"].strip())
        for r in take("tatsu-lab/alpaca", None, "train", n // 2, seed):
            out.append((r["instruction"] + ("\n\n" + r["input"] if r.get("input") else "")).strip())
        for r in take("databricks/databricks-dolly-15k", None, "train", n // 2, seed):
            out.append((r["instruction"] + ("\n\n" + r["context"] if r.get("context") else "")).strip())
        out = [p for p in out if 20 <= len(p) <= 1500]
    else:
        raise ValueError(domain)
    out = list(dict.fromkeys(out))
    rng.shuffle(out)
    return out


def drop_dev_overlap(prompts: list[str], dev_texts: list[str]) -> list[str]:
    """Remove every training prompt whose task text also appears in the public dev set."""
    blob = " || ".join(norm(t) for t in dev_texts)
    dev_defs = {m.group(1) for t in dev_texts for m in DEF_RE.finditer(t)}
    heads = [t.split("{")[0] for t in (CODE_T, MATH_T, STORY_T)]
    kept = []
    for p in prompts:
        core = p
        for h in heads:
            if h and p.startswith(h):
                core = p[len(h):]
        core = core.split("\n\nSolve this step by step")[0]
        key = norm(core)[:160]
        names = {m.group(1) for m in DEF_RE.finditer(p)}
        if (len(key) >= 20 and key in blob) or (names & dev_defs):
            continue
        kept.append(p)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--n", type=int, default=4000, help="rows drawn per source")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.target) / "tokenizer.json"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(a.results).mkdir(parents=True, exist_ok=True)

    dev = [json.loads(line) for line in open(Path(a.kit) / "data/dev/prompts.jsonl") if line.strip()]
    pre, suf = len(CHAT_PREFIX), len(CHAT_SUFFIX)
    dev_texts = [tok.decode(p["prompt_token_ids"][pre:-suf], skip_special_tokens=False) for p in dev]
    sample = list(zip(dev, dev_texts))[::4]
    ok = sum(CHAT_PREFIX + tok.encode(t, add_special_tokens=False).ids + CHAT_SUFFIX == p["prompt_token_ids"]
             for p, t in sample)
    report = {"template_match": f"{ok}/{len(sample)}", "domains": {}}
    log(f"chat template check: {ok}/{len(sample)} dev prompts reproduced token-for-token")
    if ok < 0.9 * len(sample):
        raise SystemExit("the prompt template does not reproduce the dev prompts")

    for d in DOMAINS:
        log(f"domain {d}")
        texts = build(d, a.n, a.seed)
        kept = drop_dev_overlap(texts, dev_texts)
        rows = []
        for i in range(0, len(kept), 1000):
            for e in tok.encode_batch(kept[i:i + 1000], add_special_tokens=False):
                if 8 <= len(e.ids) <= 440:
                    rows.append(CHAT_PREFIX + e.ids + CHAT_SUFFIX)
        with open(out / f"{d}.jsonl", "w") as f:
            for ids in rows:
                f.write(json.dumps({"ids": ids}) + "\n")
        lens = sorted(len(r) for r in rows) or [0]
        report["domains"][d] = {"candidates": len(texts), "after_dev_filter": len(kept), "prompts": len(rows),
                                "median_prompt_tokens": lens[len(lens) // 2]}
        log(f"  {d}: {len(rows)} prompts (dev-overlap filter removed {len(texts) - len(kept)})")
        if len(rows) < 200:
            raise SystemExit(f"too few prompts for {d}")
    (Path(a.results) / "prompts_checks.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
