"""Training prompts for the four evaluation domains (chat, code, math, long-form writing).

Prompts come from public instruction datasets and are wrapped in the same task phrasing the
public dev set uses. Anything that overlaps the dev prompts is dropped, so the dev set stays
a clean hold-out.
"""
from __future__ import annotations

import random
import re

CODE_T = ("Complete the following Python function. Reply with the full function in a single "
          "Python code block.\n\n```python\n{stub}```")
MATH_T = ("{q}\n\nSolve this step by step, then give the final answer on its own line as "
          "'Answer: <number>'.")
STORY_T = "Write a short story of about {n} words based on this prompt:\n\n{p}"

MAX_NEW = {"chat": 512, "code": 512, "math": 384, "longform": 1024}
DEF_RE = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(.*?\)\s*(->\s*[^:\n]+)?:\s*$", re.M)
TAG_RE = re.compile(r"^\s*\[\s*([A-Za-z]{2})\s*\]\s*")
DETOK = ((" n't", "n't"), (" 's", "'s"), (" 're", "'re"), (" 've", "'ve"), (" 'm", "'m"),
         (" 'll", "'ll"), (" 'd", "'d"), (" ,", ","), (" .", "."), (" ?", "?"), (" !", "!"),
         (" ;", ";"), (" :", ":"), ("`` ", '"'), (" ''", '"'), ("( ", "("), (" )", ")"))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def take(name, config, split, n, seed, log):
    """First n rows of a shuffled stream; an unavailable source just yields nothing."""
    try:
        from datasets import load_dataset
        args = (name, config) if config else (name,)
        ds = load_dataset(*args, split=split, streaming=True).shuffle(seed=seed, buffer_size=10000)
        out = []
        for row in ds:
            out.append(row)
            if len(out) >= n:
                break
        log(f"  source {name}: {len(out)} rows")
        return out
    except Exception as e:  # noqa: BLE001
        log(f"  source {name}: unavailable ({type(e).__name__}: {str(e)[:120]})")
        return []


def make_stub(instruction: str, code: str):
    code = code.replace("\r\n", "\n").replace("\t", "    ")
    m = DEF_RE.search(code)
    instruction = instruction.strip()
    if not m or not (15 <= len(instruction) <= 500):
        return None, None
    doc = "\n    ".join(line.rstrip() for line in instruction.replace('"""', "'''").splitlines())
    return f'{m.group(0).rstrip()}\n    """{doc}\n    """\n', m.group(1)


def clean_story_prompt(p: str):
    m = TAG_RE.match(p)
    if not m or m.group(1).upper() != "WP":
        return None
    p = p[m.end():]
    for a, b in DETOK:
        p = p.replace(a, b)
    p = re.sub(r"\s+", " ", p).strip()
    return p if 15 <= len(p) <= 400 else None


def build(domain: str, n: int, seed: int, log) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    if domain == "math":
        for r in take("openai/gsm8k", "main", "train", 8000, seed, log):
            out.append(MATH_T.format(q=r["question"].strip()))
        for r in take("meta-math/MetaMathQA", None, "train", 3 * n, seed, log):
            if str(r.get("type", "")).startswith("GSM"):
                out.append(MATH_T.format(q=r["query"].strip()))
    elif domain == "code":
        pairs = []
        for r in take("iamtarun/python_code_instructions_18k_alpaca", None, "train", 18000, seed, log):
            pairs.append((r["instruction"], r["output"]))
        for r in take("sahil2801/CodeAlpaca-20k", None, "train", 20000, seed, log):
            pairs.append((r["instruction"], r["output"]))
        for r in take("google-research-datasets/mbpp", "full", "train", 1000, seed, log):
            pairs.append((r["text"], r["code"]))
        for instr, code in pairs:
            stub, _name = make_stub(instr, code)
            if stub:
                out.append(CODE_T.format(stub=stub))
        free = [r["instruction"].strip() for r in
                take("nickrosh/Evol-Instruct-Code-80k-v1", None, "train", n, seed, log)]
        out += [f for f in free if 20 <= len(f) <= 1200][: max(1, len(out) // 3)]
    elif domain == "longform":
        for r in take("euclaise/writingprompts", None, "train", 6 * n, seed, log):
            p = clean_story_prompt(r["prompt"])
            if p:
                words = 600 if rng.random() < 0.8 else rng.choice([300, 400, 500, 800])
                out.append(STORY_T.format(n=words, p=p))
    elif domain == "chat":
        for r in take("HuggingFaceH4/ultrachat_200k", None, "train_sft", 2 * n, seed, log):
            out.append(r["prompt"].strip())
        for r in take("tatsu-lab/alpaca", None, "train", n, seed, log):
            out.append((r["instruction"] + ("\n\n" + r["input"] if r.get("input") else "")).strip())
        for r in take("databricks/databricks-dolly-15k", None, "train", n // 2, seed, log):
            out.append((r["instruction"] + ("\n\n" + r["context"] if r.get("context") else "")).strip())
        out = [p for p in out if 20 <= len(p) <= 1500]
    else:
        raise ValueError(domain)
    out = list(dict.fromkeys(out))
    rng.shuffle(out)
    return out


def drop_dev_overlap(prompts: list[str], dev_texts: list[str], log) -> list[str]:
    """Remove any training prompt whose task text also appears in the public dev set."""
    blob = " || ".join(norm(t) for t in dev_texts)
    dev_defs = {m.group(1) for t in dev_texts for m in DEF_RE.finditer(t)}
    kept = []
    for p in prompts:
        core = p
        for t in (CODE_T, MATH_T, STORY_T):
            head = t.split("{")[0]
            if head and p.startswith(head):
                core = p[len(head):]
        core = core.split("\n\nSolve this step by step")[0]
        key = norm(core)[:160]
        names = {m.group(1) for m in DEF_RE.finditer(p)}
        if (len(key) >= 20 and key in blob) or (names & dev_defs):
            continue
        kept.append(p)
    log(f"  dev-overlap filter: kept {len(kept)} of {len(prompts)}")
    return kept
