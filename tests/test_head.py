"""CPU self-test with tiny dimensions: no GPU, no model download.

    python tests/test_head.py

Checks the pieces whose correctness the whole pipeline rests on:
  1. FP8 (E4M3) rounding / encoding round-trips and matches torch's own float8 cast;
  2. the vectorised K-step `unroll` equals a literal step-by-step emulation of the engine's
     drafting loop with a key/value cache;
  3. `equalize_` leaves the head's function unchanged;
  4. `walk` reproduces a hand-checked acceptance count;
  5. a few training steps run, reduce the loss, and the exported folders pass the kit's
     static validation (when a kit checkout is given via EMC2_HOME);
  6. prompt helpers, batching and the supervisor's branch lookup (needs network) behave.
"""
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emcdraft.head import (FP8_TENSORS, HeadConfig, MTPHead, e4m3_decode, e4m3_encode, e4m3_round,  # noqa: E402
                           equalize_, export_submission, load_mtp_state, unroll, walk)
from emcdraft.sim import batches, collate, simulate  # noqa: E402
from emcdraft.train import train_step  # noqa: E402

torch.manual_seed(0)
C = HeadConfig(hidden=48, inter=96, heads=6, kv_heads=2, head_dim=16, rot_dim=4, theta=1e4, eps=1e-6)
V = 211


def test_fp8():
    x = torch.cat([torch.randn(20000) * 0.02, torch.randn(20000) * 0.3, torch.randn(2000) * 50,
                   torch.tensor([0.0, 448.0, -448.0, 1e4, 2.0 ** -9, 2.0 ** -10, 3e-4, 2.0 ** -6, 0.99, 1.0])])
    q = e4m3_round(x)
    assert torch.equal(e4m3_round(q), q), "rounding must be idempotent"
    assert torch.equal(e4m3_decode(e4m3_encode(x)), q), "encode/decode must round-trip"
    rel = ((q - x).abs() / x.abs().clamp(min=2.0 ** -6))[x.abs() <= 448]
    assert rel.max() <= 0.0626, rel.max()
    try:
        ref = x.clamp(-448, 448).to(torch.float8_e4m3fn)
        assert torch.equal(ref.float(), q.clamp(-448, 448)), "values differ from torch's float8 cast"
        bits = torch.arange(256, dtype=torch.uint8)
        dec = bits.view(torch.float8_e4m3fn).float()
        ok = dec == dec                                      # skip the two NaN patterns
        assert torch.equal(dec[ok], e4m3_decode(bits)[ok]), "decoder differs from torch's float8 decode"
        print("  fp8: rounding and all 254 finite bit patterns agree with torch.float8_e4m3fn")
    except (RuntimeError, TypeError, AttributeError) as e:
        print("  fp8: torch float8 cast unavailable here, skipped cross-check:", type(e).__name__)


def sequential(head, E, tokens, hidden, i, K=3):
    """Literal emulation of the engine for one round starting at base index i."""
    T = tokens.numel()
    pos = torch.arange(i + 1)[None]
    tri = torch.ones(i + 1, i + 1, dtype=torch.bool).tril()[None, None]
    out, kv = head.step(E[tokens[1:i + 2]][None], hidden[None, :i + 1], pos, (), tri)   # verified prefix
    cache_k, cache_v = kv
    h, outs = out[:, -1:], [out[0, -1]]
    for k in range(2, K + 1):
        if i + k > T - 1:
            break
        tok = tokens[i + k]
        n_keys = cache_k.shape[2] + 1
        o, kv = head.step(E[tok][None, None], h, torch.tensor([[i + k - 1]]), [(cache_k, cache_v)],
                          torch.ones(1, 1, 1, n_keys, dtype=torch.bool))
        cache_k, cache_v = torch.cat([cache_k, kv[0]], 2), torch.cat([cache_v, kv[1]], 2)
        h = o
        outs.append(o[0, 0])
    return outs


def test_unroll(head, E):
    T = 14
    tokens = torch.randint(0, V, (T,))
    hidden = torch.randn(T - 1, C.hidden)
    par = [o for _, o in unroll(head, lambda ids: E[ids], tokens[None], hidden[None], 3)]
    worst = 0.0
    for i in range(T - 1):
        for k, o in enumerate(sequential(head, E, tokens, hidden, i)):
            worst = max(worst, float((par[k][0, i] - o).abs().max()))
    assert worst < 1e-4, worst
    print(f"  unroll == step-by-step cache emulation (max abs diff {worst:.2e})")


def test_equalize(head, E):
    tokens = torch.randint(0, V, (3, 12))
    hidden = torch.randn(3, 11, C.hidden)
    before = [o for _, o in unroll(head, lambda ids: E[ids], tokens, hidden, 3)]
    factors = equalize_(head)
    after = [o for _, o in unroll(head, lambda ids: E[ids], tokens, hidden, 3)]
    drift = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(after, before))
    assert drift < 1e-3, drift
    rms = {n: round(float(dict(head.named_parameters())[n].pow(2).mean().sqrt()), 3) for n in FP8_TENSORS}
    assert all(abs(v - 0.25) < 1e-3 for v in rms.values()), rms
    print(f"  equalize: function preserved (rel drift {drift:.1e}), FP8 tensors at RMS 0.25, s={factors['s']:.2f}")


def test_walk():
    T, P = 12, 4                           # tokens 4..11 generated
    c = [[False] * T for _ in range(3)]
    c[0][3] = c[1][3] = True               # round at base 3: tokens 5, 6 accepted, 7 rejected
    c[0][6] = c[1][6] = c[2][6] = True     # next round at base 6 (cur=7): 8, 9, 10 accepted
    acc, drafts = walk(c, P, T)
    assert (acc, drafts) == (5, 2), (acc, drafts)
    print("  walk: hand-checked example ok")


def test_train_and_export(head, E, W):
    rng = random.Random(0)
    seqs = []
    for d in ("chat", "code", "math", "longform"):
        for _ in range(6):
            T, P = rng.randint(14, 22), rng.randint(3, 6)
            tok = torch.randint(0, V, (T,))
            seqs.append({"tokens": tok, "hidden": torch.randn(T - 1, C.hidden), "P": P, "T": T, "domain": d})
    a = SimpleNamespace(beta=0.5, floor=0.3, step_w=[1.0, 0.7, 0.5])
    head.set_fake_quant(True)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3)
    embed = lambda ids: F.embedding(ids, E)
    first = last = None
    for it in range(30):
        loss, info = train_step(head, embed, W, collate(seqs), a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = info["ce1"] if first is None else first
        last = info["ce1"]
    assert last < first, (first, last)
    sim = simulate(head, embed, W, seqs)
    assert 1.0 <= sim["acceptance_length"] <= 4.0 and len(batches(seqs, 60)) > 1
    print(f"  train: ce1 {first:.3f} -> {last:.3f} with FP8 fake-quant; memorised tau {sim['acceptance_length']:.2f}")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "config.json"
        cfg.write_text(json.dumps({"text_config": {"max_position_embeddings": 262144, "mtp_num_hidden_layers": 1,
                                                   "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 16}}))
        meta = {"team_id": "t", "method_description": "m", "training_data": "d"}
        for name, fp8 in (("sub_fp8", True), ("sub_bf16", False)):
            info = export_submission(head, Path(tmp) / name, cfg, fp8, meta)
            back = load_mtp_state(Path(tmp) / name)
            for n, p in head.state_dict().items():
                want = e4m3_round(p.float()) if (fp8 and n in FP8_TENSORS) else p.float().to(torch.bfloat16).float()
                assert torch.equal(back[n], want), n
            try:
                from safetensors import safe_open
                with safe_open(str(Path(tmp) / name / "model.safetensors"), framework="pt") as f:
                    t = f.get_tensor("mtp." + FP8_TENSORS[0])
                assert ("float8" in str(t.dtype)) == fp8
            except ImportError:
                pass
            kit = os.environ.get("EMC2_HOME")
            if kit:
                sys.path.insert(0, kit)
                from emc2 import submission
                errors, kinfo = submission.check(Path(tmp) / name, json.loads((Path(kit) / "contract.json").read_text()),
                                                 json.loads((Path(kit) / "target.json").read_text()))
                assert not errors, errors
                assert kinfo["weight_bytes"] == info["weight_bytes"]
        print("  export: FP8 and bf16 folders round-trip" + (" and pass the kit's validate" if os.environ.get("EMC2_HOME") else ""))


def test_helpers():
    from emcdraft.extract import make_batches
    from emcdraft.prompts import CODE_T, clean_story_prompt, drop_dev_overlap, make_stub
    x = torch.randn(3, 1000) * 0.3
    assert torch.equal(e4m3_encode(x, chunk=777), e4m3_encode(x)), "chunked FP8 encoding differs"
    stub = make_stub("Write a function to add two numbers.", "def add(a, b):\n    return a + b\n")
    assert stub.startswith("def add(a, b):") and '"""Write a function' in stub, stub
    assert make_stub("too short", "def f(): pass") is None
    assert clean_story_prompt("[ WP ] You 've won the lottery , but there 's a catch .") == \
        "You've won the lottery, but there's a catch."
    assert clean_story_prompt("[ EU ] fan fiction") is None
    dev = ["Complete the following Python function.\n\n```python\ndef strlen(string: str) -> int:\n```",
           "Ruby is 6 times older than Sam. In 9 years, Ruby will be 3 times as old as Sam. How old is Sam now?"]
    train = [CODE_T.format(stub="def strlen(s):\n    pass\n"), CODE_T.format(stub="def other(s):\n    pass\n"),
             "Ruby is 6 times older than Sam. In 9 years, Ruby will be 3 times as old as Sam. How old is Sam now?"
             "\n\nSolve this step by step, then give the final answer on its own line as 'Answer: <number>'."]
    kept = drop_dev_overlap(train, dev)
    assert kept == [train[1]], kept
    seqs = [{"ids": [0] * n} for n in (5, 50, 7, 49, 20, 3)]
    b = make_batches(seqs, 100)
    assert sorted(i for x in b for i in x) == list(range(6))
    assert all(len(x) * max(len(seqs[i]["ids"]) for i in x) <= 100 for x in b), b
    print("  prompts / batching helpers ok")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import supervise
        sha = supervise.remote_sha()
        assert sha and len(sha) == 40
        print(f"  supervisor sees branch head {sha[:12]}")
    except OSError as e:
        print("  supervisor branch lookup skipped (no network):", e)


if __name__ == "__main__":
    head = MTPHead(C)
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.08 if p.ndim == 2 else 0.05)
    E, W = torch.randn(V, C.hidden) * 0.5, torch.randn(V, C.hidden) * 0.3
    print("fp8"); test_fp8()
    with torch.no_grad():
        print("unroll"); test_unroll(head, E)
        print("equalize"); test_equalize(head, E)
    print("walk"); test_walk()
    print("train/export"); test_train_and_export(head, E, W)
    print("helpers"); test_helpers()
    print("ALL OK")
