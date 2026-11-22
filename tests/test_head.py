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


def test_append():
    """Adding data to an earlier run: earlier answers and feature shards are kept, only new files
    are processed, and validation still comes from the first extraction."""
    from emcdraft.extract import plan_train
    from emcdraft.gen_vllm import previous_outputs
    from emcdraft.sim import load_shards, save_shard
    from emcdraft.train import split_val

    def shard(stem, n, d, seed):
        g = torch.Generator().manual_seed(seed)
        seqs = [{"tokens": torch.randint(0, V, (12,), generator=g), "hidden": torch.randn(11, C.hidden, generator=g),
                 "P": 3, "domain": d, "id": f"{seed}-{i}"} for i in range(n)]
        save_shard(str(stem), seqs)

    with tempfile.TemporaryDirectory() as tmp:
        gen, data = Path(tmp) / "gen", Path(tmp) / "data"
        gen.mkdir()
        (gen / "chat.jsonl").write_text(json.dumps({"p": [1, 2], "o": [5, 6, 248046]}) + "\n")
        (gen / "code.jsonl").write_text(json.dumps({"p": [3], "o": [7] * 9}) + "\n")
        seen, stats, total, suffix = previous_outputs(gen)
        assert seen["chat"] == {(1, 2)} and total == 12 and suffix == "-2" and stats["chat"]["stopped"] == 1

        files, prefix, consumed, kept = plan_train(data, gen, append=False)        # first extraction
        assert [p.name for p in files] == ["chat.jsonl", "code.jsonl"] and prefix == "g" and kept == 0
        shard(data / "train" / "g0000", 20, "chat", 1)                             # (no consumed list, as in run 1)

        (gen / "chat-2.jsonl").write_text(json.dumps({"p": [4], "o": [8] * 9}) + "\n")
        assert previous_outputs(gen)[2] == 21 and previous_outputs(gen)[3] == "-3"
        files, prefix, consumed, kept = plan_train(data, gen, append=True)
        assert [p.name for p in files] == ["chat-2.jsonl"] and prefix == "p2g" and kept == 1, (files, prefix, kept)
        shard(data / "train" / "p2g0000", 5, "chat", 2)
        (data / "consumed_gen.json").write_text(json.dumps(sorted(consumed | {"chat-2.jsonl"})))
        (gen / "code-3.jsonl").write_text(json.dumps({"p": [9], "o": [8] * 9}) + "\n")
        files, prefix, _, kept = plan_train(data, gen, append=True)
        assert [p.name for p in files] == ["code-3.jsonl"] and prefix == "p3g" and kept == 2

        seqs = load_shards(data / "train", "cpu")
        assert [s["part"] for s in seqs] == [1] * 20 + [2] * 5
        first = [s for s in seqs if s["part"] == 1]
        _, val_a = split_val(first, 0.05, random.Random(0))
        _, val_b = split_val(load_shards(data / "train" , "cpu")[:20], 0.05, random.Random(0))
        assert [s["id"] for s in val_a] == [s["id"] for s in val_b] and all(s["part"] == 1 for s in val_a)
        t, h, P, T = collate(seqs[:3], device="cpu")
        assert h.shape == (3, 11, C.hidden) and t.device.type == "cpu"
        plan_train(data, gen, append=False)                                         # a fresh run clears the shards
        assert not list((data / "train").glob("*"))
    print("  append: earlier answers and shards kept, new files only, same validation split")


def test_recipes(head, E, W):
    """The full training entry point with the new recipe options, the recipe comparison and
    the keep-the-better-draft rule, end to end on a tiny fake target (CPU)."""
    import subprocess
    from emcdraft.head import save_safetensors_raw
    from emcdraft.sim import save_shard
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), TRAIN_DATA_DEVICE="cpu", TRAIN_LOWRANK_RANKS="4,8")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tgt, init, data = tmp / "target", tmp / "init", tmp / "data"
        tgt.mkdir()
        (tgt / "config.json").write_text(json.dumps({"text_config": {
            "hidden_size": C.hidden, "intermediate_size": C.inter, "num_attention_heads": C.heads,
            "num_key_value_heads": C.kv_heads, "head_dim": C.head_dim, "partial_rotary_factor": C.rot_dim / C.head_dim,
            "rope_theta": C.theta, "rms_norm_eps": C.eps, "max_position_embeddings": 262144,
            "mtp_num_hidden_layers": 1}}))
        names = {"lm_head.weight": W, "model.language_model.embed_tokens.weight": E}
        save_safetensors_raw(tgt / "model.safetensors", {
            n: ("BF16", t.shape, t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes())
            for n, t in names.items()})
        (tgt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "model.safetensors" for n in names}}))
        head.set_fake_quant(False)
        export_submission(head, init, tgt / "config.json", False, {"team_id": "t", "method_description": "m", "training_data": "d"})
        g = torch.Generator().manual_seed(3)
        for split, n in (("train", 20), ("dev", 4)):
            seqs = [{"tokens": torch.randint(0, V, (16,), generator=g), "hidden": torch.randn(15, C.hidden, generator=g),
                     "P": 4, "domain": d, "id": f"{d}{i}"} for d in ("chat", "code", "math", "longform") for i in range(n)]
            (data / split).mkdir(parents=True)
            save_shard(str(data / split / ("g0000" if split == "train" else "all")), seqs)
        common = ["--target", str(tgt), "--init", str(init), "--data", str(data / "train"), "--dev", str(data / "dev")]
        run = lambda args: subprocess.run([sys.executable, "-m", *args], env=env, capture_output=True, text=True)
        r = run(["emcdraft.train", *common, "--out", str(tmp / "full"), "--qat", "--step-w", "auto", "--topk", "5",
                 "--floor", "0.1", "--max-steps", "4", "--eval-every", "2", "--batch-tokens", "200"])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        s = json.loads((tmp / "full" / "train_summary.json").read_text())
        meta = json.loads((tmp / "full" / "sub_fp8" / "metadata.json").read_text())
        assert "acceptance-aware" in meta["method_description"] and "top-5" in meta["method_description"]
        assert len(s["history"]) == 2 and (tmp / "full" / "sub_bf16" / "model.safetensors").exists()
        assert "w [" in r.stdout, "auto step weights not reported"
        assert [x["rank"] for x in s["lowrank"]] == [4, 8] and all(x["val_tau"] >= 1 for x in s["lowrank"])
        r = run(["emcdraft.train", *common, "--out", str(tmp / "lora"), "--qat", "--lora", "4", "--max-steps", "3",
                 "--eval-every", "3", "--batch-tokens", "200", "--lowrank-ranks", ""])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        merged = load_mtp_state(tmp / "lora" / "sub_bf16")
        assert set(merged) == set(head.state_dict()), "LoRA must merge into the dense layout"
        changed = [n for n in FP8_TENSORS if not torch.equal(merged[n], load_mtp_state(init)[n])]
        assert "LoRA rank 4" in r.stdout and changed, "LoRA training did not change the matrices"
        r = run(["emcdraft.ablate", *common, "--out", str(tmp / "abl"), "--results", str(tmp / "res"),
                 "--state", str(tmp), "--qat", "--steps", "2"])
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        ab = json.loads((tmp / "res" / "ablation.json").read_text())
        assert set(ab["recipes"]) == {"control", "accw_topk", "accw_topk_mask", "accw_topk_lr2", "accw_topk_lora"}
        assert (tmp / "train_args").read_text().split() == ab["recipes"][ab["chosen"]]["flags"]
        assert not any((tmp / "abl" / n / "sub_fp8").exists() for n in ab["recipes"]), "ablation runs must not export"
        # keep-the-better-draft rule
        kit = os.environ.get("EMC2_HOME")
        if kit:
            res = tmp / "res"
            m = lambda tau, tpot: {"acceptance_length": tau, "tpot_median_s": tpot, "memory_bytes": 607177728}
            (res / "eval_baseline.json").write_text(json.dumps(dict(m(2.805, 0.01539), memory_bytes=916507648)))
            (res / "eval_new.json").write_text(json.dumps(m(2.95, 0.0150)))
            (res / "eval_old.json").write_text(json.dumps(m(2.90, 0.01524)))
            ch = lambda new, old: subprocess.run([sys.executable, str(root / "scripts" / "choose.py"), "--new", new,
                                                  "--new-sub", str(tmp / "full" / "sub_fp8"), "--prev", old,
                                                  "--prev-sub", str(init), "--results", str(res), "--state", str(tmp)],
                                                 env=env, capture_output=True, text=True)
            r = ch("eval_new", "eval_old")
            assert r.returncode == 0 and json.loads((res / "choice.json").read_text())["kept"] == "new", r.stderr
            assert (tmp / "final_sub").read_text().strip() == str(tmp / "full" / "sub_fp8")
            r = ch("eval_old", "eval_new")      # a worse new draft keeps the previous one
            assert json.loads((res / "choice.json").read_text())["kept"] == "previous", r.stderr
            assert (tmp / "final_sub").read_text().strip() == str(init)
    print("  recipes: training options, recipe comparison and keep-the-better rule work end to end")


def test_eagle(E, W):
    """EAGLE-3 head: the batched unroll equals a literal emulation of the engine's drafting loop
    (fused auxiliary states for step 1, pre-norm recursion afterwards); online training learns;
    the export has exactly the names the engine's loader maps and passes the kit's validate;
    the full-size draft stays inside the memory cap."""
    from emcdraft.eagle import EagleConfig, Eagle3Head, OnlineTrainer, engine_config, export_eagle
    from emcdraft.sim import collate, fp8_dequantize, fp8_quantize, load_shards, save_shard
    torch.manual_seed(1)
    cfg = EagleConfig(hidden=C.hidden, heads=4, kv_heads=2, head_dim=12, inter=64, aux_layers=(1, 2, 3), vocab=V)
    head = Eagle3Head(cfg)
    T = 12
    tokens = torch.randint(0, V, (T,))
    aux = torch.randn(T - 1, 3 * C.hidden)
    with torch.no_grad():
        par = [o for _, o in unroll(head, lambda ids: E[ids], tokens[None], aux[None], 3)]
        worst = 0.0
        for i in range(T - 1):
            pos = torch.arange(i + 1)[None]
            tri = torch.ones(i + 1, i + 1, dtype=torch.bool).tril()[None, None]
            out, (ck, cv), pre = head.step(E[tokens[1:i + 2]][None], head.combine(aux[None, :i + 1]), pos, (), tri)
            outs, h = [out[0, -1]], pre[:, -1:]
            for k in range(2, 4):
                if i + k > T - 1:
                    break
                o, kv, pre = head.step(E[tokens[i + k]][None, None], h, torch.tensor([[i + k - 1]]), [(ck, cv)],
                                       torch.ones(1, 1, 1, ck.shape[2] + 1, dtype=torch.bool))
                ck, cv = torch.cat([ck, kv[0]], 2), torch.cat([cv, kv[1]], 2)
                outs.append(o[0, 0])
                h = pre
            for k, o in enumerate(outs):
                worst = max(worst, float((par[k][0, i] - o).abs().max()))
    assert worst < 1e-4, worst
    print(f"  eagle unroll == step-by-step emulation with pre-norm recursion (max abs diff {worst:.2e})")

    tr = OnlineTrainer(cfg, "cpu", total_steps=40, lr=3e-3, warmup=5)
    toks = torch.randint(0, V, (4, 16))
    ax, teach = torch.randn(4, 15, 3 * C.hidden), torch.randn(4, 15, C.hidden)
    P, L = torch.tensor([3, 3, 4, 5]), torch.tensor([16, 14, 16, 12])
    first = tr.step(lambda ids: F.embedding(ids, E), W, toks, ax, teach, P, L)["ce1"]
    for _ in range(25):
        last = tr.step(lambda ids: F.embedding(ids, E), W, toks, ax, teach, P, L)["ce1"]
    assert last < first, (first, last)
    print(f"  eagle online training: ce1 {first:.3f} -> {last:.3f}")

    with tempfile.TemporaryDirectory() as tmp:
        info = export_eagle(tr.head, Path(tmp) / "eagle", {"team_id": "t", "method_description": "m", "training_data": "d"})
        from emcdraft.head import read_safetensors
        names = set(read_safetensors(Path(tmp) / "eagle" / "model.safetensors"))
        want = {"fc.weight", "fc_norm.0.weight", "fc_norm.1.weight", "fc_norm.2.weight", "norm.weight",
                "midlayer.input_layernorm.weight", "midlayer.hidden_norm.weight", "midlayer.post_attention_layernorm.weight",
                *(f"midlayer.self_attn.{x}_proj.weight" for x in "qkvo"),
                *(f"midlayer.mlp.{x}_proj.weight" for x in ("gate", "up", "down"))}
        assert names == want, names ^ want
        cfgj = json.loads((Path(tmp) / "eagle" / "config.json").read_text())
        assert cfgj["architectures"] == ["LlamaForCausalLMEagle3"] and cfgj["fc_norm"] is True
        assert json.loads((Path(tmp) / "eagle" / "spec.json").read_text()) == {"method": "eagle3"}
        kit = os.environ.get("EMC2_HOME")
        if kit:
            sys.path.insert(0, kit)
            from emc2 import submission
            errors, kinfo = submission.check(Path(tmp) / "eagle", json.loads((Path(kit) / "contract.json").read_text()),
                                             json.loads((Path(kit) / "target.json").read_text()))
            assert not errors, errors
            assert kinfo["weight_bytes"] == info["weight_bytes"]
        # FP8 feature shards: compact on disk, close to the original after batching
        seqs = [{"tokens": torch.randint(0, V, (9,)), "hidden": (torch.randn(8, C.hidden) * 3).to(torch.bfloat16),
                 "P": 2, "domain": "chat", "id": str(i)} for i in range(3)]
        save_shard(str(Path(tmp) / "f8"), seqs, fp8=True)
        assert (Path(tmp) / "f8.hid8.npy").exists() and not (Path(tmp) / "f8.hid.npy").exists()
        back = load_shards(tmp, "cpu")
        _, hb, _, _ = collate(back)
        rel = float((hb.float() - torch.stack([s["hidden"].float() for s in seqs])).norm()
                    / torch.stack([s["hidden"].float() for s in seqs]).norm())
        assert rel < 0.05, rel
        x = torch.randn(5, C.hidden)
        assert float((fp8_dequantize(*fp8_quantize(x), dtype=torch.float32) - x).norm() / x.norm()) < 0.05
    # the full-size draft against the memory cap (bf16 weights + KV cache at 16K tokens)
    big = EagleConfig()
    H, n = big.hidden, len(big.aux_layers)
    params = (n * H * H + 2 * H * big.heads * big.head_dim + 2 * 2 * H * big.kv_heads * big.head_dim
              + big.heads * big.head_dim * H + 3 * H * big.inter + (n + 4) * H)
    memory = 2 * params + 1 * big.kv_heads * big.head_dim * 2 * 2 * 16384
    assert 916507648 / memory >= 1.5, (params, memory)
    from emcdraft.gen_vllm import share_from_env
    sh = share_from_env("longform=0.6,chat=0.2,code=0.1,math=0.1")
    assert abs(sh["longform"] - 0.6) < 1e-9 and abs(sum(sh.values()) - 1) < 1e-9
    print(f"  eagle export: engine names, validate, FP8 shards (rel err {rel:.3f}); full size {params / 1e6:.0f} M params, "
          f"memory {memory / 1e9:.3f} GB (ratio {916507648 / memory:.2f} >= 1.5)")


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
    print("append"); test_append()
    print("recipes"); test_recipes(head, E, W)
    print("eagle"); test_eagle(E, W)
    print("ALL OK")
