# EMC — a draft model for the Speculative Decoding Championship

Work for the **Speculative Decoding Championship** of the emc2 workshop at AAAI 2027:
build the best draft model for a fixed target (**Qwen3.8-27B**, bf16, 16K context) served by an
unmodified **vLLM v0.31.0**, with `K = 3` draft tokens per round.

```
score = 100 × ( 0.50·(τ−1)/(τ_B−1) + 0.25·min(1.5, M_B/M) + 0.25·min(1.5, TPOT_B/TPOT) )
```

τ is the acceptance length, M the draft memory (stored weight bytes + draft KV cache at 16K
tokens), TPOT the median time per output token; `B` is the baseline, the target's built-in
MTP head (τ_B = 2.802, M_B = 0.917 GB, score 100). A draft is ranked only if τ ≥ 0.9·τ_B.

**Hardware.** Everything here — data generation, training and every evaluation — runs on a
single **NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)**.

## The draft

A **replacement MTP head in the baseline layout** (`method: mtp`): the same 15 `mtp.*`
tensors as the target's built-in head, the target's `config.json` unchanged, no code.

| | baseline head | this draft |
|---|---|---|
| layout | `mtp.*`, 424.7 M params | identical |
| weights | as shipped with the target | fine-tuned for 3-step recursive drafting on the target's own outputs |
| storage | bf16, 0.849 GB | large matrices in FP8 (E4M3), rest bf16: **0.540 GB** (a plain bf16 variant is exported too) |
| memory metric (weights + KV) | 0.917 GB | **0.607 GB** → M_B/M = 1.51, i.e. the 1.5 cap |

### Why this design

Three things were read out of the engine source (`qwen3_5_mtp.py`, `llm_base_proposer.py`)
before choosing it:

1. **The head's shape is fixed by the target.** vLLM builds the MTP module from the *target's*
   config, not the submission's, so an `mtp` submission cannot be narrower or shallower. A
   replacement head is therefore a question of *better weights*, and of how they are stored.
2. **Drafting is recursive.** Step 1 sees the target's final hidden state; steps 2 and 3 feed the
   head its *own* previous output (and write their own keys/values into the draft KV cache).
   A head trained only for one-step prediction is off-distribution for two thirds of what it
   is asked to do.
3. **The LM head and embeddings are shared with the target**, so they cost nothing in the memory
   metric, and the per-round draft cost is dominated by the three LM-head evaluations — the
   head's own size barely moves TPOT. Higher acceptance is what moves it.

### Training

* **Data — the target's own greedy answers.** Fresh prompts in the four evaluated domains
  (chat, code, math, long-form writing), phrased like the public dev tasks, are answered
  greedily by the target running in the serving engine itself (vLLM 0.31.0), so the training
  text is exactly the kind of text the draft has to predict. The target's final hidden state
  at every position is then recomputed with the reference implementation in one
  teacher-forced pass. Two checks guard this step: the reference implementation must agree
  with the engine's greedy choices along the dev references, and the offline simulator
  (below) must reproduce the kit-measured acceptance of the baseline head. Prompts come from
  public instruction sets (GSM8K / Orca-Math, WritingPrompts, UltraChat / Alpaca / Dolly,
  Python instruction sets, MBPP); anything overlapping the dev prompts is removed and the
  dev set is never trained on.
* **Objective — the engine's drafting loop.** `emcdraft.head.unroll` runs all three draft steps
  for every position of a sequence in parallel, with exactly the key/value visibility the
  engine's draft cache has (verified prefix, then the round's own entries). Each step is
  trained on the token the target really emitted and also distils the target's full
  distribution for that token; later steps are weighted by whether the earlier ones were
  right, because the engine only reaches them in that case.
* **FP8 with QAT.** An exact, function-preserving re-scaling (`equalize_`) first moves the big
  matrices into the range where E4M3 has relative precision; they are then trained through
  E4M3 rounding with a straight-through estimator, so the compact file is the model that was
  optimised. The stock engine up-casts FP8 tensors to bf16 at load time — the saving is in
  stored bytes, which is what the competition's memory metric counts (`emc2/submission.py`);
  runtime GPU memory of the head is unchanged. Both variants are exported and reported.
* **Offline simulator.** `emcdraft.sim.simulate` replays greedy speculative decoding from stored
  hidden states and returns τ without starting a server. It is checked against the kit's
  measured baseline acceptance before any training, and used for checkpoint selection on a
  held-out slice of the *training* data (the dev set is only reported, never selected on).

## Results

Official kit `eval` (vLLM 0.31.0, all 160 dev prompts, K = 3), run **r2**, every row measured
in the same run on the same **NVIDIA RTX PRO 6000 Blackwell**:

| draft | τ | chat / code / math / long-form | TPOT | memory | score, same GPU | score, published baseline |
|---|---|---|---|---|---|---|
| baseline (the target's built-in MTP head) | 2.805 | 3.07 / 3.77 / 3.63 / 2.36 | 15.39 ms | 0.917 GB | 100.0 | 96.2 |
| **this draft, FP8 (submitted variant)** | **2.900** | 3.08 / 3.77 / 3.70 / 2.50 | 15.24 ms | **0.607 GB** | **115.4** | **111.6** |

* *Score, same GPU* uses the baseline as measured above; *score, published baseline* is the
  kit's own number, whose speed term compares against the organisers' TPOT from different
  hardware (that is why the baseline itself gets 96.2 there).
* The same-GPU score splits into acceptance 1.053 (×0.5), memory 1.5 (×0.25, the cap) and
  speed 1.010 (×0.25). Fine-tuning added +0.095 to τ, mostly in long-form writing (+0.14) and
  math (+0.07); chat and code were already close to what one head layer can reach.
* The offline simulator predicted the dev acceptance within 0.02 of the engine (2.884 vs
  2.900 for the draft, 2.793 vs 2.805 for the baseline).
* r2 trained on 4.06 M generated tokens (9,634 answers) for 1,190 steps (2 epochs, 27 min);
  validation τ was still rising at the last step. The whole pipeline, including two full
  evaluations, took about 3 hours of GPU time.

Earlier run **r1** (2.02 M tokens, 584 steps) reached τ 2.868 and a same-GPU score of 114.4
(bf16 variant: τ 2.885, score 102.3). Twice the data gave +0.03 τ.

Raw kit outputs, per-stage logs, machine statistics and training summaries are in the
dataset repo under `runs/r1/` and `runs/r2/`; the r2 draft folders (FP8 and bf16, each with
`model.safetensors`) under `submissions/r2/`.

## Layout

```
emcdraft/head.py      MTP head re-implementation, K-step unroll, FP8 rounding/encoding, export
emcdraft/prompts.py   training prompts per domain, dev-overlap filter, template check
emcdraft/gen_vllm.py  greedy answers of the target with the serving engine
emcdraft/target.py    reference implementation wrapper: final hidden states (teacher forcing)
emcdraft/extract.py   hidden-state shards, agreement check, simulator check on the baseline
emcdraft/sim.py       offline acceptance simulator, feature shards
emcdraft/train.py     fine-tuning (recursive objective, optional FP8 QAT), export
emcdraft/probe.py     untrained FP8 copy of the baseline (engine loading probe)
scripts/kit.py        runs the official kit commands without a container runtime
scripts/pipeline.sh   end-to-end, resumable pipeline (one stage per step, logs per stage)
scripts/supervise.py  unattended driver: code updates, control requests, log uploads
scripts/launch.py     starts the driver on a GPU machine (or prints its status)
scripts/sync.py       pushes logs / results / draft folders to the Hugging Face dataset repo
scripts/collect.py    results/summary.{json,md}
control/              stage re-run requests and budget settings read by the driver
tests/test_head.py    CPU self-test (FP8 codec, unroll vs. step-by-step emulation, export)
```

## Reproduce

Linux, one GPU with at least 80 GB, about 120 GB of disk. In a Python shell or notebook on
that machine:

```python
import os, urllib.request
os.environ["HF_TOKEN"] = "<token with write access to the results repo>"
exec(urllib.request.urlopen(
    "https://raw.githubusercontent.com/Mudassiruddin7/EMC/main/scripts/launch.py").read(),
    {"__name__": "__main__"})
```

This starts `scripts/supervise.py`, which runs `scripts/pipeline.sh` stage by stage:

| stage | what it does |
|---|---|
| env | private Python 3.12 environment with `vllm==0.31.0` |
| download | the kit (pinned commit) and the target at the kit's pinned revision |
| baseline | `emc2 prepare-baseline` + `validate` |
| probe, smoke_fp8 | FP8 copy of the untrained baseline head through the official `eval` (2 prompts per domain) |
| prompts | training prompts (runs in parallel with the GPU checks) |
| generate | greedy answers with the engine (token budget `GEN_TOKENS`) |
| extract | hidden states + agreement and simulator checks |
| train | fine-tuning, export of `sub_fp8` / `sub_bf16`, `validate` |
| eval_main, eval_baseline, eval_alt | full official `eval` (160 dev prompts) of the draft, the baseline and the second variant |

Every stage resumes after an interruption, and logs, kit results, generated data and the
draft folders are pushed to the dataset repo as they are produced. Settings in
`control/run.env` extend a finished run instead of repeating it: `GEN_APPEND=1` and
`EXTRACT_APPEND=1` keep earlier answers and features, `TRAIN_INIT` starts from an earlier
head, `RUN_NAME` names the new training run.

The engine is the pinned version installed from PyPI rather than the container image; the
results JSON records this. The pipeline sets `VLLM_USE_FLASHINFER_SAMPLER=0`, so the engine
uses its built-in sampler instead of compiling FlashInfer's at start-up; decoding is greedy,
so outputs and draft behaviour are unaffected. `python tests/test_head.py` runs the CPU
self-test (no GPU, no downloads).

## Submission format and rules

Each exported folder (`runs/main/sub_fp8`, `runs/main/sub_bf16`) contains `config.json` (the
target's, byte-identical), `model.safetensors` (only `mtp.*` tensors), `spec.json`
(`{"method": "mtp"}`) and `metadata.json`, and passes `emc2 validate`. No code, no pickle,
no state across requests; verification is untouched. The head is initialised from the
target's built-in MTP head (the baseline draft itself) and shares the target's embedding and
output layer, as the baseline does; no target decoder layers are copied.

## License

Apache-2.0, like the target model and the engine.
