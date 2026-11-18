#!/usr/bin/env bash
# End-to-end pipeline on one GPU:
#   environment -> target + baseline -> engine checks -> prompts -> generation (engine)
#   -> hidden states -> training -> official kit evaluation -> summary
# Each stage is resumable (state/<stage>.done) and logs to logs/<stage>.log. Logs and results
# are pushed to the Hugging Face dataset repo after every stage (scripts/sync.py).
# Normally started by scripts/supervise.py; can also be run directly.
set -uo pipefail

export EMC_ROOT="${EMC_ROOT:-$HOME/emc}"
REPO="${EMC_REPO:-$EMC_ROOT/repo}"
KIT="$EMC_ROOT/kit"; MODELS="$EMC_ROOT/models"; WORK="$EMC_ROOT/work"; DATA="$EMC_ROOT/data"
TARGET="$MODELS/target"; BASELINE="$MODELS/baseline-mtp"; PROBE="$MODELS/probe-fp8"
RUNS="$EMC_ROOT/runs"; STATE="$EMC_ROOT/state"; LOGS="$EMC_ROOT/logs"; RESULTS="$EMC_ROOT/results"
VENV="$EMC_ROOT/venv"; PY="$VENV/bin/python"; UV="${UV:-uv}"
KIT_SHA=de57e284ae1b3d72ed7e06f00e030b2ba797adf1
mkdir -p "$MODELS" "$WORK" "$DATA" "$RUNS" "$STATE" "$LOGS" "$RESULTS" "$EMC_ROOT/tmp"
export PATH="$VENV/bin:$PATH" PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="$EMC_ROOT/hf" HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TMPDIR="$EMC_ROOT/tmp"
export UV_CACHE_DIR="$EMC_ROOT/uv-cache" UV_PYTHON_INSTALL_DIR="$EMC_ROOT/uv-python" UV_LINK_MODE=copy
export EMC2_HOME="$KIT" EMC_RESULTS="$RESULTS"
export XDG_CACHE_HOME="$WORK/cache" VLLM_CACHE_ROOT="$WORK/cache/vllm" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/inductor" TRITON_CACHE_DIR="$WORK/cache/triton"
# greedy decoding only: use the built-in sampler instead of compiling FlashInfer kernels at start-up
export VLLM_USE_FLASHINFER_SAMPLER=0
[ -f "$STATE/run_id" ] || echo "${RUN_ID:-$(date +%Y%m%d-%H%M)}" > "$STATE/run_id"
if [ -n "${RUN_ID_FORCE:-}" ]; then echo "$RUN_ID_FORCE" > "$STATE/run_id"; fi   # e.g. a new run on a fresh machine
# training run name (runs/<name>), its starting weights and whether to also evaluate the bf16 variant
RUN_NAME="${RUN_NAME:-main}"; TRAIN_INIT="${TRAIN_INIT:-$BASELINE}"; EVAL_ALT="${EVAL_ALT:-1}"
case "$TRAIN_INIT" in /*) ;; *) TRAIN_INIT="$EMC_ROOT/$TRAIN_INIT" ;; esac
OUTRUN="$RUNS/$RUN_NAME"
# the previous best draft (kept unless the new one scores higher on this machine)
PREV_TAG="${PREV_TAG:-}"; PREV_SUB="${PREV_SUB:-}"
case "$PREV_SUB" in ""|/*) ;; *) PREV_SUB="$EMC_ROOT/$PREV_SUB" ;; esac
if [ "$RUN_NAME" = main ]; then EVAL_PREFIX=eval_draft; else EVAL_PREFIX="eval_$RUN_NAME"; fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGS/pipeline.log"; }
sync_hf() { if [ -x "$PY" ]; then "$PY" "$REPO/scripts/sync.py" "$@" >> "$LOGS/sync.log" 2>&1 || true; fi; }
is_done() { [ -f "$STATE/$1.done" ]; }
stats() { if [ -x "$PY" ]; then "$PY" "$REPO/scripts/sysstats.py" "$EMC_ROOT" >> "$LOGS/stats.log" 2>/dev/null || true; fi; }

stage() {  # stage <name> <required: 0|1> <function>
  local name=$1 required=$2 fn=$3 t0=$SECONDS rc
  if is_done "$name"; then log "skip  $name (done)"; return 0; fi
  log "start $name"
  stats
  ( set -e; "$fn" ) > "$LOGS/$name.log" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then
    echo $((SECONDS - t0)) > "$STATE/$name.done"; rm -f "$STATE/$name.failed"
    log "done  $name ($((SECONDS - t0))s)"
  else
    echo "exit $rc after $((SECONDS - t0))s" > "$STATE/$name.failed"
    log "FAIL  $name (exit $rc after $((SECONDS - t0))s); end of logs/$name.log:"
    tail -n 25 "$LOGS/$name.log" | sed 's/^/      /' | tee -a "$LOGS/pipeline.log"
  fi
  sync_hf
  if [ "$rc" != 0 ] && [ "$required" = 1 ]; then log "stopping: $name is required"; exit 1; fi
  return "$rc"
}

kit() {  # the official kit commands (scripts/kit.py only maps its container paths)
  env EMC_TARGET="$TARGET" EMC_SUB="${SUB:-$BASELINE}" EMC2_WORK="$WORK" EMC_TAG="${TAG:-}" \
    "$PY" "$REPO/scripts/kit.py" "$@"
}

s_env() {
  nvidia-smi || true
  df -h "$EMC_ROOT" || true; free -g || true; nproc || true
  # a Python install or environment left half-present by a machine reset is rebuilt
  for d in "$UV_PYTHON_INSTALL_DIR"/cpython-3.12*; do
    if [ -d "$d" ] && [ ! -x "$d/bin/python3.12" ]; then echo "removing incomplete $d"; rm -rf "$UV_PYTHON_INSTALL_DIR"; break; fi
  done
  if ! "$PY" -c "import sys" >/dev/null 2>&1; then rm -rf "$VENV"; "$UV" venv --python 3.12 "$VENV"; fi
  "$UV" pip install --python "$PY" "vllm==0.31.0" accelerate datasets
  "$UV" cache clean >/dev/null 2>&1 || true
  "$PY" - <<'EOF'
import torch, transformers, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "cuda", torch.version.cuda,
      "| transformers", transformers.__version__)
print(torch.cuda.get_device_name(0), round(torch.cuda.mem_get_info(0)[1] / 2**30, 1), "GiB")
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", bool(torch.isfinite((x @ x).float()).all()))
EOF
}

s_download() {
  if [ ! -f "$KIT/contract.json" ]; then
    "$PY" - "https://codeload.github.com/emc2-competition/emc2-kit/tar.gz/$KIT_SHA" "$KIT" <<'EOF'
import io, sys, tarfile, urllib.request
from pathlib import Path
data = urllib.request.urlopen(sys.argv[1], timeout=300).read()
dst = Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data)) as tf:
    ms = []
    for m in tf.getmembers():
        if "/" in m.name and m.name.split("/", 1)[1]:
            m.name = m.name.split("/", 1)[1]; ms.append(m)
    try:
        tf.extractall(dst, members=ms, filter="data")
    except TypeError:
        tf.extractall(dst, members=ms)
print("kit ready:", sorted(p.name for p in dst.iterdir()))
EOF
  fi
  local rev; rev=$("$PY" -c "import json; print(json.load(open('$KIT/target.json'))['revision'])")
  "$VENV/bin/hf" download Qwen/Qwen3.8-27B --revision "$rev" --local-dir "$TARGET" \
    || "$PY" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.8-27B', revision='$rev', local_dir='$TARGET')"
  du -sh "$TARGET"; df -h "$EMC_ROOT"
}

s_baseline() { kit prepare-baseline --models-dir "$MODELS"; SUB="$BASELINE" kit validate; }
s_checkenv() { kit check-env; }
s_probe() {
  "$PY" -m emcdraft.probe --target "$TARGET" --baseline "$BASELINE" --out "$PROBE"
  SUB="$PROBE" kit validate
}
s_smoke_fp8() {   # does the stock engine load an FP8-typed head, and what does rounding alone cost?
  SUB="$PROBE" TAG=smoke_fp8 kit eval --limit-per-domain 2
  "$PY" - "$RESULTS/smoke_fp8.json" "$KIT/data/dev/baseline.json" "$STATE/fp8_ok" <<'EOF'
import json, sys
f, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ratio = (f["acceptance_length"] - 1) / (b["acceptance_length"] - 1)
print(f"FP8 probe acceptance {f['acceptance_length']:.3f} vs published baseline {b['acceptance_length']:.3f} (ratio {ratio:.3f})")
if ratio >= 0.85:
    open(sys.argv[3], "w").write(f"{ratio:.4f}\n")
else:
    print("acceptance collapsed: FP8 storage will not be used")
EOF
}
s_smoke_base() { SUB="$BASELINE" TAG=smoke_baseline kit eval --limit-per-domain 2; }
s_prompts()  { "$PY" -m emcdraft.prompts --target "$TARGET" --kit "$KIT" --out "$DATA/prompts" --results "$RESULTS"; }
s_generate() { "$PY" -m emcdraft.gen_vllm --target "$TARGET" --prompts "$DATA/prompts" --out "$DATA/gen" --results "$RESULTS"; }
s_extract()  { "$PY" -m emcdraft.extract --target "$TARGET" --kit "$KIT" --baseline "$BASELINE" --gen "$DATA/gen" --out "$DATA" --results "$RESULTS"; }
fetch_hf() {   # fetch_hf <dataset-repo>:<folder> <destination>: a draft folder from the dataset repo
  "$PY" - "$1" "$2" <<'PYEOF'
import shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo, sub = sys.argv[1].split(":", 1)
src = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[sub + "/*"])) / sub
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(src, sys.argv[2])
print("downloaded", sys.argv[1])
PYEOF
}
init_weights() {   # starting weights for training (downloaded on a fresh machine)
  if [ ! -d "$TRAIN_INIT" ] && [ -n "${TRAIN_INIT_HF:-}" ]; then fetch_hf "$TRAIN_INIT_HF" "$TRAIN_INIT"; fi
  [ -d "$TRAIN_INIT" ] || { echo "starting weights $TRAIN_INIT not found"; return 1; }
}
s_ablate() {   # short runs comparing training recipes; the best one is used by s_train
  if [ "${ABLATE:-0}" != 1 ]; then echo "no recipe comparison requested (ABLATE=${ABLATE:-0})"; return 0; fi
  local qat=""; [ -f "$STATE/fp8_ok" ] && qat="--qat"
  init_weights
  "$PY" -m emcdraft.ablate --target "$TARGET" --init "$TRAIN_INIT" --data "$DATA/train" --dev "$DATA/dev" \
    --out "$RUNS/ablate" --results "$RESULTS" --state "$STATE" --steps "${ABLATE_STEPS:-300}" $qat
}
s_train() {
  local qat=""; [ -f "$STATE/fp8_ok" ] && qat="--qat"
  local extra=""; [ -f "$STATE/train_args" ] && extra=$(cat "$STATE/train_args")
  echo "training recipe: ${extra:-defaults}"
  init_weights
  rm -rf "$OUTRUN"
  "$PY" -m emcdraft.train --target "$TARGET" --init "$TRAIN_INIT" --data "$DATA/train" --dev "$DATA/dev" --out "$OUTRUN" $qat $extra
  cat "$STATE/run_id" > "$OUTRUN/run_id"      # which run the exported folders belong to (for uploads)
  for s in "$OUTRUN"/sub_*; do SUB="$s" kit validate; done
}
primary() { if [ -d "$OUTRUN/sub_fp8" ]; then echo "$OUTRUN/sub_fp8"; else echo "$OUTRUN/sub_bf16"; fi; }
s_eval_main() { local p; p=$(primary); SUB="$p" TAG="${EVAL_PREFIX}_$(basename "$p" | sed 's/^sub_//')" kit eval; }
s_eval_baseline() { SUB="$BASELINE" TAG=eval_baseline kit eval; }
s_eval_alt() {
  if [ "$EVAL_ALT" != 1 ]; then echo "bf16 variant not evaluated (EVAL_ALT=$EVAL_ALT)"; return 0; fi
  if [ ! -d "$OUTRUN/sub_fp8" ]; then echo "only one variant"; return 0; fi
  SUB="$OUTRUN/sub_bf16" TAG="${EVAL_PREFIX}_bf16" kit eval
}
s_report() { "$PY" "$REPO/scripts/collect.py"; }
new_tag() { echo "${EVAL_PREFIX}_$(basename "$(primary)" | sed 's/^sub_//')"; }
s_eval_prev() {   # the previous best draft on this machine, when its result is not here yet
  if [ -z "$PREV_TAG" ]; then echo "no previous draft to compare with"; return 0; fi
  if [ -f "$RESULTS/$PREV_TAG.json" ]; then echo "previous draft already evaluated on this machine"; return 0; fi
  if [ ! -d "$PREV_SUB" ] && [ -n "${PREV_SUB_HF:-}" ]; then fetch_hf "$PREV_SUB_HF" "$PREV_SUB"; fi
  SUB="$PREV_SUB" TAG="$PREV_TAG" kit eval
}
s_choose() {
  "$PY" "$REPO/scripts/choose.py" --new "$(new_tag)" --new-sub "$(primary)" --prev "$PREV_TAG"     --prev-sub "$PREV_SUB" --results "$RESULTS" --state "$STATE"
}
bench() { env EMC_TARGET="$TARGET" EMC_SUB="${SUB:-}" EMC2_WORK="$WORK" "$PY" "$REPO/scripts/bench.py" "$@"; }
final_sub() { if [ -f "$STATE/final_sub" ]; then cat "$STATE/final_sub"; else primary; fi; }
s_bench_draft() { SUB="$(final_sub)" bench --tag draft --what batch,long,k; }
s_bench_baseline() { SUB="$BASELINE" bench --tag baseline --what batch,long; }
s_bench_target() { SUB="" bench --tag target_only --target-only --what batch,long; }

log "pipeline start: run $(cat "$STATE/run_id"), code $(cat "$REPO/.commit" 2>/dev/null | cut -c1-12), training run $RUN_NAME"
# a vLLM server left behind by an interrupted run would still hold the GPU
if pkill -f -- "--served-model-name target" 2>/dev/null; then log "stopped a leftover vLLM server"; sleep 20; fi
stage env 1 s_env
stage download 1 s_download
stage baseline 1 s_baseline
prompts_pid=""
if ! is_done prompts; then (stage prompts 1 s_prompts) & prompts_pid=$!; fi
stage checkenv 0 s_checkenv || true
stage probe 0 s_probe || true
if is_done probe; then stage smoke_fp8 0 s_smoke_fp8 || true; fi
if ! is_done smoke_fp8; then stage smoke_base 1 s_smoke_base; fi   # engine itself must work
if [ -n "$prompts_pid" ]; then wait "$prompts_pid"; fi
is_done prompts || { log "stopping: prompts failed"; exit 1; }
stage generate 1 s_generate
stage extract 1 s_extract
stage ablate 0 s_ablate || true
stage train 1 s_train
sync_hf --submissions
stage eval_main 0 s_eval_main || true
stage report_main 0 s_report || true
stage eval_baseline 0 s_eval_baseline || true
stage report_baseline 0 s_report || true
stage eval_alt 0 s_eval_alt || true
stage eval_prev 0 s_eval_prev || true
stage choose 0 s_choose || true
stage report 0 s_report || true
if [ "${BENCH:-0}" = 1 ]; then
  stage bench_draft 0 s_bench_draft || true
  stage bench_baseline 0 s_bench_baseline || true
  stage bench_target 0 s_bench_target || true
  stage report_bench 0 s_report || true
fi
sync_hf --submissions
log "pipeline finished"
