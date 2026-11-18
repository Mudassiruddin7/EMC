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
[ -f "$STATE/run_id" ] || echo "${RUN_ID:-$(date +%Y%m%d-%H%M)}" > "$STATE/run_id"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGS/pipeline.log"; }
sync_hf() { if [ -x "$PY" ]; then "$PY" "$REPO/scripts/sync.py" "$@" >> "$LOGS/sync.log" 2>&1 || true; fi; }
is_done() { [ -f "$STATE/$1.done" ]; }

stage() {  # stage <name> <required: 0|1> <function>
  local name=$1 required=$2 fn=$3 t0=$SECONDS rc
  if is_done "$name"; then log "skip  $name (done)"; return 0; fi
  log "start $name"
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
  [ -x "$PY" ] || "$UV" venv --python 3.12 "$VENV"
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
s_train() {
  local qat=""; [ -f "$STATE/fp8_ok" ] && qat="--qat"
  rm -rf "$RUNS/main"
  "$PY" -m emcdraft.train --target "$TARGET" --init "$BASELINE" --data "$DATA/train" --dev "$DATA/dev" --out "$RUNS/main" $qat
  for s in "$RUNS"/main/sub_*; do SUB="$s" kit validate; done
}
primary() { if [ -d "$RUNS/main/sub_fp8" ]; then echo "$RUNS/main/sub_fp8"; else echo "$RUNS/main/sub_bf16"; fi; }
s_eval_main() { local p; p=$(primary); SUB="$p" TAG="eval_draft_$(basename "$p" | sed 's/^sub_//')" kit eval; }
s_eval_baseline() { SUB="$BASELINE" TAG=eval_baseline kit eval; }
s_eval_alt() {
  if [ ! -d "$RUNS/main/sub_fp8" ]; then echo "only one variant"; return 0; fi
  SUB="$RUNS/main/sub_bf16" TAG=eval_draft_bf16 kit eval
}
s_report() { "$PY" "$REPO/scripts/collect.py"; }

log "pipeline start: run $(cat "$STATE/run_id"), code $(cat "$REPO/.commit" 2>/dev/null | cut -c1-12)"
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
stage train 1 s_train
sync_hf --submissions
stage eval_main 0 s_eval_main || true
stage report_main 0 s_report || true
stage eval_baseline 0 s_eval_baseline || true
stage report_baseline 0 s_report || true
stage eval_alt 0 s_eval_alt || true
stage report 0 s_report || true
sync_hf --submissions
log "pipeline finished"
