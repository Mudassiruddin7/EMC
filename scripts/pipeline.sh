#!/usr/bin/env bash
# End-to-end pipeline on one GPU: environment -> baseline checks -> data -> training -> evaluation.
# Every stage is resumable (state/<stage>.done) and logs to logs/<stage>.log; logs and results
# are pushed to the Hugging Face dataset repo after each stage when a token is available.
#
#   EMC_ROOT        working folder (default: ~/emc)
#   STAGES          optional space-separated subset of stages to run
#   GEN_MINUTES / TRAIN_MINUTES / GEN_BATCH / TRAIN_EPOCHS   budgets (see emcdraft/*.py)
set -uo pipefail

export EMC_ROOT="${EMC_ROOT:-$HOME/emc}"
REPO="$EMC_ROOT/repo"; KIT="$EMC_ROOT/kit"; MODELS="$EMC_ROOT/models"; WORK="$EMC_ROOT/work"
TARGET="$MODELS/target"; BASELINE="$MODELS/baseline-mtp"; DATA="$EMC_ROOT/data"; RUNS="$EMC_ROOT/runs"
STATE="$EMC_ROOT/state"; LOGS="$EMC_ROOT/logs"; VENV="$EMC_ROOT/venv"; RESULTS="$EMC_ROOT/results"
mkdir -p "$MODELS" "$WORK" "$DATA" "$RUNS" "$STATE" "$LOGS" "$RESULTS"
export PATH="$VENV/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 EMC2_HOME="$KIT" EMC_RESULTS="$RESULTS"
[ -f "$STATE/run_id" ] || date +%Y%m%d-%H%M > "$STATE/run_id"
PY="$VENV/bin/python"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/pipeline.log"; }
sync_hf() { [ -x "$PY" ] && "$PY" "$REPO/scripts/sync.py" "$@" >> "$LOGS/sync.log" 2>&1 || true; }
wanted() { [ -z "${STAGES:-}" ] || [[ " $STAGES " == *" $1 "* ]]; }

stage() {  # stage <name> <required:0|1> <function>
  local name=$1 required=$2; shift 2
  wanted "$name" || return 0
  if [ -f "$STATE/$name.done" ]; then log "skip  $name (already done)"; return 0; fi
  log "start $name"; local t0=$SECONDS rc=0
  ( set -e; "$@" ) > "$LOGS/$name.log" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then
    echo $((SECONDS - t0)) > "$STATE/$name.done"; rm -f "$STATE/$name.failed"
    log "done  $name ($((SECONDS - t0))s)"
  else
    echo $((SECONDS - t0)) > "$STATE/$name.failed"
    log "FAIL  $name ($((SECONDS - t0))s); last lines of logs/$name.log:"
    tail -n 30 "$LOGS/$name.log" | tee -a "$LOGS/pipeline.log"
    sync_hf
    if [ "$required" = 1 ]; then log "stopping: $name is required"; exit 1; fi
    return 1
  fi
  sync_hf
}

kit() {
  env EMC_TARGET="$TARGET" EMC_SUB="${SUB:-$BASELINE}" EMC2_WORK="$WORK" EMC_TAG="${EMC_TAG:-}" \
    "$PY" "$REPO/scripts/kit.py" "$@"
}

s_env() {
  nvidia-smi || true; df -h "$EMC_ROOT" || true; free -g || true; nproc || true
  if ! command -v uv >/dev/null; then
    python3 -m pip install -q --user uv || pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  [ -x "$PY" ] || uv venv --python 3.12 "$VENV"
  uv pip install --python "$PY" "vllm==0.31.0" --torch-backend=auto || uv pip install --python "$PY" "vllm==0.31.0"
  uv pip install --python "$PY" accelerate datasets safetensors numpy "huggingface_hub[cli]"
  "$PY" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__); print(torch.cuda.get_device_name(0), round(torch.cuda.mem_get_info(0)[1] / 2**30, 1), 'GiB')"
  command -v gcc || echo "note: no gcc on PATH"
}

s_download() {
  [ -d "$KIT/.git" ] || git clone -q https://github.com/emc2-competition/emc2-kit.git "$KIT"
  local rev; rev=$("$PY" -c "import json; print(json.load(open('$KIT/target.json'))['revision'])")
  "$VENV/bin/hf" download Qwen/Qwen3.8-27B --revision "$rev" --local-dir "$TARGET" \
    || "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.8-27B', revision='$rev', local_dir='$TARGET')"
  du -sh "$TARGET"
}

s_baseline() { kit prepare-baseline --models-dir "$MODELS" && SUB="$BASELINE" kit validate; }
s_checkenv() { kit check-env; }
s_probe()    { "$PY" -m emcdraft.probe --target "$TARGET" --baseline "$BASELINE" --out "$MODELS/probe-fp8" && SUB="$MODELS/probe-fp8" kit validate; }
s_smoke_base() { SUB="$BASELINE" EMC_TAG=smoke_baseline kit eval --limit-per-domain 2; }
s_smoke_fp8() {
  SUB="$MODELS/probe-fp8" EMC_TAG=smoke_fp8 kit eval --limit-per-domain 2 || return 1
  "$PY" - <<EOF
import json
b = json.load(open("$RESULTS/smoke_baseline.json")); f = json.load(open("$RESULTS/smoke_fp8.json"))
ratio = (f["acceptance_length"] - 1) / (b["acceptance_length"] - 1)
print(f"FP8 probe: acceptance {f['acceptance_length']:.3f} vs baseline {b['acceptance_length']:.3f} (ratio {ratio:.3f})")
if ratio >= 0.93: open("$STATE/fp8_ok", "w").write(f"{ratio:.4f}\n")
EOF
}
s_datagen() { "$PY" -m emcdraft.datagen --target "$TARGET" --kit "$KIT" --baseline "$BASELINE" --out "$DATA" --results "$RESULTS"; }
s_train() {
  local qat=""; [ -f "$STATE/fp8_ok" ] && qat="--qat"
  "$PY" -m emcdraft.train --target "$TARGET" --init "$BASELINE" --data "$DATA/train" --dev "$DATA/dev" --out "$RUNS/main" $qat \
    && for s in "$RUNS"/main/sub_*; do SUB="$s" kit validate; done
}
primary() { if [ -d "$RUNS/main/sub_fp8" ]; then echo "$RUNS/main/sub_fp8"; else echo "$RUNS/main/sub_bf16"; fi; }
s_eval_main()     { SUB="$(primary)" EMC_TAG="eval_$(basename "$(primary)" | sed 's/sub_/draft_/')" kit eval; }
s_eval_baseline() { SUB="$BASELINE" EMC_TAG=eval_baseline kit eval; }
s_eval_alt() {
  [ -d "$RUNS/main/sub_fp8" ] || { echo "no second variant"; return 0; }
  SUB="$RUNS/main/sub_bf16" EMC_TAG=eval_draft_bf16 kit eval
}
s_report() { "$PY" "$REPO/scripts/collect.py"; }

log "pipeline start (root $EMC_ROOT, run $(cat "$STATE/run_id"))"
stage env 1 s_env
stage download 1 s_download
stage baseline 1 s_baseline
stage checkenv 0 s_checkenv
stage probe 0 s_probe
stage smoke_base 0 s_smoke_base
stage smoke_fp8 0 s_smoke_fp8
stage datagen 1 s_datagen
stage train 1 s_train
sync_hf --submissions
stage eval_main 0 s_eval_main
stage report_1 0 s_report
stage eval_baseline 0 s_eval_baseline
stage report_2 0 s_report
stage eval_alt 0 s_eval_alt
stage report 0 s_report
sync_hf --submissions
log "pipeline finished"
