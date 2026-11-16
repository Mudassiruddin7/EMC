#!/usr/bin/env bash
# Fetch (or update) the code and start the pipeline detached from the calling shell.
#   bash bootstrap.sh            run everything that is not done yet
#   STAGES="train eval_main" bash bootstrap.sh
set -euo pipefail
export EMC_ROOT="${EMC_ROOT:-$HOME/emc}"
mkdir -p "$EMC_ROOT/logs"
if [ -d "$EMC_ROOT/repo/.git" ]; then
  git -C "$EMC_ROOT/repo" pull -q --ff-only || true
else
  git clone -q https://github.com/Mudassiruddin7/EMC.git "$EMC_ROOT/repo"
fi
if [ -f "$EMC_ROOT/pipeline.pid" ] && kill -0 "$(cat "$EMC_ROOT/pipeline.pid")" 2>/dev/null; then
  echo "pipeline already running (pid $(cat "$EMC_ROOT/pipeline.pid"))"; exit 0
fi
launcher=(nohup)
command -v setsid >/dev/null && launcher=(setsid nohup)
"${launcher[@]}" bash "$EMC_ROOT/repo/scripts/pipeline.sh" >> "$EMC_ROOT/logs/pipeline.out" 2>&1 < /dev/null &
echo $! > "$EMC_ROOT/pipeline.pid"
echo "pipeline started (pid $!); progress: tail -f $EMC_ROOT/logs/pipeline.log"
