#!/usr/bin/env python3
"""Run the official emc2 kit commands without a container runtime.

The kit expects fixed container paths (/models/target, /submission, /work). This launcher
points those names at local folders and then calls the kit's own entry point unchanged, so
validation, the engine command line, the metrics and the score are exactly the kit's.

  EMC2_HOME   kit checkout            EMC_TARGET  target model folder
  EMC_SUB     submission folder       EMC2_WORK   scratch / logs / results
  EMC_TAG     optional: copy the newest result file to $EMC_RESULTS/<tag>.json
"""
import os
import shutil
import sys
import time
from pathlib import Path

kit = Path(os.environ["EMC2_HOME"]).resolve()
work = Path(os.environ.get("EMC2_WORK", "emc2-work")).resolve()
work.mkdir(parents=True, exist_ok=True)
cache = work / "cache"
os.environ.update({
    "EMC2_HOME": str(kit), "EMC2_WORK": str(work), "PYTHONDONTWRITEBYTECODE": "1",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
    "XDG_CACHE_HOME": str(cache), "HF_HOME": str(cache / "hf"), "VLLM_CACHE_ROOT": str(cache / "vllm"),
    "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "TRITON_CACHE_DIR": str(cache / "triton"),
    "EMC2_IMAGE_USED": os.environ.get("EMC2_IMAGE_USED", "pip:vllm==0.31.0"),
    "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
})
sys.path.insert(0, str(kit))

from emc2 import common, evaluate, organizer, server  # noqa: E402
from emc2 import __main__ as cli  # noqa: E402

target = Path(os.environ["EMC_TARGET"]).resolve() if os.environ.get("EMC_TARGET") else common.TARGET_DIR
sub = Path(os.environ["EMC_SUB"]).resolve() if os.environ.get("EMC_SUB") else common.SUBMISSION_DIR
for mod in (common, evaluate, organizer, server, cli):
    for name, value in (("TARGET_DIR", target), ("SUBMISSION_DIR", sub), ("WORK", work)):
        if hasattr(mod, name):
            setattr(mod, name, value)
evaluate.TARGET_ONLY_LOCAL = work / "target_only_local.json"

started = time.time()
sys.argv = ["emc2", *sys.argv[1:]]
code = 0
try:
    cli.main()
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
tag, dest = os.environ.get("EMC_TAG"), os.environ.get("EMC_RESULTS")
if tag and dest and code == 0:
    new = [p for p in (work / "results").glob("*.json") if p.stat().st_mtime >= started]
    if new:
        Path(dest).mkdir(parents=True, exist_ok=True)
        shutil.copy(max(new, key=lambda p: p.stat().st_mtime), Path(dest) / f"{tag}.json")
sys.exit(code)
