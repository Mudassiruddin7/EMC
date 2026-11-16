#!/usr/bin/env python3
"""Push logs, results and (optionally) submission folders to the Hugging Face dataset repo.

Reads the token from $HF_TOKEN, $EMC_ROOT/.hf_token, ~/.hf_token or the hub's cached login;
without a token it does nothing (the run itself never depends on the upload).

  sync.py                 logs + results + state for the current run
  sync.py --submissions   additionally upload every runs/*/sub_* folder (weights)
"""
import os
import re
import shutil
import socket
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EMC_ROOT", Path.home() / "emc"))
REPO_ID = os.environ.get("HF_REPO", "mudassiruddin21/EMC")
TEXT = {".log", ".out", ".json", ".md", ".txt", ".done", ".failed"}
SCRUB = [(re.compile(r"hf_[A-Za-z0-9]{16,}"), "hf_***"),
         (re.compile(re.escape(str(ROOT))), "/workspace/emc"),
         (re.compile(re.escape(str(Path.home()))), "/workspace"),
         (re.compile(re.escape(socket.gethostname())), "gpu-node")]


def token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    for p in (ROOT / ".hf_token", Path.home() / ".hf_token"):
        if p.exists():
            return p.read_text().strip()
    return None


def scrub_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in TEXT and src.stat().st_size < 64 * 2 ** 20:
        text = src.read_text(errors="replace")
        for pat, rep in SCRUB:
            text = pat.sub(rep, text)
        dst.write_text(text)
    else:
        shutil.copy(src, dst)


def main():
    from huggingface_hub import HfApi
    tok = token()
    api = HfApi(token=tok)
    try:
        api.whoami()
    except Exception:  # noqa: BLE001
        print("no usable Hugging Face token: skipping upload")
        return 0
    run_id = (ROOT / "state" / "run_id").read_text().strip() if (ROOT / "state" / "run_id").exists() else "run"
    stage = ROOT / "export"
    shutil.rmtree(stage, ignore_errors=True)
    for pattern, sub in (("logs/*", "logs"), ("results/*.json", "results"), ("results/*.md", "results"),
                         ("state/*", "state"), ("work/logs/*.log", "engine_logs"),
                         ("work/results/*.json", "kit_results"), ("runs/*/train_summary.json", "training")):
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file():
                name = f"{p.parent.name}-{p.name}" if sub == "training" else p.name
                scrub_copy(p, stage / sub / name)
    if stage.exists():
        api.upload_folder(folder_path=str(stage), repo_id=REPO_ID, repo_type="dataset",
                          path_in_repo=f"runs/{run_id}", commit_message=f"Run {run_id}: logs and results")
        print(f"uploaded logs/results to {REPO_ID}:runs/{run_id}")
    if "--submissions" in sys.argv:
        for sub in sorted(ROOT.glob("runs/*/sub_*")):
            name = f"{sub.parent.name}-{sub.name[4:]}"
            api.upload_folder(folder_path=str(sub), repo_id=REPO_ID, repo_type="dataset",
                              path_in_repo=f"submissions/{run_id}/{name}",
                              commit_message=f"Run {run_id}: draft model {name}")
            print(f"uploaded {sub} to {REPO_ID}:submissions/{run_id}/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
