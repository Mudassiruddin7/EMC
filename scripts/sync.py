#!/usr/bin/env python3
"""Push logs, results and (optionally) the draft-model folders to the Hugging Face dataset repo.

The token comes from $HF_TOKEN or $EMC_ROOT/.hf_token; without one nothing is uploaded (the run
never depends on the upload). Paths and host names are scrubbed from text files.

  sync.py                 logs + results + state + generated token data of the current run
  sync.py --submissions   additionally every runs/*/sub_* folder (weights)
"""
import os
import re
import shutil
import socket
import sys
from pathlib import Path

try:
    import fcntl
except ImportError:      # not on Linux: no cross-process lock
    fcntl = None

ROOT = Path(os.environ.get("EMC_ROOT", str(Path.home() / "emc")))
REPO_ID = os.environ.get("HF_REPO", "mudassiruddin21/EMC")
TEXT = {".log", ".out", ".json", ".jsonl", ".md", ".txt", ".done", ".failed", ""}
SCRUB = [(re.compile(r"hf_[A-Za-z0-9]{16,}"), "hf_***"),
         (re.compile(re.escape(str(ROOT.resolve()))), "/workspace/emc"),
         (re.compile(re.escape(str(ROOT))), "/workspace/emc"),
         (re.compile(re.escape(str(Path.home()))), "/workspace"),
         (re.compile(r"\b" + re.escape(socket.gethostname()) + r"\b"), "gpu-node")]
# extra words to keep out of published logs (comma separated), set at launch time
_words = os.environ.get("EMC_SCRUB") or ((ROOT / ".scrub").read_text() if (ROOT / ".scrub").exists() else "")
_words = [w.strip() for w in _words.split(",") if w.strip()]
if _words:
    SCRUB.append((re.compile("|".join(map(re.escape, _words)), re.I), "host"))
PATTERNS = (("logs/*", "logs"), ("results/*.json", "results"), ("results/*.md", "results"),
            ("state/*", "state"), ("work/logs/*.log", "engine_logs"), ("work/results/*.json", "kit_results"),
            ("runs/*/train_summary.json", "training"), ("data/gen/*.jsonl", "data/gen"))


def token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    p = ROOT / ".hf_token"
    return p.read_text().strip() if p.exists() else None


def scrub_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in TEXT and src.stat().st_size < 64 * 2 ** 20 and src.suffix != ".jsonl":
        text = src.read_text(errors="replace")
        for pat, rep in SCRUB:
            text = pat.sub(rep, text)
        dst.write_text(text)
    else:
        shutil.copy(src, dst)


def record_stats():
    """Append a machine snapshot (memory, disk, GPU) to logs/stats.log; never blocks an upload."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sysstats import snapshot
        (ROOT / "logs").mkdir(parents=True, exist_ok=True)
        with open(ROOT / "logs" / "stats.log", "a") as f:
            f.write(snapshot(ROOT) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"machine stats unavailable: {e}")


def main():
    record_stats()
    tok = token()
    if not tok:
        print("no Hugging Face token: skipping upload")
        return 0
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    run_id = (ROOT / "state" / "run_id").read_text().strip() if (ROOT / "state" / "run_id").exists() else "run"
    with open(ROOT / ".sync.lock", "w") as lock:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        stage = ROOT / f"export-{os.getpid()}"
        shutil.rmtree(stage, ignore_errors=True)
        try:
            for pattern, sub in PATTERNS:
                for p in sorted(ROOT.glob(pattern)):
                    if p.is_file() and not p.name.startswith("."):
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
                    print(f"uploaded {sub.name} to {REPO_ID}:submissions/{run_id}/{name}")
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
