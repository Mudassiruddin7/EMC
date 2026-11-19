#!/usr/bin/env python3
"""Unattended driver for scripts/pipeline.sh (standard library only).

Every minute it asks the public code repository for the head of the branch. When it moved,
the new code is installed as a fresh snapshot behind the `repo` symlink (running processes are
not disturbed). It then applies control requests committed to the repository and (re)starts
the pipeline when it is not running and either the code changed or a reset asked for it:

  control/reset.txt   lines "<id> [kill] <stage> ..." - forget those stages so they run again
                      ("kill" first stops a running pipeline); each id is applied once
  control/run.env     KEY=VALUE lines passed to the pipeline (budgets etc.)

While the pipeline runs, logs are pushed every 10 minutes. Create $EMC_ROOT/STOP to end it.
"""
import io
import os
import re
import shutil
import signal
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("EMC_ROOT", str(Path.home() / "emc"))).resolve()
SOURCE = os.environ.get("EMC_SOURCE", "Mudassiruddin7/EMC")
BRANCH = os.environ.get("EMC_BRANCH", "main")
UV = os.environ.get("UV", "uv")
LOGS, STATE, CODE = ROOT / "logs", ROOT / "state", ROOT / "code"
VENV_PY = ROOT / "venv" / "bin" / "python"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def tool_env():
    return dict(os.environ, UV_CACHE_DIR=str(ROOT / "uv-cache"), UV_PYTHON_INSTALL_DIR=str(ROOT / "uv-python"),
                UV_LINK_MODE="copy", TMPDIR=str(ROOT / "tmp"), HF_HOME=str(ROOT / "hf"))


def remote_sha():
    req = urllib.request.Request(f"https://github.com/{SOURCE}.git/info/refs?service=git-upload-pack",
                                 headers={"User-Agent": "git/2.45.0"})
    body = urllib.request.urlopen(req, timeout=30).read().decode("latin-1")
    m = re.search(r"([0-9a-f]{40}) refs/heads/" + re.escape(BRANCH) + r"(?![\w./-])", body)
    return m.group(1) if m else None


def complete(code_dir):
    """A code snapshot is usable only with its commit stamp and the pipeline script (folders
    can survive a machine change without their files)."""
    return (code_dir / ".commit").is_file() and (code_dir / "scripts" / "pipeline.sh").is_file()


def install(sha):
    dest = CODE / sha[:12]
    if dest.exists() and not complete(dest):
        shutil.rmtree(dest, ignore_errors=True)
    if not dest.exists():
        data = urllib.request.urlopen(f"https://codeload.github.com/{SOURCE}/tar.gz/{sha}", timeout=300).read()
        tmp = CODE / f".{sha[:12]}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            members = []
            for m in tf.getmembers():
                name = m.name.split("/", 1)[1] if "/" in m.name else ""
                if name and not name.startswith("/") and ".." not in Path(name).parts:
                    m.name = name
                    members.append(m)
            try:
                tf.extractall(tmp, members=members, filter="data")
            except TypeError:
                tf.extractall(tmp, members=members)
        (tmp / ".commit").write_text(sha + "\n")
        tmp.rename(dest)
    link = ROOT / "repo"
    if link.exists() and not link.is_symlink():
        link.rename(CODE / f"legacy-{int(time.time())}")
    tmp_link = ROOT / ".repo-link"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(dest, target_is_directory=True)
    os.replace(tmp_link, link)


def ensure_venv():
    """A private Python with huggingface_hub, so logs can be pushed from the very first stage."""
    if not VENV_PY.exists():
        subprocess.run([UV, "venv", "--python", "3.12", str(ROOT / "venv")], check=True, env=tool_env())
    subprocess.run([UV, "pip", "install", "--python", str(VENV_PY), "huggingface_hub"], check=True,
                   env=tool_env(), stdout=subprocess.DEVNULL)


def read_control():
    env, resets = {}, []
    p = ROOT / "repo" / "control" / "run.env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    applied = set((STATE / "applied_resets").read_text().split()) if (STATE / "applied_resets").exists() else set()
    p = ROOT / "repo" / "control" / "reset.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split()
            if parts and not parts[0].startswith("#") and parts[0] not in applied:
                resets.append((parts[0], "kill" in parts[1:], [s for s in parts[1:] if s != "kill"]))
    return env, resets


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def sync():
    if VENV_PY.exists() and (ROOT / "repo" / "scripts" / "sync.py").exists():
        try:
            with open(LOGS / "sync.log", "a") as f:
                subprocess.run([str(VENV_PY), str(ROOT / "repo" / "scripts" / "sync.py")], stdout=f,
                               stderr=subprocess.STDOUT, env=dict(tool_env(), EMC_ROOT=str(ROOT)), timeout=1200)
        except Exception as e:  # noqa: BLE001 - an upload problem never stops the supervisor
            log(f"log push failed: {type(e).__name__}: {e}")


def start_pipeline(extra):
    run_sh = ROOT / "pipeline.run.sh"            # a copy: bash reads scripts while running them
    shutil.copy(ROOT / "repo" / "scripts" / "pipeline.sh", run_sh)
    env = tool_env()
    env.update(extra)
    env.update(EMC_ROOT=str(ROOT), EMC_REPO=str(ROOT / "repo"), UV=UV)
    out = open(LOGS / "pipeline.out", "a")
    return subprocess.Popen(["bash", str(run_sh)], stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env, cwd=str(ROOT), start_new_session=True)


def main():
    for d in (LOGS, STATE, CODE, ROOT / "tmp"):
        d.mkdir(parents=True, exist_ok=True)
    (ROOT / "supervisor.pid").write_text(str(os.getpid()))
    if os.environ.get("RUN_ID") and not (STATE / "run_id").exists():
        (STATE / "run_id").write_text(os.environ["RUN_ID"] + "\n")
    head = ROOT / "repo" / ".commit"
    current = head.read_text().strip() if complete(ROOT / "repo") else ""
    ran = (STATE / "pipeline_commit").read_text().strip() if (STATE / "pipeline_commit").exists() else ""
    # force: (re)start the pipeline once on every supervisor start - finished stages are skipped,
    # and a machine restart in the middle of a stage resumes there.
    proc, last_sync, force, reported, venv_ok = None, 0.0, True, True, False
    log(f"supervisor started in {ROOT}")
    while not (ROOT / "STOP").exists():
        try:
            sha = remote_sha()
            if sha and sha != current:
                install(sha)
                current = sha
                log(f"code updated to {sha[:12]}")
        except Exception as e:  # noqa: BLE001 - network hiccups are retried next minute
            log(f"code check failed: {type(e).__name__}: {e}")
        if not venv_ok:
            try:
                ensure_venv()
                venv_ok = True
            except Exception as e:  # noqa: BLE001
                log(f"python environment not ready yet: {e}")
        env, resets = read_control()
        running = proc is not None and proc.poll() is None
        for rid, kill, stages in resets:
            if running and not kill:
                continue
            if running:
                stop(proc)
                running = False
                log("pipeline stopped for a reset")
            for s in stages:
                for suffix in ("done", "failed"):
                    (STATE / f"{s}.{suffix}").unlink(missing_ok=True)
            with open(STATE / "applied_resets", "a") as f:
                f.write(rid + "\n")
            log(f"reset {rid}: {' '.join(stages)}")
            force = True
        if not running:
            if proc is not None and not reported:
                log(f"pipeline exited with code {proc.returncode}")
                reported = True
                sync()
            if current and not complete(ROOT / "repo"):
                current = ""                      # snapshot lost its files: fetch it again next minute
                log("code snapshot incomplete; reinstalling")
            elif current and (current != ran or force):
                try:
                    proc = start_pipeline(env)
                except Exception as e:  # noqa: BLE001 - retried next minute
                    log(f"pipeline start failed: {type(e).__name__}: {e}")
                    time.sleep(60)
                    continue
                ran, force, reported, last_sync = current, False, False, time.time()
                (STATE / "pipeline_commit").write_text(current + "\n")
                log(f"pipeline started (pid {proc.pid}) at {current[:12]}")
        elif time.time() - last_sync > 600:
            try:
                sync()
            except Exception as e:  # noqa: BLE001
                log(f"log push failed: {e}")
            last_sync = time.time()
        time.sleep(60)
    log("stop requested")


if __name__ == "__main__":
    main()
