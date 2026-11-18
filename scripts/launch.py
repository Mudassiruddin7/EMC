"""Start the unattended pipeline on a GPU machine, or show its status if it is already running.

From a notebook cell (or `python3 launch.py` in a shell):

    import os as _os, urllib.request as _u
    _os.environ["HF_TOKEN"] = "<Hugging Face write token>"      # needed once; kept in a private file
    exec(_u.urlopen("https://raw.githubusercontent.com/Mudassiruddin7/EMC/main/scripts/launch.py").read(),
         {"__name__": "__main__"})

Picks the writable folder with the most free space, stores the token (mode 600), fetches the
`uv` tool if needed and starts scripts/supervise.py in its own session, so it keeps running
after the cell returns. Re-running the cell only prints the status.
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

SOURCE = "Mudassiruddin7/EMC"


def pick_root():
    if os.environ.get("EMC_ROOT"):
        return Path(os.environ["EMC_ROOT"])
    for c in (Path.cwd(), Path.home(), Path("/workspace"), Path("/data"), Path("/mnt/data"), Path("/mnt"), Path("/tmp")):
        if (c / "emc" / "supervisor.pid").exists():
            return c / "emc"
    best, free = Path.home(), -1
    for c in (Path.cwd(), Path.home(), Path("/workspace"), Path("/data"), Path("/mnt/data"), Path("/mnt"), Path("/tmp")):
        try:
            if c.is_dir() and os.access(c, os.W_OK) and shutil.disk_usage(c).free > free:
                best, free = c, shutil.disk_usage(c).free
        except OSError:
            pass
    return best / "emc"


def get_uv(root):
    found = shutil.which("uv")
    if found:
        return found
    exe = root / "tools" / "uv"
    if not exe.exists():
        exe.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
        with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url, timeout=300).read())) as tf:
            for m in tf.getmembers():
                if m.isfile() and Path(m.name).name in ("uv", "uvx"):
                    p = exe.parent / Path(m.name).name
                    p.write_bytes(tf.extractfile(m).read())
                    p.chmod(0o755)
    return str(exe)


def alive(pidfile):
    try:
        os.kill(int(pidfile.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def status(root):
    print(f"working folder : {root}  ({shutil.disk_usage(root).used / 1e9:.0f} GB used)")
    print(f"supervisor     : {'running' if alive(root / 'supervisor.pid') else 'not running'}")
    try:
        print("GPU            :", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                                  "--format=csv,noheader"], capture_output=True, text=True,
                                                 timeout=20).stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    stats = root / "repo" / "scripts" / "sysstats.py"
    if stats.exists():
        try:
            print("machine        :", subprocess.run([sys.executable, str(stats), str(root)], capture_output=True,
                                                     text=True, timeout=60).stdout.strip())
        except Exception:  # noqa: BLE001
            pass
    done = sorted(p.stem for p in (root / "state").glob("*.done"))
    failed = sorted(p.stem for p in (root / "state").glob("*.failed"))
    print("stages done    :", " ".join(done) or "-")
    if failed:
        print("stages failed  :", " ".join(failed))
    for name in ("pipeline.log", "supervisor.log"):
        p = root / "logs" / name
        if p.exists():
            print(f"--- last lines of {name}")
            print("\n".join(p.read_text(errors="replace").splitlines()[-12:]))


def main():
    root = pick_root()
    for sub in ("logs", "state", "tmp"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "").strip()
    if token.startswith("hf_"):
        p = root / ".hf_token"
        p.write_text(token)
        p.chmod(0o600)
    if os.environ.get("EMC_SCRUB"):
        (root / ".scrub").write_text(os.environ["EMC_SCRUB"])
    if not alive(root / "supervisor.pid"):
        uv = get_uv(root)
        req = urllib.request.Request(f"https://github.com/{SOURCE}.git/info/refs?service=git-upload-pack",
                                     headers={"User-Agent": "git/2.45.0"})
        sha = re.search(r"([0-9a-f]{40}) refs/heads/main", urllib.request.urlopen(req, timeout=30)
                        .read().decode("latin-1")).group(1)
        sup = root / "supervise.py"
        sup.write_bytes(urllib.request.urlopen(
            f"https://raw.githubusercontent.com/{SOURCE}/{sha}/scripts/supervise.py", timeout=60).read())
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        env.update(EMC_ROOT=str(root), UV=uv, RUN_ID=os.environ.get("RUN_ID", "r1"))
        with open(root / "logs" / "supervisor.log", "a") as out:
            proc = subprocess.Popen([sys.executable, str(sup)], stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, cwd=str(root), start_new_session=True)
        (root / "supervisor.pid").write_text(str(proc.pid))
        print(f"started the supervisor (pid {proc.pid}); re-run this cell any time to see progress\n")
    status(root)


if __name__ == "__main__":
    main()
