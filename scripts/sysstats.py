#!/usr/bin/env python3
"""One-line snapshot of the machine: memory, disk, GPU, load and the largest processes.

Standard library only. Shown by the status view of the launch cell and appended to
logs/stats.log by every log upload, so memory use can be followed over the whole run.

  python sysstats.py [root]
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

GB = 1e9


def meminfo():
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return info


def cgroup_memory():
    """(used, limit or None) of the container's memory cgroup, when it is visible."""
    for cur, lim in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                     ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
                      "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            used = int(Path(cur).read_text())
            raw = Path(lim).read_text().strip()
            return used, (None if raw == "max" or int(raw) >= 1 << 60 else int(raw))
        except (OSError, ValueError):
            continue
    return None


def top_processes(n=3):
    procs = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return procs
    for d in entries:
        if not d.name.isdigit():
            continue
        try:
            name, rss = None, 0
            for line in (d / "status").read_text().splitlines():
                if line.startswith("Name:"):
                    name = line.split(None, 1)[1].strip()
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
            if name and rss:
                procs.append((rss, name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(procs, reverse=True)[:n]


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
        used, total, util, power = (x.strip() for x in out.splitlines()[0].split(","))
        return f"gpu {float(used) / 1024:.1f}/{float(total) / 1024:.1f} GiB, {util}%, {float(power):.0f} W"
    except Exception:  # noqa: BLE001 - no GPU tool, no line
        return "gpu n/a"


def snapshot(root):
    parts = []
    m = meminfo()
    if m.get("MemTotal"):
        used = m["MemTotal"] - m.get("MemAvailable", m.get("MemFree", 0))
        parts.append(f"ram {used / GB:.1f}/{m['MemTotal'] / GB:.0f} GB used "
                     f"(cache {m.get('Cached', 0) / GB:.1f}, shmem {m.get('Shmem', 0) / GB:.1f})")
    cg = cgroup_memory()
    if cg:
        parts.append(f"cgroup {cg[0] / GB:.1f}" + (f"/{cg[1] / GB:.0f} GB" if cg[1] else " GB (no limit)"))
    try:
        parts.append(f"disk {shutil.disk_usage(root).used / GB:.0f} GB used")
    except OSError:
        pass
    parts.append(gpu())
    try:
        parts.append(f"load {os.getloadavg()[0]:.1f}")
    except (OSError, AttributeError):  # no load average on this system
        pass
    top = top_processes()
    if top:
        parts.append("largest " + ", ".join(f"{name} {rss / GB:.1f} GB" for rss, name in top))
    return f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " + " | ".join(parts)


if __name__ == "__main__":
    print(snapshot(Path(sys.argv[1]) if len(sys.argv) > 1
                   else Path(os.environ.get("EMC_ROOT", str(Path.home() / "emc")))))
