#!/usr/bin/env python3
"""Progress bars for fine-tune runs launched by scripts/run_ft1.sh.

Reads the per-step records the Trainer appends to
runs/<model>_<tag>/train_log.jsonl, and the planned step counts / bench
status from logs/<tag>_gpu*.log.

    python scripts/watch_progress.py              # print once
    python scripts/watch_progress.py --watch 10   # refresh every 10s, Ctrl+C to quit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

LOGS = Path("/home/calypso/aic/logs")
RUNS = Path("/home/calypso/aic/runs")
BAR = 30


def planned_runs(tag: str) -> dict:
    runs: dict = {}
    for log in sorted(LOGS.glob(f"{tag}_gpu*.log")):
        gpu = log.stem.split("gpu")[-1]
        name = None
        for line in log.read_text(errors="ignore").splitlines():
            if m := re.search(r"\] (\S+) train start", line):
                name = m.group(1)
                runs[name] = {"gpu": gpu, "steps": None, "batch": None, "status": "training"}
            elif (m := re.search(r"batch=(\d+) accum=\d+ steps=(\d+)", line)) and name:
                runs[name].update(batch=int(m.group(1)), steps=int(m.group(2)))
            elif m := re.search(r"\] (\S+) train exit=(\d+)", line):
                runs.setdefault(m.group(1), {"gpu": gpu, "steps": None, "batch": None})
                runs[m.group(1)]["status"] = "benchmarking" if m.group(2) == "0" else f"TRAIN FAILED (exit {m.group(2)})"
            elif m := re.search(r"\] (\S+) bench exit=(\d+)", line):
                runs[m.group(1)]["status"] = "done" if m.group(2) == "0" else f"BENCH FAILED (exit {m.group(2)})"
    return runs


def last_records(path: Path, n: int = 25) -> list:
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()[-n:]
    return [json.loads(l) for l in lines if l]


def fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def render(tag: str) -> str:
    runs = planned_runs(tag)
    if not runs:
        return f"No runs found for tag '{tag}' in {LOGS}"
    out = [f"Fine-tune '{tag}'  —  {time.strftime('%H:%M:%S')}", ""]
    for name, info in runs.items():
        recs = last_records(RUNS / f"{name}_{tag}" / "train_log.jsonl")
        total = info.get("steps") or 0
        step = recs[-1]["step"] if recs else 0
        frac = min(step / total, 1.0) if total else 0.0
        filled = int(frac * BAR)
        bar = "█" * filled + "░" * (BAR - filled)
        line = f"GPU{info['gpu']} {name:<18} [{bar}] {frac * 100:5.1f}%  {step:>4}/{total:<4}"
        if recs and step:
            loss = sum(r["loss"] for r in recs) / len(recs)
            elapsed = recs[-1]["elapsed_sec"]
            rate = step * (info.get("batch") or 0) / elapsed if elapsed else 0
            eta = (total - step) * elapsed / step if step < total else 0
            line += f"  loss {loss:.3f}  {rate:5.1f} img/s  ETA {fmt_eta(eta)}"
        line += f"  [{info.get('status', 'training')}]"
        out.append(line)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="ft1")
    ap.add_argument("--watch", type=float, default=0, metavar="SEC", help="Refresh interval; 0 prints once.")
    args = ap.parse_args()
    if not args.watch:
        print(render(args.tag))
        return
    try:
        while True:
            os.system("clear")
            print(render(args.tag))
            print("\n(Ctrl+C to quit)")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
