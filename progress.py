"""
progress.py — Shared progress state manager.

Writes/reads a progress.json file so any process (monitor.py, dashboard) 
can show live status of downloads and training.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import CACHE_DIR

PROGRESS_FILE = CACHE_DIR / "progress.json"

# ── Default state structure ───────────────────────────────────────────
def _empty_state() -> dict:
    return {
        "downloader": {
            "active":           False,
            "started_at":       None,
            "updated_at":       None,
            "total_tasks":      0,
            "completed_tasks":  0,
            "current_task":     "",
            "current_symbol":   "",
            "current_tf":       "",
            "current_month":    "",
            "total_months":     0,
            "done_months":      0,
            "symbols_done":     [],
            "symbols_pending":  [],
            "bytes_downloaded":0,
            "errors":           [],
        },
        "trainer": {
            "active":           False,
            "started_at":       None,
            "updated_at":       None,
            "total_symbols":    0,
            "completed_symbols":0,
            "current_symbol":   "",
            "current_fold":     0,
            "total_folds":      0,
            "fold_results":     [],   # [{fold, train_acc, test_acc}]
            "symbols_done":     [],   # [{symbol, best_acc, folds}]
            "errors":           [],
        },
    }


# ── Read / Write ──────────────────────────────────────────────────────

def read() -> dict:
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return _empty_state()


def write(state: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def update(section: str, **kwargs) -> None:
    """Update one section of the progress state atomically."""
    state = read()
    now   = datetime.now(timezone.utc).isoformat()
    state[section]["updated_at"] = now
    for k, v in kwargs.items():
        state[section][k] = v
    write(state)


def start_downloader(symbols: list[str], tfs: list[str]) -> None:
    state = read()
    total = len(symbols) * len(tfs)
    state["downloader"].update({
        "active":          True,
        "started_at":      datetime.now(timezone.utc).isoformat(),
        "updated_at":      datetime.now(timezone.utc).isoformat(),
        "total_tasks":     total,
        "completed_tasks": 0,
        "symbols_done":    [],
        "symbols_pending": symbols[:],
        "errors":          [],
        "bytes_downloaded":0,
    })
    write(state)


def finish_downloader() -> None:
    update("downloader", active=False, current_task="✅ Complete")


def start_trainer(symbols: list[str]) -> None:
    state = read()
    state["trainer"].update({
        "active":           True,
        "started_at":       datetime.now(timezone.utc).isoformat(),
        "updated_at":       datetime.now(timezone.utc).isoformat(),
        "total_symbols":    len(symbols),
        "completed_symbols":0,
        "current_symbol":   "",
        "current_fold":     0,
        "total_folds":      0,
        "fold_results":     [],
        "symbols_done":     [],
        "errors":           [],
    })
    write(state)


def finish_trainer() -> None:
    update("trainer", active=False, current_symbol="✅ Complete")
