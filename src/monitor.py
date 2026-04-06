"""
Monitor — structured logging and console status.

Writes JSON-lines to life.jsonl for machine-readable metrics.
Prints concise console status for human monitoring.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


class Monitor:
    """Structured metric recording and display."""

    def __init__(self, log_dir: str = "life", console: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "life.jsonl"
        self.console = console
        self._start_time = time.time()

    def log(self, event: str, data: dict[str, Any] | None = None):
        """Write a structured event to the JSON-lines log."""
        record = {
            "ts": time.time(),
            "elapsed": time.time() - self._start_time,
            "event": event,
        }
        if data:
            record.update(data)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def task_result(
        self,
        task_type: str,
        task_id: str,
        correct: bool,
        reward: float,
        mean_change: float,
        age: int,
        extra: dict | None = None,
    ):
        """Log a task attempt result."""
        data = {
            "task_type": task_type,
            "task_id": task_id,
            "correct": correct,
            "reward": reward,
            "mean_change": mean_change,
            "age": age,
        }
        if extra:
            data.update(extra)
        self.log("task_result", data)

        if self.console:
            status = "PASS" if correct else "fail"
            print(f"  [{status}] {task_type}/{task_id}  reward={reward:.3f}  "
                  f"dw={mean_change:.5f}  age={age}")

    def day_summary(
        self,
        day: int,
        tasks_attempted: int,
        tasks_solved: int,
        types_mastered: int,
        n_edges: int,
        mean_fatigue: float,
        age: int,
    ):
        """Log end-of-day summary."""
        data = {
            "day": day,
            "tasks_attempted": tasks_attempted,
            "tasks_solved": tasks_solved,
            "types_mastered": types_mastered,
            "n_edges": n_edges,
            "mean_fatigue": mean_fatigue,
            "age": age,
        }
        self.log("day_summary", data)

        if self.console:
            pct = (tasks_solved / max(1, tasks_attempted)) * 100
            print(f"\n{'='*50}")
            print(f"Day {day} | solved {tasks_solved}/{tasks_attempted} ({pct:.0f}%) "
                  f"| mastered {types_mastered} types | edges {n_edges:,} | age {age}")
            print(f"{'='*50}\n")

    def sleep_event(self, stats: dict, age: int):
        """Log a sleep event."""
        data = {"age": age}
        data.update({k: v for k, v in stats.items() if k != "ema_r"})
        self.log("sleep", data)
        if self.console:
            print(f"  [ZZZ] sleep: replays={stats.get('replays', 0)}, "
                  f"pruned={stats.get('pruned', 0)}, grown={stats.get('grown', 0)}")

    def status(self, msg: str):
        """Log a freeform status message."""
        self.log("status", {"msg": msg})
        if self.console:
            print(f"  >> {msg}")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")
