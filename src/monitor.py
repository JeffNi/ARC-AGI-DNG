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
                  f"dw={mean_change:.5f}  age={age}", flush=True)

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
            print(f"{'='*50}\n", flush=True)

    def brain_snapshot(self, brain, label: str = "periodic"):
        """
        Log a snapshot of brain health metrics: synapse count, weight
        distribution, consolidation, DA state, firing rates, etc.
        Called periodically during training for monitoring.
        """
        from .graph import Region

        net = brain.net
        ne = net._edge_count
        w = net._edge_w[:ne]
        cons = net._edge_consolidation[:ne]
        elig = net._edge_eligibility[:ne]

        # Weight stats
        w_abs = np.abs(w)
        n_instinct = int(np.sum(cons >= 10.0))

        # Firing rate stats
        from .graph import internal_mask as _int_mask
        r = net.r
        region_list = list(Region)
        internal_mask = _int_mask(net.regions)
        motor_mask = net.regions == region_list.index(Region.MOTOR)

        data = {
            "label": label,
            "age": brain.age,
            "stage": brain.stage_manager.current_stage,
            # Network structure
            "n_edges": ne,
            "n_nodes": net.n_nodes,
            "n_instinct_edges": n_instinct,
            # Weights
            "w_mean": float(w_abs.mean()),
            "w_median": float(np.median(w_abs)),
            "w_max": float(w_abs.max()),
            "w_std": float(w_abs.std()),
            # Consolidation
            "consolidation_mean": float(cons.mean()) if ne > 0 else 0.0,
            "n_consolidated": int(np.sum(cons > 1.0)),
            # Eligibility
            "elig_mean": float(elig.mean()) if ne > 0 else 0.0,
            "elig_nonzero": int(np.sum(elig > 0.01)),
            # DA
            "da": brain.neuromod.da,
            "da_baseline": brain.neuromod.da_baseline,
            # Firing rates
            "r_mean_all": float(r.mean()),
            "r_mean_internal": float(r[internal_mask].mean()) if internal_mask.any() else 0.0,
            "r_mean_motor": float(r[motor_mask].mean()) if motor_mask.any() else 0.0,
            # Fatigue
            "fatigue": brain.fatigue.level,
        }
        self.log("brain_snapshot", data)

        if self.console:
            print(f"  [SNAP] edges={ne:,} instinct={n_instinct} "
                  f"w_mean={data['w_mean']:.4f} "
                  f"consolidated={data['n_consolidated']} "
                  f"r_int={data['r_mean_internal']:.3f} "
                  f"DA={data['da']:.3f}", flush=True)

    def infancy_snapshot(
        self, brain, day: int, birth_edges: int,
        day_sleeps: int = 0, day_pruned: int = 0, day_grown: int = 0,
        day_secs: float = 0.0, day_steps: int = 0, repr_sim: float = 1.0,
    ):
        """
        Extended snapshot for infancy phase with diagnostically useful metrics.

        Key questions answered:
          - Is the network growing? (edge count, growth ratio)
          - Are weights differentiating? (w_p90, weight spread ratio)
          - Are neurons specializing? (selectivity = std of ema_rates)
          - Are neurons dying? (dead count = ema_rate < 0.01)
          - Is edge health stable? (mean health, low-health count)
        """
        from .graph import Region

        net = brain.net
        ne = net._edge_count
        w_abs = np.abs(net._edge_w[:ne])
        cons = net._edge_consolidation[:ne]
        health = net._edge_health[:ne]
        n_instinct = int(np.sum(cons >= 10.0))

        from .graph import internal_mask as _int_mask
        region_list = list(Region)
        internal_mask = _int_mask(net.regions)
        n_internal = int(internal_mask.sum())

        ema = brain.homeostasis.ema_rate
        internal_ema = ema[internal_mask]
        n_dead = int((internal_ema < 0.01).sum())
        n_active = int((internal_ema > 0.05).sum())
        selectivity = float(internal_ema.std()) if n_internal > 0 else 0.0

        growth_ratio = ne / max(1, birth_edges)
        w_spread = float(np.percentile(w_abs, 90) / max(np.percentile(w_abs, 50), 1e-8)) if ne > 0 else 1.0

        mean_health = float(health.mean()) if ne > 0 else 1.0
        n_low_health = int((health < 0.3).sum()) if ne > 0 else 0

        data = {
            "day": day,
            "age": brain.age,
            "stage": brain.stage_manager.current_stage,
            "n_edges": ne,
            "birth_edges": birth_edges,
            "growth_ratio": growth_ratio,
            "n_instinct": n_instinct,
            "w_p50": float(np.percentile(w_abs, 50)) if ne > 0 else 0.0,
            "w_p90": float(np.percentile(w_abs, 90)) if ne > 0 else 0.0,
            "w_mean": float(w_abs.mean()) if ne > 0 else 0.0,
            "w_spread": w_spread,
            "n_internal": n_internal,
            "n_dead": n_dead,
            "n_active": n_active,
            "selectivity": selectivity,
            "ema_mean": float(internal_ema.mean()),
            "ema_max": float(internal_ema.max()) if n_internal > 0 else 0.0,
            "mean_health": mean_health,
            "n_low_health": n_low_health,
            "da": brain.neuromod.da,
            "fatigue": brain.fatigue.level,
            "day_sleeps": day_sleeps,
            "day_pruned": day_pruned,
            "day_grown": day_grown,
            "day_secs": day_secs,
            "day_steps": day_steps,
            "repr_sim": repr_sim,
        }
        self.log("infancy_snapshot", data)

        if self.console:
            print(
                f"  [DAY {day:2d} | {day_secs:.0f}s] "
                f"edges={ne:,}(x{growth_ratio:.2f}) "
                f"w90={data['w_p90']:.3f} "
                f"alive={n_active}/{n_internal} "
                f"sel={selectivity:.3f} "
                f"sim={repr_sim:.3f} "
                f"hlth={mean_health:.2f} "
                f"slp={day_sleeps}(+{day_grown:,}/-{day_pruned:,})",
                flush=True,
            )

        return data

    def eval_report(self, report, log_dir: str | None = None):
        """Log an evaluation report summary."""
        data = {
            "checkpoint_age": report.checkpoint_age,
            "stage": report.stage,
            "total_tasks": report.total_tasks,
            "total_solved": report.total_solved,
            "overall_accuracy": report.overall_accuracy,
            "mean_cell_accuracy": report.mean_cell_accuracy,
            "by_type": {
                k: {"solved": v.solved, "total": v.total, "rate": v.solve_rate}
                for k, v in report.by_type.items()
            },
        }
        self.log("eval_report", data)

        if self.console:
            print(f"\n  [EVAL] age={report.checkpoint_age} "
                  f"{report.total_solved}/{report.total_tasks} "
                  f"({report.overall_accuracy*100:.1f}%) "
                  f"cell_acc={report.mean_cell_accuracy*100:.1f}%")
            for ttype in sorted(report.by_type.keys()):
                tr = report.by_type[ttype]
                print(f"    {ttype}: {tr.solved}/{tr.total} ({tr.solve_rate*100:.0f}%)")
            print(flush=True)

        if log_dir:
            report_path = Path(log_dir) / f"eval_age_{report.checkpoint_age}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(data, f, indent=2, default=_json_default)

    def sleep_event(self, stats: dict, age: int):
        """Log a sleep event (JSON only, no console — use day_summary for console)."""
        data = {"age": age}
        data.update({k: v for k, v in stats.items() if k != "ema_r"})
        self.log("sleep", data)

    def status(self, msg: str):
        """Log a freeform status message."""
        self.log("status", {"msg": msg})
        if self.console:
            print(f"  >> {msg}", flush=True)


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
