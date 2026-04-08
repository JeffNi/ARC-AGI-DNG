"""
Evaluator — snapshot evaluation of brain capability.

Loads a checkpoint, runs all tasks with full neural dynamics but no
permanent learning. The circuit executes (signals propagate, WTA competes,
activity settles) but weights are never modified. State is reset between
tasks so each eval starts from the same baseline.

This answers: "if I froze this brain right now, what can it actually do?"
"""

from __future__ import annotations

import json
import copy
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .brain import Brain
from .genome import Genome
from .encoding import grid_to_signal, pad_grid, NUM_COLORS
from .perception.encoder import FEATURES_PER_CELL


@dataclass
class EvalResult:
    """Result from evaluating one task."""
    task_type: str
    task_id: str
    correct: bool
    cell_accuracy: float
    n_demos_shown: int


@dataclass
class TypeReport:
    """Aggregate results for one task type."""
    task_type: str
    total: int = 0
    solved: int = 0
    mean_cell_acc: float = 0.0

    @property
    def solve_rate(self) -> float:
        return self.solved / max(1, self.total)


@dataclass
class EvalReport:
    """Full evaluation report."""
    checkpoint_age: int
    stage: str
    n_edges: int
    n_nodes: int
    by_type: dict[str, TypeReport] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total_tasks(self) -> int:
        return len(self.results)

    @property
    def total_solved(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def overall_accuracy(self) -> float:
        return self.total_solved / max(1, self.total_tasks)

    @property
    def mean_cell_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.cell_accuracy for r in self.results]))

    def summary_str(self) -> str:
        lines = [
            f"=== Eval @ age {self.checkpoint_age} ({self.stage}) ===",
            f"Network: {self.n_nodes} nodes, {self.n_edges:,} edges",
            f"Overall: {self.total_solved}/{self.total_tasks} "
            f"({self.overall_accuracy*100:.1f}%) | "
            f"cell acc: {self.mean_cell_accuracy*100:.1f}%",
            "",
        ]
        for ttype in sorted(self.by_type.keys()):
            tr = self.by_type[ttype]
            lines.append(
                f"  {ttype}: {tr.solved}/{tr.total} "
                f"({tr.solve_rate*100:.0f}%) cell_acc={tr.mean_cell_acc*100:.1f}%"
            )
        return "\n".join(lines)


class Evaluator:
    """
    Snapshot evaluator. Loads a brain checkpoint, runs tasks with full
    neural dynamics (thinking) but no learning (no reward, no CHL,
    no weight modification). State reset between tasks.
    """

    def __init__(
        self,
        brain: Brain,
        task_dir: str = "micro_tasks",
        observe_steps: int = 50,
        attempt_steps: int = 80,
        n_demos: int | None = None,
    ):
        self.brain = brain
        self.task_dir = Path(task_dir)
        self.observe_steps = observe_steps
        self.attempt_steps = attempt_steps
        # If n_demos is None, use the brain's current stage setting
        self._n_demos_override = n_demos

        self._tasks_by_type: dict[str, list[dict]] = {}
        self._load_tasks()

        # Snapshot the initial state for resets between tasks
        self._save_transient_state()

    def _load_tasks(self):
        """Load all task files, filtering oversized ones."""
        if not self.task_dir.exists():
            return
        max_h = self.brain.net.max_h
        max_w = self.brain.net.max_w

        for tier_dir in sorted(self.task_dir.iterdir()):
            if not tier_dir.is_dir():
                continue
            tasks = []
            for task_file in sorted(tier_dir.glob("*.json")):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                    task["_file"] = task_file.name
                    task["_tier"] = tier_dir.name
                    if self._task_fits(task, max_h, max_w):
                        tasks.append(task)
                except (json.JSONDecodeError, KeyError):
                    continue
            if tasks:
                self._tasks_by_type[tier_dir.name] = tasks

    @staticmethod
    def _task_fits(task: dict, max_h: int, max_w: int) -> bool:
        for pair in task.get("train", []) + task.get("test", []):
            for key in ("input", "output"):
                g = pair.get(key, [])
                if len(g) > max_h or (g and len(g[0]) > max_w):
                    return False
        return True

    def _save_transient_state(self):
        """Snapshot firing rates, voltages, eligibility, facilitation, adaptation.
        Weights are NOT saved here — they're already frozen (never modified)."""
        net = self.brain.net
        self._snap_r = net.r.copy()
        self._snap_V = net.V.copy()
        self._snap_prev_r = net.prev_r.copy()
        self._snap_f = net.f.copy()
        self._snap_adaptation = net.adaptation.copy()
        ne = net._edge_count
        self._snap_elig = net._edge_eligibility[:ne].copy()
        self._snap_da = self.brain.neuromod.da

    def _restore_transient_state(self):
        """Reset to pre-eval baseline. Weights untouched (never modified)."""
        net = self.brain.net
        net.r[:] = self._snap_r
        net.V[:] = self._snap_V
        net.prev_r[:] = self._snap_prev_r
        net.f[:] = self._snap_f
        net.adaptation[:] = self._snap_adaptation
        ne = net._edge_count
        net._edge_eligibility[:ne] = self._snap_elig
        self.brain.neuromod.da = self._snap_da
        self.brain.clear_signal()

    def _eval_n_demos(self) -> int:
        if self._n_demos_override is not None:
            return self._n_demos_override
        sp = self.brain.stage_manager.current_setpoints()
        return sp.n_demos

    def eval_single(self, task_type: str, task: dict) -> EvalResult:
        """
        Evaluate one task. Full dynamics, no learning.

        Flow:
          1. Reset transient state to clean baseline
          2. Show demos (full dynamics — brain "thinks" about examples)
          3. Show test input, let brain settle, read motor output
          4. Compare — no reward, no CHL, no weight changes
        """
        self._restore_transient_state()

        h = self.brain.net.max_h
        w = self.brain.net.max_w
        pairs = task.get("train", [])
        if not pairs:
            return EvalResult(task_type, task.get("_file", "?"), False, 0.0, 0)

        n_demos = min(self._eval_n_demos(), len(pairs))

        # --- Demos: brain sees examples with full dynamics ---
        demo_pairs = pairs[:n_demos]
        for dp in demo_pairs:
            dp_in = pad_grid(np.array(dp["input"], dtype=np.int32), h, w)
            dp_out = pad_grid(np.array(dp["output"], dtype=np.int32), h, w)

            sensory = grid_to_signal(dp_in, max_h=h, max_w=w)
            motor = self._motor_signal(dp_out, h, w)

            self.brain.inject_teaching_signal(sensory, motor)
            self.brain.step(n_steps=self.observe_steps)
            self.brain.clear_signal()
            self.brain.step(n_steps=5)

        # --- Attempt: input only, brain produces output ---
        remaining = [p for p in pairs if p not in demo_pairs]
        test_pair = remaining[0] if remaining else pairs[-1]
        input_grid = pad_grid(np.array(test_pair["input"], dtype=np.int32), h, w)
        expected_grid = pad_grid(np.array(test_pair["output"], dtype=np.int32), h, w)
        out_h, out_w = expected_grid.shape

        signal = grid_to_signal(input_grid, max_h=h, max_w=w)
        self.brain.inject_signal(signal)
        self.brain.step(n_steps=self.attempt_steps)

        output_grid = self.brain.read_motor(out_h, out_w)
        self.brain.clear_signal()

        # --- Score (no reward, no CHL) ---
        correct = np.array_equal(output_grid, expected_grid)
        n_cells = out_h * out_w
        cell_acc = float(np.sum(output_grid == expected_grid)) / n_cells

        return EvalResult(
            task_type=task_type,
            task_id=task.get("_file", "unknown"),
            correct=correct,
            cell_accuracy=cell_acc,
            n_demos_shown=n_demos,
        )

    def _motor_signal(self, grid: np.ndarray, h: int, w: int) -> np.ndarray:
        """One-hot motor signal from grid."""
        n_cells = h * w
        motor = np.zeros(n_cells * NUM_COLORS, dtype=np.float64)
        for i in range(n_cells):
            r, c = divmod(i, w)
            color = int(grid[r, c])
            motor[i * NUM_COLORS + color] = 1.0
        return motor

    def run_full_eval(self, console: bool = True) -> EvalReport:
        """
        Evaluate ALL tasks across ALL types.
        Returns a structured report.
        """
        report = EvalReport(
            checkpoint_age=self.brain.age,
            stage=self.brain.stage_manager.current_stage,
            n_edges=self.brain.net._edge_count,
            n_nodes=self.brain.net.n_nodes,
        )

        total = sum(len(ts) for ts in self._tasks_by_type.values())
        done = 0

        for task_type in sorted(self._tasks_by_type.keys()):
            tasks = self._tasks_by_type[task_type]
            type_report = TypeReport(task_type=task_type)
            cell_accs = []

            for task in tasks:
                result = self.eval_single(task_type, task)
                report.results.append(result)
                type_report.total += 1
                if result.correct:
                    type_report.solved += 1
                cell_accs.append(result.cell_accuracy)

                done += 1
                if console:
                    tag = "PASS" if result.correct else "fail"
                    print(f"  [{tag}] {task_type}/{result.task_id} "
                          f"cell_acc={result.cell_accuracy:.1%}"
                          f"  ({done}/{total})", flush=True)

            type_report.mean_cell_acc = float(np.mean(cell_accs)) if cell_accs else 0.0
            report.by_type[task_type] = type_report

        if console:
            print()
            print(report.summary_str())

        return report

    def save_report(self, report: EvalReport, path: str | Path):
        """Save eval report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "checkpoint_age": report.checkpoint_age,
            "stage": report.stage,
            "n_edges": report.n_edges,
            "n_nodes": report.n_nodes,
            "overall_accuracy": report.overall_accuracy,
            "mean_cell_accuracy": report.mean_cell_accuracy,
            "total_tasks": report.total_tasks,
            "total_solved": report.total_solved,
            "by_type": {
                k: {
                    "total": v.total,
                    "solved": v.solved,
                    "solve_rate": v.solve_rate,
                    "mean_cell_acc": v.mean_cell_acc,
                }
                for k, v in report.by_type.items()
            },
            "results": [
                {
                    "task_type": r.task_type,
                    "task_id": r.task_id,
                    "correct": r.correct,
                    "cell_accuracy": r.cell_accuracy,
                    "n_demos": r.n_demos_shown,
                }
                for r in report.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def evaluate_checkpoint(
    checkpoint_dir: str,
    genome: Genome | None = None,
    task_dir: str = "micro_tasks",
    n_demos: int | None = None,
    console: bool = True,
) -> EvalReport:
    """
    Convenience: load a checkpoint and run full evaluation.
    Returns the report. Doesn't modify the checkpoint.
    """
    if genome is None:
        genome = Genome()
    brain = Brain.resume(genome, checkpoint_dir=checkpoint_dir)
    if console:
        print(f"Loaded brain: age={brain.age}, stage={brain.stage_manager.current_stage}, "
              f"edges={brain.net._edge_count:,}")

    evaluator = Evaluator(brain, task_dir=task_dir, n_demos=n_demos)
    return evaluator.run_full_eval(console=console)
