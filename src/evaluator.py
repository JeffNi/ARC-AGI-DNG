"""
Evaluator — snapshot evaluation of brain capability.

Runs tasks with full neural dynamics but no learning or reward.
The brain perceives through gaze, acts via sequential motor commits,
and its canvas is scored against the expected output.

No weight changes, no reward delivery during evaluation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from .brain import Brain
from .display_buffer import DisplayBuffer
from .genome import Genome
from .encoding import pad_grid, NUM_COLORS


@dataclass
class EvalResult:
    task_type: str
    task_id: str
    correct: bool
    cell_accuracy: float
    n_demos_shown: int
    total_commits: int = 0


@dataclass
class TypeReport:
    task_type: str
    total: int = 0
    solved: int = 0
    mean_cell_acc: float = 0.0

    @property
    def solve_rate(self) -> float:
        return self.solved / max(1, self.total)


@dataclass
class EvalReport:
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
    Snapshot evaluator. Full neural dynamics, no learning or reward.
    Brain perceives through gaze, commits actions to canvas, scored at end.
    """

    def __init__(
        self,
        brain: Brain,
        task_dir: str = "micro_tasks",
        observe_steps: int = 40,
        action_steps: int = 150,
        n_demos: int | None = None,
    ):
        self.brain = brain
        self.task_dir = Path(task_dir)
        self.observe_steps = observe_steps
        self.action_steps = action_steps
        self._n_demos_override = n_demos

        self.display_buffer = DisplayBuffer(
            brain.net.max_h, brain.net.max_w,
        )

        self._tasks_by_type: dict[str, list[dict]] = {}
        self._load_tasks()

        self._save_transient_state()

    def _load_tasks(self):
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
        net = self.brain.net
        net.r[:] = self._snap_r
        net.V[:] = self._snap_V
        net.prev_r[:] = self._snap_prev_r
        net.f[:] = self._snap_f
        net.adaptation[:] = self._snap_adaptation
        ne = net._edge_count
        n_snap = len(self._snap_elig)
        n_copy = min(ne, n_snap)
        net._edge_eligibility[:n_copy] = self._snap_elig[:n_copy]
        self.brain.neuromod.da = self._snap_da
        self.brain.clear_signal()
        self.brain.reset_canvas()

    def _eval_n_demos(self) -> int:
        if self._n_demos_override is not None:
            return self._n_demos_override
        sp = self.brain.stage_manager.current_setpoints()
        return sp.n_demos

    def eval_single(self, task_type: str, task: dict) -> EvalResult:
        """
        Evaluate one task. Full dynamics, no learning, no reward.

        Flow:
          1. Reset transient state + canvas
          2. Load task into display buffer
          3. Guide gaze through demo pairs (observe via gaze)
          4. Guide gaze to test input, let brain act autonomously
          5. Read canvas when DONE or max steps, compare to expected
        """
        self._restore_transient_state()

        h = self.brain.net.max_h
        w = self.brain.net.max_w
        pairs = task.get("train", [])
        test_pairs = task.get("test", [])
        if not pairs and not test_pairs:
            return EvalResult(task_type, task.get("_file", "?"), False, 0.0, 0)

        n_demos = min(self._eval_n_demos(), len(pairs))

        # Load task
        n_loaded = self.display_buffer.load_task(task)

        # --- Demos via gaze ---
        steps_per_slot = max(5, self.observe_steps // max(1, n_loaded * 2 + 1))
        for slot in range(n_loaded * 2):
            self.brain.guide_gaze(slot, strength=0.8)
            self.brain.step_with_gaze(
                self.display_buffer,
                n_steps=steps_per_slot,
                learn=False,
            )

        # Guide to test input
        test_slot = n_loaded * 2
        self.brain.guide_gaze(test_slot, strength=0.8)
        self.brain.step_with_gaze(
            self.display_buffer,
            n_steps=steps_per_slot,
            learn=False,
        )

        # --- Action phase: deterministic position cycling, no reward ---
        # Use the same motor loop as training so the brain's color
        # selection is exercised at every position. No reward or
        # depression — this purely reads out what the brain would pick.
        self.brain.reset_canvas()

        test_input = pad_grid(
            np.array((test_pairs[0] if test_pairs else pairs[-1])["input"],
                     dtype=np.int32),
            h, w,
        )
        self.brain.set_observed_colors(test_input)

        self.brain.clear_gaze_bias()
        self.brain.guide_gaze(test_slot, strength=0.4)
        self.brain.apply_gaze(self.display_buffer)

        n_cells = h * w
        total_commits = n_cells
        for pos in range(n_cells):
            self.brain.commit_at_position(pos, expected_grid=None)

        # --- Score ---
        if test_pairs:
            expected = pad_grid(np.array(test_pairs[0]["output"], dtype=np.int32), h, w)
        else:
            expected = pad_grid(np.array(pairs[-1]["output"], dtype=np.int32), h, w)

        canvas = self.brain.get_canvas_grid(h, w)
        self.brain.clear_signal()
        self.brain.clear_gaze_bias()

        correct = np.array_equal(canvas, expected)
        n_cells = h * w
        cell_acc = float(np.sum(canvas == expected)) / n_cells

        return EvalResult(
            task_type=task_type,
            task_id=task.get("_file", "unknown"),
            correct=correct,
            cell_accuracy=cell_acc,
            n_demos_shown=n_demos,
            total_commits=total_commits,
        )

    def run_full_eval(self, console: bool = True) -> EvalReport:
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
                          f"cell_acc={result.cell_accuracy:.1%} "
                          f"commits={result.total_commits}"
                          f"  ({done}/{total})", flush=True)

            type_report.mean_cell_acc = float(np.mean(cell_accs)) if cell_accs else 0.0
            report.by_type[task_type] = type_report

        if console:
            print()
            print(report.summary_str())

        return report

    def save_report(self, report: EvalReport, path: str | Path):
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
                    "commits": r.total_commits,
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
    if genome is None:
        genome = Genome()
    brain = Brain.resume(genome, checkpoint_dir=checkpoint_dir)
    if console:
        print(f"Loaded brain: age={brain.age}, stage={brain.stage_manager.current_stage}, "
              f"edges={brain.net._edge_count:,}")

    evaluator = Evaluator(brain, task_dir=task_dir, n_demos=n_demos)
    return evaluator.run_full_eval(console=console)
