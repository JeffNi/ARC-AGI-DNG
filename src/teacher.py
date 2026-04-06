"""
Teacher — external caregiver that feeds tasks and manages reward.

The Teacher:
  - Loads tasks from the micro_tasks directory
  - Manages a mastery-based curriculum (must solve ALL tasks of a type)
  - Feeds input signals, observes output, computes reward
  - Sets DA based on reward prediction error
  - Tracks per-type performance
  - Periodically retests previously mastered types
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .brain import Brain
from .encoding import grid_to_signal, pad_grid, NUM_COLORS
from .monitor import Monitor


@dataclass
class TypeTracker:
    """Track performance for one task type."""
    attempts: int = 0
    solves: int = 0
    consecutive_solves: int = 0
    mastered: bool = False
    last_tested_age: int = 0

    @property
    def solve_rate(self) -> float:
        return self.solves / max(1, self.attempts)


@dataclass
class CurriculumState:
    """State of the mastery-based curriculum."""
    current_tier: int = 0
    type_trackers: dict[str, TypeTracker] = field(default_factory=dict)
    day: int = 0
    tasks_today: int = 0
    solves_today: int = 0


class Teacher:
    """External caregiver for the brain."""

    def __init__(
        self,
        brain: Brain,
        monitor: Monitor,
        task_dir: str = "micro_tasks",
        mastery_threshold: int = 3,
        retest_interval: int = 50,
        reward_correct: float = 0.3,
        reward_wrong: float = -0.05,
        observe_steps: int = 50,
        attempt_steps: int = 80,
    ):
        self.brain = brain
        self.monitor = monitor
        self.task_dir = Path(task_dir)
        self.mastery_threshold = mastery_threshold
        self.retest_interval = retest_interval
        self.reward_correct = reward_correct
        self.reward_wrong = reward_wrong
        self.observe_steps = observe_steps
        self.attempt_steps = attempt_steps

        self.state = CurriculumState()
        self._tasks_by_type: dict[str, list[dict]] = {}
        self._tier_order: list[str] = []

        self._load_tasks()

    def _task_fits(self, task: dict) -> bool:
        """Check if all grids in a task fit within the brain's max dimensions."""
        max_h = self.brain.net.max_h
        max_w = self.brain.net.max_w
        for pair in task.get("train", []) + task.get("test", []):
            for key in ("input", "output"):
                g = pair.get(key, [])
                if len(g) > max_h or (g and len(g[0]) > max_w):
                    return False
        return True

    def _load_tasks(self):
        """Load all tasks from the micro_tasks directory structure."""
        if not self.task_dir.exists():
            self.monitor.status(f"Task directory not found: {self.task_dir}")
            return

        skipped = 0
        tier_dirs = sorted(self.task_dir.iterdir())
        for tier_dir in tier_dirs:
            if not tier_dir.is_dir():
                continue
            tier_name = tier_dir.name
            tasks = []
            for task_file in sorted(tier_dir.glob("*.json")):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                    task["_file"] = task_file.name
                    task["_tier"] = tier_name
                    if not self._task_fits(task):
                        skipped += 1
                        continue
                    tasks.append(task)
                except (json.JSONDecodeError, KeyError):
                    continue
            if tasks:
                self._tasks_by_type[tier_name] = tasks
                self._tier_order.append(tier_name)

        total = sum(len(t) for t in self._tasks_by_type.values())
        self.monitor.status(f"Loaded {total} tasks across {len(self._tier_order)} types"
                           f" (filtered {skipped} oversized)")

    def current_type(self) -> str | None:
        """Get the current task type to train on."""
        if not self._tier_order:
            return None
        idx = min(self.state.current_tier, len(self._tier_order) - 1)
        return self._tier_order[idx]

    def pick_task(self) -> tuple[str, dict] | None:
        """Pick the next task: current type, retest, or advance."""
        if not self._tier_order:
            return None

        # Occasionally retest a mastered type
        mastered = [t for t, tr in self.state.type_trackers.items()
                    if tr.mastered and (self.brain.age - tr.last_tested_age) > self.retest_interval]
        if mastered and random.random() < 0.15:
            retest_type = random.choice(mastered)
            task = random.choice(self._tasks_by_type[retest_type])
            return retest_type, task

        # Current type
        task_type = self.current_type()
        if task_type is None:
            return None

        tasks = self._tasks_by_type.get(task_type, [])
        if not tasks:
            return None

        task = random.choice(tasks)
        return task_type, task

    def run_task(self, task_type: str, task: dict) -> bool:
        """
        Present a single task to the brain.

        The Teacher interacts through senses only:
          - inject_signal: show something to the brain's eyes
          - clear_signal: stop showing
          - apply_reward: external reward/punishment (brain converts to DA internally)

        The Teacher NEVER directly sets DA or other brain internals.
        """
        h = self.brain.net.max_h
        w = self.brain.net.max_w

        pairs = task.get("train", [])
        if not pairs:
            return False

        pair = random.choice(pairs)
        input_grid = np.array(pair["input"], dtype=np.int32)
        expected_grid = np.array(pair["output"], dtype=np.int32)

        # Ensure grids fit the brain's max dimensions
        input_grid = pad_grid(input_grid, h, w)
        expected_grid = pad_grid(expected_grid, h, w)
        out_h, out_w = expected_grid.shape

        # Phase 1: Show input — brain looks at the stimulus
        signal = grid_to_signal(input_grid, max_h=h, max_w=w)
        self.brain.store_signal(signal)
        self.brain.inject_signal(signal)
        self.brain.step(n_steps=self.observe_steps)

        # Phase 2: Attempt — brain produces output (input stays visible)
        # Capture "free" correlations: what the brain does on its own
        free_corr = self.brain.snapshot_correlations(n_steps=self.attempt_steps)
        output_grid = self.brain.read_motor(out_h, out_w)

        # Phase 3: Evaluate and reward
        correct = np.array_equal(output_grid, expected_grid)
        n_cells = out_h * out_w
        cell_acc = float(np.sum(output_grid == expected_grid)) / n_cells

        if correct:
            reward = self.reward_correct
        else:
            reward = self.reward_wrong * (1.0 - cell_acc)

        mean_change = self.brain.apply_reward(reward)

        # Phase 4: If wrong, show the correct answer and capture "clamped"
        # correlations — what the brain does when seeing the right answer.
        # CHL uses the difference to compute an error-corrective weight update.
        if not correct:
            teach_signal = grid_to_signal(expected_grid, max_h=h, max_w=w)
            self.brain.inject_signal(teach_signal)
            clamped_corr = self.brain.snapshot_correlations(
                n_steps=self.observe_steps // 2,
            )
            chl_change = self.brain.apply_chl(free_corr, clamped_corr)
        else:
            # Correct: clamped == free, store for replay but no CHL needed
            self.brain.store_replay(free_corr, free_corr)
            chl_change = 0.0

        # Phase 5: Rest — clear stimulus, let activity settle, spontaneous replay
        self.brain.clear_signal()
        self.brain.step(n_steps=10)
        self.brain.spontaneous_replay(n_steps=10, strength=0.2)

        # Check for sleep
        sleep_stats = self.brain.try_sleep()
        if sleep_stats:
            self.monitor.sleep_event(sleep_stats, self.brain.age)

        # Update tracking
        tracker = self.state.type_trackers.setdefault(task_type, TypeTracker())
        tracker.attempts += 1
        tracker.last_tested_age = self.brain.age
        if correct:
            tracker.solves += 1
            tracker.consecutive_solves += 1
        else:
            tracker.consecutive_solves = 0

        # Check mastery
        if not tracker.mastered and tracker.consecutive_solves >= self.mastery_threshold:
            all_tasks = self._tasks_by_type.get(task_type, [])
            if tracker.solves >= len(all_tasks):
                tracker.mastered = True
                self.monitor.status(f"MASTERED: {task_type}")
                self._advance_tier()

        self.state.tasks_today += 1
        if correct:
            self.state.solves_today += 1

        self.monitor.task_result(
            task_type=task_type,
            task_id=task.get("_file", "unknown"),
            correct=correct,
            reward=reward,
            mean_change=mean_change,
            age=self.brain.age,
            extra={"cell_accuracy": cell_acc},
        )

        return correct

    def _advance_tier(self):
        """Advance to next tier if current is mastered."""
        current = self.current_type()
        tracker = self.state.type_trackers.get(current)
        if tracker and tracker.mastered:
            if self.state.current_tier < len(self._tier_order) - 1:
                self.state.current_tier += 1
                new_type = self.current_type()
                self.monitor.status(f"Advanced to tier {self.state.current_tier}: {new_type}")

    def run_day(self, max_tasks: int = 50) -> dict:
        """Run a full day of tasks."""
        self.state.tasks_today = 0
        self.state.solves_today = 0
        self.state.day += 1

        for _ in range(max_tasks):
            pick = self.pick_task()
            if pick is None:
                break
            task_type, task = pick
            self.run_task(task_type, task)

        n_mastered = sum(1 for t in self.state.type_trackers.values() if t.mastered)
        self.monitor.day_summary(
            day=self.state.day,
            tasks_attempted=self.state.tasks_today,
            tasks_solved=self.state.solves_today,
            types_mastered=n_mastered,
            n_edges=self.brain.net._edge_count,
            mean_fatigue=self.brain.fatigue.level,
            age=self.brain.age,
        )

        # Autosave
        if self.brain.checkpointer.should_autosave():
            self.brain.save()

        return {
            "day": self.state.day,
            "attempted": self.state.tasks_today,
            "solved": self.state.solves_today,
            "mastered": n_mastered,
        }
