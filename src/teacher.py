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
from .display_buffer import DisplayBuffer
from .encoding import grid_to_signal, pad_grid, NUM_COLORS
from .episodic_memory import EpisodicMemory
from .gaze_log import GazeLogger
from .monitor import Monitor
from .rule_verifiers import verify_rule


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
        display_buffer: DisplayBuffer | None = None,
        gaze_logger: GazeLogger | None = None,
        task_dir: str = "micro_tasks",
        mastery_threshold: int = 3,
        retest_interval: int = 50,
        observe_steps: int = 50,
        attempt_steps: int = 80,
        **kwargs,
    ):
        self.brain = brain
        self.monitor = monitor
        self.display_buffer = display_buffer or DisplayBuffer(
            brain.net.max_h, brain.net.max_w,
        )
        self.gaze_logger = gaze_logger
        self.task_dir = Path(task_dir)
        self.mastery_threshold = mastery_threshold
        self.retest_interval = retest_interval
        self.observe_steps = observe_steps
        self.attempt_steps = attempt_steps

        self.state = CurriculumState()
        self._tasks_by_type: dict[str, list[dict]] = {}
        self._tier_order: list[str] = []

        self.episodic_memory = EpisodicMemory(
            max_h=brain.net.max_h, max_w=brain.net.max_w,
        )

        self._load_tasks()
        self._generate_micro_tasks()

    def _generate_micro_tasks(self):
        """Generate simple pixel-manipulation tasks programmatically.

        These are the earliest curriculum tier — they test whether the
        learned motor pathways can produce specific outputs.  Micro-tasks
        are deliberately trivial so the network can succeed early in
        childhood and build momentum for harder tasks.
        """
        h, w = self.brain.net.max_h, self.brain.net.max_w
        rng = np.random.default_rng(42)
        tasks: list[dict] = []

        for i in range(20):
            kind = i % 3
            if kind == 0:
                # Single pixel: blank -> one pixel colored
                train_pairs, test_pair = [], None
                for _ in range(3):
                    inp = [[0] * w for _ in range(h)]
                    out = [[0] * w for _ in range(h)]
                    r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
                    color = int(rng.integers(1, NUM_COLORS))
                    out[r][c] = color
                    train_pairs.append({"input": inp, "output": out})
                inp = [[0] * w for _ in range(h)]
                out = [[0] * w for _ in range(h)]
                r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
                out[r][c] = int(rng.integers(1, NUM_COLORS))
                test_pair = {"input": inp, "output": out}
            elif kind == 1:
                # Pixel copy: sparse input -> same output (mini-identity)
                train_pairs, test_pair = [], None
                for _ in range(3):
                    grid = [[0] * w for _ in range(h)]
                    n_px = int(rng.integers(1, 4))
                    for _ in range(n_px):
                        r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
                        grid[r][c] = int(rng.integers(1, NUM_COLORS))
                    train_pairs.append({"input": grid, "output": grid})
                grid = [[0] * w for _ in range(h)]
                for _ in range(int(rng.integers(1, 4))):
                    r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
                    grid[r][c] = int(rng.integers(1, NUM_COLORS))
                test_pair = {"input": grid, "output": grid}
            else:
                # Color fill: blank -> one row filled
                train_pairs, test_pair = [], None
                for _ in range(3):
                    inp = [[0] * w for _ in range(h)]
                    out = [[0] * w for _ in range(h)]
                    row = int(rng.integers(0, h))
                    color = int(rng.integers(1, NUM_COLORS))
                    out[row] = [color] * w
                    train_pairs.append({"input": inp, "output": out})
                inp = [[0] * w for _ in range(h)]
                out = [[0] * w for _ in range(h)]
                row = int(rng.integers(0, h))
                out[row] = [int(rng.integers(1, NUM_COLORS))] * w
                test_pair = {"input": inp, "output": out}

            tasks.append({
                "train": train_pairs,
                "test": [test_pair],
                "_file": f"micro_{i:03d}.json",
                "_tier": "tier_micro",
            })

        self._tasks_by_type["tier_micro"] = tasks
        self._tier_order.append("tier_micro")

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

    def _build_spotlight_signal(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        h: int, w: int,
        strength: float = 1.5,
    ) -> np.ndarray:
        """
        Build a sensory signal with attention spotlighting: cells that
        CHANGE between input and output get boosted signal strength.
        Like a parent pointing: "look HERE at what changed."
        """
        signal = grid_to_signal(input_grid, max_h=h, max_w=w)
        if strength <= 1.0:
            return signal

        from .perception.encoder import FEATURES_PER_CELL
        n_cells = h * w
        for i in range(min(n_cells, input_grid.size, output_grid.size)):
            r, c = divmod(i, w)
            if r < input_grid.shape[0] and c < input_grid.shape[1]:
                in_val = input_grid[r, c]
            else:
                in_val = 0
            if r < output_grid.shape[0] and c < output_grid.shape[1]:
                out_val = output_grid[r, c]
            else:
                out_val = 0
            if in_val != out_val:
                base = i * FEATURES_PER_CELL
                end = base + FEATURES_PER_CELL
                signal[base:end] *= strength
        return signal

    def _motor_signal_from_grid(self, grid: np.ndarray, h: int, w: int) -> np.ndarray:
        """Encode an output grid as a motor teaching signal (one-hot per cell)."""
        padded = pad_grid(grid, h, w)
        n_cells = h * w
        motor = np.zeros(n_cells * NUM_COLORS, dtype=np.float64)
        for i in range(n_cells):
            r, c = divmod(i, w)
            color = int(padded[r, c])
            motor[i * NUM_COLORS + color] = 1.0
        return motor

    def _gaze_step(self, slot_idx: int, n_steps: int,
                   strength: float = 0.5, **step_kwargs) -> int:
        """Guide gaze to a slot, apply it, step, and log.

        Returns the slot the network actually fixated on (may differ
        from slot_idx if guidance strength is low and the network
        overrides).
        """
        self.brain.guide_gaze(slot_idx, strength)
        actual_slot = self.brain.apply_gaze(self.display_buffer)
        self.brain.step(n_steps=n_steps, **step_kwargs)
        if self.gaze_logger:
            stype = self.display_buffer.slot_types[actual_slot]
            self.gaze_logger.record(self.brain.age, actual_slot, stype)
        return actual_slot

    def run_task(self, task_type: str, task: dict) -> bool:
        """
        Present a single task via display buffer + guided gaze.

        Visual-only teaching -- no external motor injection. The copy
        pathway drives motor neurons when the network sees the answer.

        Flow:
          1. Load task into display buffer (slots: [in1, out1, ...test, answer]).
          2. Sequential demos: guide gaze to input slot, then output slot.
             Copy pathway sees output and drives motor naturally.
          3. Episodic recall.
          4. Attempt loop (max 3 tries): guide gaze to test input,
             read motor output, evaluate. If wrong: global DA + show
             correct answer visually + CHL (free vs clamped).
          5. Rest + consolidation.
        """
        h = self.brain.net.max_h
        w = self.brain.net.max_w

        pairs = task.get("train", [])
        if not pairs:
            return False

        sp = self.brain.stage_manager.current_setpoints()
        n_demos = sp.n_demos

        # --- Load task into display buffer ---
        n_loaded = self.display_buffer.load_task(task)
        gaze_strength = 0.5  # moderate guidance during childhood

        # --- Phase 1: Sequential demonstrations via gaze ---
        # Visual only: guide gaze to input, then output. The copy pathway
        # sees the output and drives motor neurons naturally -- no external
        # motor injection needed.
        self.episodic_memory.clear()
        demo_pairs = pairs[:n_demos] if n_demos > 0 else []
        for pair_idx, dp in enumerate(demo_pairs):
            dp_in = pad_grid(np.array(dp["input"], dtype=np.int32), h, w)
            dp_out = pad_grid(np.array(dp["output"], dtype=np.int32), h, w)
            self.episodic_memory.store(dp_in, dp_out)

            in_slot = pair_idx * 2
            out_slot = pair_idx * 2 + 1

            self._gaze_step(in_slot, n_steps=15, strength=gaze_strength)
            self._gaze_step(out_slot, n_steps=20, strength=gaze_strength)

            self.brain.clear_signal()
            self.brain.clear_gaze_bias()
            self.brain.step(n_steps=5)

        # --- Phase 2: Episodic recall (sensory, not motor) ---
        remaining = [p for p in pairs if p not in demo_pairs]
        test_pair = random.choice(remaining) if remaining else random.choice(pairs)
        input_grid = pad_grid(np.array(test_pair["input"], dtype=np.int32), h, w)
        expected_grid = pad_grid(np.array(test_pair["output"], dtype=np.int32), h, w)
        out_h, out_w = expected_grid.shape

        recalled = self.episodic_memory.recall_as_sensory(input_grid, h, w)
        if recalled is not None:
            self.brain.inject_signal(recalled)
            self.brain.step(n_steps=5, learn=False)
            self.brain.clear_signal()
            self.brain.step(n_steps=5)

        # --- Phase 3: Attempt loop (max 3 tries) ---
        # Visual-only teaching: network attempts freely, then sees the
        # correct answer via gaze. CHL captures the contrast. Global DA
        # signals overall accuracy; three-factor learning + CHL provide
        # cell-level specificity.
        test_slot = next(
            (i for i, t in enumerate(self.display_buffer.slot_types)
             if t == "test_input"),
            0,
        )
        answer_slot = self.display_buffer.answer_slot

        motor_nodes = self.brain.net.output_nodes

        correct = False
        n_attempts = 0
        cell_acc = 0.0
        mean_change = 0.0
        last_free_corr = None
        max_attempts = 3

        for attempt in range(max_attempts):
            n_attempts = attempt + 1

            self.brain.net.V[motor_nodes] = 0.0
            self.brain.net.r[motor_nodes] = 0.0
            self.brain.net.adaptation[motor_nodes] = 0.0

            ne = self.brain.net._edge_count
            self.brain.net._edge_eligibility[:ne] = 0.0

            # Free phase: guide gaze to test input, let network process
            self._gaze_step(test_slot, n_steps=10,
                            strength=gaze_strength, clamp_sensory=True)
            output_grid = self.brain.read_motor(out_h, out_w)

            self.display_buffer.write_answer(output_grid)
            if self.gaze_logger:
                self.gaze_logger.record(
                    self.brain.age, answer_slot, "answer", motor_event=True,
                )

            rule_type = task.get("type", "")
            correct = verify_rule(rule_type, input_grid, output_grid,
                                  expected_grid, task)
            n_cells = out_h * out_w
            cell_acc = float(np.sum(output_grid == expected_grid)) / n_cells

            # Global DA based on overall accuracy
            da = 0.1 + 0.3 * cell_acc
            if correct:
                da += 0.1
            self.brain.set_da(da)

            if correct:
                break

            # Snapshot free-phase correlations (network's own attempt)
            last_free_corr = self.brain.snapshot_correlations(n_steps=15)

            # Clamped phase: show correct answer visually via gaze.
            # Copy pathway sees the answer and drives motor naturally.
            self.display_buffer.write_answer(expected_grid)
            self._gaze_step(answer_slot, n_steps=20, strength=gaze_strength)
            clamped_corr = self.brain.snapshot_correlations(n_steps=15)

            mean_change = self.brain.apply_chl(last_free_corr, clamped_corr)

            self.brain.clear_signal()
            self.brain.clear_gaze_bias()
            self.brain.step(n_steps=5)

        if correct and last_free_corr is None:
            free_corr = self.brain.snapshot_correlations(n_steps=15)
            self.brain.store_replay(free_corr, free_corr)

        # --- Phase 4: Rest + spontaneous replay ---
        self.brain.clear_signal()
        self.brain.clear_gaze_bias()
        self.brain.step(n_steps=10)
        self.brain.spontaneous_replay(n_steps=10, strength=0.2)

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

        if not tracker.mastered and tracker.consecutive_solves >= self.mastery_threshold:
            all_tasks = self._tasks_by_type.get(task_type, [])
            if tracker.solves >= len(all_tasks):
                tracker.mastered = True
                self.monitor.status(f"MASTERED: {task_type}")
                self._advance_tier()

        self.state.tasks_today += 1
        if correct:
            self.state.solves_today += 1

        pixel_correct = np.array_equal(output_grid, expected_grid)
        self.monitor.task_result(
            task_type=task_type,
            task_id=task.get("_file", "unknown"),
            correct=correct,
            reward=cell_acc,
            mean_change=mean_change,
            age=self.brain.age,
            extra={
                "cell_accuracy": cell_acc,
                "pixel_match": pixel_correct,
                "rule_type": rule_type,
                "n_demos": len(demo_pairs),
                "n_attempts": n_attempts,
            },
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
