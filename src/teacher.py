"""
Experimenter — minimal external interface for task presentation.

The experimenter can ONLY:
  - Load tasks into the display buffer
  - Clear the answer canvas to blank
  - Provide light gaze bias (pointing at demo pairs and test input)
  - Pass expected grid to the reward circuit (answer key)
  - Read the canvas when DONE fires or max steps, score it

The experimenter CANNOT: inject signals directly into the brain,
set DA, manipulate weights, or perform any "neurosurgery".
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from .brain import Brain
from .display_buffer import DisplayBuffer
from .encoding import pad_grid, NUM_COLORS
from .gaze_log import GazeLogger
from .monitor import Monitor


@dataclass
class TypeTracker:
    """Track performance for one task type."""
    attempts: int = 0
    solves: int = 0
    consecutive_solves: int = 0
    mastered: bool = False
    failed: bool = False
    last_tested_age: int = 0

    @property
    def solve_rate(self) -> float:
        return self.solves / max(1, self.attempts)


@dataclass
class CurriculumState:
    """State of the mastery-based curriculum with failure timeout."""
    current_tier: int = 0
    type_trackers: dict[str, TypeTracker] = field(default_factory=dict)
    day: int = 0
    tasks_today: int = 0
    solves_today: int = 0


class Teacher:
    """Minimal experimenter for autonomous mushroom body learning."""

    # Max attempts per task type before declaring failure and advancing
    MAX_ATTEMPTS_PER_TYPE = 30
    # Consecutive correct solves required for mastery
    MASTERY_THRESHOLD = 3

    def __init__(
        self,
        brain: Brain,
        monitor: Monitor,
        display_buffer: DisplayBuffer | None = None,
        gaze_logger: GazeLogger | None = None,
        task_dir: str = "micro_tasks",
        observe_steps: int | None = None,
        action_steps: int | None = None,
        **kwargs,
    ):
        self.brain = brain
        self.monitor = monitor
        self.display_buffer = display_buffer or DisplayBuffer(
            brain.net.max_h, brain.net.max_w,
        )
        self.gaze_logger = gaze_logger
        self.task_dir = Path(task_dir)
        # Read timing from genome unless caller overrides
        self.observe_steps = observe_steps if observe_steps is not None else brain.genome.observe_steps
        self.action_steps = action_steps if action_steps is not None else brain.genome.action_budget

        self.state = CurriculumState()
        self._tasks_by_type: dict[str, list[dict]] = {}
        self._tier_order: list[str] = []
        self._last_task_type: str | None = None

        self._load_tasks()
        self._generate_micro_tasks()

    def _generate_micro_tasks(self):
        """Generate tiered tasks scaled to grid size."""
        h, w = self.brain.net.max_h, self.brain.net.max_w
        rng = np.random.default_rng(42)

        def _rand_grid(rng, h, w, n_colors=3, density=0.5):
            g = [[0] * w for _ in range(h)]
            colors = [int(rng.integers(1, NUM_COLORS)) for _ in range(n_colors)]
            for r in range(h):
                for c in range(w):
                    if rng.random() < density:
                        g[r][c] = colors[int(rng.integers(0, len(colors)))]
            return g

        def _copy_grid(g):
            return [row[:] for row in g]

        # --- Tier 0: Identity ---
        t0 = []
        for i in range(20):
            pairs = []
            for _ in range(3):
                g = _rand_grid(rng, h, w, n_colors=2, density=0.4)
                pairs.append({"input": g, "output": _copy_grid(g)})
            tg = _rand_grid(rng, h, w, n_colors=2, density=0.4)
            t0.append({
                "train": pairs, "test": [{"input": tg, "output": _copy_grid(tg)}],
                "_file": f"identity_{i:03d}", "_tier": "gen_identity", "type": "identity",
            })
        self._tasks_by_type["gen_identity"] = t0
        self._tier_order.append("gen_identity")

        # --- Tier 1: Solid fill ---
        t1 = []
        for i in range(20):
            pairs = []
            for _ in range(3):
                fill_color = int(rng.integers(1, NUM_COLORS))
                g = [[0] * w for _ in range(h)]
                for r in range(h):
                    for c in range(w):
                        if rng.random() < 0.6:
                            g[r][c] = fill_color
                        elif rng.random() < 0.3:
                            g[r][c] = int(rng.integers(1, NUM_COLORS))
                out = [[fill_color] * w for _ in range(h)]
                pairs.append({"input": g, "output": out})
            fill_color = int(rng.integers(1, NUM_COLORS))
            g = [[0] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    if rng.random() < 0.6:
                        g[r][c] = fill_color
                    elif rng.random() < 0.3:
                        g[r][c] = int(rng.integers(1, NUM_COLORS))
            out = [[fill_color] * w for _ in range(h)]
            t1.append({
                "train": pairs, "test": [{"input": tg, "output": _copy_grid(tg)}],
                "_file": f"solid_fill_{i:03d}", "_tier": "gen_solid_fill", "type": "solid_fill",
            })
            # Fix: use the actual fill output for this task
            t1[-1]["test"] = [{"input": g, "output": out}]
        self._tasks_by_type["gen_solid_fill"] = t1
        self._tier_order.append("gen_solid_fill")

        # --- Tier 2: Color swap ---
        t2 = []
        for i in range(20):
            ca = int(rng.integers(1, NUM_COLORS))
            cb = int(rng.integers(1, NUM_COLORS))
            while cb == ca:
                cb = int(rng.integers(1, NUM_COLORS))
            pairs = []
            for _ in range(3):
                g = _rand_grid(rng, h, w, n_colors=2, density=0.5)
                g[0][0] = ca
                g[h-1][w-1] = cb
                out = _copy_grid(g)
                for r in range(h):
                    for c in range(w):
                        if out[r][c] == ca:
                            out[r][c] = cb
                        elif out[r][c] == cb:
                            out[r][c] = ca
                pairs.append({"input": g, "output": out})
            tg = _rand_grid(rng, h, w, n_colors=2, density=0.5)
            tg[0][0] = ca
            tg[h-1][w-1] = cb
            to = _copy_grid(tg)
            for r in range(h):
                for c in range(w):
                    if to[r][c] == ca:
                        to[r][c] = cb
                    elif to[r][c] == cb:
                        to[r][c] = ca
            t2.append({
                "train": pairs, "test": [{"input": tg, "output": to}],
                "_file": f"color_swap_{i:03d}", "_tier": "gen_color_swap", "type": "color_swap",
            })
        self._tasks_by_type["gen_color_swap"] = t2
        self._tier_order.append("gen_color_swap")

        # --- Tier 3: Horizontal flip ---
        t3 = []
        for i in range(20):
            pairs = []
            for _ in range(3):
                g = _rand_grid(rng, h, w, n_colors=3, density=0.5)
                out = [row[::-1] for row in g]
                pairs.append({"input": g, "output": out})
            tg = _rand_grid(rng, h, w, n_colors=3, density=0.5)
            to = [row[::-1] for row in tg]
            t3.append({
                "train": pairs, "test": [{"input": tg, "output": to}],
                "_file": f"flip_h_{i:03d}", "_tier": "gen_flip_h", "type": "flip_h",
            })
        self._tasks_by_type["gen_flip_h"] = t3
        self._tier_order.append("gen_flip_h")

        # --- Tier 4: Color extract ---
        t4 = []
        for i in range(20):
            target = int(rng.integers(1, NUM_COLORS))
            pairs = []
            for _ in range(3):
                g = _rand_grid(rng, h, w, n_colors=3, density=0.6)
                g[int(rng.integers(0, h))][int(rng.integers(0, w))] = target
                out = [[g[r][c] if g[r][c] == target else 0 for c in range(w)] for r in range(h)]
                pairs.append({"input": g, "output": out})
            tg = _rand_grid(rng, h, w, n_colors=3, density=0.6)
            tg[int(rng.integers(0, h))][int(rng.integers(0, w))] = target
            to = [[tg[r][c] if tg[r][c] == target else 0 for c in range(w)] for r in range(h)]
            t4.append({
                "train": pairs, "test": [{"input": tg, "output": to}],
                "_file": f"color_extract_{i:03d}", "_tier": "gen_color_extract", "type": "color_extract",
            })
        self._tasks_by_type["gen_color_extract"] = t4
        self._tier_order.append("gen_color_extract")

    def _task_fits(self, task: dict) -> bool:
        max_h = self.brain.net.max_h
        max_w = self.brain.net.max_w
        for pair in task.get("train", []) + task.get("test", []):
            for key in ("input", "output"):
                g = pair.get(key, [])
                if len(g) > max_h or (g and len(g[0]) > max_w):
                    return False
        return True

    def _load_tasks(self):
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
        if not self._tier_order:
            return None
        idx = min(self.state.current_tier, len(self._tier_order) - 1)
        return self._tier_order[idx]

    def pick_task(self) -> tuple[str, dict] | None:
        """Pick next task: varied within tier, no blocked practice."""
        if not self._tier_order:
            return None

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
        Autonomous learning loop. The experimenter is NOT a neurosurgeon.

        Flow:
          1. Load task into display buffer, clear canvas, reset eligibility
          2. Light gaze bias through demos (observation phase)
          3. Gaze bias to test input, then brain explores freely (action phase)
          4. Brain commits actions autonomously, gets per-action food/shock
          5. DONE fires or max steps -> read canvas, compute accuracy
          6. Global DA for DONE + GAZE learning
          7. Brief rest
        """
        h = self.brain.net.max_h
        w = self.brain.net.max_w

        self._last_task_type = task_type

        pairs = task.get("train", [])
        test_pairs = task.get("test", [])
        if not pairs and not test_pairs:
            return False

        # Use last training pair as test if no explicit test
        if test_pairs:
            test_pair = test_pairs[0]
        else:
            test_pair = pairs[-1]

        expected_grid = pad_grid(np.array(test_pair["output"], dtype=np.int32), h, w)

        # ── 1. Load task, clear canvas, reset traces ──
        n_demos = self.display_buffer.load_task(task)
        self.brain.reset_canvas()
        self.brain.reset_eligibility_traces()

        # ── 2. Observation phase: gaze at test input only ──
        # The brain observes the test input to build an L3 representation.
        # Demo grids are skipped — the brain doesn't yet understand what
        # "demo pairs" mean, so gazing at them adds noise, not signal.
        test_slot = n_demos * 2
        self.brain.guide_gaze(test_slot, strength=0.4)
        self.brain.step_with_gaze(
            self.display_buffer,
            gaze_logger=self.gaze_logger,
            n_steps=self.observe_steps,
            learn=True,
        )

        # ── 3. Action phase: deterministic position cycling ──
        # The teacher walks the brain through every cell, like a bee
        # visiting each flower. The brain picks a color at each stop.
        # This guarantees full grid coverage so depression can accumulate.
        self.brain.reset_canvas()

        # Set observed-color constraint AFTER reset_canvas (which clears it).
        test_grid = pad_grid(np.array(test_pair["input"], dtype=np.int32), h, w)
        self.brain.set_observed_colors(test_grid)

        # Keep test input visible — the sensory signal drives copy bias
        # and L3 representation. Gaze stays on test input throughout.
        self.brain.clear_gaze_bias()
        self.brain.guide_gaze(test_slot, strength=0.4)
        self.brain.apply_gaze(self.display_buffer)

        n_cells = h * w
        commit_log = []
        total_dw = 0.0
        max_dw = 0.0

        for pos in range(n_cells):
            detail = self.brain.commit_at_position(pos, expected_grid)
            commit_log.append(detail)
            dw = abs(detail.get("mean_dw", 0.0))
            total_dw += dw
            if dw > max_dw:
                max_dw = dw

        # ── 4. Score the canvas ──
        canvas = self.brain.get_canvas_grid(h, w)
        cell_acc = float(np.sum(canvas == expected_grid)) / n_cells
        correct = np.array_equal(canvas, expected_grid)

        # ── 5. Global DA for DONE + GAZE learning ──
        g = self.brain.genome
        global_da = g.da_reward_offset + g.da_reward_slope * cell_acc
        if correct:
            global_da = max(global_da, g.da_reward_floor)
        self.brain.apply_reward(global_da)

        # ── 6. Rest period ──
        self.brain.clear_gaze_bias()
        self.brain.clear_signal()
        self.brain.step(n_steps=5)

        sleep_stats = self.brain.try_sleep()
        if sleep_stats:
            self.monitor.sleep_event(sleep_stats, self.brain.age)

        # ── 7. Update tracking ──
        tracker = self.state.type_trackers.setdefault(task_type, TypeTracker())
        tracker.attempts += 1
        tracker.last_tested_age = self.brain.age
        if correct:
            tracker.solves += 1
            tracker.consecutive_solves += 1
        else:
            tracker.consecutive_solves = 0

        # Mastery check
        if not tracker.mastered and not tracker.failed:
            if tracker.consecutive_solves >= self.MASTERY_THRESHOLD:
                tracker.mastered = True
                self.monitor.status(f"MASTERED: {task_type}")
                self._advance_tier()
            elif tracker.attempts >= self.MAX_ATTEMPTS_PER_TYPE:
                tracker.failed = True
                self.monitor.status(f"FAILED (timeout): {task_type} after {tracker.attempts} attempts")
                self._advance_tier()

        self.state.tasks_today += 1
        if correct:
            self.state.solves_today += 1

        mbon_weights = self.brain.get_mbon_weight_summary()

        self.monitor.task_result(
            task_type=task_type,
            task_id=task.get("_file", "unknown"),
            correct=correct,
            reward=cell_acc,
            mean_change=max_dw,
            age=self.brain.age,
            extra={
                "cell_accuracy": cell_acc,
                "total_commits": n_cells,
                "done_fired": False,
                "mbon_weights": mbon_weights,
                "commits": commit_log,
            },
        )

        self.last_cell_acc = cell_acc
        return correct

    def _advance_tier(self):
        if self.state.current_tier < len(self._tier_order) - 1:
            self.state.current_tier += 1
            new_type = self.current_type()
            self.monitor.status(f"Advanced to tier {self.state.current_tier}: {new_type}")

    MAX_RETRIES_PER_INSTANCE = 10

    def run_day(self, max_tasks: int = 50) -> dict:
        """Run a full day: cycle through task types in blocks.

        For each task type:
          1. Reset MBON weights (fresh start for this type).
          2. Retry the *same* task instance until solved or MAX_RETRIES hit.
             Depression accumulates across retries — this is the core learning loop.
          3. Once solved (or retries exhausted), move to the next instance
             of the same type — MBON weights persist so the brain must
             generalize the learned rule.
          4. After exhausting the budget for this type, move to the next type.
        """
        self.state.tasks_today = 0
        self.state.solves_today = 0
        self.state.day += 1

        gen_types = [t for t in self._tasks_by_type if t.startswith("gen_")]
        if not gen_types:
            gen_types = list(self._tasks_by_type.keys())

        trials_per_type = max(1, max_tasks // max(1, len(gen_types)))

        for task_type in gen_types:
            tasks = self._tasks_by_type.get(task_type, [])
            if not tasks:
                continue
            trials_used = 0
            task_idx = 0

            while trials_used < trials_per_type and task_idx < len(tasks):
                task = tasks[task_idx]
                for retry in range(self.MAX_RETRIES_PER_INSTANCE):
                    if trials_used >= trials_per_type:
                        break
                    solved = self.run_task(task_type, task)
                    trials_used += 1
                    if solved:
                        break
                task_idx += 1

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

        if self.brain.checkpointer.should_autosave():
            self.brain.save()

        return {
            "day": self.state.day,
            "attempted": self.state.tasks_today,
            "solved": self.state.solves_today,
            "mastered": n_mastered,
        }
