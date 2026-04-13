"""
Display buffer: the visual world the network can saccade across.

Holds multiple grid slots that the network observes through gaze selection.
During infancy, slots contain structured pattern pairs for exploration.
During childhood, slots hold task input/output pairs for learning.

The last slot is always the "answer canvas" where motor output is written.
"""

from __future__ import annotations

import numpy as np

from .encoding import pad_grid
from .stimuli import GENERATORS

MAX_SLOTS = 8


class DisplayBuffer:
    """Multi-slot visual display that the network scans via gaze."""

    def __init__(self, max_h: int, max_w: int, n_slots: int = MAX_SLOTS):
        self.max_h = max_h
        self.max_w = max_w
        self.n_slots = n_slots
        self.answer_slot = n_slots - 1
        self.grids: list[np.ndarray] = [
            np.zeros((max_h, max_w), dtype=np.int32) for _ in range(n_slots)
        ]
        self.slot_types: list[str] = ["empty"] * n_slots
        self.slot_types[self.answer_slot] = "answer"

    def get_slot(self, idx: int) -> np.ndarray:
        """Return the grid at slot idx (clamped to valid range)."""
        idx = max(0, min(idx, self.n_slots - 1))
        return self.grids[idx]

    def write_answer(self, grid: np.ndarray) -> None:
        """Write decoded motor output to the answer canvas slot."""
        g = pad_grid(np.asarray(grid, dtype=np.int32), self.max_h, self.max_w)
        self.grids[self.answer_slot] = g

    def clear(self) -> None:
        """Reset all slots to empty (black) grids."""
        for i in range(self.n_slots):
            self.grids[i] = np.zeros((self.max_h, self.max_w), dtype=np.int32)
            self.slot_types[i] = "empty"
        self.slot_types[self.answer_slot] = "answer"

    def load_task(self, task: dict) -> int:
        """Populate slots from a task's demo pairs + test input.

        Layout: [in1, out1, in2, out2, ..., test_in, answer]
        Returns the number of demo pairs loaded.
        """
        self.clear()
        train_pairs = task.get("train", [])
        test_pairs = task.get("test", [])

        slot = 0
        n_loaded = 0
        for pair in train_pairs:
            if slot + 1 >= self.answer_slot:
                break
            inp = pad_grid(np.array(pair["input"], dtype=np.int32),
                           self.max_h, self.max_w)
            out = pad_grid(np.array(pair["output"], dtype=np.int32),
                           self.max_h, self.max_w)
            self.grids[slot] = inp
            self.slot_types[slot] = "input"
            self.grids[slot + 1] = out
            self.slot_types[slot + 1] = "output"
            slot += 2
            n_loaded += 1

        if test_pairs and slot < self.answer_slot:
            test_in = pad_grid(np.array(test_pairs[0]["input"], dtype=np.int32),
                               self.max_h, self.max_w)
            self.grids[slot] = test_in
            self.slot_types[slot] = "test_input"

        return n_loaded

    def load_infancy_stimuli(self, rng: np.random.Generator) -> None:
        """Populate slots with related pattern pairs for pre-task exploration.

        Adjacent slots get related patterns (same shape with a variation),
        mimicking the structured visual environment of an infant's world.
        The answer slot stays empty (motor babble writes there).
        """
        h, w = self.max_h, self.max_w
        n_pairs = (self.n_slots - 1) // 2

        for pair_idx in range(n_pairs):
            slot_a = pair_idx * 2
            slot_b = pair_idx * 2 + 1
            if slot_b >= self.answer_slot:
                break

            gen = rng.choice(GENERATORS)
            seed_val = int(rng.integers(0, 2**31))

            base_grid = gen(h, w, np.random.default_rng(seed_val))
            self.grids[slot_a] = base_grid
            self.slot_types[slot_a] = "stimulus"

            variant = _make_variant(base_grid, rng)
            self.grids[slot_b] = variant
            self.slot_types[slot_b] = "stimulus"

        self.grids[self.answer_slot] = np.zeros((h, w), dtype=np.int32)
        self.slot_types[self.answer_slot] = "answer"


def _make_variant(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create a related variant of a grid pattern.

    Applies one random transformation: color swap, rotation, flip, or shift.
    This creates the statistical regularity that adjacent slots contain
    related patterns, without explicit labeling.
    """
    variant = grid.copy()
    transform = rng.integers(0, 5)

    if transform == 0:
        # Color swap: replace one non-background color with another
        unique = np.unique(variant)
        if len(unique) > 1:
            old_c = int(rng.choice(unique))
            new_c = (old_c + int(rng.integers(1, 10))) % 10
            variant[variant == old_c] = new_c

    elif transform == 1:
        # Horizontal flip
        variant = np.fliplr(variant)

    elif transform == 2:
        # Vertical flip
        variant = np.flipud(variant)

    elif transform == 3:
        # 90-degree rotation (pad back to original shape if needed)
        rotated = np.rot90(variant, k=int(rng.integers(1, 4)))
        h, w = grid.shape
        rh, rw = rotated.shape
        variant = np.zeros_like(grid)
        mh, mw = min(h, rh), min(w, rw)
        variant[:mh, :mw] = rotated[:mh, :mw]

    elif transform == 4:
        # Shift by 1-2 cells in a random direction
        shift = int(rng.integers(1, 3))
        direction = int(rng.integers(0, 4))
        if direction == 0:
            variant = np.roll(variant, shift, axis=0)
        elif direction == 1:
            variant = np.roll(variant, -shift, axis=0)
        elif direction == 2:
            variant = np.roll(variant, shift, axis=1)
        else:
            variant = np.roll(variant, -shift, axis=1)

    return np.ascontiguousarray(variant)
