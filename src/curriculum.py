"""
Curriculum: order ARC tasks from easiest to hardest.

Difficulty heuristics (lower = easier):
  1. Grid size (smaller grids are simpler)
  2. Number of unique colors used
  3. Whether output = input (identity is trivial)
  4. Whether all outputs are the same color (fill is simple)
  5. Grid size change (input != output size adds complexity)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def task_difficulty(train_pairs, test_in, test_out) -> float:
    """Estimate difficulty of a task (lower = easier)."""
    score = 0.0

    all_grids = [test_in, test_out]
    for inp, out in train_pairs:
        all_grids.extend([inp, out])

    # Grid size
    max_cells = max(g.shape[0] * g.shape[1] for g in all_grids)
    score += max_cells * 0.5

    # Unique colors across all grids
    all_colors = set()
    for g in all_grids:
        all_colors.update(g.ravel().tolist())
    score += len(all_colors) * 2

    # Identity check (output == input for all examples)
    is_identity = all(
        np.array_equal(inp, out) and inp.shape == out.shape
        for inp, out in train_pairs
    )
    if is_identity:
        score -= 50  # very easy

    # Single-color fill check
    is_fill = all(
        len(np.unique(out)) == 1
        for _, out in train_pairs
    )
    if is_fill:
        score -= 20  # easy

    # Size change between input and output
    size_changes = any(
        inp.shape != out.shape
        for inp, out in train_pairs
    )
    if size_changes:
        score += 15

    # More training examples = potentially more complex rule
    score += len(train_pairs) * 2

    return score


def sort_by_difficulty(
    tasks: List[Tuple],
) -> List[Tuple]:
    """Sort tasks by estimated difficulty (easiest first)."""
    scored = []
    for task in tasks:
        train_pairs, test_in, test_out = task
        d = task_difficulty(train_pairs, test_in, test_out)
        scored.append((d, task))

    scored.sort(key=lambda x: x[0])
    return [t for _, t in scored]
