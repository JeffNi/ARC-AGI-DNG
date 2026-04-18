"""
Perception encoder — one-hot color encoding for insect-baseline architecture.

Each cell gets a 10-element one-hot vector (one per ARC color).
No boundary, object, or global features — the mushroom body does
pattern separation on raw sensory input.
"""

from __future__ import annotations

import numpy as np

FEATURES_PER_CELL = 10
GLOBAL_FEATURES = 0


def perceive(grid: np.ndarray) -> np.ndarray:
    """
    One-hot color encoding: raw grid -> sensory signal vector.

    Returns a 1D float64 array of length (h * w * 10).

    Layout:
      For cell (r, c) at flat index i = r * w + c:
        signal[i * 10 : i * 10 + 10] = one-hot color
    """
    grid = np.asarray(grid, dtype=np.int32)
    h, w = grid.shape
    n_cells = h * w

    signal = np.zeros(n_cells * FEATURES_PER_CELL, dtype=np.float64)
    flat = grid.ravel()
    for i in range(n_cells):
        signal[i * FEATURES_PER_CELL + flat[i]] = 1.0

    return signal


def sensory_size(grid_h: int, grid_w: int) -> int:
    """Total number of sensory nodes for a grid of given dimensions."""
    return grid_h * grid_w * FEATURES_PER_CELL + GLOBAL_FEATURES


def decode_colors(signal: np.ndarray, h: int, w: int) -> np.ndarray:
    """Extract color grid from a perception signal via argmax on one-hot slots."""
    n_cells = h * w
    grid = np.empty(n_cells, dtype=np.int32)
    for i in range(n_cells):
        base = i * FEATURES_PER_CELL
        grid[i] = int(np.argmax(signal[base:base + 10]))
    return grid.reshape(h, w)
