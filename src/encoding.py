"""
ARC grid <-> DNG signal encoding / decoding.

Uses the perception module for rich feature extraction on encode,
and argmax over color nodes for decode.

Supports variable-size grids by padding to the network's max dimensions.
"""

from __future__ import annotations

import numpy as np

from .perception import perceive, FEATURES_PER_CELL, GLOBAL_FEATURES
from .perception.encoder import sensory_size, decode_colors

NUM_COLORS = 10


def pad_grid(grid: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    """Pad a grid to (max_h, max_w) with zeros (background color)."""
    grid = np.asarray(grid)
    h, w = grid.shape
    if h > max_h or w > max_w:
        grid = grid[:max_h, :max_w]
        h, w = grid.shape
    if h == max_h and w == max_w:
        return grid
    padded = np.zeros((max_h, max_w), dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded


def grid_to_signal(
    grid: np.ndarray | list,
    n_total_nodes: int | None = None,
    max_h: int | None = None,
    max_w: int | None = None,
) -> np.ndarray:
    """
    Encode an ARC grid into a rich perception signal vector.

    If max_h/max_w are provided, the grid is padded to that size first.
    Returns a signal of length sensory_size(h, w) or n_total_nodes if specified.
    """
    grid = np.asarray(grid, dtype=np.int32)

    if max_h is not None and max_w is not None:
        grid = pad_grid(grid, max_h, max_w)

    signal = perceive(grid)

    if n_total_nodes is not None and n_total_nodes > len(signal):
        full = np.zeros(n_total_nodes)
        full[:len(signal)] = signal
        return full

    return signal


def signal_to_grid(
    rates: np.ndarray,
    h: int,
    w: int,
    max_h: int | None = None,
    max_w: int | None = None,
) -> np.ndarray:
    """
    Decode motor firing rates to an ARC grid via argmax on color nodes.

    Motor neurons use simple one-hot color encoding (10 per cell).
    If max_h/max_w are provided, decodes the full padded grid and crops.
    """
    decode_h = max_h if max_h is not None else h
    decode_w = max_w if max_w is not None else w
    n_cells = decode_h * decode_w

    motor_r = rates[:n_cells * NUM_COLORS]
    motor_r = motor_r.reshape(n_cells, NUM_COLORS)
    full_grid = np.argmax(motor_r, axis=1).reshape(decode_h, decode_w).astype(int)

    if max_h is not None and max_w is not None:
        return full_grid[:h, :w]
    return full_grid
