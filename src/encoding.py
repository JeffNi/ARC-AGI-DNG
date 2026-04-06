"""
ARC grid <-> DNG signal encoding / decoding (vectorized).

Supports variable-size grids by padding to the network's max dimensions.
Unused sensory/motor nodes receive zero (background) signal.

See docs/04_ARC_Strategy.md Section 1.
"""

from __future__ import annotations

import numpy as np


NUM_COLORS = 10


def pad_grid(grid: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    """Pad a grid to (max_h, max_w) with zeros (background color)."""
    grid = np.asarray(grid)
    h, w = grid.shape
    if h == max_h and w == max_w:
        return grid
    padded = np.zeros((max_h, max_w), dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded


def grid_to_signal(
    grid: np.ndarray | list,
    node_offset: int = 0,
    n_total_nodes: int | None = None,
    max_h: int | None = None,
    max_w: int | None = None,
) -> np.ndarray:
    """
    One-hot encode an ARC grid into a signal vector (vectorized).

    If max_h/max_w are provided, the grid is padded to that size first.
    Each cell (r, c) with color v produces a +1 at the node for (r, c, v)
    and 0 elsewhere.
    """
    grid = np.asarray(grid)

    if max_h is not None and max_w is not None:
        grid = pad_grid(grid, max_h, max_w)

    h, w = grid.shape
    n_cells = h * w

    if n_total_nodes is None:
        n_total_nodes = node_offset + n_cells * NUM_COLORS

    signal = np.zeros(n_total_nodes)

    flat = grid.ravel()
    cell_indices = np.arange(n_cells)
    active_idx = node_offset + cell_indices * NUM_COLORS + flat
    signal[active_idx] = 1.0

    return signal


def signal_to_grid(
    values: np.ndarray,
    h: int,
    w: int,
    node_offset: int = 0,
    max_h: int | None = None,
    max_w: int | None = None,
) -> np.ndarray:
    """
    Decode motor neuron values to an ARC grid via argmax (vectorized).

    *values* can be firing rates (r) or membrane voltages (V).  The
    accumulation model uses V so that the decision reflects total
    integrated evidence rather than saturated rates.

    If max_h/max_w are provided, decodes the full padded grid and then
    crops to the requested (h, w).
    """
    decode_h = max_h if max_h is not None else h
    decode_w = max_w if max_w is not None else w
    n_cells = decode_h * decode_w

    motor_r = values[node_offset : node_offset + n_cells * NUM_COLORS]
    motor_r = motor_r.reshape(n_cells, NUM_COLORS)
    full_grid = np.argmax(motor_r, axis=1).reshape(decode_h, decode_w).astype(int)

    if max_h is not None and max_w is not None:
        return full_grid[:h, :w]
    return full_grid
