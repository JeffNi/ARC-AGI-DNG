"""
Perception encoder — assembles all feature levels into a single sensory vector.

This is the main entry point for the perception module. Takes a raw ARC grid,
runs all feature extractors, and produces a flat signal vector ready for the
sensory layer of the DNG.
"""

from __future__ import annotations

import numpy as np

from .features import compute_local_features
from .objects import compute_object_features, _connected_components
from .global_features import compute_global_features

# Per-cell: 10 one-hot + 4 ortho boundary + 4 diag boundary
#         + 8 object features + 1 border flag + 1 neighbor fraction
#         + 2 position encoding (row, col) = 30
FEATURES_PER_CELL = 30

# Global features: 28 (histogram + bg + symmetry + stats)
GLOBAL_FEATURES = 28


def perceive(grid: np.ndarray) -> np.ndarray:
    """
    Full perception pipeline: raw grid -> rich sensory signal vector.

    Returns a 1D float64 array of length (h * w * FEATURES_PER_CELL + GLOBAL_FEATURES).

    Layout:
      For cell (r, c) at flat index i = r * w + c:
        signal[i * 30 : i * 30 + 10]  = one-hot color
        signal[i * 30 + 10 : i * 30 + 14] = ortho boundary
        signal[i * 30 + 14 : i * 30 + 18] = diag boundary
        signal[i * 30 + 18 : i * 30 + 26] = object features
        signal[i * 30 + 26] = border flag
        signal[i * 30 + 27] = same-color neighbor fraction
        signal[i * 30 + 28] = normalized row position
        signal[i * 30 + 29] = normalized column position
      signal[h*w*30 : h*w*30 + 28] = global features
    """
    grid = np.asarray(grid, dtype=np.int32)
    h, w = grid.shape
    n_cells = h * w

    # Level 1: local features (fills slots 0:10, 10:18, 26, 27)
    local = compute_local_features(grid)

    # Level 2: object features (8 features per cell)
    obj = compute_object_features(grid)

    # Merge object features into local feature array slots [18:26]
    local[:, :, 18:26] = obj

    # Level 3: global features (needs object count)
    _, n_objects = _connected_components(grid)
    glob = compute_global_features(grid, n_objects)

    # Flatten: all cells row-major, then global
    signal = np.empty(n_cells * FEATURES_PER_CELL + GLOBAL_FEATURES, dtype=np.float64)
    signal[:n_cells * FEATURES_PER_CELL] = local.reshape(-1)
    signal[n_cells * FEATURES_PER_CELL:] = glob

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
