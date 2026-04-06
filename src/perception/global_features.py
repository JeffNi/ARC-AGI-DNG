"""
Grid-level global features — color histogram, symmetry, background.

Level 3 of the perception module. These are broadcast to dedicated
sensory nodes so every part of the network knows the global context.
"""

from __future__ import annotations

import numpy as np

NUM_COLORS = 10


def compute_global_features(grid: np.ndarray, n_objects: int) -> np.ndarray:
    """
    Compute global grid features.

    Returns a vector of length 28:
      [0:10]  - color histogram (fraction of grid per color, sums to 1.0)
      [10:20] - background color one-hot (most common color)
      [20]    - horizontal symmetry (1.0 if grid == fliplr(grid))
      [21]    - vertical symmetry (1.0 if grid == flipud(grid))
      [22]    - 180° rotational symmetry (1.0 if grid == rot180(grid))
      [23]    - transpose symmetry (1.0 if grid == grid.T, only for square grids)
      [24]    - distinct color count (normalized: count / 10)
      [25]    - total object count (normalized: min(count, 30) / 30)
      [26]    - grid aspect ratio (h / max(h, w))
      [27]    - fill ratio (non-background cells / total cells)
    """
    grid = np.asarray(grid, dtype=np.int32)
    h, w = grid.shape
    total = h * w
    features = np.zeros(28, dtype=np.float64)

    # [0:10] Color histogram
    for color in range(NUM_COLORS):
        features[color] = np.sum(grid == color) / total

    # [10:20] Background color (most common) as one-hot
    bg_color = int(np.argmax(features[:NUM_COLORS]))
    features[10 + bg_color] = 1.0

    # [20] Horizontal symmetry (left-right)
    features[20] = 1.0 if np.array_equal(grid, np.fliplr(grid)) else 0.0

    # [21] Vertical symmetry (top-bottom)
    features[21] = 1.0 if np.array_equal(grid, np.flipud(grid)) else 0.0

    # [22] 180° rotational symmetry
    features[22] = 1.0 if np.array_equal(grid, np.rot90(grid, 2)) else 0.0

    # [23] Transpose symmetry (only meaningful for square grids)
    if h == w:
        features[23] = 1.0 if np.array_equal(grid, grid.T) else 0.0
    else:
        features[23] = 0.0

    # [24] Distinct color count (normalized)
    n_distinct = len(np.unique(grid))
    features[24] = n_distinct / NUM_COLORS

    # [25] Total object count (normalized, cap at 30)
    features[25] = min(n_objects, 30) / 30.0

    # [26] Grid aspect ratio
    features[26] = h / max(h, w)

    # [27] Fill ratio (non-background cells / total)
    n_bg = np.sum(grid == bg_color)
    features[27] = (total - n_bg) / total if total > 0 else 0.0

    return features
