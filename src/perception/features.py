"""
Per-cell local features — edge detection, neighbor analysis, border flags.

Level 1 of the perception module. Analogous to retinal ganglion cells
and V1 simple cells: detects local contrast and spatial position.
"""

from __future__ import annotations

import numpy as np

NUM_COLORS = 10

# Orthogonal directions: up, down, left, right
_ORTHO_DR = np.array([-1, 1, 0, 0], dtype=np.int32)
_ORTHO_DC = np.array([0, 0, -1, 1], dtype=np.int32)

# Diagonal directions: top-left, top-right, bottom-left, bottom-right
_DIAG_DR = np.array([-1, -1, 1, 1], dtype=np.int32)
_DIAG_DC = np.array([-1, 1, -1, 1], dtype=np.int32)


def compute_local_features(grid: np.ndarray) -> np.ndarray:
    """
    Compute per-cell local features from a raw ARC grid.

    For each cell, produces a vector of length 28:
      [0:10]  - one-hot color encoding
      [10:14] - orthogonal boundary signals (1.0 if neighbor differs)
      [14:18] - diagonal boundary signals (1.0 if neighbor differs)
      [18:22] - reserved for object features (filled by objects.py)
      [22:26] - reserved for object features (filled by objects.py)
      [26]    - border flag (1.0 if cell is on grid edge)
      [27]    - same-color orthogonal neighbor fraction

    Returns:
        np.ndarray of shape (h, w, 28) with values in [0, 1].
    """
    grid = np.asarray(grid, dtype=np.int32)
    h, w = grid.shape
    features = np.zeros((h, w, 28), dtype=np.float64)

    # One-hot color encoding [0:10]
    for r in range(h):
        for c in range(w):
            color = grid[r, c]
            if 0 <= color < NUM_COLORS:
                features[r, c, color] = 1.0

    # Boundary signals [10:14] orthogonal, [14:18] diagonal
    for r in range(h):
        for c in range(w):
            color = grid[r, c]

            # Orthogonal boundaries
            for d in range(4):
                nr, nc = r + _ORTHO_DR[d], c + _ORTHO_DC[d]
                if 0 <= nr < h and 0 <= nc < w:
                    if grid[nr, nc] != color:
                        features[r, c, 10 + d] = 1.0
                # Out of bounds: no boundary signal (not a color difference)

            # Diagonal boundaries
            for d in range(4):
                nr, nc = r + _DIAG_DR[d], c + _DIAG_DC[d]
                if 0 <= nr < h and 0 <= nc < w:
                    if grid[nr, nc] != color:
                        features[r, c, 14 + d] = 1.0

    # Border flag [26]
    for r in range(h):
        for c in range(w):
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                features[r, c, 26] = 1.0

    # Same-color orthogonal neighbor fraction [27]
    for r in range(h):
        for c in range(w):
            color = grid[r, c]
            n_neighbors = 0
            n_same = 0
            for d in range(4):
                nr, nc = r + _ORTHO_DR[d], c + _ORTHO_DC[d]
                if 0 <= nr < h and 0 <= nc < w:
                    n_neighbors += 1
                    if grid[nr, nc] == color:
                        n_same += 1
            features[r, c, 27] = n_same / max(n_neighbors, 1)

    return features
