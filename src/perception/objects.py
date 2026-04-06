"""
Per-cell object/region features via connected component analysis.

Level 2 of the perception module. Analogous to V2/V4 figure-ground
segregation: gives each cell awareness of which object it belongs to,
the object's size, shape, and the cell's position within it.

Uses 4-connectivity (orthogonal only) — matching ARC conventions.
"""

from __future__ import annotations

import numpy as np


def compute_object_features(grid: np.ndarray) -> np.ndarray:
    """
    Compute per-cell object features using connected component analysis.

    For each cell, produces 8 features written into slots [18:26]:
      [18] - object size (normalized by grid area)
      [19] - boundary flag (1.0 if cell is on object boundary, 0.0 if interior)
      [20] - relative position top (distance from top of bounding box, normalized)
      [21] - relative position bottom
      [22] - relative position left
      [23] - relative position right
      [24] - same-color object count (normalized by max 10)
      [25] - object compactness (area / bounding_box_area)

    Returns:
        np.ndarray of shape (h, w, 8) with values in [0, 1].
    """
    grid = np.asarray(grid, dtype=np.int32)
    h, w = grid.shape
    total_cells = h * w
    features = np.zeros((h, w, 8), dtype=np.float64)

    labels, n_objects = _connected_components(grid)

    if n_objects == 0:
        return features

    # Compute per-object stats
    obj_sizes = np.zeros(n_objects, dtype=np.int32)
    obj_min_r = np.full(n_objects, h, dtype=np.int32)
    obj_max_r = np.full(n_objects, -1, dtype=np.int32)
    obj_min_c = np.full(n_objects, w, dtype=np.int32)
    obj_max_c = np.full(n_objects, -1, dtype=np.int32)
    obj_color = np.full(n_objects, -1, dtype=np.int32)

    for r in range(h):
        for c in range(w):
            lbl = labels[r, c]
            obj_sizes[lbl] += 1
            obj_color[lbl] = grid[r, c]
            if r < obj_min_r[lbl]:
                obj_min_r[lbl] = r
            if r > obj_max_r[lbl]:
                obj_max_r[lbl] = r
            if c < obj_min_c[lbl]:
                obj_min_c[lbl] = c
            if c > obj_max_c[lbl]:
                obj_max_c[lbl] = c

    # Count objects per color
    color_obj_count = np.zeros(10, dtype=np.int32)
    for i in range(n_objects):
        col = obj_color[i]
        if 0 <= col < 10:
            color_obj_count[col] += 1

    # Bounding box areas and compactness
    bbox_h = obj_max_r - obj_min_r + 1
    bbox_w = obj_max_c - obj_min_c + 1
    bbox_area = bbox_h * bbox_w
    compactness = np.where(bbox_area > 0, obj_sizes / bbox_area, 1.0)

    # Fill per-cell features
    for r in range(h):
        for c in range(w):
            lbl = labels[r, c]

            # [0] Object size normalized by grid area
            features[r, c, 0] = obj_sizes[lbl] / total_cells

            # [1] Object boundary flag
            is_boundary = False
            color = grid[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    is_boundary = True
                    break
                if labels[nr, nc] != lbl:
                    is_boundary = True
                    break
            features[r, c, 1] = 1.0 if is_boundary else 0.0

            # [2:6] Relative position within bounding box
            bb_h = bbox_h[lbl]
            bb_w = bbox_w[lbl]
            if bb_h > 1:
                features[r, c, 2] = (r - obj_min_r[lbl]) / (bb_h - 1)
                features[r, c, 3] = (obj_max_r[lbl] - r) / (bb_h - 1)
            # else: single row, distances stay 0.0

            if bb_w > 1:
                features[r, c, 4] = (c - obj_min_c[lbl]) / (bb_w - 1)
                features[r, c, 5] = (obj_max_c[lbl] - c) / (bb_w - 1)
            # else: single col, distances stay 0.0

            # [6] Same-color object count (normalized, cap at 10)
            col = grid[r, c]
            if 0 <= col < 10:
                features[r, c, 6] = min(color_obj_count[col], 10) / 10.0

            # [7] Object compactness
            features[r, c, 7] = compactness[lbl]

    return features


def _connected_components(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """
    4-connected component labeling using union-find.

    Returns:
        labels: (h, w) array of component IDs (0-indexed, consecutive)
        n_components: total number of components
    """
    h, w = grid.shape
    parent = np.arange(h * w, dtype=np.int32)
    rank = np.zeros(h * w, dtype=np.int32)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for r in range(h):
        for c in range(w):
            idx = r * w + c
            color = grid[r, c]
            # Check right neighbor
            if c + 1 < w and grid[r, c + 1] == color:
                union(idx, idx + 1)
            # Check down neighbor
            if r + 1 < h and grid[r + 1, c] == color:
                union(idx, idx + w)

    # Relabel to consecutive IDs
    label_map = {}
    labels = np.empty((h, w), dtype=np.int32)
    next_id = 0
    for r in range(h):
        for c in range(w):
            root = find(r * w + c)
            if root not in label_map:
                label_map[root] = next_id
                next_id += 1
            labels[r, c] = label_map[root]

    return labels, next_id
