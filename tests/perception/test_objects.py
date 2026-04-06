"""Tests for connected component analysis and object features."""

import numpy as np
import pytest
from src.perception.objects import compute_object_features, _connected_components


class TestConnectedComponents:
    def test_uniform_grid(self):
        grid = np.ones((3, 3), dtype=int)
        labels, n = _connected_components(grid)
        assert n == 1
        assert np.all(labels == labels[0, 0])

    def test_two_rectangles(self):
        grid = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [2, 2, 2, 2]])
        labels, n = _connected_components(grid)
        assert n == 3  # 1-block, 0-block, 2-block
        assert labels[0, 0] == labels[1, 1]  # same 1-group
        assert labels[0, 2] == labels[0, 3]  # same 0-group
        assert labels[0, 0] != labels[0, 2]  # different groups

    def test_diagonal_not_connected(self):
        """ARC uses 4-connectivity, diagonals don't connect."""
        grid = np.array([[1, 0],
                         [0, 1]])
        labels, n = _connected_components(grid)
        # The two 1s are NOT connected (diagonal only)
        assert labels[0, 0] != labels[1, 1]
        assert n == 4  # each cell is its own component

    def test_l_shape(self):
        grid = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [1, 1, 1]])
        _, n = _connected_components(grid)
        labels_1 = set()
        labels_0 = set()
        for r in range(3):
            for c in range(3):
                if grid[r, c] == 1:
                    labels_1.add(_connected_components(grid)[0][r, c])
                else:
                    labels_0.add(_connected_components(grid)[0][r, c])
        # Recompute once for consistency
        labels, n = _connected_components(grid)
        ones_labels = {labels[r, c] for r in range(3) for c in range(3) if grid[r, c] == 1}
        assert len(ones_labels) == 1  # all 1s form one component

    def test_all_different_colors(self):
        grid = np.arange(6).reshape(2, 3)
        _, n = _connected_components(grid)
        assert n == 6

    def test_single_cell(self):
        grid = np.array([[5]])
        labels, n = _connected_components(grid)
        assert n == 1
        assert labels[0, 0] == 0


class TestObjectFeatures:
    def test_output_shape(self):
        grid = np.ones((3, 4), dtype=int)
        feat = compute_object_features(grid)
        assert feat.shape == (3, 4, 8)

    def test_uniform_grid_size(self):
        grid = np.ones((3, 3), dtype=int)
        feat = compute_object_features(grid)
        # One object covering entire grid -> size/total = 1.0
        assert feat[0, 0, 0] == 1.0

    def test_boundary_vs_interior(self):
        grid = np.ones((3, 3), dtype=int)
        feat = compute_object_features(grid)
        # Interior cell (1,1) should be 0.0 (not boundary)
        assert feat[1, 1, 1] == 0.0
        # Edge cells should be 1.0 (boundary)
        assert feat[0, 0, 1] == 1.0
        assert feat[0, 1, 1] == 1.0

    def test_single_cell_boundary(self):
        grid = np.array([[1, 0],
                         [0, 0]])
        feat = compute_object_features(grid)
        # The single 1-cell object: always boundary
        assert feat[0, 0, 1] == 1.0

    def test_line_all_boundary(self):
        grid = np.array([[1, 1, 1, 1, 1]])
        feat = compute_object_features(grid)
        for c in range(5):
            assert feat[0, c, 1] == 1.0  # all boundary (width=1)

    def test_bounding_box_position(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[1:4, 1:4] = 1  # 3x3 block in center
        feat = compute_object_features(grid)
        # Center of the 1-block: (2,2) relative to bbox (1:3, 1:3)
        assert abs(feat[2, 2, 2] - 0.5) < 1e-10  # top dist
        assert abs(feat[2, 2, 3] - 0.5) < 1e-10  # bottom dist
        assert abs(feat[2, 2, 4] - 0.5) < 1e-10  # left dist
        assert abs(feat[2, 2, 5] - 0.5) < 1e-10  # right dist
        # Top-left of block: (1,1)
        assert feat[1, 1, 2] == 0.0  # top dist = 0
        assert feat[1, 1, 4] == 0.0  # left dist = 0
        assert feat[1, 1, 3] == 1.0  # bottom dist = 1.0
        assert feat[1, 1, 5] == 1.0  # right dist = 1.0

    def test_single_cell_bbox(self):
        grid = np.array([[0, 1],
                         [0, 0]])
        feat = compute_object_features(grid)
        # Single cell object: all distances = 0
        assert feat[0, 1, 2] == 0.0
        assert feat[0, 1, 3] == 0.0
        assert feat[0, 1, 4] == 0.0
        assert feat[0, 1, 5] == 0.0

    def test_compactness_square(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[1:4, 1:4] = 1
        feat = compute_object_features(grid)
        # 3x3 square: area=9, bbox=9 -> compactness=1.0
        assert abs(feat[2, 2, 7] - 1.0) < 1e-10

    def test_compactness_l_shape(self):
        grid = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [1, 1, 1]])
        feat = compute_object_features(grid)
        # L-shape: 5 cells in 3x3 bbox -> 5/9
        assert abs(feat[0, 0, 7] - 5 / 9) < 1e-10

    def test_compactness_line(self):
        grid = np.array([[1, 1, 1, 1, 1]])
        feat = compute_object_features(grid)
        # Line: 5 cells in 1x5 bbox -> 5/5 = 1.0
        assert abs(feat[0, 0, 7] - 1.0) < 1e-10

    def test_same_color_object_count(self):
        grid = np.array([[1, 0, 1],
                         [0, 0, 0],
                         [1, 0, 0]])
        feat = compute_object_features(grid)
        # Color 1: three separate blobs (none 4-connected)
        assert abs(feat[0, 0, 6] - 3 / 10) < 1e-10
        assert abs(feat[0, 2, 6] - 3 / 10) < 1e-10

    def test_mixed_color_counts(self):
        grid = np.array([[1, 0, 1],
                         [2, 2, 0]])
        feat = compute_object_features(grid)
        # Color 1: 2 separate blobs
        # Color 2: 1 blob
        assert abs(feat[0, 0, 6] - 2 / 10) < 1e-10  # red cells
        assert abs(feat[1, 0, 6] - 1 / 10) < 1e-10  # blue cells

    def test_enclosed_region(self):
        grid = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]])
        labels, n = _connected_components(grid)
        # The 0 cell at (1,1) is enclosed but still a separate component
        assert labels[1, 1] != labels[0, 0]

    def test_output_range(self):
        grid = np.random.randint(0, 10, size=(5, 5))
        feat = compute_object_features(grid)
        assert np.all(feat >= 0.0)
        assert np.all(feat <= 1.0)
