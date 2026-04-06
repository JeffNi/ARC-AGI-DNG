"""Tests for grid-level global features."""

import numpy as np
import pytest
from src.perception.global_features import compute_global_features


class TestColorHistogram:
    def test_uniform_red(self):
        grid = np.full((3, 3), 1, dtype=int)
        feat = compute_global_features(grid, n_objects=1)
        assert feat[1] == 1.0  # color 1 = 100%
        assert feat[0] == 0.0
        assert abs(np.sum(feat[:10]) - 1.0) < 1e-10

    def test_fifty_fifty(self):
        grid = np.array([[0, 0, 1, 1],
                         [0, 0, 1, 1]])
        feat = compute_global_features(grid, n_objects=2)
        assert abs(feat[0] - 0.5) < 1e-10
        assert abs(feat[1] - 0.5) < 1e-10
        assert abs(np.sum(feat[:10]) - 1.0) < 1e-10

    def test_sums_to_one(self):
        grid = np.random.randint(0, 10, size=(4, 4))
        feat = compute_global_features(grid, n_objects=5)
        assert abs(np.sum(feat[:10]) - 1.0) < 1e-10


class TestBackgroundColor:
    def test_mostly_blue(self):
        grid = np.ones((5, 5), dtype=int)
        grid[0, 0] = 3
        feat = compute_global_features(grid, n_objects=2)
        # Background = color 1 (most common)
        assert feat[11] == 1.0  # one-hot at index 10+1
        assert np.sum(feat[10:20]) == 1.0

    def test_deterministic_tiebreak(self):
        # Two equally common colors -> argmax picks lowest index
        grid = np.array([[0, 1],
                         [0, 1]])
        feat = compute_global_features(grid, n_objects=2)
        # Both 50%, argmax picks 0
        assert feat[10] == 1.0


class TestSymmetry:
    def test_horizontal_symmetric(self):
        grid = np.array([[1, 2, 1],
                         [3, 4, 3]])
        feat = compute_global_features(grid, n_objects=4)
        assert feat[20] == 1.0  # h_sym
        assert feat[21] == 0.0  # v_sym (rows differ)

    def test_vertical_symmetric(self):
        grid = np.array([[1, 2],
                         [3, 4],
                         [1, 2]])
        feat = compute_global_features(grid, n_objects=4)
        assert feat[20] == 0.0  # not h_sym
        assert feat[21] == 1.0  # v_sym

    def test_rot180_symmetric(self):
        grid = np.array([[1, 2],
                         [2, 1]])
        feat = compute_global_features(grid, n_objects=2)
        assert feat[22] == 1.0

    def test_transpose_symmetric_square(self):
        grid = np.array([[1, 2],
                         [2, 1]])
        feat = compute_global_features(grid, n_objects=2)
        assert feat[23] == 1.0  # symmetric along diagonal

    def test_transpose_nonsquare(self):
        grid = np.array([[1, 2, 3],
                         [1, 2, 3]])
        feat = compute_global_features(grid, n_objects=3)
        assert feat[23] == 0.0  # not square -> always 0

    def test_asymmetric(self):
        grid = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        feat = compute_global_features(grid, n_objects=9)
        assert feat[20] == 0.0
        assert feat[21] == 0.0
        assert feat[22] == 0.0
        assert feat[23] == 0.0

    def test_all_symmetric_uniform(self):
        grid = np.ones((3, 3), dtype=int)
        feat = compute_global_features(grid, n_objects=1)
        assert feat[20] == 1.0
        assert feat[21] == 1.0
        assert feat[22] == 1.0
        assert feat[23] == 1.0

    def test_near_symmetric_strict(self):
        grid = np.array([[1, 2, 1],
                         [3, 4, 3]])
        grid_off = grid.copy()
        grid_off[0, 2] = 5  # break h-symmetry by one cell
        feat = compute_global_features(grid_off, n_objects=5)
        assert feat[20] == 0.0  # strict, not fuzzy


class TestOtherGlobals:
    def test_distinct_colors(self):
        grid = np.ones((3, 3), dtype=int)
        feat = compute_global_features(grid, n_objects=1)
        assert abs(feat[24] - 1 / 10) < 1e-10

        grid = np.arange(10).reshape(2, 5)
        feat = compute_global_features(grid, n_objects=10)
        assert abs(feat[24] - 1.0) < 1e-10

    def test_object_count(self):
        feat = compute_global_features(np.ones((2, 2), dtype=int), n_objects=5)
        assert abs(feat[25] - 5 / 30) < 1e-10

    def test_aspect_ratio(self):
        grid = np.ones((3, 5), dtype=int)
        feat = compute_global_features(grid, n_objects=1)
        assert abs(feat[26] - 3 / 5) < 1e-10

        grid = np.ones((5, 3), dtype=int)
        feat = compute_global_features(grid, n_objects=1)
        assert abs(feat[26] - 5 / 5) < 1e-10  # h/max(h,w) = 5/5

    def test_fill_ratio(self):
        grid = np.zeros((3, 3), dtype=int)
        grid[0, 0] = 1
        grid[1, 1] = 2
        feat = compute_global_features(grid, n_objects=3)
        # Background=0 (7 cells), non-bg=2 cells
        assert abs(feat[27] - 2 / 9) < 1e-10

    def test_output_range(self):
        grid = np.random.randint(0, 10, size=(5, 5))
        feat = compute_global_features(grid, n_objects=10)
        assert np.all(feat >= 0.0)
        assert np.all(feat <= 1.0)
        assert len(feat) == 28
