"""Tests for per-cell local features (edge detection, neighbors, borders)."""

import numpy as np
import pytest
from src.perception.features import compute_local_features


class TestOneHotColor:
    def test_uniform_grid(self):
        grid = np.full((3, 3), 1, dtype=int)
        feat = compute_local_features(grid)
        for r in range(3):
            for c in range(3):
                assert feat[r, c, 1] == 1.0
                assert feat[r, c, 0] == 0.0
                assert np.sum(feat[r, c, :10]) == 1.0

    def test_all_colors(self):
        grid = np.arange(10).reshape(2, 5)
        feat = compute_local_features(grid)
        for i in range(10):
            r, c = divmod(i, 5)
            assert feat[r, c, i] == 1.0
            assert np.sum(feat[r, c, :10]) == 1.0


class TestBoundarySignals:
    def test_uniform_no_boundaries(self):
        grid = np.ones((3, 3), dtype=int)
        feat = compute_local_features(grid)
        assert np.all(feat[:, :, 10:18] == 0.0)

    def test_checkerboard_all_boundaries(self):
        grid = np.array([[0, 1, 0, 1],
                         [1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [1, 0, 1, 0]])
        feat = compute_local_features(grid)
        for r in range(4):
            for c in range(4):
                # Every in-bounds neighbor differs
                for d in range(4):
                    dr = [-1, 1, 0, 0][d]
                    dc = [0, 0, -1, 1][d]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 4 and 0 <= nc < 4:
                        assert feat[r, c, 10 + d] == 1.0

    def test_single_red_in_blue(self):
        grid = np.ones((3, 3), dtype=int)  # all blue (1)
        grid[1, 1] = 2  # red center
        feat = compute_local_features(grid)
        # The red cell: all 4 ortho neighbors differ
        for d in range(4):
            assert feat[1, 1, 10 + d] == 1.0
        # Cell (0,1) has boundary toward (1,1) which is direction down (d=1)
        assert feat[0, 1, 11] == 1.0  # down neighbor differs
        assert feat[0, 1, 10] == 0.0  # up neighbor out of bounds -> 0

    def test_horizontal_stripe(self):
        grid = np.array([[2, 2, 2],
                         [1, 1, 1],
                         [1, 1, 1]])
        feat = compute_local_features(grid)
        # Row 0: boundary toward row 1 (down, d=1)
        for c in range(3):
            assert feat[0, c, 11] == 1.0  # down differs
        # Row 1: boundary toward row 0 (up, d=0) but not toward row 2
        for c in range(3):
            assert feat[1, c, 10] == 1.0  # up differs
            assert feat[1, c, 11] == 0.0  # down same

    def test_ortho_and_diag_separate(self):
        grid = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]])
        feat = compute_local_features(grid)
        # Center cell (0): all ortho and diag boundaries = 1
        assert np.all(feat[1, 1, 10:14] == 1.0)
        assert np.all(feat[1, 1, 14:18] == 1.0)
        # Corner cell (0,0): its diag neighbor (1,1) differs
        assert feat[0, 0, 17] == 1.0  # bottom-right diag


class TestNeighborFraction:
    def test_uniform_interior(self):
        grid = np.ones((3, 3), dtype=int)
        feat = compute_local_features(grid)
        # Interior cell: 4/4 neighbors same
        assert feat[1, 1, 27] == 1.0

    def test_uniform_corner(self):
        grid = np.ones((3, 3), dtype=int)
        feat = compute_local_features(grid)
        # Corner: 2 neighbors, both same -> 2/2 = 1.0
        assert feat[0, 0, 27] == 1.0

    def test_isolated_cell(self):
        grid = np.array([[0, 1, 0],
                         [1, 5, 1],
                         [0, 1, 0]])
        feat = compute_local_features(grid)
        # Center: 4 neighbors, all differ -> 0/4 = 0.0
        assert feat[1, 1, 27] == 0.0

    def test_corner_partial(self):
        grid = np.array([[1, 1, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
        feat = compute_local_features(grid)
        # (0,0): neighbors (0,1)=1 same, (1,0)=1 same -> 2/2 = 1.0
        assert feat[0, 0, 27] == 1.0
        # (0,1): neighbors (0,0)=1 same, (0,2)=0 diff, (1,1)=0 diff -> 1/3
        assert abs(feat[0, 1, 27] - 1 / 3) < 1e-10


class TestBorderFlag:
    def test_3x3_border(self):
        grid = np.ones((3, 3), dtype=int)
        feat = compute_local_features(grid)
        # All edge cells = 1.0
        for r in range(3):
            for c in range(3):
                expected = 1.0 if (r in (0, 2) or c in (0, 2)) else 0.0
                assert feat[r, c, 26] == expected, f"({r},{c})"

    def test_1x1_grid(self):
        grid = np.array([[5]])
        feat = compute_local_features(grid)
        assert feat[0, 0, 26] == 1.0

    def test_5x5_interior(self):
        grid = np.ones((5, 5), dtype=int)
        feat = compute_local_features(grid)
        # Interior cells
        for r in range(1, 4):
            for c in range(1, 4):
                assert feat[r, c, 26] == 0.0


class TestEdgeCases:
    def test_1x1_grid(self):
        grid = np.array([[3]])
        feat = compute_local_features(grid)
        assert feat.shape == (1, 1, 28)
        assert feat[0, 0, 3] == 1.0  # color 3
        assert np.all(feat[0, 0, 10:18] == 0.0)  # no neighbors
        assert feat[0, 0, 26] == 1.0  # border
        assert feat[0, 0, 27] == 0.0  # no neighbors -> 0/0 handled as 0

    def test_1xN_grid(self):
        grid = np.array([[0, 1, 0]])
        feat = compute_local_features(grid)
        assert feat.shape == (1, 3, 28)
        # Middle cell: left/right neighbors exist
        assert feat[0, 1, 12] == 1.0  # left neighbor differs
        assert feat[0, 1, 13] == 1.0  # right neighbor differs

    def test_Nx1_grid(self):
        grid = np.array([[0], [1], [0]])
        feat = compute_local_features(grid)
        assert feat.shape == (3, 1, 28)
        assert feat[1, 0, 10] == 1.0  # up differs
        assert feat[1, 0, 11] == 1.0  # down differs

    def test_all_10_colors(self):
        grid = np.arange(10).reshape(2, 5)
        feat = compute_local_features(grid)
        assert feat.shape == (2, 5, 28)
        for i in range(10):
            r, c = divmod(i, 5)
            assert np.sum(feat[r, c, :10]) == 1.0

    def test_output_range(self):
        grid = np.random.randint(0, 10, size=(5, 5))
        feat = compute_local_features(grid)
        assert np.all(feat >= 0.0)
        assert np.all(feat <= 1.0)
