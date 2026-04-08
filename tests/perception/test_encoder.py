"""Tests for the full perception encoder pipeline."""

import numpy as np
import pytest
from src.perception.encoder import perceive, sensory_size, decode_colors, FEATURES_PER_CELL, GLOBAL_FEATURES


class TestDimensions:
    def test_3x3(self):
        grid = np.ones((3, 3), dtype=int)
        signal = perceive(grid)
        expected = 9 * FEATURES_PER_CELL + GLOBAL_FEATURES
        assert len(signal) == expected
        assert len(signal) == sensory_size(3, 3)

    def test_5x5(self):
        grid = np.ones((5, 5), dtype=int)
        signal = perceive(grid)
        expected = 25 * FEATURES_PER_CELL + GLOBAL_FEATURES
        assert len(signal) == expected

    def test_1x1(self):
        grid = np.array([[5]])
        signal = perceive(grid)
        expected = 1 * FEATURES_PER_CELL + GLOBAL_FEATURES
        assert len(signal) == expected

    def test_rectangular(self):
        grid = np.ones((3, 7), dtype=int)
        signal = perceive(grid)
        assert len(signal) == sensory_size(3, 7)


class TestSegmentPlacement:
    def test_color_onehot_location(self):
        grid = np.array([[3, 7],
                         [0, 5]])
        signal = perceive(grid)
        F = FEATURES_PER_CELL
        # Cell (0,0) = color 3
        assert signal[3] == 1.0
        assert np.sum(signal[0:10]) == 1.0
        # Cell (0,1) = color 7
        assert signal[F + 7] == 1.0
        assert np.sum(signal[F:F + 10]) == 1.0

    def test_boundary_location(self):
        grid = np.array([[1, 0],
                         [0, 0]])
        signal = perceive(grid)
        # Cell (0,0): ortho boundaries at [10:14], diag at [14:18]
        # Right neighbor (0,1)=0 differs -> d=3 (right)
        assert signal[13] == 1.0  # 0*FEATURES_PER_CELL + 10 + 3

    def test_object_features_location(self):
        grid = np.ones((3, 3), dtype=int)
        signal = perceive(grid)
        base = 4 * FEATURES_PER_CELL
        obj_size = signal[base + 18]
        assert obj_size == 1.0  # uniform grid, one object covering all

    def test_border_flag_location(self):
        grid = np.ones((3, 3), dtype=int)
        signal = perceive(grid)
        F = FEATURES_PER_CELL
        assert signal[4 * F + 26] == 0.0  # interior cell (1,1)
        assert signal[26] == 1.0  # corner cell (0,0)

    def test_global_features_at_end(self):
        grid = np.ones((2, 2), dtype=int)
        signal = perceive(grid)
        n_cells = 4
        global_start = n_cells * FEATURES_PER_CELL
        # Global: color histogram should have color 1 = 1.0
        assert signal[global_start + 1] == 1.0


class TestValueRanges:
    def test_all_in_01(self):
        for _ in range(5):
            h, w = np.random.randint(1, 8, size=2)
            grid = np.random.randint(0, 10, size=(h, w))
            signal = perceive(grid)
            assert np.all(signal >= 0.0), f"Min={signal.min()}"
            assert np.all(signal <= 1.0), f"Max={signal.max()}"

    def test_onehot_exactly_one(self):
        grid = np.random.randint(0, 10, size=(4, 4))
        signal = perceive(grid)
        for i in range(16):
            base = i * FEATURES_PER_CELL
            color_sum = np.sum(signal[base:base + 10])
            assert abs(color_sum - 1.0) < 1e-10


class TestRoundTrip:
    def test_decode_colors(self):
        grid = np.array([[0, 3, 7],
                         [9, 1, 5]])
        signal = perceive(grid)
        recovered = decode_colors(signal, 2, 3)
        np.testing.assert_array_equal(grid, recovered)

    def test_roundtrip_random(self):
        for _ in range(10):
            h, w = np.random.randint(1, 8, size=2)
            grid = np.random.randint(0, 10, size=(h, w))
            signal = perceive(grid)
            recovered = decode_colors(signal, h, w)
            np.testing.assert_array_equal(grid, recovered)
