"""Tests for DNG graph modifications — eligibility, per-node params, save/load."""

import tempfile
import numpy as np
import pytest
from src.graph import DNG, NodeType, Region, _NTYPE_E, _NTYPE_I


def _make_small_dng(n=20):
    """Create a small DNG for testing."""
    types = np.full(n, _NTYPE_E, dtype=int)
    types[n // 2:] = _NTYPE_I
    regions = np.zeros(n, dtype=int)
    net = DNG(
        n_nodes=n,
        node_types=types,
        regions=regions,
        input_nodes=np.arange(5, dtype=np.int32),
        output_nodes=np.arange(15, 20, dtype=np.int32),
    )
    return net


class TestEligibilityArray:
    def test_exists_at_creation(self):
        net = _make_small_dng()
        assert net._edge_eligibility is not None
        assert len(net._edge_eligibility) >= net._edge_capacity

    def test_all_zeros_initially(self):
        net = _make_small_dng()
        assert np.all(net._edge_eligibility[:net._edge_count] == 0.0)

    def test_grows_with_edges(self):
        net = _make_small_dng()
        srcs = np.array([0, 1, 2], dtype=np.int32)
        dsts = np.array([3, 4, 5], dtype=np.int32)
        ws = np.array([0.1, 0.2, 0.3])
        net.add_edges_batch(srcs, dsts, ws)
        assert net._edge_eligibility is not None
        assert len(net._edge_eligibility) >= net._edge_count

    def test_new_edges_have_zero_eligibility(self):
        net = _make_small_dng()
        net.add_edges_batch(
            np.array([0, 1], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
            np.array([0.1, 0.2]),
        )
        n = net._edge_count
        assert np.all(net._edge_eligibility[:n] == 0.0)

    def test_survives_compact(self):
        net = _make_small_dng()
        net.add_edges_batch(
            np.array([0, 1, 2], dtype=np.int32),
            np.array([3, 4, 5], dtype=np.int32),
            np.array([0.1, 0.0, 0.3]),  # middle edge has w=0 (dead)
        )
        net._edge_eligibility[0] = 0.5
        net._edge_eligibility[2] = 0.7
        net.compact()
        assert net._edge_count == 2
        assert net._edge_eligibility[0] == 0.5
        assert net._edge_eligibility[1] == 0.7


class TestPerNodeParams:
    def test_max_rate_is_array(self):
        net = _make_small_dng(20)
        assert isinstance(net.max_rate, np.ndarray)
        assert len(net.max_rate) == 20

    def test_adapt_rate_is_array(self):
        net = _make_small_dng(20)
        assert isinstance(net.adapt_rate, np.ndarray)
        assert len(net.adapt_rate) == 20

    def test_default_values(self):
        net = _make_small_dng(20)
        # By default, all nodes get the same value
        assert np.all(net.max_rate == 1.0)
        assert np.all(net.adapt_rate == 0.01)

    def test_different_e_i_values(self):
        net = _make_small_dng(20)
        e_mask = net.node_types == _NTYPE_E
        i_mask = net.node_types == _NTYPE_I
        # Set different values per type
        net.max_rate[e_mask] = 1.0
        net.max_rate[i_mask] = 1.5
        net.adapt_rate[e_mask] = 0.01
        net.adapt_rate[i_mask] = 0.005
        # Verify they stuck
        assert np.all(net.max_rate[e_mask] == 1.0)
        assert np.all(net.max_rate[i_mask] == 1.5)
        assert np.all(net.adapt_rate[e_mask] == 0.01)
        assert np.all(net.adapt_rate[i_mask] == 0.005)


class TestSaveLoadRoundTrip:
    def test_basic_roundtrip(self):
        net = _make_small_dng(20)
        # Add some edges with known values
        srcs = np.array([0, 1, 2, 3], dtype=np.int32)
        dsts = np.array([10, 11, 12, 13], dtype=np.int32)
        ws = np.array([0.1, -0.2, 0.3, -0.4])
        net.add_edges_batch(srcs, dsts, ws)

        # Set some state
        net._edge_eligibility[0] = 0.99
        net._edge_eligibility[2] = 0.55
        net.max_rate[:10] = 1.0
        net.max_rate[10:] = 1.5
        net.adapt_rate[:10] = 0.01
        net.adapt_rate[10:] = 0.005
        net.da = 0.5
        net.da_baseline = 0.3
        net.ne = 0.1

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            net.save(f.name)
            loaded = DNG.load(f.name)

        assert loaded.n_nodes == net.n_nodes
        assert loaded._edge_count == net._edge_count
        np.testing.assert_array_equal(loaded._edge_w[:4], ws)
        np.testing.assert_array_almost_equal(
            loaded._edge_eligibility[:4],
            net._edge_eligibility[:4],
        )
        np.testing.assert_array_almost_equal(loaded.max_rate, net.max_rate)
        np.testing.assert_array_almost_equal(loaded.adapt_rate, net.adapt_rate)
        assert loaded.da == net.da
        assert loaded.da_baseline == net.da_baseline
        assert loaded.ne == net.ne

    def test_metadata_preserved(self):
        net = _make_small_dng(20)
        net.max_h = 5
        net.max_w = 5
        net.inh_scale = 1.3

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            net.save(f.name)
            loaded = DNG.load(f.name)

        assert loaded.max_h == 5
        assert loaded.max_w == 5
        assert loaded.inh_scale == 1.3
        np.testing.assert_array_equal(loaded.node_types, net.node_types)
        np.testing.assert_array_equal(loaded.regions, net.regions)
