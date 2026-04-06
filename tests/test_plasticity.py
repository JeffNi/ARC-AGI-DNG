"""Tests for plasticity rules — eligibility, DA, CHL, Dale's law, consolidation."""

import numpy as np
import pytest
from src.graph import DNG, _NTYPE_E, _NTYPE_I
from src.plasticity import (
    eligibility_modulated_update,
    contrastive_hebbian_update,
    consolidate_synapses,
    get_weight_snapshot,
)


def _make_test_net():
    """Create a DNG with known edge structure for plasticity testing."""
    n = 10
    types = np.full(n, _NTYPE_E, dtype=int)
    types[5:8] = _NTYPE_I
    regions = np.zeros(n, dtype=int)
    net = DNG(
        n_nodes=n,
        node_types=types,
        regions=regions,
        input_nodes=np.arange(3, dtype=np.int32),
        output_nodes=np.arange(8, 10, dtype=np.int32),
    )
    # 4 edges: 2 excitatory (E->E), 2 inhibitory (I->E)
    srcs = np.array([0, 1, 5, 6], dtype=np.int32)
    dsts = np.array([3, 4, 3, 4], dtype=np.int32)
    ws = np.array([0.5, 0.3, -0.5, -0.3])
    net.add_edges_batch(srcs, dsts, ws)
    return net


class TestNegativeDAWeakens:
    """CRITICAL: Past failure mode — negative DA must WEAKEN eligible synapses."""

    def test_negative_da_weakens_excitatory(self):
        net = _make_test_net()
        net._edge_eligibility[:4] = np.array([1.0, 1.0, 1.0, 1.0])
        w_before = net._edge_w[:4].copy()

        eligibility_modulated_update(net, DA=-0.5, eta=0.1, w_max=5.0)

        # Excitatory edges (indices 0,1): should DECREASE
        assert net._edge_w[0] < w_before[0], \
            f"Excitatory edge should weaken: {w_before[0]} -> {net._edge_w[0]}"
        assert net._edge_w[1] < w_before[1]

    def test_negative_da_weakens_inhibitory(self):
        net = _make_test_net()
        net._edge_eligibility[:4] = np.array([1.0, 1.0, 1.0, 1.0])
        w_before = net._edge_w[:4].copy()

        eligibility_modulated_update(net, DA=-0.5, eta=0.1, w_max=5.0)

        # Inhibitory edges (indices 2,3): "weakening" = less negative
        assert net._edge_w[2] > w_before[2], \
            f"Inhibitory edge should become less negative: {w_before[2]} -> {net._edge_w[2]}"
        assert net._edge_w[3] > w_before[3]


class TestPositiveDAStrengthens:
    def test_positive_da_strengthens_excitatory(self):
        net = _make_test_net()
        net._edge_eligibility[:4] = np.array([1.0, 1.0, 1.0, 1.0])
        w_before = net._edge_w[:4].copy()

        eligibility_modulated_update(net, DA=0.5, eta=0.1, w_max=5.0)

        assert net._edge_w[0] > w_before[0]
        assert net._edge_w[1] > w_before[1]

    def test_positive_da_strengthens_inhibitory(self):
        net = _make_test_net()
        net._edge_eligibility[:4] = np.array([1.0, 1.0, 1.0, 1.0])
        w_before = net._edge_w[:4].copy()

        eligibility_modulated_update(net, DA=0.5, eta=0.1, w_max=5.0)

        # "Strengthening" inhibition = more negative
        assert net._edge_w[2] < w_before[2]
        assert net._edge_w[3] < w_before[3]


class TestZeroDA:
    def test_zero_da_no_change(self):
        net = _make_test_net()
        net._edge_eligibility[:4] = np.array([1.0, 1.0, 1.0, 1.0])
        w_before = net._edge_w[:4].copy()

        eligibility_modulated_update(net, DA=0.0, eta=0.1, w_max=5.0)

        np.testing.assert_array_almost_equal(net._edge_w[:4], w_before)


class TestEligibilityRequired:
    def test_no_eligibility_no_change(self):
        net = _make_test_net()
        # Zero eligibility
        net._edge_eligibility[:4] = 0.0
        w_before = net._edge_w[:4].copy()

        eligibility_modulated_update(net, DA=1.0, eta=0.1, w_max=5.0)

        np.testing.assert_array_almost_equal(net._edge_w[:4], w_before)


class TestDalesLaw:
    def test_excitatory_stays_positive(self):
        net = _make_test_net()
        net._edge_eligibility[:4] = np.array([1.0, 1.0, 1.0, 1.0])

        # Strong negative DA to try to push excitatory below 0
        eligibility_modulated_update(net, DA=-10.0, eta=1.0, w_max=5.0)

        # Excitatory edges must remain positive
        assert net._edge_w[0] > 0, f"Excitatory went negative: {net._edge_w[0]}"
        assert net._edge_w[1] > 0

    def test_inhibitory_stays_negative(self):
        net = _make_test_net()
        net._edge_eligibility[:4] = np.array([1.0, 1.0, 1.0, 1.0])

        # Strong positive DA to try to push inhibitory above 0
        eligibility_modulated_update(net, DA=10.0, eta=1.0, w_max=5.0)

        # Inhibitory edges must remain negative
        assert net._edge_w[2] < 0, f"Inhibitory went positive: {net._edge_w[2]}"
        assert net._edge_w[3] < 0


class TestCHLErrorCorrection:
    def test_chl_strengthens_correct(self):
        net = _make_test_net()
        net.da = 0.5

        # Free phase: wrong correlations (low)
        free = np.zeros(4)
        free[0] = 0.1

        # Clamped phase: correct correlations (high)
        clamped = np.zeros(4)
        clamped[0] = 0.9

        w_before = net._edge_w[0]
        contrastive_hebbian_update(net, free, clamped, eta=0.1, w_max=5.0)

        # Edge 0 should be strengthened (clamped > free, positive DA)
        assert net._edge_w[0] > w_before

    def test_chl_weakens_wrong(self):
        net = _make_test_net()
        net.da = 0.5

        # Free phase: wrong pattern is strong
        free = np.zeros(4)
        free[1] = 0.9

        # Clamped phase: that correlation is weak
        clamped = np.zeros(4)
        clamped[1] = 0.1

        w_before = net._edge_w[1]
        contrastive_hebbian_update(net, free, clamped, eta=0.1, w_max=5.0)

        # Edge 1 should be weakened (free > clamped)
        assert net._edge_w[1] < w_before


class TestAllEdgesPlastic:
    """Regression test: we previously excluded sensory edges from plasticity."""

    def test_no_region_exclusion(self):
        from src.graph import Region
        _reg = list(Region)
        n = 20
        types = np.full(n, _NTYPE_E, dtype=int)
        regions = np.zeros(n, dtype=int)
        regions[:5] = _reg.index(Region.SENSORY)
        regions[5:15] = _reg.index(Region.INTERNAL)
        regions[15:] = _reg.index(Region.MOTOR)

        net = DNG(
            n_nodes=n,
            node_types=types,
            regions=regions,
            input_nodes=np.arange(5, dtype=np.int32),
            output_nodes=np.arange(15, 20, dtype=np.int32),
        )
        # Sensory -> Internal edge
        net.add_edges_batch(
            np.array([2], dtype=np.int32),
            np.array([7], dtype=np.int32),
            np.array([0.5]),
        )
        net._edge_eligibility[0] = 1.0

        w_before = net._edge_w[0]
        eligibility_modulated_update(net, DA=0.5, eta=0.1, w_max=5.0)

        assert net._edge_w[0] != w_before, "Sensory->Internal edge must be plastic"


class TestConsolidationProtects:
    def test_consolidated_changes_less(self):
        net = _make_test_net()
        net._edge_eligibility[:4] = np.array([1.0, 1.0, 1.0, 1.0])

        # Consolidate edge 0 heavily
        net._edge_consolidation[0] = 10.0  # strong consolidation
        net._edge_consolidation[1] = 0.0   # no consolidation

        w_before = net._edge_w[:2].copy()
        eligibility_modulated_update(net, DA=0.5, eta=0.1, w_max=5.0)

        change_0 = abs(net._edge_w[0] - w_before[0])
        change_1 = abs(net._edge_w[1] - w_before[1])

        assert change_0 < change_1, \
            f"Consolidated edge changed more: {change_0} vs {change_1}"
