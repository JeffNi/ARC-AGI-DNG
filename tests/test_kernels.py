"""Tests for numba kernel changes — refractory, eligibility, per-node dynamics."""

import numpy as np
import pytest
from src.graph import DNG, _NTYPE_E, _NTYPE_I
from src.numba_kernels import run_steps_plastic


def _make_tiny_net(n=10):
    """Create a tiny DNG with a few edges for kernel testing."""
    types = np.full(n, _NTYPE_E, dtype=int)
    types[n // 2:] = _NTYPE_I
    regions = np.zeros(n, dtype=int)
    net = DNG(
        n_nodes=n,
        node_types=types,
        regions=regions,
        input_nodes=np.arange(3, dtype=np.int32),
        output_nodes=np.arange(7, 10, dtype=np.int32),
    )
    return net


def _run_one_plastic_step(net, n_steps=1, DA=0.0, signal=None):
    """Helper to run the plastic kernel on a tiny DNG."""
    n = net.n_nodes
    ne = net._edge_count
    if signal is None:
        signal = np.zeros(n)
        has_signal = False
    else:
        has_signal = True

    noise = np.zeros((n_steps, n))
    edge_plastic = np.ones(max(ne, 1), dtype=numba_bool())

    bcm_theta = np.full(n, 0.1, dtype=np.float64)

    # Refractory applies only to INTERNAL/MEMORY; use regions to build mask
    from src.graph import Region
    _reg = list(Region)
    int_idx = _reg.index(Region.INTERNAL)
    mem_idx = _reg.index(Region.MEMORY)
    refractory_mask = (net.regions == int_idx) | (net.regions == mem_idx)

    run_steps_plastic(
        net.V, net.r, net.prev_r, net.f,
        net.threshold, net.leak_rates, net.excitability, net.adaptation,
        net._edge_src[:ne], net._edge_dst[:ne], net._edge_w[:ne], ne,
        net.inh_scale,
        signal, 0.0,
        DA, 0.01, 5.0, bcm_theta, 0,
        net.max_rate, 0.05, 0.02, 3.0,
        net.adapt_rate, 0.1,
        np.array([], dtype=np.int64), 0,
        np.array([], dtype=np.int64), 0,
        has_signal, n, n_steps,
        noise,
        n, 0, 10,  # motor_start=n (past end), n_cells=0 -> no motor WTA
        edge_plastic,
        net._edge_eligibility[:ne], 0.95,
        refractory_mask,
    )


def numba_bool():
    return np.bool_


class TestRefractorySuppression:
    def test_high_rate_suppressed(self):
        from src.graph import Region
        net = _make_tiny_net(10)
        # Mark nodes as INTERNAL so refractory applies
        _reg = list(Region)
        net.regions[:] = _reg.index(Region.INTERNAL)

        # Wire edge: node 0 -> node 1
        net.add_edges_batch(
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1.0]),
        )

        # Set node 0 to have been strongly active
        net.r[0] = 0.9  # above 0.8 * max_rate(1.0)
        net.prev_r[0] = 0.9
        net.V[0] = 1.0

        # Node 2 was moderately active (below threshold)
        net.r[2] = 0.3
        net.prev_r[2] = 0.3
        net.V[2] = 0.4

        _run_one_plastic_step(net, n_steps=1)

        # Node 0 should be strongly suppressed (refractory)
        assert net.r[0] < 0.15, f"Refractory failed: r[0]={net.r[0]}"

    def test_low_rate_not_suppressed(self):
        from src.graph import Region
        net = _make_tiny_net(10)
        net.regions[:] = list(Region).index(Region.INTERNAL)

        net.add_edges_batch(
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1.0]),
        )

        # Set node 0 below refractory threshold
        net.r[0] = 0.3
        net.prev_r[0] = 0.3
        net.V[0] = 0.5

        # Drive node 0 with strong signal
        signal = np.zeros(10)
        signal[0] = 2.0

        _run_one_plastic_step(net, n_steps=1, signal=signal)

        # Should not be suppressed (prev_r < 0.8 * max_rate)
        assert net.r[0] > 0.1, f"Incorrectly suppressed: r[0]={net.r[0]}"

    def test_sensory_neurons_not_refractory(self):
        """Sensory neurons (region=SENSORY) should not be subject to refractory."""
        from src.graph import Region
        net = _make_tiny_net(10)
        # Default regions=0 (SENSORY) — no refractory should apply

        net.add_edges_batch(
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1.0]),
        )

        # Drive node 0 with strong continuous signal
        signal = np.zeros(10)
        signal[0] = 2.0

        _run_one_plastic_step(net, n_steps=20, signal=signal)

        # Sensory neuron should sustain high rate (no refractory)
        assert net.r[0] > 0.8, f"Sensory refractory should not apply: r[0]={net.r[0]}"


class TestEligibilityAccumulation:
    def test_coactive_builds_eligibility(self):
        net = _make_tiny_net(10)
        net.add_edges_batch(
            np.array([0, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.5, 0.3]),
        )

        # Drive both pre and post neurons
        signal = np.zeros(10)
        signal[0] = 1.0
        signal[1] = 1.0

        _run_one_plastic_step(net, n_steps=5, signal=signal)

        # Edge 0->1: both driven, should have eligibility > 0
        assert net._edge_eligibility[0] > 0, "Eligibility should accumulate"

    def test_silent_neurons_no_eligibility(self):
        net = _make_tiny_net(10)
        net.add_edges_batch(
            np.array([0], dtype=np.int32),
            np.array([5], dtype=np.int32),
            np.array([0.5]),
        )

        # No signal, everything silent
        _run_one_plastic_step(net, n_steps=5)

        assert net._edge_eligibility[0] < 1e-6, "Silent neurons shouldn't build eligibility"

    def test_eligibility_decays(self):
        net = _make_tiny_net(10)
        net.add_edges_batch(
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([0.5]),
        )

        # Manually set eligibility to a known value
        net._edge_eligibility[0] = 1.0

        # Run steps with no activity
        _run_one_plastic_step(net, n_steps=10)

        # Should have decayed: 1.0 * 0.95^10 ≈ 0.60
        assert net._edge_eligibility[0] < 0.7, f"Eligibility didn't decay: {net._edge_eligibility[0]}"
        assert net._edge_eligibility[0] > 0.3, f"Eligibility decayed too fast: {net._edge_eligibility[0]}"


class TestPerNodeDynamics:
    def test_different_max_rates(self):
        net = _make_tiny_net(10)
        # E neurons: max_rate=1.0, I neurons: max_rate=1.5
        e_mask = net.node_types == _NTYPE_E
        i_mask = net.node_types == _NTYPE_I
        net.max_rate[e_mask] = 1.0
        net.max_rate[i_mask] = 1.5

        # Drive all neurons to max with strong signal
        signal = np.full(10, 5.0)
        _run_one_plastic_step(net, n_steps=20, signal=signal)

        # E neurons should be capped at ~1.0, I at ~1.5
        # (refractory will reduce them, so check they're roughly in range)
        for i in range(10):
            assert net.r[i] <= net.max_rate[i] + 1e-6, \
                f"Node {i} exceeds max_rate: {net.r[i]} > {net.max_rate[i]}"
