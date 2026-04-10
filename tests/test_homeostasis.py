"""
Tests for the continuous homeostasis system.

Unit tests:
  - Synaptic scaling preserves weight ratios
  - Synaptic scaling respects Dale's law
  - Consolidated edges resist scaling
  - BCM theta tracks activity
  - Intrinsic plasticity adjusts excitability
  - E/I balance adjusts inh_scale
  - Stage transitions are smooth

Integration tests:
  - Identity tasks survive reward events with homeostasis active
"""

import tempfile
import numpy as np
import pytest

from src.graph import DNG, _NTYPE_E, _NTYPE_I
from src.genome import Genome
from src.brain import Brain
from src.encoding import grid_to_signal
from src.homeostasis import Homeostasis, HomeostasisSetpoints, StageManager, STAGES
from src.homeostasis.scaling import synaptic_scaling
from src.homeostasis.intrinsic import intrinsic_plasticity
from src.homeostasis.ei_balance import ei_balance_update


def _make_test_net(n=20, n_edges=40):
    """Create a small DNG with known edge structure."""
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

    rng = np.random.default_rng(42)
    for _ in range(n_edges):
        s, d = rng.integers(0, n, size=2)
        while s == d:
            d = rng.integers(0, n)
        w = rng.normal(0, 0.5)
        if types[s] == _NTYPE_I:
            w = -abs(w)
        else:
            w = abs(w) + 0.01
        net.add_edge(s, d, w)

    return net


class TestSynapticScaling:
    def test_ratio_preservation(self):
        """After scaling, ratio between incoming weights to a neuron should be ~preserved."""
        net = _make_test_net()
        ne = net._edge_count

        # Find a neuron with multiple incoming edges
        dst = net._edge_dst[:ne]
        counts = np.bincount(dst, minlength=net.n_nodes)
        target = int(np.argmax(counts))
        incoming = np.where(dst == target)[0]
        assert len(incoming) >= 2, "Need a neuron with at least 2 incoming edges"

        w_before = net._edge_w[incoming].copy()
        ratio_before = w_before[0] / (w_before[1] + 1e-10)

        # Simulate high activity on target
        ema = np.zeros(net.n_nodes)
        ema[target] = 0.5  # well above default target of 0.15
        sp = HomeostasisSetpoints(scaling_gain=0.05)

        synaptic_scaling(net, ema, sp)

        w_after = net._edge_w[incoming]
        ratio_after = w_after[0] / (w_after[1] + 1e-10)

        # Ratios should be approximately preserved (multiplicative scaling)
        assert abs(ratio_before - ratio_after) < abs(ratio_before) * 0.2, \
            f"Ratio changed too much: {ratio_before:.4f} -> {ratio_after:.4f}"

    def test_dale_law_preserved(self):
        """Scaling should never flip edge signs."""
        net = _make_test_net()
        ne = net._edge_count
        w_before = net._edge_w[:ne].copy()

        ema = np.full(net.n_nodes, 0.3)
        sp = HomeostasisSetpoints(scaling_gain=0.1)

        synaptic_scaling(net, ema, sp)

        w_after = net._edge_w[:ne]
        pos_before = w_before > 0
        neg_before = w_before < 0

        assert np.all(w_after[pos_before] > 0), "Positive edges flipped sign"
        assert np.all(w_after[neg_before] < 0), "Negative edges flipped sign"

    def test_consolidated_edges_resist(self):
        """Edges with high consolidation should change less than unconsolidated."""
        net = _make_test_net()
        ne = net._edge_count

        # Set first half consolidated, second half not
        net._edge_consolidation[:ne // 2] = 50.0
        net._edge_consolidation[ne // 2:ne] = 0.0

        w_before = net._edge_w[:ne].copy()

        ema = np.full(net.n_nodes, 0.5)
        sp = HomeostasisSetpoints(
            scaling_gain=0.1,
            scaling_protect_consolidated=True,
        )

        synaptic_scaling(net, ema, sp)

        w_after = net._edge_w[:ne]
        change_cons = np.mean(np.abs(w_after[:ne // 2] - w_before[:ne // 2]))
        change_free = np.mean(np.abs(w_after[ne // 2:ne] - w_before[ne // 2:ne]))

        assert change_cons < change_free, \
            f"Consolidated should change less: {change_cons:.6f} >= {change_free:.6f}"

    def test_overactive_scaled_down(self):
        """Neurons above target rate should have incoming weights scaled down."""
        net = _make_test_net()
        ne = net._edge_count

        ema = np.full(net.n_nodes, 0.3)  # above target
        sp = HomeostasisSetpoints(target_rate=0.15, scaling_gain=0.02)

        w_before_abs = np.abs(net._edge_w[:ne].copy())

        synaptic_scaling(net, ema, sp)

        w_after_abs = np.abs(net._edge_w[:ne])
        assert np.mean(w_after_abs) < np.mean(w_before_abs), \
            "Overactive neurons should have weights scaled down"

    def test_underactive_scaled_up(self):
        """Neurons below target rate should have incoming weights scaled up."""
        net = _make_test_net()
        ne = net._edge_count

        ema = np.full(net.n_nodes, 0.05)  # below target
        sp = HomeostasisSetpoints(target_rate=0.15, scaling_gain=0.02)

        w_before_abs = np.abs(net._edge_w[:ne].copy())

        synaptic_scaling(net, ema, sp)

        w_after_abs = np.abs(net._edge_w[:ne])
        assert np.mean(w_after_abs) > np.mean(w_before_abs), \
            "Underactive neurons should have weights scaled up"


class TestIntrinsicPlasticity:
    def test_overactive_reduces_excitability(self):
        """Neurons above target should have excitability reduced."""
        net = _make_test_net()
        net.excitability[:] = 1.0
        ema = np.full(net.n_nodes, 0.3)
        sp = HomeostasisSetpoints(target_rate=0.15, intrinsic_eta=0.01)

        intrinsic_plasticity(net, ema, sp)

        assert np.mean(net.excitability) < 1.0, "Excitability should decrease"

    def test_underactive_increases_excitability(self):
        """Neurons below target should have excitability increased."""
        net = _make_test_net()
        net.excitability[:] = 1.0
        ema = np.full(net.n_nodes, 0.05)
        sp = HomeostasisSetpoints(target_rate=0.15, intrinsic_eta=0.01)

        intrinsic_plasticity(net, ema, sp)

        assert np.mean(net.excitability) > 1.0, "Excitability should increase"

    def test_excitability_clipped(self):
        """Excitability should stay within bounds."""
        net = _make_test_net()
        net.excitability[:] = 0.05  # below min
        ema = np.full(net.n_nodes, 0.5)
        sp = HomeostasisSetpoints(intrinsic_eta=1.0, intrinsic_min=0.1, intrinsic_max=5.0)

        intrinsic_plasticity(net, ema, sp)

        assert np.all(net.excitability >= 0.1)
        assert np.all(net.excitability <= 5.0)


class TestEIBalance:
    def test_adjusts_inh_scale(self):
        """E/I balance should adjust inh_scale."""
        net = _make_test_net()
        net.inh_scale = 1.0
        net.r[:] = 0.2

        sp = HomeostasisSetpoints(ei_target_ratio=0.8, ei_adjustment_rate=0.01)

        old_scale = net.inh_scale
        ei_balance_update(net, sp)

        assert net.inh_scale != old_scale, "inh_scale should change"

    def test_inh_scale_bounded(self):
        """inh_scale should stay within [0.5, 3.0]."""
        net = _make_test_net()
        net.r[:] = 0.2
        sp = HomeostasisSetpoints(ei_adjustment_rate=10.0)

        for _ in range(100):
            ei_balance_update(net, sp)

        assert 0.5 <= net.inh_scale <= 3.0


class TestStageManager:
    def test_initial_stage(self):
        mgr = StageManager(initial_stage="infancy")
        sp = mgr.current_setpoints()
        assert sp.target_rate == STAGES["infancy"].target_rate

    def test_transition_is_gradual(self):
        mgr = StageManager(initial_stage="infancy", transition_tau=100)
        mgr.transition_to("childhood")

        assert mgr.is_transitioning

        # At t=tau (100 steps), exponential approach gives ~63%
        for _ in range(100):
            mgr.step()

        sp = mgr.current_setpoints()
        infancy_rate = STAGES["infancy"].target_rate
        childhood_rate = STAGES["childhood"].target_rate
        expected_at_tau = infancy_rate + 0.632 * (childhood_rate - infancy_rate)

        assert abs(sp.target_rate - expected_at_tau) < 0.02, \
            f"At t=tau, rate should be ~{expected_at_tau:.3f}: got {sp.target_rate:.3f}"

    def test_transition_completes(self):
        mgr = StageManager(initial_stage="infancy", transition_tau=100)
        mgr.transition_to("childhood")

        # At ~5.3*tau, progress > 0.995 → snaps to 1.0
        for _ in range(600):
            mgr.step()

        assert not mgr.is_transitioning
        sp = mgr.current_setpoints()
        assert sp.target_rate == STAGES["childhood"].target_rate

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError):
            StageManager(initial_stage="nonexistent")

    def test_state_dict_round_trip(self):
        mgr = StageManager(initial_stage="childhood", transition_tau=200)
        mgr.transition_to("adolescence")
        for _ in range(50):
            mgr.step()

        state = mgr.state_dict()

        mgr2 = StageManager()
        mgr2.load_state_dict(state)

        assert mgr2.current_stage == "adolescence"
        assert abs(mgr2._transition_progress - mgr._transition_progress) < 0.01


class TestHomeostasisOrchestrator:
    def test_ema_updates_every_step(self):
        """EMA should update every step, not just at intervals."""
        net = _make_test_net()
        net.r[:] = 0.5
        h = Homeostasis(net, HomeostasisSetpoints(), interval=100)

        h.step()
        assert h.ema_rate.mean() > 0, "EMA should update after one step"

    def test_mechanisms_run_at_interval(self):
        """Scaling/intrinsic/EI should run at the specified interval."""
        from src.graph import Region
        net = _make_test_net()
        # Mark all nodes as INTERNAL so they're regulated
        _reg = list(Region)
        net.regions[:] = _reg.index(Region.INTERNAL)
        net.r[:] = 0.3
        sp = HomeostasisSetpoints(scaling_gain=0.05, intrinsic_eta=0.01)
        h = Homeostasis(net, sp, interval=5)

        excit_before = net.excitability.copy()

        for _ in range(4):
            h.step()

        # After 4 steps, should NOT have run yet
        assert np.array_equal(net.excitability, excit_before), \
            "Should not run mechanisms before interval"

        h.step()  # 5th step: should fire

        assert not np.array_equal(net.excitability, excit_before), \
            "Mechanisms should run at interval"

    def test_state_dict_round_trip(self):
        net = _make_test_net()
        net.r[:] = 0.2
        h = Homeostasis(net, HomeostasisSetpoints(), interval=10)

        for _ in range(15):
            h.step()

        state = h.state_dict()

        h2 = Homeostasis(net, HomeostasisSetpoints(), interval=10)
        h2.load_state_dict(state)

        np.testing.assert_array_almost_equal(h2.ema_rate, h.ema_rate)
        assert h2._step_counter == h._step_counter


class TestSetpointsInterpolation:
    def test_interpolate_at_zero(self):
        a = STAGES["infancy"]
        b = STAGES["childhood"]
        result = a.interpolate(b, 0.0)
        assert result.target_rate == a.target_rate

    def test_interpolate_at_one(self):
        a = STAGES["infancy"]
        b = STAGES["childhood"]
        result = a.interpolate(b, 1.0)
        assert result.target_rate == b.target_rate

    def test_interpolate_midpoint(self):
        a = STAGES["infancy"]
        b = STAGES["childhood"]
        result = a.interpolate(b, 0.5)
        expected = (a.target_rate + b.target_rate) / 2
        assert abs(result.target_rate - expected) < 1e-6


class TestIdentitySurvivesReward:
    """Integration test: run identity tasks with DA, verify brain still works."""

    def test_identity_stable_after_rewards(self):
        """After several reward events, the brain should still solve identity."""
        genome = Genome(
            n_internal=50, n_memory=20,
            max_h=3, max_w=3, wta_k=5,
            homeostasis_interval=10,
        )
        brain = Brain.birth(
            genome, grid_h=3, grid_w=3, seed=42,
            checkpoint_dir=tempfile.mkdtemp(),
        )

        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        signal = grid_to_signal(grid, max_h=3, max_w=3)

        # Run 10 rounds of: observe -> attempt -> reward -> homeostasis catches up
        for trial in range(10):
            brain.inject_signal(signal)
            brain.step(n_steps=50)

            output = brain.read_motor(3, 3)

            # Alternate positive and negative rewards
            da = 0.3 if (output == grid).all() else -0.05
            brain.apply_reward(DA=da)
            brain.step(n_steps=20)
            brain.clear_signal()
            brain.step(n_steps=10)

        # Final test: does identity still work?
        brain.inject_signal(signal)
        brain.step(n_steps=50)
        final_output = brain.read_motor(3, 3)

        # The copy pathway should still dominate
        accuracy = (final_output == grid).mean()
        assert accuracy >= 0.5, \
            f"Identity accuracy collapsed to {accuracy:.1%} after rewards"

    def test_homeostasis_prevents_runaway(self):
        """Large DA spikes should not cause runaway activity."""
        genome = Genome(
            n_internal=50, n_memory=20,
            max_h=3, max_w=3, wta_k=5,
            homeostasis_interval=5,
        )
        brain = Brain.birth(
            genome, grid_h=3, grid_w=3, seed=42,
            checkpoint_dir=tempfile.mkdtemp(),
        )

        grid = np.ones((3, 3), dtype=int) * 3
        signal = grid_to_signal(grid, max_h=3, max_w=3)

        brain.inject_signal(signal)
        brain.step(n_steps=30)

        # Large reward spike
        brain.apply_reward(DA=5.0)
        brain.step(n_steps=100)

        mean_r = float(np.mean(brain.net.r))
        assert mean_r < 1.0, f"Activity should be bounded: mean_r={mean_r:.3f}"
        assert np.all(np.isfinite(brain.net._edge_w[:brain.net._edge_count])), \
            "Weights should remain finite"
