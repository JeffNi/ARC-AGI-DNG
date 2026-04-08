"""Tests for Brain engine — checkpoint, signal injection, DA, fatigue, motor readout."""

import tempfile
import numpy as np
import pytest
from src.genome import Genome
from src.brain import Brain, NeuromodState, FatigueTracker


def _birth_brain(seed=42, grid_h=3, grid_w=3, tmpdir=None):
    """Create a small brain for testing."""
    genome = Genome(
        n_internal=50,
        n_memory=20,
        max_h=3,
        max_w=3,
        wta_k=10,
    )
    return Brain.birth(
        genome, grid_h=grid_h, grid_w=grid_w, seed=seed,
        checkpoint_dir=tmpdir or tempfile.mkdtemp(),
    )


class TestBrainBirth:
    def test_creates_network(self):
        brain = _birth_brain()
        assert brain.net.n_nodes > 0
        assert brain.net._edge_count > 0
        assert brain.age == 0

    def test_has_io_nodes(self):
        brain = _birth_brain()
        assert len(brain.net.input_nodes) > 0
        assert len(brain.net.output_nodes) > 0


class TestSignalInjection:
    def test_inject_drives_activity(self):
        brain = _birth_brain()
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        from src.encoding import grid_to_signal
        signal = grid_to_signal(grid, max_h=3, max_w=3)
        brain.inject_signal(signal)
        brain.step(n_steps=20)

        mean_r = float(np.mean(brain.net.r))
        assert mean_r > 0, "Signal injection should drive activity"

    def test_clear_signal(self):
        brain = _birth_brain()
        grid = np.ones((3, 3), dtype=int)
        from src.encoding import grid_to_signal
        signal = grid_to_signal(grid, max_h=3, max_w=3)
        brain.inject_signal(signal)
        brain.step(n_steps=5)
        brain.clear_signal()
        assert brain._current_signal is None


class TestDADecay:
    def test_da_decays_toward_baseline(self):
        brain = _birth_brain()
        brain.set_da(1.0)
        brain.neuromod.da_baseline = 0.05
        brain.step(n_steps=10)
        assert brain.neuromod.da < 1.0, "DA should decay"
        assert brain.neuromod.da > 0.0, "DA shouldn't undershoot"

    def test_da_baseline_rest(self):
        brain = _birth_brain()
        brain.set_da(0.0)
        brain.neuromod.da_baseline = 0.5
        for _ in range(100):
            brain.neuromod.decay()
        assert abs(brain.neuromod.da - 0.5) < 0.1, \
            f"DA should converge to baseline: {brain.neuromod.da}"


class TestFatigue:
    def test_fatigue_accumulates(self):
        brain = _birth_brain()
        grid = np.ones((3, 3), dtype=int)
        from src.encoding import grid_to_signal
        signal = grid_to_signal(grid, max_h=3, max_w=3)
        brain.inject_signal(signal)
        brain.fatigue.rate = 1.0  # fast for testing
        brain.step(n_steps=20)
        assert brain.fatigue.level > 0, "Fatigue should accumulate"

    def test_sleep_resets_fatigue(self):
        brain = _birth_brain()
        brain.fatigue.level = 100.0
        brain.fatigue.threshold = 50.0
        assert brain.fatigue.needs_sleep()
        stats = brain.try_sleep()
        assert stats is not None
        assert brain.fatigue.level < 50.0, "Sleep should reduce fatigue"


class TestMotorReadout:
    def test_read_motor_shape(self):
        brain = _birth_brain(grid_h=3, grid_w=3)
        brain.step(n_steps=10)
        output = brain.read_motor(3, 3)
        assert output.shape == (3, 3)

    def test_read_motor_valid_colors(self):
        brain = _birth_brain(grid_h=3, grid_w=3)
        brain.step(n_steps=10)
        output = brain.read_motor(3, 3)
        assert np.all(output >= 0) and np.all(output < 10)


class TestApplyReward:
    def test_positive_reward_changes_weights(self):
        brain = _birth_brain()
        # Advance to childhood so plasticity_rate > 0 (infancy has no BCM)
        brain.stage_manager.transition_to("childhood")
        brain.stage_manager._transition_progress = 1.0
        grid = np.ones((3, 3), dtype=int)
        from src.encoding import grid_to_signal
        signal = grid_to_signal(grid, max_h=3, max_w=3)
        brain.inject_signal(signal)
        brain.step(n_steps=20)

        # Set some eligibility manually for the test
        ne = brain.net._edge_count
        brain.net._edge_eligibility[:ne] = 0.5

        w_before = brain.net._edge_w[:ne].copy()
        brain.apply_reward(DA=1.0)
        w_after = brain.net._edge_w[:ne]

        assert not np.array_equal(w_before, w_after), "Reward should change weights"

    def test_negative_reward_weakens(self):
        brain = _birth_brain()
        brain.stage_manager.transition_to("childhood")
        brain.stage_manager._transition_progress = 1.0
        ne = brain.net._edge_count
        brain.net._edge_eligibility[:ne] = 0.5
        w_before = brain.net._edge_w[:ne].copy()

        brain.apply_reward(DA=-0.5)

        # Excitatory edges should decrease
        exc_mask = w_before > 0
        if exc_mask.any():
            assert np.mean(brain.net._edge_w[:ne][exc_mask]) < np.mean(w_before[exc_mask])


class TestCheckpointRoundTrip:
    def test_save_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            genome = Genome(n_internal=30, n_memory=10, max_h=3, max_w=3, wta_k=5)
            brain = Brain.birth(genome, grid_h=3, grid_w=3, seed=42, checkpoint_dir=tmpdir)

            # Run some steps to change state
            grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            from src.encoding import grid_to_signal
            signal = grid_to_signal(grid, max_h=3, max_w=3)
            brain.inject_signal(signal)
            brain.step(n_steps=20)
            brain.age = 500
            brain.fatigue.level = 25.0

            brain.save(tag="test")

            # Resume
            brain2 = Brain.resume(genome, checkpoint_dir=tmpdir)
            assert brain2.age == 500
            assert abs(brain2.fatigue.level - 25.0) < 0.1
            assert brain2.net.n_nodes == brain.net.n_nodes
            assert brain2.net._edge_count == brain.net._edge_count

    def test_milestone_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            genome = Genome(n_internal=30, n_memory=10, max_h=3, max_w=3, wta_k=5)
            brain = Brain.birth(genome, grid_h=3, grid_w=3, seed=42, checkpoint_dir=tmpdir)
            path = brain.save_milestone("test_milestone")
            assert path.exists()
            assert "milestone" in path.name
