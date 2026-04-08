"""
Integration test: end-to-end birth -> task -> learning signals -> checkpoint -> resume.

Tests the full lifecycle pipeline without relying on external task files.
"""

import tempfile
import json
import numpy as np
import pytest

from src.genome import Genome
from src.brain import Brain
from src.encoding import grid_to_signal, NUM_COLORS
from src.monitor import Monitor


def _make_identity_task(h=3, w=3):
    """Create a simple identity task: output = input."""
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 10, size=(h, w)).tolist()
    return {
        "train": [{"input": grid, "output": grid}],
        "_file": "identity_test",
        "_tier": "identity",
    }


class TestFullLifecycle:
    def test_birth_step_read(self):
        """Brain can be born, stepped, and read from."""
        genome = Genome(n_internal=50, n_memory=20, max_h=3, max_w=3, wta_k=5)
        brain = Brain.birth(genome, grid_h=3, grid_w=3, seed=42,
                           checkpoint_dir=tempfile.mkdtemp())

        assert brain.age == 0
        assert brain.net._edge_count > 0

        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        signal = grid_to_signal(grid, max_h=3, max_w=3)
        brain.inject_signal(signal)
        brain.step(n_steps=30)

        output = brain.read_motor(3, 3)
        assert output.shape == (3, 3)
        assert brain.age == 30

    def test_task_produces_learning(self):
        """A task attempt should produce eligibility and weight changes."""
        genome = Genome(n_internal=200, n_memory=20, max_h=5, max_w=5, wta_k=10)
        brain = Brain.birth(genome, grid_h=5, grid_w=5, seed=42,
                           checkpoint_dir=tempfile.mkdtemp())

        # Strong signal + enough steps for internal neurons to activate
        grid = np.ones((5, 5), dtype=int) * 3
        signal = grid_to_signal(grid, max_h=5, max_w=5)
        brain.inject_signal(signal)
        brain.step(n_steps=100)

        # Check eligibility accumulated (on any edge — the mechanism is what matters)
        ne = brain.net._edge_count
        elig_sum = float(np.sum(brain.net._edge_eligibility[:ne]))
        assert elig_sum > 0, "Eligibility should accumulate during observation"

        # Apply reward
        w_before = brain.net._edge_w[:ne].copy()
        change = brain.apply_reward(DA=1.0)
        w_after = brain.net._edge_w[:ne]

        assert not np.array_equal(w_before, w_after), "Weights should change"
        assert change > 0, "Mean change should be positive"

    def test_checkpoint_round_trip(self):
        """Save and resume should preserve full brain state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            genome = Genome(n_internal=50, n_memory=20, max_h=3, max_w=3, wta_k=5)
            brain = Brain.birth(genome, grid_h=3, grid_w=3, seed=42,
                               checkpoint_dir=tmpdir)

            # Run some computation
            grid = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
            signal = grid_to_signal(grid, max_h=3, max_w=3)
            brain.inject_signal(signal)
            brain.step(n_steps=20)

            # Save
            brain.age = 1000
            brain.save(tag="test")

            # Resume
            brain2 = Brain.resume(genome, checkpoint_dir=tmpdir)
            assert brain2.age == 1000
            assert brain2.net.n_nodes == brain.net.n_nodes
            assert brain2.net._edge_count == brain.net._edge_count
            np.testing.assert_array_almost_equal(
                brain2.net._edge_w[:brain.net._edge_count],
                brain.net._edge_w[:brain.net._edge_count],
            )

    def test_da_affects_learning_direction(self):
        """Positive DA should strengthen, negative should weaken."""
        genome = Genome(n_internal=50, n_memory=20, max_h=3, max_w=3, wta_k=5)

        # Trial 1: positive DA — advance to childhood so plasticity_rate > 0
        brain1 = Brain.birth(genome, grid_h=3, grid_w=3, seed=42,
                            checkpoint_dir=tempfile.mkdtemp())
        brain1.stage_manager.transition_to("childhood")
        brain1.stage_manager._transition_progress = 1.0
        grid = np.ones((3, 3), dtype=int) * 3
        signal = grid_to_signal(grid, max_h=3, max_w=3)
        brain1.inject_signal(signal)
        brain1.step(n_steps=20)
        ne = brain1.net._edge_count
        brain1.net._edge_eligibility[:ne] = 0.5
        w_before_pos = brain1.net._edge_w[:ne].copy()
        brain1.apply_reward(DA=1.0)
        w_after_pos = brain1.net._edge_w[:ne]

        # Trial 2: negative DA (same starting state)
        brain2 = Brain.birth(genome, grid_h=3, grid_w=3, seed=42,
                            checkpoint_dir=tempfile.mkdtemp())
        brain2.stage_manager.transition_to("childhood")
        brain2.stage_manager._transition_progress = 1.0
        brain2.inject_signal(signal)
        brain2.step(n_steps=20)
        ne2 = brain2.net._edge_count
        brain2.net._edge_eligibility[:ne2] = 0.5
        w_before_neg = brain2.net._edge_w[:ne2].copy()
        brain2.apply_reward(DA=-0.5)
        w_after_neg = brain2.net._edge_w[:ne2]

        # Excitatory weights: positive DA should increase, negative should decrease
        exc_mask = w_before_pos > 0
        if exc_mask.any():
            mean_delta_pos = float(np.mean(w_after_pos[exc_mask] - w_before_pos[exc_mask]))
            mean_delta_neg = float(np.mean(w_after_neg[exc_mask] - w_before_neg[exc_mask]))
            assert mean_delta_pos > mean_delta_neg, \
                f"Positive DA should produce larger changes: {mean_delta_pos} vs {mean_delta_neg}"

    def test_sleep_cycle(self):
        """Fatigue-triggered sleep should execute without error."""
        genome = Genome(n_internal=50, n_memory=20, max_h=3, max_w=3, wta_k=5)
        brain = Brain.birth(genome, grid_h=3, grid_w=3, seed=42,
                           checkpoint_dir=tempfile.mkdtemp())

        brain.fatigue.level = 100.0
        brain.fatigue.threshold = 50.0

        stats = brain.try_sleep()
        assert stats is not None
        assert brain.fatigue.level < 50.0
        assert "replays" in stats
        assert "pruned" in stats

    def test_monitor_writes_log(self):
        """Monitor should create a log file with valid JSON-lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = Monitor(log_dir=tmpdir, console=False)
            monitor.task_result("identity", "test.json", True, 1.0, 0.001, 100)
            monitor.day_summary(1, 10, 5, 1, 50000, 10.0, 500)

            log_path = monitor.log_path
            assert log_path.exists()

            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            for line in lines:
                record = json.loads(line)
                assert "event" in record
                assert "ts" in record
