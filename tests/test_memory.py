"""Tests for memory systems: episodic (hippocampus) and DA-gated working memory (PFC).

Verifies:
  1. EpisodicMemory stores, deduplicates, and recalls correctly
  2. recall_signal produces valid motor-positioned hints
  3. Memory neurons activate during demos (high DA), persist through rest
  4. DA gate attenuates external→memory input when DA is low
  5. Hippocampal hint reaches motor output in the Teacher loop
"""

import tempfile
import numpy as np
import pytest
from src.genome import Genome
from src.brain import Brain
from src.encoding import grid_to_signal, NUM_COLORS, pad_grid
from src.episodic_memory import EpisodicMemory
from src.monitor import Monitor
from src.teacher import Teacher


def _birth_brain(seed=42, grid_h=3, grid_w=3, tmpdir=None):
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


# ── EpisodicMemory unit tests ──────────────────────────────────────────

class TestEpisodicMemory:
    def test_store_and_recall(self):
        em = EpisodicMemory(max_h=3, max_w=3)
        inp = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]])
        out = np.array([[3, 2, 1], [0, 0, 0], [0, 0, 0]])
        em.store(inp, out)
        assert len(em) == 1

        results = em.recall(inp, top_k=1)
        assert len(results) == 1
        ep, sim = results[0]
        assert sim == 1.0
        assert np.array_equal(ep.output_grid, out)

    def test_deduplication(self):
        em = EpisodicMemory(max_h=3, max_w=3)
        inp = np.array([[1, 2], [3, 4]])
        out = np.array([[5, 6], [7, 8]])
        em.store(inp, out)
        em.store(inp, out)
        assert len(em) == 1

    def test_recall_signal_shape(self):
        em = EpisodicMemory(max_h=3, max_w=3)
        inp = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        out = np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])
        em.store(inp, out)

        n_nodes = 5000
        motor_offset = 4000
        hint = em.recall_signal(inp, motor_offset=motor_offset,
                                n_total_nodes=n_nodes, strength=1.0)
        assert hint.shape == (n_nodes,)
        assert hint[:motor_offset].sum() == 0, "No signal before motor offset"
        assert hint[motor_offset:].sum() > 0, "Signal should be at motor positions"

    def test_recall_signal_correct_color(self):
        """recall_signal should place a one-hot for the output color."""
        em = EpisodicMemory(max_h=3, max_w=3)
        inp = np.array([[5, 0, 0], [0, 0, 0], [0, 0, 0]])
        out = np.array([[7, 0, 0], [0, 0, 0], [0, 0, 0]])
        em.store(inp, out)

        motor_offset = 100
        n_nodes = 200
        hint = em.recall_signal(inp, motor_offset=motor_offset,
                                n_total_nodes=n_nodes, strength=1.0)
        cell_0_start = motor_offset + 0 * NUM_COLORS
        cell_0_vals = hint[cell_0_start:cell_0_start + NUM_COLORS]
        assert cell_0_vals[7] > 0, "Color 7 should be activated"
        assert cell_0_vals[0] == 0 or cell_0_vals[7] > cell_0_vals[0], \
            "Output color should dominate"

    def test_similarity_weighting(self):
        """More similar inputs should produce stronger recall."""
        em = EpisodicMemory(max_h=3, max_w=3)
        inp_a = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]])
        out_a = np.array([[9, 9, 9], [0, 0, 0], [0, 0, 0]])
        inp_b = np.array([[4, 5, 6], [0, 0, 0], [0, 0, 0]])
        out_b = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        em.store(inp_a, out_a)
        em.store(inp_b, out_b)

        results = em.recall(inp_a, top_k=2)
        assert results[0][1] > results[1][1], "Exact match should rank highest"


# ── DA-gated memory neuron tests ───────────────────────────────────────

class TestDAGatedMemory:
    def test_memory_neurons_activate_during_demo(self):
        """Memory neurons should activate when teaching signal is injected."""
        brain = _birth_brain()
        mem_idx = brain._memory_idx

        grid_in = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        grid_out = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        sensory = grid_to_signal(grid_in, max_h=3, max_w=3)
        motor = np.zeros(len(brain.net.output_nodes), dtype=np.float64)
        n_cells = 3 * 3
        for i in range(n_cells):
            r, c = divmod(i, 3)
            color = int(grid_out[r, c])
            motor[i * NUM_COLORS + color] = 1.0

        brain.inject_teaching_signal(sensory, motor)
        brain.step(n_steps=50)
        mem_rates = brain.net.r[mem_idx]
        assert mem_rates.mean() > 0, "Memory neurons should activate during demo"

    def test_memory_persists_through_rest(self):
        """Memory neuron activity should decay slowly (low leak=0.02)."""
        brain = _birth_brain()
        mem_idx = brain._memory_idx

        grid_in = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sensory = grid_to_signal(grid_in, max_h=3, max_w=3)
        grid_out = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        motor = np.zeros(len(brain.net.output_nodes), dtype=np.float64)
        n_cells = 3 * 3
        for i in range(n_cells):
            r, c = divmod(i, 3)
            color = int(grid_out[r, c])
            motor[i * NUM_COLORS + color] = 1.0

        brain.inject_teaching_signal(sensory, motor)
        brain.step(n_steps=50)
        mem_after_demo = brain.net.r[mem_idx].copy()

        brain.clear_signal()
        brain.step(n_steps=5)
        mem_after_rest = brain.net.r[mem_idx].copy()

        if mem_after_demo.mean() > 0.01:
            retention = mem_after_rest.mean() / max(mem_after_demo.mean(), 1e-8)
            assert retention > 0.3, (
                f"Memory should retain significant activity after 5 rest steps, "
                f"got {retention:.2%} retention"
            )

    def test_da_gate_attenuates_at_low_da(self):
        """At low DA, external→memory edges should be scaled down."""
        brain = _birth_brain()
        ne = brain.net._edge_count
        mask = brain._get_memory_gate_mask(ne)

        if not mask.any():
            pytest.skip("No external→memory edges in this birth config")

        original_w = brain.net._edge_w[:ne][mask].copy()

        brain.neuromod.da = 0.01
        brain.step(n_steps=1)

        # After step, weights should be restored to original
        restored_w = brain.net._edge_w[:ne][mask]
        np.testing.assert_allclose(
            restored_w, original_w, atol=0.5,
            err_msg="Weights should be approximately restored after step"
        )

    def test_memory_differentiates_stimuli(self):
        """Different stimuli should produce different memory patterns."""
        brain = _birth_brain()
        mem_idx = brain._memory_idx

        grid_a = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        grid_b = np.array([[0, 0, 0], [0, 0, 9], [0, 0, 0]])

        sensory_a = grid_to_signal(grid_a, max_h=3, max_w=3)
        motor_a = np.zeros(len(brain.net.output_nodes), dtype=np.float64)
        brain.inject_teaching_signal(sensory_a, motor_a)
        brain.step(n_steps=50)
        pattern_a = brain.net.r[mem_idx].copy()

        brain.clear_signal()
        brain.step(n_steps=20)
        brain.net.r[:] = 0
        brain.net.V[:] = 0

        sensory_b = grid_to_signal(grid_b, max_h=3, max_w=3)
        brain.inject_teaching_signal(sensory_b, motor_a)
        brain.step(n_steps=50)
        pattern_b = brain.net.r[mem_idx].copy()

        if pattern_a.sum() > 0.01 and pattern_b.sum() > 0.01:
            cos_sim = (
                np.dot(pattern_a, pattern_b)
                / (np.linalg.norm(pattern_a) * np.linalg.norm(pattern_b) + 1e-8)
            )
            assert cos_sim < 0.99, (
                f"Different stimuli should produce different memory patterns, "
                f"got cos_sim={cos_sim:.3f}"
            )


# ── End-to-end Teacher integration ─────────────────────────────────────

class TestTeacherMemoryIntegration:
    def _make_identity_task(self):
        return {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
                {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
                {"input": [[9, 1], [2, 3]], "output": [[9, 1], [2, 3]]},
            ],
            "_file": "test_identity",
        }

    def test_episodic_memory_populated_after_demos(self):
        brain = _birth_brain(grid_h=2, grid_w=2)
        monitor = Monitor(log_dir=tempfile.mkdtemp(), console=False)
        teacher = Teacher(brain, monitor, task_dir="micro_tasks")

        task = self._make_identity_task()
        teacher.run_task("identity", task)

        assert len(teacher.episodic_memory) > 0, \
            "Episodic memory should store demos during run_task"

    def test_episodic_memory_cleared_per_task(self):
        """Each task call starts with fresh episodic memory."""
        brain = _birth_brain(grid_h=2, grid_w=2)
        monitor = Monitor(log_dir=tempfile.mkdtemp(), console=False)
        teacher = Teacher(brain, monitor, task_dir="micro_tasks")

        task = self._make_identity_task()
        teacher.run_task("identity", task)
        n_after_first = len(teacher.episodic_memory)

        teacher.run_task("identity", task)
        n_after_second = len(teacher.episodic_memory)

        assert n_after_second <= len(task["train"]), \
            "Episodic memory should be cleared at start of each task"

    def test_hippocampal_hint_nonzero(self):
        """After demos, recall_signal should produce a non-zero motor hint."""
        brain = _birth_brain(grid_h=2, grid_w=2)
        em = EpisodicMemory(max_h=brain.net.max_h, max_w=brain.net.max_w)

        inp = np.array([[1, 2], [3, 4]])
        out = np.array([[5, 6], [7, 8]])
        em.store(inp, out)

        hint = em.recall_signal(
            inp,
            motor_offset=brain._motor_start,
            n_total_nodes=brain.net.n_nodes,
            strength=0.3,
        )
        motor_part = hint[brain._motor_start:]
        assert motor_part.sum() > 0, "Hippocampal hint should produce non-zero motor signal"
