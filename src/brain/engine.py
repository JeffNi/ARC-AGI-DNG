"""
Brain engine — the always-on continuous neural system.

This is the core loop. When the brain is "on", it steps continuously.
The Teacher feeds it stimuli and rewards. DA gates learning.
Sleep happens when fatigue builds up.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..graph import DNG, Region
from ..genome import Genome
from ..encoding import grid_to_signal, signal_to_grid, NUM_COLORS
from ..numba_kernels import run_steps_plastic, wta_motor_cells, accumulate_corr
from ..plasticity import (
    eligibility_modulated_update,
    contrastive_hebbian_update,
    consolidate_synapses,
    get_weight_snapshot,
    synaptogenesis,
)
from ..homeostasis import Homeostasis, HomeostasisSetpoints, StageManager
from .neuromodulators import NeuromodState
from .fatigue import FatigueTracker
from .sleep import nrem_sleep
from .checkpoint import Checkpointer


class Brain:
    """
    Continuous brain engine. Always on until externally stopped.

    The Brain doesn't know about tasks or curricula — it just processes
    signals, learns from DA, and sleeps when tired. The Teacher handles
    everything pedagogical.
    """

    def __init__(
        self,
        net: DNG,
        genome: Genome,
        rng: np.random.Generator | None = None,
        checkpoint_dir: str = "life",
        stage_manager: StageManager | None = None,
    ):
        self.net = net
        self.genome = genome
        self.rng = rng or np.random.default_rng()

        self.neuromod = NeuromodState(
            da_baseline=genome.da_baseline_rest,
            da_decay=genome.da_decay,
        )
        self.fatigue = FatigueTracker(
            rate=genome.fatigue_rate,
            threshold=genome.fatigue_threshold,
            sleep_reset=genome.fatigue_sleep_reset,
        )
        self.checkpointer = Checkpointer(checkpoint_dir=checkpoint_dir)

        # Homeostasis: continuous self-regulation
        self.stage_manager = stage_manager or StageManager(initial_stage="childhood")
        self.homeostasis = Homeostasis(
            net,
            self.stage_manager.current_setpoints(),
            interval=genome.homeostasis_interval,
        )

        self.age = 0  # total steps lived
        self._replay_buffer: list = []  # (free_corr, clamped_corr) for CHL replay
        self._signal_buffer: list = []  # recent sensory signals for spontaneous replay
        self._max_replay = 50

        # Pre-compute region indices
        _reg = list(Region)
        self._internal_idx = np.where(net.regions == _reg.index(Region.INTERNAL))[0].astype(np.int64)
        self._memory_idx = np.where(net.regions == _reg.index(Region.MEMORY))[0].astype(np.int64)
        self._motor_start = int(net.output_nodes[0]) if len(net.output_nodes) > 0 else net.n_nodes
        self._n_motor_cells = len(net.output_nodes) // NUM_COLORS if len(net.output_nodes) > 0 else 0

        # Refractory suppression only for internal + memory neurons.
        # Sensory neurons need sustained rates for signal fidelity;
        # motor neurons need sustained output for readout.
        self._refractory_mask = np.zeros(net.n_nodes, dtype=np.bool_)
        self._refractory_mask[self._internal_idx] = True
        self._refractory_mask[self._memory_idx] = True

    def inject_signal(self, signal: np.ndarray):
        """Inject a sensory signal into the network."""
        n_sensory = len(self.net.input_nodes)
        sig = np.zeros(self.net.n_nodes)
        sig[:n_sensory] = signal[:n_sensory]
        self._current_signal = sig

    def clear_signal(self):
        self._current_signal = None

    def set_da(self, level: float):
        """Teacher sets DA (reward prediction error)."""
        self.neuromod.set_da(level)
        self.net.da = level

    def step(self, n_steps: int = 10):
        """
        Run the brain for n_steps continuous ticks.

        Dynamics, eligibility traces, and continuous Hebbian+DA plasticity
        all happen inside the kernel. DA decays each call. Homeostasis
        runs after the kernel on its own schedule.
        """
        n = self.net.n_nodes
        ne = self.net._edge_count

        if ne == 0:
            self.age += n_steps
            return

        signal = self._current_signal if hasattr(self, '_current_signal') and self._current_signal is not None else np.zeros(n)
        has_signal = hasattr(self, '_current_signal') and self._current_signal is not None

        noise = self.rng.normal(0, 1, (n_steps, n)).astype(np.float64)

        # Motor neurons learn through eligibility traces, not continuous Hebbian.
        # Edges TO motor are non-plastic in the kernel to prevent noisy corruption.
        edge_plastic = np.ones(ne, dtype=np.bool_)
        _reg = list(Region)
        motor_idx = _reg.index(Region.MOTOR)
        dst_is_motor = self.net.regions[self.net._edge_dst[:ne]] == motor_idx
        edge_plastic[dst_is_motor] = False

        # Update stage manager and sync setpoints
        self.stage_manager.step()
        self.homeostasis.setpoints = self.stage_manager.current_setpoints()

        run_steps_plastic(
            self.net.V, self.net.r, self.net.prev_r, self.net.f,
            self.net.threshold, self.net.leak_rates, self.net.excitability,
            self.net.adaptation,
            self.net._edge_src[:ne], self.net._edge_dst[:ne],
            self.net._edge_w[:ne], ne,
            self.net.inh_scale,
            signal, self.genome.noise_std,
            self.neuromod.da, self.genome.elig_eta, self.genome.w_max,
            self.homeostasis.bcm_theta, self.genome.plasticity_interval,
            self.net.max_rate, self.genome.f_rate,
            self.genome.f_decay, self.genome.f_max,
            self.net.adapt_rate, 0.1,
            self._internal_idx, self.genome.wta_k,
            self._memory_idx, max(1, len(self._memory_idx) // 3),
            has_signal, n, n_steps,
            noise,
            self._motor_start, self._n_motor_cells, NUM_COLORS,
            edge_plastic,
            self.net._edge_eligibility[:ne], self.genome.elig_decay,
            self._refractory_mask,
        )

        # Continuous homeostasis — the core stabilizer
        self.homeostasis.step()

        # DA decay toward baseline
        self.neuromod.decay()
        self.net.da = self.neuromod.da

        # Fatigue accumulates from mean activity
        mean_r = float(np.mean(self.net.r))
        self.fatigue.accumulate(mean_r)

        self.age += n_steps

    def apply_reward(self, DA: float) -> float:
        """
        Teacher calls this after observing the brain's output.
        Cashes in eligibility traces with the given DA signal.
        """
        self.set_da(DA)
        return eligibility_modulated_update(
            self.net, DA=DA, eta=self.genome.elig_eta, w_max=self.genome.w_max,
        )

    def snapshot_correlations(self, n_steps: int = 20) -> np.ndarray:
        """
        Run the brain for n_steps while accumulating pre*post correlations
        on every edge. Returns the mean correlation vector (one value per edge).

        Used for CHL: capture correlations during the "free" phase (brain's
        own output) and "clamped" phase (correct answer shown), then compare.
        """
        ne = self.net._edge_count
        corr = np.zeros(ne, dtype=np.float64)
        for _ in range(n_steps):
            self.step(n_steps=1)
            accumulate_corr(
                self.net.r,
                self.net._edge_src[:ne],
                self.net._edge_dst[:ne],
                corr, ne,
            )
        corr /= max(n_steps, 1)
        return corr

    def apply_chl(self, free_corr: np.ndarray, clamped_corr: np.ndarray) -> float:
        """
        Apply Contrastive Hebbian Learning: adjust weights based on the
        difference between clamped (correct) and free (brain's guess)
        correlation patterns.

        Also stores the pair for sleep replay.
        """
        from ..plasticity import contrastive_hebbian_update
        mean_change = contrastive_hebbian_update(
            self.net, free_corr, clamped_corr,
            eta=self.genome.chl_eta,
        )
        self.store_replay(free_corr, clamped_corr)
        return mean_change

    def read_motor(self, h: int, w: int) -> np.ndarray:
        """Read the brain's output: decode motor firing rates to a grid."""
        motor_r = self.net.r[self.net.output_nodes]
        return signal_to_grid(
            motor_r, h, w,
            max_h=self.net.max_h, max_w=self.net.max_w,
        )

    def try_sleep(self) -> dict | None:
        """Sleep if fatigued. Returns sleep stats or None."""
        if not self.fatigue.needs_sleep():
            return None

        stats = nrem_sleep(
            self.net,
            self._replay_buffer,
            chl_eta=self.genome.chl_eta * 0.5,
            shy_downscale=0.97,
        )
        self.fatigue.reset()

        # Synaptogenesis after sleep
        n_grown = synaptogenesis(self.net, growth_rate=0.3, rng=self.rng)
        stats["grown"] = n_grown

        return stats

    def store_replay(self, free_corr: np.ndarray, clamped_corr: np.ndarray):
        """Store correlation pair for sleep replay."""
        self._replay_buffer.append((free_corr.copy(), clamped_corr.copy()))
        if len(self._replay_buffer) > self._max_replay:
            self._replay_buffer.pop(0)

    def store_signal(self, signal: np.ndarray):
        """Store a sensory signal for spontaneous replay during rest."""
        self._signal_buffer.append(signal.copy())
        if len(self._signal_buffer) > self._max_replay:
            self._signal_buffer.pop(0)

    def spontaneous_replay(self, n_steps: int = 30, strength: float = 0.3):
        """
        During rest, replay a recent sensory memory at low intensity.
        Mimics the brain's tendency to "rehearse" recent experiences
        between tasks. Helps consolidate without explicit teaching.
        """
        if not self._signal_buffer:
            self.step(n_steps=n_steps)
            return

        # Pick a random recent memory
        idx = self.rng.integers(0, len(self._signal_buffer))
        memory = self._signal_buffer[idx] * strength
        self.inject_signal(memory)
        self.step(n_steps=n_steps)
        self.clear_signal()

    def save(self, tag: str = "auto") -> Path:
        return self.checkpointer.save(self, tag)

    def save_milestone(self, label: str) -> Path:
        return self.checkpointer.save_milestone(self, label)

    @classmethod
    def birth(
        cls,
        genome: Genome,
        grid_h: int = 5,
        grid_w: int = 5,
        seed: int | None = None,
        checkpoint_dir: str = "life",
    ) -> "Brain":
        """Create a new brain from scratch."""
        from ..template import create_dng
        rng = np.random.default_rng(seed)
        net = create_dng(genome, grid_h, grid_w, rng=rng)
        return cls(net, genome, rng=rng, checkpoint_dir=checkpoint_dir)

    @classmethod
    def resume(
        cls,
        genome: Genome,
        checkpoint_dir: str = "life",
    ) -> "Brain":
        """Resume from the latest checkpoint."""
        checkpointer = Checkpointer(checkpoint_dir=checkpoint_dir)
        ckpt = checkpointer.find_latest()
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

        net = DNG.load(str(ckpt / "network.npz"))
        with open(ckpt / "brain_meta.json") as f:
            meta = json.load(f)

        rng_state = meta.get("rng_state")
        rng = np.random.default_rng()
        if rng_state:
            rng.bit_generator.state = rng_state

        # Restore stage manager
        stage_mgr = StageManager()
        if "stage_manager" in meta:
            stage_mgr.load_state_dict(meta["stage_manager"])

        brain = cls(net, genome, rng=rng, checkpoint_dir=checkpoint_dir,
                     stage_manager=stage_mgr)
        brain.age = meta.get("age", 0)
        brain.neuromod = NeuromodState.from_dict(meta.get("neuromod", {}))
        brain.fatigue = FatigueTracker.from_dict(meta.get("fatigue", {}))

        # Restore homeostasis state
        if "homeostasis" in meta:
            brain.homeostasis.load_state_dict(meta["homeostasis"])

        return brain
