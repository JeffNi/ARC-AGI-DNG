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
    lateral_decorrelation,
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

        # Pre-WTA firing rates (dendritic calcium analog).
        # The kernel writes here before WTA suppresses r. Used by
        # homeostasis for EMA tracking of actual input drive.
        self._r_pre_wta = np.zeros(net.n_nodes, dtype=np.float64)

        # Homeostasis: continuous self-regulation
        self.stage_manager = stage_manager or StageManager(initial_stage="infancy")
        self.homeostasis = Homeostasis(
            net,
            self.stage_manager.current_setpoints(),
            interval=genome.homeostasis_interval,
            r_pre_wta=self._r_pre_wta,
        )

        self.age = 0  # total steps lived
        self._replay_buffer: list = []  # (free_corr, clamped_corr) for CHL replay
        self._signal_buffer: list = []  # recent sensory signals for spontaneous replay
        self._max_replay = 50

        # Novelty detector: automatic DA from stimulus unfamiliarity.
        # Brain compares incoming signals to a running EMA of recent input.
        n_sensory = len(net.input_nodes)
        self._recent_signal_ema = np.zeros(n_sensory, dtype=np.float64)
        self._novelty_sensitivity = 0.5  # scaled by StageManager
        self._novelty_ema_rate = 0.1

        # Pre-compute region indices
        from ..graph import internal_mask as _internal_mask
        _reg = list(Region)
        self._internal_idx = np.where(_internal_mask(net.regions))[0].astype(np.int64)
        self._memory_idx = np.where(net.regions == _reg.index(Region.MEMORY))[0].astype(np.int64)
        self._motor_start = int(net.output_nodes[0]) if len(net.output_nodes) > 0 else net.n_nodes
        self._n_motor_cells = len(net.output_nodes) // NUM_COLORS if len(net.output_nodes) > 0 else 0

        # Per-layer indices for diagnostics
        self._l1_idx = np.where(net.regions == _reg.index(Region.LOCAL_DETECT))[0].astype(np.int64)
        self._l2_idx = np.where(net.regions == _reg.index(Region.MID_LEVEL))[0].astype(np.int64)
        self._l3_idx = np.where(net.regions == _reg.index(Region.ABSTRACT))[0].astype(np.int64)

        # Per-column WTA pools: L1 and L2 have separate column spaces
        # (L1: 0..n_cells-1, L2: n_cells..2*n_cells-1) so they compete
        # within-layer, not across layers. L3 is non-topographic (col=-1).
        max_col = int(net.column_ids.max()) + 1 if len(net.column_ids) > 0 else 0
        n_cols = max(max_col, net.max_h * net.max_w)
        col_buckets = [[] for _ in range(n_cols)]
        for idx in self._internal_idx:
            col = net.column_ids[idx]
            if col >= 0:
                col_buckets[col].append(int(idx))
        self._col_pool = np.concatenate(
            [np.array(b, dtype=np.int64) for b in col_buckets if b]
            or [np.array([], dtype=np.int64)]
        )
        self._col_sizes = np.array([len(b) for b in col_buckets], dtype=np.int64)
        offsets = np.zeros(n_cols, dtype=np.int64)
        running = 0
        for i in range(n_cols):
            offsets[i] = running
            running += len(col_buckets[i])
        self._col_offsets = offsets
        self._n_cols = n_cols

        # Per-column mean post-WTA rate buffer (written by kernel each step)
        self._col_mean_r = np.zeros(n_cols, dtype=np.float64)
        # Per-node column ID for the kernel to look up column mean
        self._node_col_id = net.column_ids.astype(np.int64)

        # Refractory suppression for cortical + memory neurons.
        self._refractory_mask = np.zeros(net.n_nodes, dtype=np.bool_)
        self._refractory_mask[self._internal_idx] = True
        self._refractory_mask[self._memory_idx] = True

    def inject_signal(self, signal: np.ndarray):
        """
        Inject a sensory signal into the network.

        Automatically computes novelty (how different this signal is from
        recent input) and spikes DA proportionally. This is the basal
        ganglia's novelty/orienting response — no Teacher involvement.
        """
        n_sensory = len(self.net.input_nodes)
        sensory_part = signal[:n_sensory]

        # Novelty = mean absolute difference from recent signal EMA
        novelty = float(np.mean(np.abs(sensory_part - self._recent_signal_ema)))
        self._recent_signal_ema = (
            (1.0 - self._novelty_ema_rate) * self._recent_signal_ema
            + self._novelty_ema_rate * sensory_part
        )

        # DA spike proportional to novelty, capped
        da_spike = min(novelty * self._novelty_sensitivity, 0.4)
        if da_spike > 0.01:
            new_da = self.neuromod.da_baseline + da_spike
            self.neuromod.set_da(new_da)
            self.net.da = new_da

        sig = np.zeros(self.net.n_nodes)
        sig[:n_sensory] = sensory_part
        self._current_signal = sig

    def inject_teaching_signal(
        self,
        sensory_signal: np.ndarray,
        motor_signal: np.ndarray,
    ):
        """
        Inject both input (sensory) and correct output (motor) simultaneously.
        Used during demonstrations: the brain sees the answer being produced
        while observing the input, like a parent guiding a child's hand.
        Novelty DA is computed from the sensory part only.
        """
        n_sensory = len(self.net.input_nodes)
        sensory_part = sensory_signal[:n_sensory]

        # Novelty from sensory input
        novelty = float(np.mean(np.abs(sensory_part - self._recent_signal_ema)))
        self._recent_signal_ema = (
            (1.0 - self._novelty_ema_rate) * self._recent_signal_ema
            + self._novelty_ema_rate * sensory_part
        )
        da_spike = min(novelty * self._novelty_sensitivity, 0.4)
        if da_spike > 0.01:
            new_da = self.neuromod.da_baseline + da_spike
            self.neuromod.set_da(new_da)
            self.net.da = new_da

        sig = np.zeros(self.net.n_nodes)
        sig[:n_sensory] = sensory_part
        # Drive motor neurons with the correct output
        n_motor = len(self.net.output_nodes)
        motor_part = motor_signal[:n_motor]
        motor_start = int(self.net.output_nodes[0])
        sig[motor_start:motor_start + n_motor] = motor_part
        self._current_signal = sig

    def clear_signal(self):
        self._current_signal = None

    def set_da(self, level: float):
        """Teacher sets DA (reward prediction error)."""
        self.neuromod.set_da(level)
        self.net.da = level

    def _normalize_incoming_weights(self, ne: int):
        """Normalize each internal neuron's incoming excitatory weights to a fixed L2 norm.

        Vectorized: computes per-neuron squared norms via np.add.at, then
        rescales all edges in one pass. Only scales DOWN neurons that exceed
        the target norm — never amplifies weak neurons.
        """
        w = self.net._edge_w[:ne]
        dst = self.net._edge_dst[:ne]
        cons = self.net._edge_consolidation[:ne]

        # Only normalize edges from sensory neurons — memory and lateral
        # edges serve different purposes and shouldn't be constrained.
        _reg = list(Region)
        sensory_idx = _reg.index(Region.SENSORY)
        src = self.net._edge_src[:ne]
        src_is_sensory = self.net.regions[src] == sensory_idx

        exc_mask = (w > 0) & (cons < 1.0) & src_is_sensory
        if exc_mask.sum() == 0:
            return

        target_norm_sq = (self.genome.weight_scale * 30.0) ** 2

        sq_sums = np.zeros(self.net.n_nodes, dtype=np.float64)
        np.add.at(sq_sums, dst[exc_mask], w[exc_mask] ** 2)

        needs_scale = sq_sums > target_norm_sq * 1.44  # 1.2^2
        if not np.any(needs_scale):
            return

        scale_factors = np.ones(self.net.n_nodes, dtype=np.float64)
        mask = needs_scale & (sq_sums > 1e-16)
        scale_factors[mask] = np.sqrt(target_norm_sq / sq_sums[mask])

        per_edge_scale = scale_factors[dst]
        w[exc_mask] *= per_edge_scale[exc_mask]
        self.net._csr_dirty = True

    def decorrelate_layers(self, eta: float = 0.005, sim_threshold: float = 0.5):
        """Apply anti-Hebbian lateral decorrelation to each cortical layer.

        Should be called once per stimulus presentation (not every step).
        Pushes co-active neurons' weight vectors apart, preventing the
        representational collapse where all neurons converge to detecting
        the same features.

        Returns total number of repelled pairs across all layers.
        """
        total = 0
        for layer_idx in (self._l1_idx, self._l2_idx, self._l3_idx):
            if len(layer_idx) > 0:
                total += lateral_decorrelation(
                    self.net, layer_idx,
                    eta=eta, sim_threshold=sim_threshold,
                )
        return total

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

        _reg = list(Region)
        edge_plastic = np.ones(ne, dtype=np.bool_)

        # Motor neurons learn through eligibility traces, not continuous Hebbian.
        motor_idx = _reg.index(Region.MOTOR)
        dst_is_motor = self.net.regions[self.net._edge_dst[:ne]] == motor_idx
        edge_plastic[dst_is_motor] = False

        # Update stage manager and sync setpoints
        self.stage_manager.step()
        sp = self.stage_manager.current_setpoints()
        self.homeostasis.setpoints = sp

        # Sync developmental parameters from genetic clock
        self.neuromod.da_baseline = sp.da_baseline
        self._novelty_sensitivity = sp.da_sensitivity
        self.fatigue.threshold = sp.fatigue_threshold

        run_steps_plastic(
            self.net.V, self.net.r, self.net.prev_r, self.net.f,
            self.net.threshold, self.net.leak_rates, self.net.excitability,
            self.net.adaptation,
            self.net._edge_src[:ne], self.net._edge_dst[:ne],
            self.net._edge_w[:ne], ne,
            self.net.inh_scale,
            signal, sp.noise_std,
            self.neuromod.da, self.genome.elig_eta * sp.plasticity_rate, self.genome.w_max,
            self.genome.plasticity_interval,
            self.net.max_rate, self.genome.f_rate,
            self.genome.f_decay, self.genome.f_max,
            self.net.adapt_rate, 0.1,
            self._col_pool, self._col_sizes, self._col_offsets,
            self._n_cols, sp.wta_active_frac,
            self._memory_idx, max(1, len(self._memory_idx) // 3),
            has_signal, n, n_steps,
            noise,
            self._motor_start, self._n_motor_cells, NUM_COLORS,
            edge_plastic,
            self.net._edge_eligibility[:ne], self.genome.elig_decay,
            self._refractory_mask,
            self.genome.eta_local * sp.plasticity_rate,
            self._r_pre_wta,
            self._node_col_id,
            self._col_mean_r,
            self.homeostasis.ema_rate_dendritic,
        )

        # Per-neuron weight normalization for competitive learning.
        # K-H theory requires weight vectors on a unit sphere so competition
        # is based on angular similarity, not raw magnitude. Without this,
        # monopolist neurons accumulate weight and win for all stimuli.
        if sp.plasticity_rate > 0:
            self._normalize_incoming_weights(ne)

        # EMA tracking (continuous) — scaling/intrinsic deferred to sleep
        self.homeostasis.step()

        # Continuous synaptogenesis — growth cones extend during waking activity
        if sp.synaptogenesis_candidates > 0:
            synaptogenesis(
                self.net,
                growth_rate=sp.synaptogenesis_rate,
                n_candidates=sp.synaptogenesis_candidates,
                activity_ema=self.homeostasis.ema_rate,
                rng=self.rng,
            )

        # DA decay toward baseline
        self.neuromod.decay()
        self.net.da = self.neuromod.da

        # Fatigue accumulates proportional to both activity level and duration
        mean_r = float(np.mean(self.net.r))
        self.fatigue.accumulate(mean_r, n_steps)

        self.age += n_steps

    def apply_reward(self, DA: float) -> float:
        """
        Teacher calls this after observing the brain's output.
        Cashes in eligibility traces with the given DA signal.
        Learning rate scaled by the developmental plasticity multiplier.
        """
        self.set_da(DA)
        sp = self.stage_manager.current_setpoints()
        return eligibility_modulated_update(
            self.net, DA=DA,
            eta=self.genome.elig_eta * sp.plasticity_rate,
            w_max=self.genome.w_max,
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
        sp = self.stage_manager.current_setpoints()
        mean_change = contrastive_hebbian_update(
            self.net, free_corr, clamped_corr,
            eta=self.genome.chl_eta * sp.plasticity_rate,
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

        sp = self.stage_manager.current_setpoints()

        stats = nrem_sleep(
            self.net,
            self._replay_buffer,
            chl_eta=self.genome.chl_eta * 0.5 * sp.plasticity_rate,
            shy_downscale=0.97,
            health_decay_rate=sp.health_decay_rate,
            ema_rate=self.homeostasis.ema_rate_dendritic,
            target_rate=sp.target_rate,
        )

        # Homeostatic corrections run during sleep, not waking.
        self.homeostasis.sleep_correction()

        self.fatigue.reset()

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
        from ..stimuli import solid_fill, checkerboard
        from ..encoding import grid_to_signal

        rng = np.random.default_rng(seed)
        net = create_dng(genome, grid_h, grid_w, rng=rng)

        brain = cls(net, genome, rng=rng, checkpoint_dir=checkpoint_dir)

        # Warm up EMA rates so homeostasis starts from realistic values.
        cal_rng = np.random.default_rng(0)
        for gen in [solid_fill, checkerboard]:
            grid = gen(grid_h, grid_w, cal_rng)
            sig = grid_to_signal(grid, max_h=grid_h, max_w=grid_w)
            brain.inject_signal(sig)
            brain.step(n_steps=30)
            brain.clear_signal()
        brain.homeostasis.calibrate_from_rates(brain._r_pre_wta)
        brain.net.V[:] = 0.0
        brain.net.r[:] = 0.0
        brain.net.adaptation[:] = 0.0
        brain.net.f[:] = 0.0
        brain.clear_signal()
        brain.age = 0
        brain.fatigue.reset()
        brain.homeostasis._step_counter = 0

        return brain

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
