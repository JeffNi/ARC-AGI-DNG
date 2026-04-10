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
    eligibility_modulated_update_percell,
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
        self.birth_edges = net._edge_count  # snapshot at birth for density regulation
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

        # Region indices for edge masking
        self._sensory_reg = _reg.index(Region.SENSORY)
        self._motor_reg = _reg.index(Region.MOTOR)

        # Per-layer indices for diagnostics
        self._l1_idx = np.where(net.regions == _reg.index(Region.LOCAL_DETECT))[0].astype(np.int64)
        self._l2_idx = np.where(net.regions == _reg.index(Region.MID_LEVEL))[0].astype(np.int64)
        self._l3_idx = np.where(net.regions == _reg.index(Region.ABSTRACT))[0].astype(np.int64)

        # Per-column WTA pools: L1 and L2 have separate column spaces
        # (L1: 0..n_cells-1, L2: n_cells..2*n_cells-1) so they compete
        # within-layer, not across layers.
        #
        # L3 is non-topographic — assign all L3 neurons to a single virtual
        # column so the K-H kernel computes a proper plasticity threshold
        # (h_star) for them instead of defaulting to 0.
        max_col = int(net.column_ids.max()) + 1 if len(net.column_ids) > 0 else 0
        l3_virtual_col = max(max_col, net.max_h * net.max_w)
        n_cols = l3_virtual_col + 1

        # Override L3 column IDs from -1 to the virtual column
        self._node_col_id = net.column_ids.astype(np.int64).copy()
        for idx in self._l3_idx:
            self._node_col_id[idx] = l3_virtual_col

        col_buckets = [[] for _ in range(n_cols)]
        for idx in self._internal_idx:
            col = self._node_col_id[idx]
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

        # Refractory suppression for cortical + memory neurons.
        self._refractory_mask = np.zeros(net.n_nodes, dtype=np.bool_)
        self._refractory_mask[self._internal_idx] = True
        self._refractory_mask[self._memory_idx] = True

    def inject_signal(self, signal: np.ndarray):
        """
        Inject a sensory signal into the network.

        Resets sensory V/r before applying the new signal so the retinal
        input immediately represents the new stimulus. Without this,
        residual sensory state from the previous stimulus persists and
        corrupts the copy pathway's color selection.

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

        # Clear sensory residuals — new retinal input replaces old
        self.net.V[:n_sensory] = 0.0
        self.net.r[:n_sensory] = 0.0

        # Boost color channel signal to dominate L1→sensory feedback.
        # L1 feedback grows to ±2-5 via K-H plasticity during infancy.
        # Color channels at 1.0 get overwhelmed. At 5.0, retinal input
        # always wins (like real thalamocortical drive to V1).
        from ..perception.encoder import FEATURES_PER_CELL as _FPC
        boosted = sensory_part.copy()
        n_cells = self.net.max_h * self.net.max_w
        for c in range(n_cells):
            base = c * _FPC
            if base + NUM_COLORS <= n_sensory:
                boosted[base:base + NUM_COLORS] *= 5.0

        sig = np.zeros(self.net.n_nodes)
        sig[:n_sensory] = boosted
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

        # Boost color channels (same as inject_signal)
        from ..perception.encoder import FEATURES_PER_CELL as _FPC
        boosted = sensory_part.copy()
        n_cells = self.net.max_h * self.net.max_w
        for c in range(n_cells):
            base = c * _FPC
            if base + NUM_COLORS <= n_sensory:
                boosted[base:base + NUM_COLORS] *= 5.0

        sig = np.zeros(self.net.n_nodes)
        sig[:n_sensory] = boosted
        # Drive motor neurons with the correct output
        n_motor = len(self.net.output_nodes)
        motor_part = motor_signal[:n_motor]
        motor_start = int(self.net.output_nodes[0])
        sig[motor_start:motor_start + n_motor] = motor_part
        self._current_signal = sig

    def add_motor_hint(self, motor_signal: np.ndarray, strength: float = 0.3):
        """Blend a soft motor bias into the current signal (hippocampal recall)."""
        if self._current_signal is None:
            return
        n_motor = len(self.net.output_nodes)
        self._current_signal[self._motor_start:self._motor_start + n_motor] += (
            motor_signal[:n_motor] * strength
        )

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

    def decorrelate_layers(self, eta: float = 0.01, sim_threshold: float = 0.4):
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

    def _get_memory_gate_mask(self, ne: int) -> np.ndarray:
        """Cached mask: external→memory edges (excludes memory→memory self-loops).

        Used by DA-gated memory write: these edges are scaled by DA level
        so high DA (demos) opens the gate and low DA (attempt/rest) lets
        the attractor maintain its state.
        """
        cached = getattr(self, '_memory_gate_cache', None)
        if cached is not None and len(cached) == ne:
            return cached
        dst = self.net._edge_dst[:ne]
        src = self.net._edge_src[:ne]
        mem_reg = list(Region).index(Region.MEMORY)
        dst_is_mem = self.net.regions[dst] == mem_reg
        src_is_mem = self.net.regions[src] == mem_reg
        mask = dst_is_mem & ~src_is_mem
        self._memory_gate_cache = mask
        return mask

    def _get_edge_plastic(self, ne: int) -> np.ndarray:
        """Cached boolean mask: True for edges eligible for Hebbian plasticity.

        Motor-targeted edges learn via eligibility traces only, so they
        are excluded here.  Invalidated when edges are added/removed.
        """
        cached = getattr(self, '_edge_plastic_cache', None)
        if cached is not None and len(cached) == ne:
            return cached
        mask = np.ones(ne, dtype=np.bool_)
        motor_idx = list(Region).index(Region.MOTOR)
        mask[self.net.regions[self.net._edge_dst[:ne]] == motor_idx] = False
        self._edge_plastic_cache = mask
        return mask

    def step(self, n_steps: int = 10, learn: bool = True,
             clamp_sensory: bool = False):
        """
        Run the brain for n_steps continuous ticks.

        Dynamics, eligibility traces, and continuous Hebbian+DA plasticity
        all happen inside the kernel. DA decays each call. Homeostasis
        runs after the kernel on its own schedule.

        When learn=False, network dynamics run normally (activation
        propagation, WTA) but no weight updates occur. Used during
        motor feedback phases — analogous to corollary discharge
        suppressing sensory plasticity during self-generated actions.

        When clamp_sensory=True, runs single-step kernel calls with
        sensory color channels re-clamped to signal values after each
        step. Prevents L1→sensory feedback from corrupting bottom-up
        color perception. Only used during task attempts where motor
        accuracy matters — NOT during infancy/mimicry where L1→sensory
        feedback is part of the learning process.
        """
        n = self.net.n_nodes
        ne = self.net._edge_count

        if ne == 0:
            self.age += n_steps
            return

        signal = self._current_signal if hasattr(self, '_current_signal') and self._current_signal is not None else np.zeros(n)
        has_signal = hasattr(self, '_current_signal') and self._current_signal is not None

        noise = self.rng.normal(0, 1, (n_steps, n)).astype(np.float64)

        if learn:
            edge_plastic = self._get_edge_plastic(ne)
        else:
            edge_plastic = np.zeros(ne, dtype=np.bool_)

        # Update stage manager and sync setpoints
        self.stage_manager.step(n_steps)
        sp = self.stage_manager.current_setpoints()
        self.homeostasis.setpoints = sp

        # Sync developmental parameters from genetic clock
        self.neuromod.da_baseline = sp.da_baseline
        self._novelty_sensitivity = sp.da_sensitivity
        self.fatigue.threshold = sp.fatigue_threshold

        # DA-gated memory write (O'Reilly & Frank 2006): scale external→memory
        # edge weights by DA so demos (high DA) encode while attempt/rest
        # (low DA) lets the attractor maintain its state.
        gate_threshold = 0.15
        gate_min = 0.15
        mem_gate_mask = self._get_memory_gate_mask(ne)
        da_gate = min(1.0, self.neuromod.da / gate_threshold)
        effective_gate = gate_min + da_gate * (1.0 - gate_min)
        saved_mem_w = None
        if mem_gate_mask.any() and effective_gate < 0.99:
            saved_mem_w = self.net._edge_w[:ne][mem_gate_mask].copy()
            self.net._edge_w[:ne][mem_gate_mask] *= effective_gate
            self.net._csr_dirty = True

        # Build CSR once per step() for the parallel matmul path.
        # get_weight_matrix() caches and only rebuilds when edges change.
        W_csr = self.net.get_weight_matrix()
        csr_indptr = W_csr.indptr
        csr_indices = W_csr.indices
        csr_data = W_csr.data

        _kernel_args = (
            self.net.V, self.net.r, self.net.prev_r, self.net.f,
            self.net.threshold, self.net.leak_rates, self.net.excitability,
            self.net.adaptation,
            self.net._edge_src[:ne], self.net._edge_dst[:ne],
            self.net._edge_w[:ne], ne,
            self.net.inh_scale,
            signal, sp.noise_std,
            self.neuromod.da, self.genome.elig_eta * sp.plasticity_rate,
            self.genome.w_max, self.genome.plasticity_interval,
            self.net.max_rate, self.genome.f_rate,
            self.genome.f_decay, self.genome.f_max,
            self.net.adapt_rate, 0.1,
            self._col_pool, self._col_sizes, self._col_offsets,
            self._n_cols, sp.wta_active_frac,
            self._memory_idx, max(1, len(self._memory_idx) // 3),
        )

        if clamp_sensory and has_signal:
            from ..perception.encoder import FEATURES_PER_CELL as _FPC
            n_sensory = len(self.net.input_nodes)
            n_cells = self.net.max_h * self.net.max_w
            color_idx = np.concatenate([
                np.arange(c * _FPC, c * _FPC + NUM_COLORS)
                for c in range(n_cells)
                if c * _FPC + NUM_COLORS <= n_sensory
            ])
            color_vals = signal[color_idx]

            for s_i in range(n_steps):
                run_steps_plastic(
                    *_kernel_args,
                    True, n, 1,
                    noise[s_i:s_i+1],
                    self._motor_start, self._n_motor_cells, NUM_COLORS,
                    edge_plastic,
                    self.net._edge_eligibility[:ne], self.genome.elig_decay,
                    self._refractory_mask,
                    self.genome.eta_local * sp.plasticity_rate,
                    self._r_pre_wta,
                    self._node_col_id,
                    self._col_mean_r,
                    self.homeostasis.ema_rate_dendritic,
                    csr_indptr, csr_indices, csr_data,
                )
                self.net.V[color_idx] = color_vals
                self.net.r[color_idx] = color_vals
        else:
            run_steps_plastic(
                *_kernel_args,
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
                csr_indptr, csr_indices, csr_data,
            )

        # Restore DA-gated memory weights after kernel.
        if saved_mem_w is not None:
            self.net._edge_w[:ne][mem_gate_mask] = saved_mem_w
            self.net._csr_dirty = True

        # Per-neuron weight normalization for competitive learning.
        if learn and sp.plasticity_rate > 0:
            self._normalize_incoming_weights(ne)
            self.net._csr_dirty = True

        # EMA tracking (continuous) — scaling/intrinsic deferred to sleep
        self.homeostasis.step()

        # Continuous synaptogenesis — modulated by density ceiling + activity demand.
        # Stage setpoints provide the base/max rate; actual rate scales down
        # automatically as the network fills up and neurons become well-served.
        if sp.synaptogenesis_candidates > 0 and self.birth_edges > 0:
            growth_ratio = self.net._edge_count / self.birth_edges
            peak = self.genome.peak_growth_target
            density_factor = max(0.0, 1.0 - growth_ratio / peak)

            ema_d = self.homeostasis.ema_rate_dendritic
            demand_factor = float((ema_d < sp.target_rate).mean())

            eff_rate = sp.synaptogenesis_rate * density_factor * demand_factor
            eff_cand = max(1, int(sp.synaptogenesis_candidates * density_factor))

            if eff_rate > 0.001:
                # Motor cortex matures later than sensory cortex — block
                # new synapses targeting motor neurons until childhood.
                stage = self.stage_manager.current_stage
                motor_ok = stage not in ("infancy", "late_infancy")
                old_ne = self.net._edge_count
                synaptogenesis(
                    self.net,
                    growth_rate=eff_rate,
                    n_candidates=eff_cand,
                    activity_ema=ema_d,
                    rng=self.rng,
                    allow_motor_target=motor_ok,
                )
                if self.net._edge_count != old_ne:
                    self._edge_plastic_cache = None
                    self._memory_gate_cache = None
                    self._sensory_motor_cache = None

        # Copy pathway maintenance: re-pin instinct weights every step.
        # Without this, SHY sleep downscaling erodes them nightly.
        # When copy_strength=1.0 (infancy), weights stay at 2.0.
        # When copy_strength decays (childhood), weights decay with it.
        copy_mask = self.net._edge_consolidation[:ne] >= 20.0
        if copy_mask.any():
            target_w = 5.0 * sp.copy_strength
            self.net._edge_w[:ne][copy_mask] = target_w
            self.net._csr_dirty = True

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

    def apply_cell_reward(self, cell_rewards: np.ndarray) -> float:
        """
        Per-cell motor reward — topographic DA analog.

        cell_rewards: array of shape (n_motor_cells,) where positive = correct,
        negative = wrong. Each cell's NUM_COLORS motor neurons receive the
        same DA value. Non-motor nodes get zero DA.

        Uses 1/10th the kernel's learning rate. The kernel's Hebbian+DA
        plasticity runs continuously at low effective DA and is balanced
        by homeostasis. This per-cell reward is a concentrated pulse —
        full eta would let a single trial make random internal->motor
        edges as strong as the copy pathway.
        """
        da_per_node = np.zeros(self.net.n_nodes, dtype=np.float64)
        n_cells = min(len(cell_rewards), self._n_motor_cells)
        for c in range(n_cells):
            start = self._motor_start + c * NUM_COLORS
            da_per_node[start:start + NUM_COLORS] = cell_rewards[c]
        sp = self.stage_manager.current_setpoints()
        reward_eta = self.genome.elig_eta * sp.plasticity_rate * 0.01
        return eligibility_modulated_update_percell(
            self.net, da_per_node,
            eta=reward_eta,
            w_max=self.genome.w_max,
        )

    def _get_sensory_motor_mask(self, ne: int) -> np.ndarray:
        """Cached mask: learnable 1:1 copy-topology edges only.

        Selects sensory→motor edges where the source is a color channel
        (not a spatial/boundary feature) mapping to the SAME cell and
        SAME color in motor. Excludes the consolidated copy pathway
        (re-pinned) and diffuse random sensory→motor edges.
        """
        cached = getattr(self, '_sensory_motor_cache', None)
        if cached is not None and len(cached) == ne:
            return cached

        from ..perception.encoder import FEATURES_PER_CELL

        regions = self.net.regions
        src = self.net._edge_src[:ne]
        dst = self.net._edge_dst[:ne]

        is_sensory_motor = (
            (regions[src] == self._sensory_reg)
            & (regions[dst] == self._motor_reg)
        )
        # Exclude consolidated copy pathway
        copy = self.net._edge_consolidation[:ne] >= 20.0
        is_sensory_motor = is_sensory_motor & ~copy

        mask = np.zeros(ne, dtype=np.bool_)
        sm_idx = np.where(is_sensory_motor)[0]
        if len(sm_idx) > 0:
            s = src[sm_idx]
            d = dst[sm_idx]
            src_cell = s // FEATURES_PER_CELL
            src_feat = s % FEATURES_PER_CELL
            dst_cell = (d - self._motor_start) // NUM_COLORS
            dst_color = (d - self._motor_start) % NUM_COLORS
            # 1:1 topology: same cell, feature IS the matching color
            topo_match = (src_cell == dst_cell) & (src_feat < NUM_COLORS) & (src_feat == dst_color)
            mask[sm_idx[topo_match]] = True

        self._sensory_motor_cache = mask
        return mask

    def apply_mimicry_reward(self, da: float = 0.05) -> float:
        """Cash in eligibility traces ONLY on sensory→motor edges.

        Called at the end of mimicry: the copy pathway produced the correct
        output, sensory and motor nodes were co-active, so traces on the
        correct sensory→motor edges are strong. Cashing them in teaches the
        direct sensory→motor pathway to reproduce identity without the copy
        pathway.

        Restricted to sensory→motor to avoid the color-specificity problem
        with L1→Motor edges (which connect to all color channels and create
        statistical biases toward frequent colors).
        """
        ne = self.net._edge_count
        if ne == 0:
            return 0.0
        mask = self._get_sensory_motor_mask(ne)
        if not mask.any():
            return 0.0

        elig = self.net._edge_eligibility[:ne]
        active = mask & (elig > 1e-6)
        if not active.any():
            return 0.0

        sp = self.stage_manager.current_setpoints()
        eta = self.genome.elig_eta * sp.plasticity_rate

        w = self.net._edge_w[:ne]
        dw = np.zeros(ne, dtype=np.float64)
        dw[active] = eta * da * elig[active]

        new_w = w.copy()
        new_w[active] = np.clip(w[active] + dw[active], 1e-6, self.genome.w_max)

        change = np.abs(new_w - w)
        self.net._edge_w[:ne] = new_w
        self.net._edge_tag[:ne] += change
        self.net._csr_dirty = True

        # Partial decay of cashed traces
        self.net._edge_eligibility[:ne][active] *= 0.5

        return float(change[active].mean()) if active.any() else 0.0

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

    # ------------------------------------------------------------------
    #  Motor babbling & mimicry
    # ------------------------------------------------------------------

    def motor_babble(self, noise_std: float = 0.5,
                     observe_steps: int = 20) -> np.ndarray:
        """Random motor exploration with dual-channel feedback.

        Phase A — inject noise, let motor WTA pick winners.
        Phase B — proprioceptive: hold motor at output, let feedback
                  edges propagate the signal internally (efference copy).
        Phase C — visual: decode motor -> grid -> full perception ->
                  inject as sensory.  The network "sees" what it did.
        Phase D — rest.
        """
        motor_nodes = self.net.output_nodes

        # Phase A: random activation
        self.net.V[motor_nodes] += self.rng.normal(0, noise_std, len(motor_nodes))
        self.step(n_steps=5)

        motor_rates = self.net.r[motor_nodes].copy()
        grid = self.read_motor(self.net.max_h, self.net.max_w)

        # Phase B: proprioceptive feedback (efference copy)
        # learn=False: corollary discharge suppresses sensory plasticity
        saved_v = self.net.V[motor_nodes].copy()
        self.net.r[motor_nodes] = motor_rates
        self.step(n_steps=5, learn=False)
        self.net.V[motor_nodes] = saved_v

        # Phase C: visual feedback through full perception
        # learn=False: L1 propagates but doesn't learn from self-generated input
        feedback = grid_to_signal(grid, max_h=self.net.max_h,
                                  max_w=self.net.max_w)
        self.inject_signal(feedback)
        self.step(n_steps=observe_steps, learn=False)

        # Phase D: rest
        self.clear_signal()
        self.step(n_steps=5)
        return grid

    def stimulus_with_feedback(self, signal: np.ndarray,
                               observe_steps: int = 25) -> np.ndarray:
        """Mimicry: show a stimulus, let copy pathway fire, feed back the
        motor output so the network sees what it produced.

        The copy pathway fires within ~3 steps and produces the correct
        output. We capture that output quickly, then clamp motor to the
        correct state while L1/L2 continue activating. This ensures
        Hebbian co-activation pairs the right sensory features with the
        right motor output — like a reflexive motor program the infant
        can't override yet.
        """
        motor_nodes = self.net.output_nodes

        # Phase 1: Quick copy — let copy pathway fire (5 steps is enough)
        self.inject_signal(signal)
        self.step(n_steps=5)

        # Capture the copy pathway's output before internal pathways corrupt it
        motor_rates = self.net.r[motor_nodes].copy()
        grid = self.read_motor(self.net.max_h, self.net.max_w)

        # Phase 2: Clamp motor to correct output while sensory features
        # develop. Eligibility traces accumulate from co-activation of
        # sensory color nodes and correct motor output.
        remaining = max(1, observe_steps - 5)
        for _ in range(remaining):
            self.net.r[motor_nodes] = motor_rates
            self.step(n_steps=1)

        # Mimicry reward: cash in eligibility traces on sensory→motor edges
        # NOW, while traces are fresh (elig_decay=0.85 means they vanish
        # within ~30 steps). The copy pathway guaranteed correct co-activation
        # between sensory color nodes and matching motor color nodes.
        self.apply_mimicry_reward(da=1.0)

        # Proprioceptive phase: hold motor at output state
        # learn=False: corollary discharge during self-generated feedback
        self.net.r[motor_nodes] = motor_rates
        self.step(n_steps=5, learn=False)

        # Visual feedback: see own output through full perception
        # learn=False: don't corrupt L1 features with self-generated input
        feedback = grid_to_signal(grid, max_h=self.net.max_h,
                                  max_w=self.net.max_w)
        self.inject_signal(feedback)
        self.step(n_steps=observe_steps, learn=False)

        self.clear_signal()
        self.step(n_steps=5)
        return grid

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
            w_max=self.genome.w_max,
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
        return cls._load_checkpoint(genome, ckpt, checkpoint_dir)

    @classmethod
    def load_from(
        cls,
        genome: Genome,
        checkpoint_path: str,
        checkpoint_dir: str | None = None,
    ) -> "Brain":
        """Load from a specific checkpoint directory."""
        from pathlib import Path
        ckpt = Path(checkpoint_path)
        if not (ckpt / "network.npz").exists():
            raise FileNotFoundError(f"No network.npz in {ckpt}")
        ckpt_dir = checkpoint_dir or str(ckpt.parent)
        return cls._load_checkpoint(genome, ckpt, ckpt_dir)

    @classmethod
    def _load_checkpoint(
        cls,
        genome: Genome,
        ckpt: "Path",
        checkpoint_dir: str,
    ) -> "Brain":
        net = DNG.load(str(ckpt / "network.npz"))
        with open(ckpt / "brain_meta.json") as f:
            meta = json.load(f)

        rng_state = meta.get("rng_state")
        rng = np.random.default_rng()
        if rng_state:
            rng.bit_generator.state = rng_state

        stage_mgr = StageManager()
        if "stage_manager" in meta:
            stage_mgr.load_state_dict(meta["stage_manager"])

        brain = cls(net, genome, rng=rng, checkpoint_dir=checkpoint_dir,
                     stage_manager=stage_mgr)
        brain.age = meta.get("age", 0)
        brain.birth_edges = meta.get("birth_edges", net._edge_count)
        brain.neuromod = NeuromodState.from_dict(meta.get("neuromod", {}))
        brain.fatigue = FatigueTracker.from_dict(meta.get("fatigue", {}))

        if "homeostasis" in meta:
            brain.homeostasis.load_state_dict(meta["homeostasis"])

        return brain
