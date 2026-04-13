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

from ..graph import DNG, Region, layer_index
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
        self._replay_buffer: list = []  # (free_corr, clamped_corr, da_tag) for CHL replay
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
        self._gate_idx = np.where(net.regions == _reg.index(Region.GATE))[0].astype(np.int64)
        self._gaze_idx = np.where(net.regions == _reg.index(Region.GAZE))[0].astype(np.int64)
        self._gaze_bias = np.zeros(len(self._gaze_idx), dtype=np.float64)
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
        # L3 (mushroom body / Kenyon cells) has NO column assignment.
        # Competition is handled by APL-style global inhibition instead
        # of columnar WTA. L3 neurons keep column_id = -1, so the K-H
        # kernel uses h_star = 0 for them (plasticity is mostly frozen
        # on L2->L3 edges anyway; learning happens at L3->MOTOR).
        max_col = int(net.column_ids.max()) + 1 if len(net.column_ids) > 0 else 0
        n_cols = max(max_col, 1)

        self._node_col_id = net.column_ids.astype(np.int64).copy()

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
        self._refractory_mask[self._gate_idx] = True
        self._refractory_mask[self._gaze_idx] = True

        # Cached per-edge cortical layer classification (invalidated on edge count change)
        self._edge_layer_cache: np.ndarray | None = None

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
        """Apply anti-Hebbian lateral decorrelation with layer-specific strength.

        L2 gets stronger decorrelation to counteract the trace rule's
        convergence pressure. L1 and L3 use the default parameters.
        """
        total = 0
        layer_params = [
            (self._l1_idx, eta, sim_threshold),
            (self._l2_idx, eta * 3.0, max(sim_threshold - 0.1, 0.15)),
            (self._l3_idx, eta, sim_threshold),
        ]
        for layer_idx, layer_eta, layer_thresh in layer_params:
            if len(layer_idx) > 0:
                total += lateral_decorrelation(
                    self.net, layer_idx,
                    eta=layer_eta, sim_threshold=layer_thresh,
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
        are excluded here.  After infancy, L1 edges are also locked —
        V1 critical period closes early (Hensch 2005) and L1 should be
        a stable feature extractor, not rewritten by task-driven DA.

        Invalidated when edges are added/removed or stage transitions.
        """
        cached = getattr(self, '_edge_plastic_cache', None)
        if cached is not None and len(cached) == ne:
            return cached
        mask = np.ones(ne, dtype=np.bool_)
        motor_idx = list(Region).index(Region.MOTOR)
        mask[self.net.regions[self.net._edge_dst[:ne]] == motor_idx] = False

        stage = self.stage_manager.current_stage
        if stage not in ("infancy",):
            edge_layers = self._get_edge_layer_class()
            mask[edge_layers == 0] = False

        self._edge_plastic_cache = mask
        return mask

    def _get_edge_l1_to_l2(self, ne: int) -> np.ndarray:
        """Cached boolean mask: True for edges where src is L1 and dst is L2."""
        cached = getattr(self, '_edge_l1_to_l2_cache', None)
        if cached is not None and len(cached) == ne:
            return cached
        l1_reg = list(Region).index(Region.LOCAL_DETECT)
        l2_reg = list(Region).index(Region.MID_LEVEL)
        src_is_l1 = self.net.regions[self.net._edge_src[:ne]] == l1_reg
        dst_is_l2 = self.net.regions[self.net._edge_dst[:ne]] == l2_reg
        mask = src_is_l1 & dst_is_l2
        self._edge_l1_to_l2_cache = mask
        return mask

    def _get_edge_into_l3(self, ne: int) -> np.ndarray:
        """Cached boolean mask: True for edges where dst is L3 (Kenyon cells).

        Used to freeze PN->KC (L2->L3, L1->L3, sensory->L3) plasticity
        after infancy — the random conjunction wiring IS the representation.
        """
        cached = getattr(self, '_edge_into_l3_cache', None)
        if cached is not None and len(cached) == ne:
            return cached
        l3_reg = list(Region).index(Region.ABSTRACT)
        mask = self.net.regions[self.net._edge_dst[:ne]] == l3_reg
        self._edge_into_l3_cache = mask
        return mask

    def _get_edge_layer_class(self) -> np.ndarray:
        """Classify each edge by the highest cortical layer it touches.

        0 = L1 (or non-cortical), 1 = L2, 2 = L3. Used to build per-edge
        decay scale arrays for layer-aware pruning: higher layers get
        reduced decay during early development.
        """
        n = self.net._edge_count
        if self._edge_layer_cache is not None and len(self._edge_layer_cache) == n:
            return self._edge_layer_cache
        src_layers = np.array([layer_index(int(r)) for r in self.net.regions[self.net._edge_src[:n]]])
        dst_layers = np.array([layer_index(int(r)) for r in self.net.regions[self.net._edge_dst[:n]]])
        src_layers[src_layers < 0] = 0
        dst_layers[dst_layers < 0] = 0
        self._edge_layer_cache = np.maximum(src_layers, dst_layers)
        return self._edge_layer_cache

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

        # Mushroom body: freeze PN->KC (input to L3) plasticity after infancy.
        # Mushroom body (L3/KC) input edges are NEVER plastic.
        # The sparse random wiring IS the representation — pattern separation
        # comes from the randomness of connectivity, not from learning.
        # Learning happens at KC->MBON (L3->MOTOR) via DA-gated eligibility.
        if learn:
            into_l3 = self._get_edge_into_l3(ne)
            if into_l3.any():
                if edge_plastic is self._edge_plastic_cache:
                    edge_plastic = edge_plastic.copy()
                edge_plastic[into_l3] = False
        self.homeostasis.setpoints = sp

        # L1 critical period depends on stage — invalidate during transitions
        if self.stage_manager.is_transitioning:
            self._edge_plastic_cache = None
            self._edge_l1_to_l2_cache = None
            self._edge_into_l3_cache = None

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
            np.concatenate([self._memory_idx, self._gate_idx]),
            max(1, (len(self._memory_idx) + len(self._gate_idx)) // 3),
        )

        _tail_args = (
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

        # Temporal trace rule for L1->L2 invariance learning.
        # Active from late_infancy onward (L1 must be stable first).
        stage = self.stage_manager.current_stage
        _use_trace = stage not in ("infancy",)
        _trace_kwargs = dict(
            r_trace=self.net.r_trace,
            trace_decay=self.genome.trace_decay,
            edge_is_l1_to_l2=self._get_edge_l1_to_l2(ne),
            use_trace_rule=_use_trace,
            trace_contrast_eta=self.genome.trace_contrast_eta,
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
                    *_kernel_args, True, n, 1, noise[s_i:s_i+1],
                    *_tail_args, **_trace_kwargs,
                )
                self.net.V[color_idx] = color_vals
                self.net.r[color_idx] = color_vals
        else:
            run_steps_plastic(
                *_kernel_args, has_signal, n, n_steps, noise,
                *_tail_args, **_trace_kwargs,
            )

        # APL feedback: mushroom body global inhibition for L3/Kenyon cells.
        # Only keep the top-k active KCs where k = target_sparseness * n_l3.
        # This mimics the biological APL: a single inhibitory neuron driven
        # by total KC activity that suppresses all but the most active KCs.
        if len(self._l3_idx) > 0:
            l3_r = self.net.r[self._l3_idx]
            k = max(1, int(self.genome.apl_target_sparseness * len(self._l3_idx)))
            n_active = int((l3_r > 0.01).sum())
            if n_active > k:
                thresh = np.partition(l3_r, -k)[-k]
                suppress = l3_r < thresh
                self.net.r[self._l3_idx[suppress]] *= 0.001

        # Gaze WTA: strict 1-winner among gaze neurons (oculomotor selection).
        # Applied post-kernel so gaze reflects the final rates of each step
        # block. With only 8 neurons this is negligible overhead.
        if len(self._gaze_idx) > 0:
            gaze_r = self.net.r[self._gaze_idx] + self._gaze_bias
            winner = int(np.argmax(gaze_r))
            for gi, idx in enumerate(self._gaze_idx):
                if gi != winner:
                    self.net.r[idx] *= 0.001

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

        # L2 excitability floor: boost quiet neurons so they can respond
        # to novel patterns (homeostatic intrinsic plasticity, Turrigiano 2011).
        if len(self._l2_idx) > 0:
            l2_ema = self.homeostasis.ema_rate_dendritic[self._l2_idx]
            quiet = l2_ema < 0.01
            self.net.excitability[self._l2_idx[quiet]] += 0.001

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

            # Birth-edge floor: boost growth when below birth count
            edge_ratio = self.net._edge_count / max(1, self.birth_edges)
            if edge_ratio < 1.0:
                floor_boost = 1.0 + 2.0 * (1.0 - edge_ratio)
                eff_rate *= floor_boost
                eff_cand = int(eff_cand * floor_boost)

            if eff_rate > 0.001:
                # Motor cortex matures later than sensory cortex — block
                # new synapses targeting motor neurons until childhood.
                stage = self.stage_manager.current_stage
                motor_ok = stage not in ("infancy", "late_infancy")
                layer_boost = {
                    1: sp.synaptogenesis_L2_boost,
                    2: sp.synaptogenesis_L3_boost,
                }
                old_ne = self.net._edge_count
                synaptogenesis(
                    self.net,
                    growth_rate=eff_rate,
                    n_candidates=eff_cand,
                    activity_ema=ema_d,
                    rng=self.rng,
                    allow_motor_target=motor_ok,
                    layer_demand_boost=layer_boost,
                )
                if self.net._edge_count != old_ne:
                    self._edge_plastic_cache = None
                    self._memory_gate_cache = None
                    self._sensory_motor_cache = None
                    self._edge_layer_cache = None
                    self._edge_l1_to_l2_cache = None
                    self._edge_into_l3_cache = None

        # Copy pathway maintenance. During infancy/late_infancy the instinct
        # is re-pinned every step to prevent SHY erosion. In childhood, the
        # copy pathway becomes plastic: consolidation drops from 20 -> 2
        # and repinning stops, allowing CHL to refine or weaken copy weights
        # based on task-specific reward.
        copy_mask = self.net._edge_consolidation[:ne] >= 20.0
        if stage in ("infancy", "late_infancy"):
            if copy_mask.any():
                target_w = 5.0 * sp.copy_strength
                self.net._edge_w[:ne][copy_mask] = target_w
                self.net._csr_dirty = True
        elif copy_mask.any():
            # Childhood/adolescence: release copy pathway for learning.
            # Reduce consolidation once (20 -> 2), then never repin again.
            self.net._edge_consolidation[:ne][copy_mask] = 2.0
            self.net._csr_dirty = True
            self._sensory_motor_cache = None

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

    # ── Gaze (active vision) ─────────────────────────────────────────

    def read_gaze(self) -> int:
        """Return the currently selected gaze slot (argmax of gaze rates)."""
        if len(self._gaze_idx) == 0:
            return 0
        gaze_r = self.net.r[self._gaze_idx] + self._gaze_bias
        return int(np.argmax(gaze_r))

    def apply_gaze(self, display_buffer) -> int:
        """Read gaze winner, fetch that slot's grid, inject into sensory.

        Returns the selected slot index.
        """
        slot = self.read_gaze()
        grid = display_buffer.get_slot(slot)
        signal = grid_to_signal(grid, max_h=self.net.max_h,
                                max_w=self.net.max_w)
        self.inject_signal(signal)
        return slot

    def guide_gaze(self, slot_idx: int, strength: float = 1.0) -> None:
        """Bias a specific gaze neuron to win WTA (teacher-guided saccade).

        strength controls how strongly the teacher overrides the network's
        own gaze preference. 1.0 = full control, 0.0 = no guidance.
        """
        self._gaze_bias[:] = 0.0
        if 0 <= slot_idx < len(self._gaze_idx) and strength > 0:
            self._gaze_bias[slot_idx] = strength * 2.0

    def clear_gaze_bias(self) -> None:
        """Remove teacher gaze guidance."""
        self._gaze_bias[:] = 0.0

    def step_with_gaze(self, display_buffer, gaze_logger=None,
                       n_steps: int = 10, learn: bool = True,
                       clamp_sensory: bool = False) -> int:
        """Step the brain and apply gaze consequence in one call.

        Runs a normal step (motor and gaze neurons both fire and compete),
        then reads the gaze winner and injects the corresponding display
        buffer slot as the next sensory input. Returns the selected slot.

        This is the standard "perceive through gaze" tick: the network's
        oculomotor output determines what it sees on the next cycle, just
        as biological saccades shift retinal input between fixations.
        """
        self.step(n_steps=n_steps, learn=learn, clamp_sensory=clamp_sensory)
        slot = self.apply_gaze(display_buffer)
        if gaze_logger is not None:
            stype = display_buffer.slot_types[slot]
            motor_nodes = self.net.output_nodes
            motor_active = float(self.net.r[motor_nodes].mean()) > 0.05
            gaze_logger.record(self.age, slot, stype, motor_event=motor_active)
        return slot

    # ------------------------------------------------------------------
    #  Motor babbling & mimicry
    # ------------------------------------------------------------------

    def motor_babble(self, noise_std: float = 0.5,
                     observe_steps: int = 20,
                     display_buffer=None,
                     gaze_logger=None) -> np.ndarray:
        """Random motor exploration with gaze-routed visual feedback.

        Phase A — inject noise, let motor fire. Instinct connections
                  excite the answer gaze neuron during motor activity.
        Phase B — discrete saccade: read gaze winner (biased toward
                  answer by instinct), inject that slot as stable sensory,
                  observe for the full fixation period.
        Phase C — rest.
        """
        motor_nodes = self.net.output_nodes

        # Phase A: random motor activation
        self.net.V[motor_nodes] += self.rng.normal(0, noise_std, len(motor_nodes))
        self.step(n_steps=5)

        motor_rates = self.net.r[motor_nodes].copy()
        grid = self.read_motor(self.net.max_h, self.net.max_w)

        # Phase B: discrete gaze selection + stable fixation
        if display_buffer is not None:
            display_buffer.write_answer(grid)
            slot = self.read_gaze()
            fixation_grid = display_buffer.get_slot(slot)
            signal = grid_to_signal(fixation_grid, max_h=self.net.max_h,
                                    max_w=self.net.max_w)
            self.inject_signal(signal)
            if gaze_logger is not None:
                stype = display_buffer.slot_types[slot]
                gaze_logger.record(self.age, slot, stype,
                                   motor_event=True)
        else:
            signal = grid_to_signal(grid, max_h=self.net.max_h,
                                    max_w=self.net.max_w)
            self.inject_signal(signal)

        self.step(n_steps=observe_steps, learn=False)

        # Phase C: rest
        self.clear_signal()
        self.clear_gaze_bias()
        self.step(n_steps=5)
        return grid

    def stimulus_with_feedback(self, signal: np.ndarray,
                               observe_steps: int = 25,
                               display_buffer=None,
                               gaze_logger=None) -> np.ndarray:
        """Mimicry: show a stimulus, let copy pathway fire, then observe
        the motor output via a single discrete gaze saccade.

        The copy pathway fires within ~3 steps. Motor is clamped so
        the instinct edges bias gaze toward the answer canvas. After
        mimicry reward, gaze selects a slot and sensory is held stable
        for the fixation period.
        """
        motor_nodes = self.net.output_nodes

        # Phase 1: Quick copy — let copy pathway fire (5 steps is enough)
        self.inject_signal(signal)
        self.step(n_steps=5)

        motor_rates = self.net.r[motor_nodes].copy()
        grid = self.read_motor(self.net.max_h, self.net.max_w)

        # Phase 2: Clamp motor to correct output while sensory features
        # develop. Eligibility traces accumulate from co-activation of
        # sensory color nodes and correct motor output.
        remaining = max(1, observe_steps - 5)
        for _ in range(remaining):
            self.net.r[motor_nodes] = motor_rates
            self.step(n_steps=1)

        self.apply_mimicry_reward(da=1.0)

        # Phase 3: discrete gaze saccade + stable fixation.
        # Motor is still active from Phase 2, so instinct edges bias
        # gaze toward the answer slot. Read gaze once, fixate.
        if display_buffer is not None:
            display_buffer.write_answer(grid)
            slot = self.read_gaze()
            fixation_grid = display_buffer.get_slot(slot)
            fb_signal = grid_to_signal(fixation_grid,
                                       max_h=self.net.max_h,
                                       max_w=self.net.max_w)
            self.inject_signal(fb_signal)
            if gaze_logger is not None:
                stype = display_buffer.slot_types[slot]
                gaze_logger.record(self.age, slot, stype,
                                   motor_event=True)
        else:
            feedback = grid_to_signal(grid, max_h=self.net.max_h,
                                      max_w=self.net.max_w)
            self.inject_signal(feedback)

        self.step(n_steps=observe_steps, learn=False)

        self.clear_signal()
        self.clear_gaze_bias()
        self.step(n_steps=5)
        return grid

    def try_sleep(self) -> dict | None:
        """Sleep if fatigued. Returns sleep stats or None."""
        if not self.fatigue.needs_sleep():
            return None

        sp = self.stage_manager.current_setpoints()

        # Layer-aware health decay: build per-edge scale from stage setpoints
        edge_layers = self._get_edge_layer_class()
        layer_decay_scales = np.ones(self.net._edge_count, dtype=np.float64)
        layer_decay_scales[edge_layers == 1] = sp.health_decay_L2_scale
        layer_decay_scales[edge_layers == 2] = sp.health_decay_L3_scale

        # Birth-edge floor: soften pruning when below birth count
        health_decay = sp.health_decay_rate
        edge_ratio = self.net._edge_count / max(1, self.birth_edges)
        if edge_ratio < 1.0:
            deficit = 1.0 - edge_ratio
            health_decay *= (1.0 - 0.5 * deficit)

        stats = nrem_sleep(
            self.net,
            self._replay_buffer,
            chl_eta=self.genome.chl_eta * 0.3 * sp.plasticity_rate,
            shy_downscale=0.97,
            health_decay_rate=health_decay,
            ema_rate=self.homeostasis.ema_rate_dendritic,
            target_rate=sp.target_rate,
            w_max=self.genome.w_max,
            layer_decay_scales=layer_decay_scales,
        )

        # Pruning may have changed edge count — invalidate caches
        if stats.get("pruned", 0) > 0:
            self._edge_plastic_cache = None
            self._memory_gate_cache = None
            self._sensory_motor_cache = None
            self._edge_layer_cache = None
            self._edge_l1_to_l2_cache = None
            self._edge_into_l3_cache = None

        # Homeostatic corrections run during sleep, not waking.
        self.homeostasis.sleep_correction()

        self.fatigue.reset()

        return stats

    def store_replay(self, free_corr: np.ndarray, clamped_corr: np.ndarray):
        """Store correlation pair for sleep replay, tagged with current DA level.

        Higher DA at storage time = more surprising experience = higher
        replay priority during sleep (mirroring hippocampal sharp-wave
        ripple prioritization of salient memories).
        """
        da_tag = float(self.neuromod.da)
        self._replay_buffer.append((free_corr.copy(), clamped_corr.copy(), da_tag))
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
