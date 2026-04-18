"""
Brain engine — the always-on continuous neural system.

Autonomous mushroom body: the brain perceives through gaze, acts one cell
at a time via POSITION+ACTION commit stability, and learns from per-action
food/shock (DA to winning neurons only). No neurosurgery.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..graph import DNG, Region, layer_index
from ..genome import Genome
from ..encoding import grid_to_signal, NUM_COLORS
from ..perception.encoder import FEATURES_PER_CELL
from ..numba_kernels import run_steps_plastic, wta_motor_cells, accumulate_corr
from ..plasticity import (
    eligibility_modulated_update,
    eligibility_modulated_update_percell,
    contrastive_hebbian_update,
    consolidate_synapses,
    get_weight_snapshot,
    synaptogenesis,
    lateral_decorrelation,
    depression_update_l3_memory,
    potentiation_update_l3_memory,
    recovery_l3_memory,
)
from ..homeostasis import Homeostasis, HomeostasisSetpoints, StageManager
from .neuromodulators import NeuromodState
from .fatigue import FatigueTracker
from .sleep import nrem_sleep
from .checkpoint import Checkpointer


class Brain:
    """
    Continuous brain engine with autonomous mushroom body architecture.

    Motor output: 9 POSITION + 11 ACTION + 1 DONE + 1 COMMIT neurons.
    Commits one cell at a time when COMMIT neuron fires above threshold
    (basal ganglia go/no-go gate). Canvas tracks the answer being built.
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

        # All behavioral constants read from genome (single source of truth)
        self.COMMIT_GAIN = genome.commit_gain
        self.COMMIT_NOISE = genome.commit_noise
        self.COMMIT_LR = genome.commit_lr
        self.REFRACTORY_STEPS = genome.refractory_steps
        self.GAZE_REFLEX_STEPS = genome.gaze_reflex_steps
        self.DONE_NOISE_BOOST = genome.done_noise
        self.MIN_COMMITS_BEFORE_DONE = genome.min_commits_before_done
        self.AUTO_SUBMIT_STEPS = genome.auto_submit_steps
        self.ACTION_NOISE_COEFF = genome.action_noise_coeff
        self.ACTION_EXPLORE_NOISE = genome.action_explore_noise
        self.SUPPRESS_STEPS = genome.suppress_steps
        self.SUPPRESS_STRENGTH = genome.suppress_strength
        self.MBON_STRENGTH = genome.mbon_strength
        self.COPY_BIAS = genome.copy_bias
        self.ATTENTION_GAIN = genome.attention_gain

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

        self._r_pre_wta = np.zeros(net.n_nodes, dtype=np.float64)

        self.stage_manager = stage_manager or StageManager(initial_stage="operational")
        self.homeostasis = Homeostasis(
            net,
            self.stage_manager.current_setpoints(),
            interval=genome.homeostasis_interval,
            r_pre_wta=self._r_pre_wta,
        )

        self.age = 0
        self.birth_edges = net._edge_count
        self._replay_buffer: list = []
        self._signal_buffer: list = []
        self._max_replay = 50

        n_sensory = len(net.input_nodes)
        self._recent_signal_ema = np.zeros(n_sensory, dtype=np.float64)
        self._novelty_sensitivity = 0.5
        self._novelty_ema_rate = 0.1

        # Pre-compute region indices
        from ..graph import internal_mask as _internal_mask
        _reg = list(Region)
        self._internal_idx = np.where(_internal_mask(net.regions))[0].astype(np.int64)
        self._memory_idx = np.where(net.regions == _reg.index(Region.MEMORY))[0].astype(np.int64)
        self._gate_idx = np.where(net.regions == _reg.index(Region.GATE))[0].astype(np.int64)
        self._gaze_idx = np.where(net.regions == _reg.index(Region.GAZE))[0].astype(np.int64)
        self._gaze_bias = np.zeros(len(self._gaze_idx), dtype=np.float64)

        # Motor pool indices (new architecture)
        self._position_idx = np.where(net.regions == _reg.index(Region.POSITION))[0].astype(np.int64)
        self._action_idx = np.where(net.regions == _reg.index(Region.ACTION))[0].astype(np.int64)
        self._done_idx = np.where(net.regions == _reg.index(Region.DONE))[0].astype(np.int64)
        self._commit_idx = np.where(net.regions == _reg.index(Region.COMMIT))[0].astype(np.int64)

        self._n_position = len(self._position_idx)
        self._n_action = len(self._action_idx)

        # MBON compartment lookup: which MEMORY neurons belong to which color group
        if hasattr(net, '_memory_group_ids') and net._memory_group_ids is not None:
            self._memory_group_ids = net._memory_group_ids
            self._n_memory_groups = getattr(net, '_n_memory_groups', NUM_COLORS)
        else:
            self._memory_group_ids = np.full(len(self._memory_idx), -1, dtype=np.int32)
            self._n_memory_groups = 0

        self._mbon_readout_cache = None

        # For backward compat with kernel motor WTA (we handle our own WTA now)
        self._motor_start = int(self._position_idx[0]) if len(self._position_idx) > 0 else net.n_nodes
        self._n_motor_cells = 0  # disable old per-cell WTA in kernel

        self._sensory_reg = _reg.index(Region.SENSORY)
        self._motor_reg = _reg.index(Region.MOTOR)

        self._l1_idx = np.where(net.regions == _reg.index(Region.LOCAL_DETECT))[0].astype(np.int64)
        self._l2_idx = np.where(net.regions == _reg.index(Region.MID_LEVEL))[0].astype(np.int64)
        self._l3_idx = np.where(net.regions == _reg.index(Region.ABSTRACT))[0].astype(np.int64)

        # L3 global WTA pool (APL analog)
        self._l3_pool_indices = self._l3_idx.copy()
        self._l3_wta_k = max(1, int(self.genome.apl_target_sparseness * len(self._l3_idx)))

        # Column pools for internal WTA
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
        self._col_mean_r = np.zeros(n_cols, dtype=np.float64)

        # Refractory suppression mask (internal + memory + gate + gaze)
        self._refractory_mask = np.zeros(net.n_nodes, dtype=np.bool_)
        self._refractory_mask[self._internal_idx] = True
        self._refractory_mask[self._memory_idx] = True
        self._refractory_mask[self._gate_idx] = True
        self._refractory_mask[self._gaze_idx] = True

        self._edge_layer_cache: np.ndarray | None = None

        # ── Sequential motor state ──
        self._canvas = np.zeros(self._n_position, dtype=np.int32)
        self._committed = np.zeros(self._n_position, dtype=np.bool_)
        self._pos_winner_history: list[int] = []
        self._act_winner_history: list[int] = []
        self._steps_since_commit = 0
        self._total_commits = 0
        self._pos_refractory = np.zeros(self._n_position, dtype=np.int32)
        self._gaze_reflex_countdown = 0
        self._done_fired = False

        # ── Lose-shift exploration (basal ganglia indirect pathway) ──
        # After a wrong commit, suppress that action for SUPPRESS_STEPS to
        # force the WTA toward alternatives. Not random — directed away from
        # the known-bad answer.
        self._suppress_group: int = -1
        self._suppress_countdown: int = 0

        # ── Observed-color constraint ──
        # Boolean mask: which colors appeared in the test input during
        # observation. ACTION neurons for unseen colors get suppressed,
        # narrowing the search space from 10 to ~2-3 candidates.
        self._observed_colors = np.ones(NUM_COLORS, dtype=np.bool_)

        # ── Spatial attention (per-position KC gain factors) ──
        # Pre-compute how strongly each KC is connected to each cell's
        # sensory nodes. Used for proportional gain modulation at commit time.
        self._attended_kc_gains: dict[int, np.ndarray] = {}
        self._build_attended_kc_gains()

    def _build_attended_kc_gains(self):
        """Pre-compute per-KC gain factors for each grid position.

        For position i, each KC gets a gain proportional to what fraction
        of its SENSORY inputs come from cell i's sensory nodes [i*10, i*10+10).
        A KC with 3/10 inputs from cell i gets gain = 1 + 0.3 * (attention_gain - 1).
        A KC with 0/10 inputs gets gain = 1.0 (no boost).

        This is more position-specific than binary masks because most KCs
        have 0-1 connections from any given cell (out of ~10 total inputs).
        """
        ne = self.net._edge_count
        src = self.net._edge_src[:ne]
        dst = self.net._edge_dst[:ne]

        _reg = list(Region)
        sensory_r = _reg.index(Region.SENSORY)
        abstract_r = _reg.index(Region.ABSTRACT)

        s2l3 = ((self.net.regions[src] == sensory_r) &
                (self.net.regions[dst] == abstract_r))
        s2l3_idx = np.where(s2l3)[0]

        l3_to_local = {int(g): i for i, g in enumerate(self._l3_idx)}

        # Count total SENSORY inputs per KC
        total_inputs = np.zeros(len(self._l3_idx), dtype=np.float64)
        for ei in s2l3_idx:
            d = int(dst[ei])
            if d in l3_to_local:
                total_inputs[l3_to_local[d]] += 1.0
        total_inputs = np.maximum(total_inputs, 1.0)

        n_positions = self._n_position
        for pos in range(n_positions):
            cell_base = pos * FEATURES_PER_CELL
            cell_end = cell_base + FEATURES_PER_CELL
            cell_inputs = np.zeros(len(self._l3_idx), dtype=np.float64)
            for ei in s2l3_idx:
                s = int(src[ei])
                if cell_base <= s < cell_end:
                    d = int(dst[ei])
                    if d in l3_to_local:
                        cell_inputs[l3_to_local[d]] += 1.0
            fraction = cell_inputs / total_inputs
            self._attended_kc_gains[pos] = fraction

    # ── Sensory injection ─────────────────────────────────────────

    def inject_signal(self, signal: np.ndarray):
        """Inject a sensory signal. Computes novelty DA automatically."""
        n_sensory = len(self.net.input_nodes)
        sensory_part = signal[:n_sensory]

        novelty = float(np.mean(np.abs(sensory_part - self._recent_signal_ema)))
        self._recent_signal_ema = (
            (1.0 - self._novelty_ema_rate) * self._recent_signal_ema
            + self._novelty_ema_rate * sensory_part
        )

        da_spike = min(novelty * self._novelty_sensitivity, self.genome.da_novelty_cap)
        if da_spike > 0.01:
            new_da = self.neuromod.da_baseline + da_spike
            self.neuromod.set_da(new_da)
            self.net.da = new_da

        self.net.V[:n_sensory] = 0.0
        self.net.r[:n_sensory] = 0.0

        sig = np.zeros(self.net.n_nodes)
        sig[:n_sensory] = sensory_part
        self._current_signal = sig

    def clear_signal(self):
        self._current_signal = None

    def set_da(self, level: float):
        self.neuromod.set_da(level)
        self.net.da = level

    # ── Canvas management ─────────────────────────────────────────

    def reset_canvas(self):
        """Clear the answer canvas to blank (all zeros)."""
        self._canvas[:] = 0
        self._committed[:] = False
        self._pos_winner_history.clear()
        self._act_winner_history.clear()
        self._steps_since_commit = 0
        self._total_commits = 0
        self._pos_refractory[:] = 0
        self._gaze_reflex_countdown = 0
        self._done_fired = False
        self._suppress_group = -1
        self._suppress_countdown = 0
        self._observed_colors[:] = True  # permissive until observation constrains it

    def get_canvas_grid(self, h: int, w: int) -> np.ndarray:
        """Read current canvas as a 2D grid."""
        n_cells = h * w
        grid = np.zeros(n_cells, dtype=np.int32)
        grid[:min(n_cells, len(self._canvas))] = self._canvas[:min(n_cells, len(self._canvas))]
        return grid.reshape(h, w)

    def reset_eligibility_traces(self):
        """Zero all eligibility traces (between-task boundary)."""
        ne = self.net._edge_count
        self.net._edge_eligibility[:ne] = 0.0

    def set_observed_colors(self, grid: np.ndarray):
        """Extract which colors appear in a grid and set the constraint mask.

        During the action phase, ACTION neurons for colors NOT in this set
        get suppressed, narrowing WTA competition to ~2-3 plausible colors.
        """
        self._observed_colors[:] = False
        unique_colors = np.unique(grid)
        for c in unique_colors:
            ci = int(c)
            if 0 <= ci < NUM_COLORS:
                self._observed_colors[ci] = True

    def reset_mbon_weights(self):
        """Restore L3->MEMORY weights to their initial values.

        Called between task types to prevent cross-task interference.
        Within-type persistence enables generalization.
        """
        if not (hasattr(self.net, '_l3_mem_initial_w') and
                hasattr(self.net, '_l3_mem_edge_slice')):
            return
        sl_start, sl_end = self.net._l3_mem_edge_slice
        ne = self.net._edge_count
        actual_end = min(sl_end, ne)
        if sl_start >= actual_end:
            return
        n = actual_end - sl_start
        self.net._edge_w[sl_start:actual_end] = self.net._l3_mem_initial_w[:n].copy()
        self.net._csr_dirty = True

    # ── Plasticity masks ──────────────────────────────────────────

    def _get_edge_plastic(self, ne: int) -> np.ndarray:
        """Boolean mask: True for edges eligible for Hebbian/eligibility plasticity.

        Plastic pathways:
          L3->POSITION, L3->DONE, L3->COMMIT  (spatial targeting + go gate)
          L3->MEMORY                           (depression site — KC→MBON)
          MEMORY->POSITION/DONE/COMMIT         (MBON output for spatial/completion/gate)
          POSITION->COMMIT, ACTION->COMMIT     (motor plan → go gate)
          L3->GAZE                             (gaze strategy)

        NOT plastic (handled by other mechanisms):
          MEMORY->ACTION: fixed structured wiring (consolidated)
        """
        cached = getattr(self, '_edge_plastic_cache', None)
        if cached is not None and len(cached) == ne:
            return cached

        _reg = list(Region)
        src_reg = self.net.regions[self.net._edge_src[:ne]]
        dst_reg = self.net.regions[self.net._edge_dst[:ne]]

        abstract_r = _reg.index(Region.ABSTRACT)
        memory_r = _reg.index(Region.MEMORY)
        position_r = _reg.index(Region.POSITION)
        action_r = _reg.index(Region.ACTION)
        done_r = _reg.index(Region.DONE)
        commit_r = _reg.index(Region.COMMIT)
        gaze_r = _reg.index(Region.GAZE)

        mask = np.zeros(ne, dtype=np.bool_)

        src_is_l3 = src_reg == abstract_r
        dst_is_commit = dst_reg == commit_r

        # L3 -> POSITION/DONE/COMMIT
        mask |= src_is_l3 & (dst_reg == position_r)
        mask |= src_is_l3 & (dst_reg == done_r)
        mask |= src_is_l3 & dst_is_commit
        # L3 -> MEMORY (depression learning site)
        mask |= src_is_l3 & (dst_reg == memory_r)

        # MEMORY -> POSITION/DONE/COMMIT
        src_is_mem = src_reg == memory_r
        mask |= src_is_mem & (dst_reg == position_r)
        mask |= src_is_mem & (dst_reg == done_r)
        mask |= src_is_mem & dst_is_commit

        # POSITION -> COMMIT, ACTION -> COMMIT (motor plan informs go gate)
        mask |= (src_reg == position_r) & dst_is_commit
        mask |= (src_reg == action_r) & dst_is_commit

        # L3 -> GAZE
        mask |= src_is_l3 & (dst_reg == gaze_r)

        # Exclude consolidated edges (instinct wiring, MEMORY→ACTION)
        cons = self.net._edge_consolidation[:ne]
        mask &= cons < 10.0

        self._edge_plastic_cache = mask
        return mask

    def _get_edge_l3_to_motor(self, ne: int) -> np.ndarray:
        """Cached mask: L3 -> any motor pool (POSITION/ACTION/DONE/COMMIT)."""
        cached = getattr(self, '_edge_l3_to_motor_cache', None)
        if cached is not None and len(cached) == ne:
            return cached
        _reg = list(Region)
        l3_reg = _reg.index(Region.ABSTRACT)
        pos_reg = _reg.index(Region.POSITION)
        act_reg = _reg.index(Region.ACTION)
        done_reg = _reg.index(Region.DONE)
        commit_reg = _reg.index(Region.COMMIT)
        src_is_l3 = self.net.regions[self.net._edge_src[:ne]] == l3_reg
        dst_reg = self.net.regions[self.net._edge_dst[:ne]]
        dst_is_motor = ((dst_reg == pos_reg) | (dst_reg == act_reg) |
                        (dst_reg == done_reg) | (dst_reg == commit_reg))
        mask = src_is_l3 & dst_is_motor
        self._edge_l3_to_motor_cache = mask
        return mask

    def _get_edge_into_l3(self, ne: int) -> np.ndarray:
        cached = getattr(self, '_edge_into_l3_cache', None)
        if cached is not None and len(cached) == ne:
            return cached
        l3_reg = list(Region).index(Region.ABSTRACT)
        mask = self.net.regions[self.net._edge_dst[:ne]] == l3_reg
        self._edge_into_l3_cache = mask
        return mask

    def _get_edge_layer_class(self) -> np.ndarray:
        n = self.net._edge_count
        if self._edge_layer_cache is not None and len(self._edge_layer_cache) == n:
            return self._edge_layer_cache
        src_layers = np.array([layer_index(int(r)) for r in self.net.regions[self.net._edge_src[:n]]])
        dst_layers = np.array([layer_index(int(r)) for r in self.net.regions[self.net._edge_dst[:n]]])
        src_layers[src_layers < 0] = 0
        dst_layers[dst_layers < 0] = 0
        self._edge_layer_cache = np.maximum(src_layers, dst_layers)
        return self._edge_layer_cache

    def _invalidate_edge_caches(self):
        """Invalidate all edge-count-dependent caches."""
        self._edge_plastic_cache = None
        self._edge_layer_cache = None
        self._edge_l3_to_motor_cache = None
        self._edge_into_l3_cache = None
        self._mbon_readout_cache = None
        attrs = ['_memory_gate_cache', '_sensory_motor_cache',
                 '_edge_l1_to_l2_cache']
        for a in attrs:
            if hasattr(self, a):
                setattr(self, a, None)

    # ── Motor WTA (custom for POSITION / ACTION / DONE) ───────────

    def _get_mbon_readout_indices(self):
        """Build or return cached L3->MEMORY edge indices per group.

        Returns list of (src_indices, weight_slice_indices) per group,
        where indices refer into the full edge arrays ([:ne]).
        Rebuilt when edges change (synaptogenesis/pruning).
        """
        if self._mbon_readout_cache is not None:
            return self._mbon_readout_cache

        ne = self.net._edge_count
        src = self.net._edge_src[:ne]
        dst = self.net._edge_dst[:ne]

        _reg = list(Region)
        abstract_r = _reg.index(Region.ABSTRACT)
        memory_r = _reg.index(Region.MEMORY)

        is_l3_to_mem = ((self.net.regions[src] == abstract_r) &
                        (self.net.regions[dst] == memory_r))
        l3m_indices = np.where(is_l3_to_mem)[0]

        group_edge_indices = []
        for k in range(self._n_memory_groups):
            gmask = self._memory_group_ids == k
            group_global = set(self._memory_idx[gmask].tolist())
            mask = np.array([dst[i] in group_global for i in l3m_indices],
                            dtype=np.bool_)
            group_edge_indices.append(l3m_indices[mask])

        self._mbon_readout_cache = group_edge_indices
        return group_edge_indices

    def _motor_wta(self):
        """Apply WTA to POSITION, ACTION, and DONE pools separately."""
        r = self.net.r

        # POSITION WTA: suppress all but winner, respecting refractory
        if len(self._position_idx) > 0:
            pos_r = r[self._position_idx].copy()
            for i in range(self._n_position):
                if self._pos_refractory[i] > 0:
                    pos_r[i] = 0.0
            winner = int(np.argmax(pos_r))
            for i, idx in enumerate(self._position_idx):
                if i != winner:
                    r[idx] *= 0.001

        # Spatial attention: proportional gain modulation on KCs for the
        # POSITION winner's cell. KCs with more inputs from the attended cell
        # get a stronger boost. Full grid context preserved (unconnected KCs = 1x).
        if len(self._position_idx) > 0 and self.ATTENTION_GAIN > 1.0:
            pos_w = int(np.argmax(pos_r))
            fractions = self._attended_kc_gains.get(pos_w)
            if fractions is not None:
                gains = 1.0 + fractions * (self.ATTENTION_GAIN - 1.0)
                r[self._l3_idx] *= gains

        # Direct MBON readout: compute KC->MBON weighted sum for each color
        # group, then subtract the cross-group mean before injecting into
        # ACTION. This models lateral inhibition between MBONs — removes
        # common-mode input (wiring density bias) while preserving the
        # relative differential created by depression-based learning.
        if self._n_memory_groups > 0 and len(self._action_idx) > 0:
            group_edges = self._get_mbon_readout_indices()
            ne = self.net._edge_count
            src = self.net._edge_src[:ne]
            w = self.net._edge_w[:ne]
            group_size = getattr(self.net, '_memory_group_size', 10) or 10

            n_groups = min(self._n_memory_groups, self._n_action)
            raw_drives = np.zeros(n_groups)
            for k in range(n_groups):
                eidx = group_edges[k]
                if len(eidx) == 0:
                    continue
                kc_rates = r[src[eidx]]
                raw_drives[k] = float(np.sum(kc_rates * w[eidx])) / group_size

            mean_drive = float(np.mean(raw_drives))
            for k in range(n_groups):
                r[self._action_idx[k]] += (raw_drives[k] - mean_drive) * self.MBON_STRENGTH

        # ACTION WTA: observed-color gated noisy competition
        if len(self._action_idx) > 0:
            act_r = r[self._action_idx].copy()
            np.maximum(act_r, 0, out=act_r)

            # Observed-color gating: zero out colors not seen in the test input.
            # Narrows the WTA from 10 candidates to ~2-3 plausible ones, making
            # depression-based elimination tractable.
            for k in range(min(NUM_COLORS, self._n_action)):
                if not self._observed_colors[k]:
                    act_r[k] = 0.0

            # Lose-shift: suppress recently punished action group
            if self._suppress_countdown > 0 and 0 <= self._suppress_group < self._n_action:
                act_r[self._suppress_group] *= (1.0 - self.SUPPRESS_STRENGTH)
                self._suppress_countdown -= 1

            noise = self.rng.normal(0, 1, size=self._n_action)
            act_r_noisy = act_r + noise * (act_r * self.ACTION_NOISE_COEFF
                                           + self.ACTION_EXPLORE_NOISE)
            np.maximum(act_r_noisy, 0, out=act_r_noisy)

            # Re-enforce gating after noise (noise can't resurrect unseen colors)
            for k in range(min(NUM_COLORS, self._n_action)):
                if not self._observed_colors[k]:
                    act_r_noisy[k] = 0.0

            winner = int(np.argmax(act_r_noisy))
            for i, idx in enumerate(self._action_idx):
                if i != winner:
                    r[idx] *= 0.001
            # Tonic motor drive: the noisy WTA value becomes the winner's rate.
            # Without this, ACTION neurons with no afferent input stay at 0
            # and can never pass the commit gate's min_act_rate threshold.
            r[self._action_idx[winner]] = max(
                float(r[self._action_idx[winner]]),
                float(act_r_noisy[winner]),
            )

        # DONE: no WTA (single neuron), but apply noise boost
        if len(self._done_idx) > 0:
            r[self._done_idx[0]] += self.rng.normal(0, self.DONE_NOISE_BOOST)
            if r[self._done_idx[0]] < 0:
                r[self._done_idx[0]] = 0.0

        # COMMIT: spontaneous noise bootstraps go/no-go learning
        if len(self._commit_idx) > 0:
            r[self._commit_idx[0]] += self.rng.normal(0, self.COMMIT_NOISE)
            if r[self._commit_idx[0]] < 0:
                r[self._commit_idx[0]] = 0.0

    def _read_motor_winners(self) -> tuple[int, int, bool]:
        """Read current POSITION winner, ACTION winner, and DONE state."""
        r = self.net.r

        pos_r = r[self._position_idx].copy()
        for i in range(self._n_position):
            if self._pos_refractory[i] > 0:
                pos_r[i] = 0.0
        pos_winner = int(np.argmax(pos_r))

        act_winner = int(np.argmax(r[self._action_idx]))

        done_fired = False
        if len(self._done_idx) > 0:
            done_fired = float(r[self._done_idx[0]]) > 0.3

        return pos_winner, act_winner, done_fired

    # ── Commit logic ──────────────────────────────────────────────

    def _check_commit(self) -> tuple[bool, int, int]:
        """Probabilistic commit gate driven by COMMIT neuron.

        The COMMIT neuron's firing rate is converted to a commit probability:
        P(commit) = clamp(rate * COMMIT_GAIN, 0, 1). The brain samples this
        probability each iteration — higher confidence = more frequent commits.

        Returns (should_commit, pos_winner, act_winner).
        """
        pos_w, act_w, done = self._read_motor_winners()

        self._pos_winner_history.append(pos_w)
        self._act_winner_history.append(act_w)

        max_hist = 5
        if len(self._pos_winner_history) > max_hist:
            self._pos_winner_history = self._pos_winner_history[-max_hist:]
            self._act_winner_history = self._act_winner_history[-max_hist:]

        if done and self._total_commits >= self.MIN_COMMITS_BEFORE_DONE:
            self._done_fired = True
            return False, pos_w, act_w

        act_rate = float(self.net.r[self._action_idx[act_w]])
        if act_rate < self.genome.min_act_rate:
            return False, pos_w, act_w

        # Probabilistic commit gate: COMMIT rate → commit probability
        commit_rate = 0.0
        if len(self._commit_idx) > 0:
            commit_rate = float(self.net.r[self._commit_idx[0]])

        p_commit = min(1.0, max(0.0, commit_rate * self.COMMIT_GAIN))
        if self.rng.random() >= p_commit:
            return False, pos_w, act_w

        return True, pos_w, act_w

    def _snapshot_mbon_drives(self, pos_winner: int) -> dict:
        """Capture current MBON readout drives for logging.

        Reports baseline-subtracted values matching what _motor_wta injects.
        """
        r = self.net.r
        n_groups = min(self._n_memory_groups, self._n_action)

        if n_groups > 0 and len(self._action_idx) > 0:
            group_edges = self._get_mbon_readout_indices()
            ne = self.net._edge_count
            src = self.net._edge_src[:ne]
            w = self.net._edge_w[:ne]
            gs = getattr(self.net, '_memory_group_size', 10) or 10

            raw = np.zeros(n_groups)
            for k in range(n_groups):
                eidx = group_edges[k]
                if len(eidx) == 0:
                    continue
                kc_rates = r[src[eidx]]
                raw[k] = float(np.sum(kc_rates * w[eidx])) / gs
            mean_d = float(np.mean(raw))
            mbon_drives = [round((raw[k] - mean_d) * self.MBON_STRENGTH, 6)
                           for k in range(n_groups)]
        else:
            mbon_drives = [0.0] * n_groups

        return {
            "mbon_drives": mbon_drives,
            "observed_colors": self._observed_colors.tolist(),
        }

    def get_mbon_weight_summary(self) -> list[float]:
        """Return mean L3->MEMORY weight per color group (10 floats).

        Used for logging weight trajectories across tasks.
        """
        if self._n_memory_groups == 0:
            return []

        ne = self.net._edge_count
        src = self.net._edge_src[:ne]
        dst = self.net._edge_dst[:ne]
        w = self.net._edge_w[:ne]

        _reg = list(Region)
        abstract_r = _reg.index(Region.ABSTRACT)
        memory_r = _reg.index(Region.MEMORY)
        is_l3m = ((self.net.regions[src] == abstract_r) &
                  (self.net.regions[dst] == memory_r))

        result = []
        for k in range(self._n_memory_groups):
            gmask = self._memory_group_ids == k
            gids = set(self._memory_idx[gmask].tolist())
            emask = is_l3m & np.array([d in gids for d in dst[:ne]],
                                       dtype=np.bool_)
            ws = w[emask]
            result.append(round(float(ws.mean()), 6) if len(ws) > 0 else 0.0)
        return result

    def _update_commit_weights(self, da: float):
        """Direct DA-modulated weight update for COMMIT neuron edges.

        Striatal D1/D2 receptor analog: DA > 0 potentiates active inputs
        to COMMIT (D1 direct pathway reinforces "go"), DA < 0 depresses
        them (D2 indirect pathway punishes premature "go").

        Only modifies edges where the source neuron was recently active,
        keeping the update Hebbian-compatible.
        """
        if len(self._commit_idx) == 0:
            return

        ne = self.net._edge_count
        dst = self.net._edge_dst[:ne]
        src = self.net._edge_src[:ne]

        commit_node = int(self._commit_idx[0])
        commit_edges = np.where(dst == commit_node)[0]
        if len(commit_edges) == 0:
            return

        # Only modify edges from active sources (Hebbian gating)
        src_rates = self.net.r[src[commit_edges]]
        active_mask = src_rates > 0.01
        if not active_mask.any():
            return

        active_edges = commit_edges[active_mask]
        dw = da * self.COMMIT_LR * src_rates[active_mask]

        w = self.net._edge_w
        w[active_edges] = np.clip(
            w[active_edges] + dw,
            0.001,  # floor: don't zero out edges completely
            self.genome.w_max,
        )

    def on_motor_commit(self, pos_winner: int, act_winner: int,
                        expected_grid: np.ndarray | None) -> dict:
        """Execute a commit: write to canvas, deliver per-action reward.

        Depression-based learning: wrong commits depress L3->MEMORY for
        the wrong color's MBON group. Correct commits potentiate the
        correct color's group.

        Returns a detail dict with commit diagnostics.
        """
        old_color = int(self._canvas[pos_winner])
        snapshot = self._snapshot_mbon_drives(pos_winner)

        commit_rate = 0.0
        if len(self._commit_idx) > 0:
            commit_rate = float(self.net.r[self._commit_idx[0]])

        detail = {
            "pos": pos_winner,
            "act": act_winner,
            "old_color": old_color,
            "expected": -1,
            "correct": False,
            "rule": "none",
            "mean_dw": 0.0,
            "commit_rate": round(commit_rate, 4),
            **snapshot,
        }

        if act_winner < NUM_COLORS:
            self._canvas[pos_winner] = act_winner

        self._total_commits += 1
        self._steps_since_commit = 0

        self._pos_refractory[pos_winner] = self.REFRACTORY_STEPS
        self._gaze_reflex_countdown = self.GAZE_REFLEX_STEPS

        self._pos_winner_history.clear()
        self._act_winner_history.clear()

        if expected_grid is None:
            return detail

        flat_expected = expected_grid.ravel()
        cell_idx = pos_winner

        if cell_idx >= len(flat_expected):
            return detail

        expected_color = int(flat_expected[cell_idx])
        new_color = int(self._canvas[cell_idx])
        detail["expected"] = expected_color

        first_commit = not self._committed[pos_winner]
        self._committed[pos_winner] = True

        if act_winner >= NUM_COLORS:
            return detail

        # Re-commits: update COMMIT neuron DA but skip MBON learning
        # to prevent weight corruption from redundant corrections.
        if not first_commit:
            is_correct = (new_color == expected_color)
            self._update_commit_weights(da=+1.0 if is_correct else -1.0)
            detail["correct"] = is_correct
            detail["rule"] = "recommit"
            return detail

        if new_color == expected_color:
            detail["correct"] = True
            correct_color = act_winner

            # No potentiation — the insect MB learns primarily through
            # depression of wrong-color groups. Correct colors win by
            # elimination (they stay undepressed). Potentiation caused
            # runaway weight growth on frequent colors (esp. color 0).
            mean_change = 0.0
            detail["rule"] = "correct_no_change"

            detail["mean_dw"] = round(float(mean_change), 6)

            da_per_node = np.zeros(self.net.n_nodes, dtype=np.float64)
            da_per_node[self._position_idx[pos_winner]] = self.genome.da_correct_commit
            sp = self.stage_manager.current_setpoints()
            eligibility_modulated_update_percell(
                self.net, da_per_node,
                eta=self.genome.elig_eta * sp.plasticity_rate,
                w_max=self.genome.w_max,
            )

            self._update_commit_weights(da=+1.0)

            self.set_da(self.genome.da_global_correct)
            return detail

        # Wrong commit
        wrong_color = act_winner
        if wrong_color >= self._n_memory_groups:
            return detail

        wrong_group_mask = np.zeros(self.net.n_nodes, dtype=np.bool_)
        for i, gid in enumerate(self._memory_group_ids):
            if gid == wrong_color:
                wrong_group_mask[self._memory_idx[i]] = True

        mean_change = depression_update_l3_memory(
            self.net,
            wrong_group_mask=wrong_group_mask,
            eta=self.genome.depression_eta,
            w_floor=self.genome.depression_floor,
        )

        detail["rule"] = "depression"
        detail["mean_dw"] = round(float(mean_change), 6)

        # Lose-shift: suppress the just-punished group so WTA explores
        self._suppress_group = wrong_color
        self._suppress_countdown = self.SUPPRESS_STEPS

        # Direct COMMIT punishment: D2 pathway depression
        self._update_commit_weights(da=-1.0)

        if old_color == expected_color:
            da_per_node = np.zeros(self.net.n_nodes, dtype=np.float64)
            da_per_node[self._position_idx[pos_winner]] = self.genome.da_wrong_commit
            sp = self.stage_manager.current_setpoints()
            eligibility_modulated_update_percell(
                self.net, da_per_node,
                eta=self.genome.elig_eta * sp.plasticity_rate,
                w_max=self.genome.w_max,
            )

        self.set_da(self.genome.da_global_wrong)

        return detail

    # ── Deterministic motor commit ─────────────────────────────────

    def commit_at_position(self, pos: int, expected_grid: np.ndarray | None,
                           n_settle_steps: int = 5) -> dict:
        """Visit a specific position, pick a color, and commit.

        Deterministic position cycling — the teacher tells the brain WHERE
        to look; the brain decides WHAT color to paint. Analogous to a bee
        visiting each flower in a patch: the environment determines the
        route, learning determines the behavior at each stop.

        1. Amplify sensory signal at the attended cell, re-inject, settle
        2. Read L3 rates (now position-specific due to amplified drive)
        3. Compute MBON readout (with lateral inhibition)
        4. Add instinctive copy bias from sensory input at this position
        5. Gate by observed colors, apply lose-shift suppression
        6. Add noise, argmax -> chosen color
        7. Write to canvas, call on_motor_commit for reward/depression

        Returns the commit detail dict from on_motor_commit.
        """
        r = self.net.r

        # 1. Attentional spotlight with KC reset.
        #
        # Biology: KCs respond to the CURRENT stimulus, not accumulated
        # past.  When an insect shifts fixation, the KC population resets
        # and forms a fresh sparse pattern from the new input.
        #
        # We clear L3 state (V, r, f, adaptation) so the commit settling
        # starts from a clean slate, then apply center-surround modulation:
        # attended cell at full gain, surround suppressed to 10%.
        signal = getattr(self, '_current_signal', None)
        if signal is not None:
            n_sensory = len(self.net.input_nodes)
            SURROUND_SUPPRESSION = 0.1

            # Reset L3 for fresh position-specific pattern
            self.net.V[self._l3_idx] = 0.0
            self.net.r[self._l3_idx] = 0.0
            self.net.f[self._l3_idx] = 0.0
            self.net.adaptation[self._l3_idx] = 0.0

            # Center-surround signal: suppress surround, amplify target
            pos_signal = signal.copy()
            pos_signal[:n_sensory] *= SURROUND_SUPPRESSION
            cell_base = pos * FEATURES_PER_CELL
            cell_end = cell_base + FEATURES_PER_CELL
            if cell_end <= n_sensory:
                pos_signal[cell_base:cell_end] = signal[cell_base:cell_end] * self.ATTENTION_GAIN
            self._current_signal = pos_signal
            self.step(n_steps=n_settle_steps, learn=True)
            self._current_signal = signal
        else:
            self.step(n_steps=n_settle_steps, learn=True)

        # 3. MBON readout with lateral inhibition
        n_groups = min(self._n_memory_groups, self._n_action)
        raw_drives = np.zeros(n_groups)

        if self._n_memory_groups > 0 and len(self._action_idx) > 0:
            group_edges = self._get_mbon_readout_indices()
            ne = self.net._edge_count
            src = self.net._edge_src[:ne]
            w = self.net._edge_w[:ne]
            group_size = getattr(self.net, '_memory_group_size', 10) or 10

            for k in range(n_groups):
                eidx = group_edges[k]
                if len(eidx) == 0:
                    continue
                kc_rates = r[src[eidx]]
                raw_drives[k] = float(np.sum(kc_rates * w[eidx])) / group_size

        mean_drive = float(np.mean(raw_drives)) if n_groups > 0 else 0.0
        color_drives = np.zeros(NUM_COLORS)
        for k in range(n_groups):
            color_drives[k] = (raw_drives[k] - mean_drive) * self.MBON_STRENGTH

        # 4. Instinctive copy bias: read the sensory input color at this position
        signal = getattr(self, '_current_signal', None)
        if signal is not None and self.COPY_BIAS > 0:
            cell_base = pos * FEATURES_PER_CELL
            cell_end = cell_base + FEATURES_PER_CELL
            n_sensory = len(self.net.input_nodes)
            if cell_end <= n_sensory:
                cell_signal = signal[cell_base:cell_end]
                input_color = int(np.argmax(cell_signal))
                if cell_signal[input_color] > 0.5:
                    color_drives[input_color] += self.COPY_BIAS

        # 5. Observed-color gating
        for k in range(NUM_COLORS):
            if not self._observed_colors[k]:
                color_drives[k] = -999.0

        # Lose-shift suppression
        if self._suppress_countdown > 0 and 0 <= self._suppress_group < NUM_COLORS:
            color_drives[self._suppress_group] *= (1.0 - self.SUPPRESS_STRENGTH)
            self._suppress_countdown -= 1

        # 6. Add noise and pick winner
        noise = self.rng.normal(0, 1, size=NUM_COLORS)
        noisy_drives = color_drives + noise * (
            np.maximum(color_drives, 0) * self.ACTION_NOISE_COEFF
            + self.ACTION_EXPLORE_NOISE
        )
        # Re-enforce gating after noise
        for k in range(NUM_COLORS):
            if not self._observed_colors[k]:
                noisy_drives[k] = -999.0

        chosen_color = int(np.argmax(noisy_drives))

        # 7. Commit: write to canvas and trigger reward/depression
        detail = self.on_motor_commit(pos, chosen_color, expected_grid)
        return detail

    # ── Core dynamics step ────────────────────────────────────────

    def step(self, n_steps: int = 10, learn: bool = True,
             clamp_sensory: bool = False):
        """Run the brain for n_steps continuous ticks."""
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

        self.stage_manager.step(n_steps)
        sp = self.stage_manager.current_setpoints()
        self.homeostasis.setpoints = sp

        self.neuromod.da_baseline = sp.da_baseline
        self._novelty_sensitivity = sp.da_sensitivity
        self.fatigue.threshold = sp.fatigue_threshold

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
            self._l3_pool_indices,
            self._l3_wta_k,
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

        _trace_kwargs = dict(
            r_trace=self.net.r_trace,
            trace_decay=self.genome.trace_decay,
            edge_is_l1_to_l2=np.zeros(ne, dtype=np.bool_),
            use_trace_rule=False,
            trace_contrast_eta=0.0,
        )

        if clamp_sensory and has_signal:
            n_sensory = len(self.net.input_nodes)
            sensory_vals = signal[:n_sensory]
            for s_i in range(n_steps):
                run_steps_plastic(
                    *_kernel_args, True, n, 1, noise[s_i:s_i+1],
                    *_tail_args, **_trace_kwargs,
                )
                self.net.V[:n_sensory] = sensory_vals
                self.net.r[:n_sensory] = sensory_vals
        else:
            run_steps_plastic(
                *_kernel_args, has_signal, n, n_steps, noise,
                *_tail_args, **_trace_kwargs,
            )

        # Custom motor WTA (POSITION/ACTION/DONE separate pools)
        self._motor_wta()

        # Tick down refractory counters
        active_refrac = self._pos_refractory > 0
        self._pos_refractory[active_refrac] -= n_steps
        np.clip(self._pos_refractory, 0, None, out=self._pos_refractory)

        # Gaze WTA
        if len(self._gaze_idx) > 0:
            gaze_r = self.net.r[self._gaze_idx] + self._gaze_bias

            # Post-commit gaze reflex: bias toward canvas (last slot)
            if self._gaze_reflex_countdown > 0:
                canvas_slot = len(self._gaze_idx) - 1
                if canvas_slot >= 0:
                    gaze_r[canvas_slot] += 1.0
                self._gaze_reflex_countdown -= n_steps
                if self._gaze_reflex_countdown < 0:
                    self._gaze_reflex_countdown = 0

            winner = int(np.argmax(gaze_r))
            for gi, idx in enumerate(self._gaze_idx):
                if gi != winner:
                    self.net.r[idx] *= 0.001

        self.homeostasis.step()

        # Synaptogenesis
        if sp.synaptogenesis_candidates > 0 and self.birth_edges > 0:
            growth_ratio = self.net._edge_count / self.birth_edges
            peak = self.genome.peak_growth_target
            density_factor = max(0.0, 1.0 - growth_ratio / peak)

            ema_d = self.homeostasis.ema_rate_dendritic
            demand_factor = float((ema_d < sp.target_rate).mean())

            eff_rate = sp.synaptogenesis_rate * density_factor * demand_factor
            eff_cand = max(1, int(sp.synaptogenesis_candidates * density_factor))

            edge_ratio = self.net._edge_count / max(1, self.birth_edges)
            if edge_ratio < 1.0:
                floor_boost = 1.0 + 2.0 * (1.0 - edge_ratio)
                eff_rate *= floor_boost
                eff_cand = int(eff_cand * floor_boost)

            if eff_rate > 0.001:
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
                    self._invalidate_edge_caches()

        # DA decay
        self.neuromod.decay()
        self.net.da = self.neuromod.da

        # Fatigue
        mean_r = float(np.mean(self.net.r))
        self.fatigue.accumulate(mean_r, n_steps)

        self.age += n_steps

    # ── Reward application ────────────────────────────────────────

    def apply_reward(self, DA: float) -> float:
        """Global DA: cashes in eligibility traces on DONE + GAZE edges only."""
        self.set_da(DA)
        ne = self.net._edge_count

        # Scope: only DONE and GAZE destination edges
        _reg = list(Region)
        done_r = _reg.index(Region.DONE)
        gaze_r = _reg.index(Region.GAZE)
        dst_reg = self.net.regions[self.net._edge_dst[:ne]]
        scope_mask = (dst_reg == done_r) | (dst_reg == gaze_r)

        if not scope_mask.any():
            return 0.0

        # Build per-node DA: only DONE and GAZE neurons get DA
        da_per_node = np.zeros(self.net.n_nodes, dtype=np.float64)
        for idx in self._done_idx:
            da_per_node[idx] = DA
        for idx in self._gaze_idx:
            da_per_node[idx] = DA

        sp = self.stage_manager.current_setpoints()
        return eligibility_modulated_update_percell(
            self.net, da_per_node,
            eta=self.genome.elig_eta * sp.plasticity_rate,
            w_max=self.genome.w_max,
        )

    # ── Gaze ──────────────────────────────────────────────────────

    def read_gaze(self) -> int:
        if len(self._gaze_idx) == 0:
            return 0
        gaze_r = self.net.r[self._gaze_idx] + self._gaze_bias
        return int(np.argmax(gaze_r))

    def apply_gaze(self, display_buffer) -> int:
        slot = self.read_gaze()
        grid = display_buffer.get_slot(slot)
        signal = grid_to_signal(grid, max_h=self.net.max_h,
                                max_w=self.net.max_w)
        self.inject_signal(signal)
        return slot

    def guide_gaze(self, slot_idx: int, strength: float = 1.0) -> None:
        self._gaze_bias[:] = 0.0
        if 0 <= slot_idx < len(self._gaze_idx) and strength > 0:
            self._gaze_bias[slot_idx] = strength * 2.0

    def clear_gaze_bias(self) -> None:
        self._gaze_bias[:] = 0.0

    def step_with_gaze(self, display_buffer, gaze_logger=None,
                       n_steps: int = 10, learn: bool = True,
                       clamp_sensory: bool = False,
                       expected_grid: np.ndarray | None = None) -> dict:
        """Step the brain with gaze + commit detection.

        If expected_grid is provided, commits trigger per-action reward.
        Updates the canvas in the display buffer's answer slot.

        Returns dict with step info (slot, committed, done_fired).
        """
        self.step(n_steps=n_steps, learn=learn, clamp_sensory=clamp_sensory)

        # Check for commit
        should_commit, pos_w, act_w = self._check_commit()
        self._steps_since_commit += n_steps

        commit_detail = None
        committed = False
        if should_commit:
            commit_detail = self.on_motor_commit(pos_w, act_w, expected_grid)
            committed = True

        # Update canvas in display buffer
        canvas_grid = self.get_canvas_grid(self.net.max_h, self.net.max_w)
        display_buffer.grids[display_buffer.answer_slot] = canvas_grid

        # Apply gaze (perceive through current gaze selection)
        slot = self.apply_gaze(display_buffer)
        if gaze_logger is not None:
            stype = display_buffer.slot_types[slot]
            gaze_logger.record(self.age, slot, stype, motor_event=committed)

        # Auto-submit if no commits for too long
        auto_submit = (self._steps_since_commit >= self.AUTO_SUBMIT_STEPS
                       and self._total_commits > 0)

        mean_change = commit_detail["mean_dw"] if commit_detail else 0.0

        return {
            "slot": slot,
            "committed": committed,
            "pos_winner": pos_w,
            "act_winner": act_w,
            "mean_change": mean_change,
            "commit_detail": commit_detail,
            "done_fired": self._done_fired,
            "auto_submit": auto_submit,
            "total_commits": self._total_commits,
        }

    # ── Sleep ─────────────────────────────────────────────────────

    def try_sleep(self) -> dict | None:
        if not self.fatigue.needs_sleep():
            return None

        sp = self.stage_manager.current_setpoints()
        edge_layers = self._get_edge_layer_class()
        layer_decay_scales = np.ones(self.net._edge_count, dtype=np.float64)
        layer_decay_scales[edge_layers == 1] = sp.health_decay_L2_scale
        layer_decay_scales[edge_layers == 2] = sp.health_decay_L3_scale

        health_decay = sp.health_decay_rate
        edge_ratio = self.net._edge_count / max(1, self.birth_edges)
        if edge_ratio < 1.0:
            deficit = 1.0 - edge_ratio
            health_decay *= (1.0 - 0.5 * deficit)

        stats = nrem_sleep(
            self.net,
            self._replay_buffer,
            chl_eta=self.genome.chl_eta * self.genome.sleep_chl_scale * sp.plasticity_rate,
            shy_downscale=self.genome.sleep_shy_downscale,
            health_decay_rate=health_decay,
            ema_rate=self.homeostasis.ema_rate_dendritic,
            target_rate=sp.target_rate,
            w_max=self.genome.w_max,
            layer_decay_scales=layer_decay_scales,
        )

        # Spontaneous recovery: depressed L3→MEMORY weights drift back
        # toward their initial strong values (biological forgetting of
        # specific depression, prevents permanent inflexibility).
        recovery_l3_memory(self.net, recovery_rate=self.genome.recovery_rate)

        if stats.get("pruned", 0) > 0:
            self._invalidate_edge_caches()

        self.homeostasis.sleep_correction()
        self.fatigue.reset()
        return stats

    # ── Persistence ───────────────────────────────────────────────

    def store_replay(self, free_corr: np.ndarray, clamped_corr: np.ndarray):
        da_tag = float(self.neuromod.da)
        self._replay_buffer.append((free_corr.copy(), clamped_corr.copy(), da_tag))
        if len(self._replay_buffer) > self._max_replay:
            self._replay_buffer.pop(0)

    def store_signal(self, signal: np.ndarray):
        self._signal_buffer.append(signal.copy())
        if len(self._signal_buffer) > self._max_replay:
            self._signal_buffer.pop(0)

    def spontaneous_replay(self, n_steps: int = 30, strength: float = 0.3):
        if not self._signal_buffer:
            self.step(n_steps=n_steps)
            return
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
        grid_h: int = 3,
        grid_w: int = 3,
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

        # Warm up EMA rates
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

    # ── Decorrelation (kept for compatibility) ────────────────────

    def decorrelate_layers(self, eta: float = 0.01, sim_threshold: float = 0.4):
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

    def _normalize_incoming_weights(self, ne: int):
        w = self.net._edge_w[:ne]
        dst = self.net._edge_dst[:ne]
        cons = self.net._edge_consolidation[:ne]

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

        needs_scale = sq_sums > target_norm_sq * 1.44
        if not np.any(needs_scale):
            return

        scale_factors = np.ones(self.net.n_nodes, dtype=np.float64)
        mask = needs_scale & (sq_sums > 1e-16)
        scale_factors[mask] = np.sqrt(target_norm_sq / sq_sums[mask])

        per_edge_scale = scale_factors[dst]
        w[exc_mask] *= per_edge_scale[exc_mask]
        self.net._csr_dirty = True

    # ── Correlation snapshots (kept for sleep replay) ─────────────

    def snapshot_correlations(self, n_steps: int = 20) -> np.ndarray:
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
