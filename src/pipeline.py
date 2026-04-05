"""
DNG lifecycle: reasoning via contrastive Hebbian learning.

Two modes:
  CHILDHOOD (weight changes via CHL):
    1. Set ACh high (learning mode)
    2. Observe training examples (facilitation builds, no weight change)
    3. Free phase: think with test input, record correlations (network's guess)
    4. Read guess, compute reward
    5. Set DA = reward prediction error, NE = surprise
    6. Clamped phase: inject correct output, record correlations
    7. CHL update: dw = eta * DA * ACh * (clamped - free)
    8. Retry if wrong (round-robin across tasks, multiple rounds)
    9. Sleep at end of day

  ADULT / SOLVE (no weight changes):
    1. Set ACh low (recall mode), no DA/NE modulation
    2. Observe examples (facilitation warms existing pathways)
    3. Think with test input
    4. Read motor output
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .dynamics import step, think, record_think
from .encoding import NUM_COLORS, grid_to_signal, signal_to_grid
from .episodic_memory import EpisodicMemory
from .genome import Genome
from .graph import DNG, Region
from .plasticity import (
    record_phase,
    contrastive_hebbian_update,
    consolidate_synapses,
    get_weight_snapshot,
    fast_hebbian_bind,
    synaptic_scaling,
    homeostatic_excitability_update,
    prune_competitive,
    sleep_selective,
    synaptogenesis,
)


@dataclass
class LifecycleConfig:

    # Thinking
    observe_steps: int = 40
    think_steps: int = 80
    consolidation_steps: int = 20

    # CHL learning (childhood only)
    eta: float = 0.05
    w_max: float = 2.5
    free_phase_steps: int = 40
    clamped_phase_steps: int = 40

    # Retry (round-robin)
    attempts_per_round: int = 3
    n_rounds: int = 3
    stuck_patience: int = 2  # skip task if no improvement for this many attempts

    # Noise
    noise_std: float = 0.02

    # Homeostatic
    eta_b: float = 0.005
    a_target: float = 0.2

    # Rest
    rest_steps: int = 30
    rest_noise_std: float = 0.03

    # Sleep (selective)
    sleep_downscale: float = 0.95
    sleep_tag_threshold: float = 0.01

    # Pruning (sustained weakness)
    prune_weak_threshold: float = 0.01
    prune_cycles_required: int = 5

    # Synaptogenesis (edge growth -- probability-based, modulated by ACh)
    growth_rate: float = 0.3          # base probability scale for new connections
    growth_candidates: int = 50000    # how many random pairs to evaluate per day

    # Readout
    theta_conf: float = 0.3
    t_max: int = 300

    # Sleep replay (consolidation)
    replay_eta: float = 0.003       # gentle CHL during replay (with DA=1,ACh=1 → effective 0.003)
    replay_steps: int = 20          # steps per replay phase
    replay_passes: int = 3          # how many times to cycle through all memories

    # Consolidation
    consolidation_decay: float = 0.98   # per-sleep decay: unreinforced synapses become plastic again
    consolidation_strength: float = 0.5 # how much to consolidate on success (0 = disabled)
    consolidation_threshold: float = 0.8  # minimum reward to trigger consolidation

    # Episodic memory (tensor buffer)
    memory_hint_strength: float = 1.0   # recalled outputs bias motor neurons (same scale as input)
    spontaneous_strength: float = 0.15  # faint associative signals during thinking

    # Focus signal
    focus_strength: float = 0.5

    # Nursery (unsupervised infancy)
    nursery_binding_eta: float = 0.08
    nursery_growth_rate: float = 0.8
    nursery_growth_candidates: int = 500000

    @classmethod
    def from_genome(cls, genome: Genome) -> "LifecycleConfig":
        return cls(
            think_steps=genome.think_steps,
            eta=genome.eta_w,
            w_max=genome.w_max,
            sleep_downscale=genome.sleep_factor,
            rest_steps=genome.rest_steps,
            rest_noise_std=genome.rest_noise_std,
        )


@dataclass
class AttemptResult:
    guess: np.ndarray
    reward: float
    n_attempts: int
    memory_snapshot: np.ndarray | None = None
    input_signal: np.ndarray | None = None
    full_signal: np.ndarray | None = None


# ── Copy pathway suppression (for CHL free phase) ─────────────────

def _suppress_sensory_to_motor(net: DNG):
    """
    Temporarily zero sensory→motor weights so internal processing
    drives motor output during the CHL free phase.

    Returns (mask, saved_weights) for restoration.
    """
    n = net._edge_count
    _sensory = list(Region).index(Region.SENSORY)
    _motor = list(Region).index(Region.MOTOR)
    src_reg = net.regions[net._edge_src[:n]]
    dst_reg = net.regions[net._edge_dst[:n]]
    copy_mask = (src_reg == _sensory) & (dst_reg == _motor)
    saved = net._edge_w[:n][copy_mask].copy()
    net._edge_w[:n][copy_mask] = 0.0
    net._csr_dirty = True
    return copy_mask, saved


def _restore_sensory_to_motor(net: DNG, copy_mask, saved):
    """Restore sensory→motor weights after free phase."""
    net._edge_w[:net._edge_count][copy_mask] = saved
    net._csr_dirty = True


_focus_cache: dict = {}

def _focus_mask(
    net: DNG,
    grid_h: int,
    grid_w: int,
    strength: float,
) -> np.ndarray:
    """Suppress unused sensory/motor nodes in padded regions (vectorized + cached)."""
    mh, mw = net.max_h, net.max_w
    if grid_h >= mh and grid_w >= mw:
        return np.zeros(net.n_nodes)

    cache_key = (grid_h, grid_w, mh, mw, strength, net.n_nodes)
    if cache_key in _focus_cache:
        return _focus_cache[cache_key]

    s0 = int(net.input_nodes[0])
    m0 = int(net.output_nodes[0])

    rows, cols = np.mgrid[0:mh, 0:mw]
    pad_mask = (rows >= grid_h) | (cols >= grid_w)
    pad_cells = rows[pad_mask] * mw + cols[pad_mask]

    # Each padded cell maps to NUM_COLORS consecutive nodes
    offsets = np.arange(NUM_COLORS)
    cell_bases_s = s0 + pad_cells * NUM_COLORS
    cell_bases_m = m0 + pad_cells * NUM_COLORS
    # Broadcast: (n_pad, 1) + (1, NUM_COLORS) → (n_pad, NUM_COLORS)
    idx_s = (cell_bases_s[:, None] + offsets[None, :]).ravel()
    idx_m = (cell_bases_m[:, None] + offsets[None, :]).ravel()

    focus = np.zeros(net.n_nodes)
    focus[idx_s] = -strength
    focus[idx_m] = -strength

    _focus_cache[cache_key] = focus
    return focus


def _cell_confidence(
    rates: np.ndarray,
    motor_offset: int,
    h: int,
    w: int,
    max_h: int,
    max_w: int,
) -> np.ndarray:
    n_cells = max_h * max_w
    motor_r = rates[motor_offset : motor_offset + n_cells * NUM_COLORS]
    motor_r = motor_r.reshape(n_cells, NUM_COLORS)
    full_conf = np.partition(motor_r, -2, axis=1)
    full_conf = (full_conf[:, -1] - full_conf[:, -2]).reshape(max_h, max_w)
    return full_conf[:h, :w]


# ── Nursery: unsupervised exposure for concept formation ─────────────

@dataclass
class NurseryResult:
    grids_seen: int
    edges_bound: int
    edges_grown: int
    edges_pruned: int
    edges_after: int
    synaptogenesis_stats: "object | None" = None  # last SynaptogenesisStats


def nursery_exposure(
    net: DNG,
    grids: List[np.ndarray],
    config: LifecycleConfig | None = None,
    rng: np.random.Generator | None = None,
    ema_r: np.ndarray | None = None,
    maturity: float = 0.0,
    base_threshold: np.ndarray | None = None,
) -> Tuple[NurseryResult, np.ndarray]:
    """
    Unsupervised infancy: present grids with no task, just look.

    Mimics biological infant visual development:
      - Spontaneous activity waves sweep through ALL internal neurons
        before each stimulus (like retinal waves), ensuring every
        neuron participates in Hebbian binding.
      - WTA competition starts weak (many neurons active) and
        gradually tightens as maturity increases (0→1).
      - Thresholds are lowered during infancy so neurons are
        easily excitable.
      - Hebbian binding + synaptogenesis build connections;
        synaptic scaling prevents runaway potentiation.

    maturity: 0.0 = newborn (weak WTA, low threshold, strong waves)
              1.0 = end of infancy (normal WTA, normal threshold, no waves)
    """
    if config is None:
        config = LifecycleConfig()
    if rng is None:
        rng = np.random.default_rng()
    if base_threshold is None:
        base_threshold = net.threshold.copy()

    mh, mw = net.max_h, net.max_w
    n_total = net.n_nodes

    net.ach = 1.0
    net.ne = 0.0
    net.da = 0.0

    abstract_region = list(Region).index(Region.ABSTRACT)
    memory_region = list(Region).index(Region.MEMORY)
    internal_nodes = np.where(net.regions == abstract_region)[0]
    memory_nodes = np.where(net.regions == memory_region)[0]
    developing_nodes = np.concatenate([internal_nodes, memory_nodes])
    n_developing = len(developing_nodes)

    # ── Developmental parameters based on maturity ─────────────────
    # SET permanently -- the network's competition and thresholds should
    # reflect its current developmental stage during measurements too.
    adult_wta_k = 0.05
    net.wta_k_frac = 0.30 * (1.0 - maturity) + adult_wta_k * maturity

    threshold_scale = 0.3 + 0.7 * maturity
    net.threshold[developing_nodes] = base_threshold[developing_nodes] * threshold_scale

    # Inhibitory scaling: in biological infants, inhibitory circuits
    # mature AFTER excitatory ones. Early E/I balance favors excitation,
    # gradually shifting toward adult balance. Without this, inhibition
    # dominates from birth, causing "pulse and crash" dynamics where
    # one burst of activity triggers overwhelming inhibitory feedback.
    net.inh_scale = 0.3 + 0.7 * maturity

    # Excitability cap: prevent homeostatic excitability from amplifying
    # the already-dense synaptic input by 5x during infancy. Starts low,
    # rises to adult levels. This prevents the all-or-nothing firing that
    # drives the pulse-crash cycle.
    max_exc = 1.5 + 3.5 * maturity
    net.excitability[developing_nodes] = np.clip(
        net.excitability[developing_nodes], 0.1, max_exc,
    )

    wave_strength = 0.3 * (1.0 - maturity)

    # Steady growth throughout infancy, no front-loading
    growth_multiplier = 1.0

    # Continuous update interval: every HOUR_SIZE grids, run synaptogenesis
    # and homeostatic updates. This means a neuron that starts firing on
    # grid 3 can grow connections by grid 5, and those new partners can
    # be active by grid 6. Real brains wire continuously, not in daily batches.
    HOUR_SIZE = 5
    infant_homeo_eta = config.eta_b * (10.0 + 40.0 * (1.0 - maturity))
    growth_per_hour = config.nursery_growth_rate * growth_multiplier
    cands_per_hour = int(config.nursery_growth_candidates * growth_multiplier
                         / max(1, len(grids) // HOUR_SIZE))

    total_bound = 0
    n_grown = 0
    hourly_peak_r = np.zeros(net.n_nodes, dtype=np.float64)

    for i, grid in enumerate(grids):
        grid = np.asarray(grid)
        gh, gw = grid.shape

        # ── Spontaneous activity wave ──────────────────────────────
        if wave_strength > 0.01:
            n_cells = mh * mw
            wave_phase = rng.random()
            cell_assignments = np.arange(n_developing) % n_cells
            cell_row = cell_assignments // mw
            cell_col = cell_assignments % mw
            wave_pos = (cell_row / max(1, mh - 1) + cell_col / max(1, mw - 1)) / 2.0
            wave_val = np.cos(2 * np.pi * (wave_pos - wave_phase))
            wave_val = np.maximum(wave_val, 0) * wave_strength
            net.V[developing_nodes] += wave_val
            net.r[developing_nodes] = np.maximum(
                net.r[developing_nodes], wave_val * 0.5,
            )

        net.V[net.output_nodes] = 0.0
        net.r[net.output_nodes] = 0.0
        net.prev_r[net.output_nodes] = 0.0

        focus = _focus_mask(net, gh, gw, config.focus_strength)
        sig = grid_to_signal(grid, 0, n_total, max_h=mh, max_w=mw) + focus

        think(net, signal=sig, steps=config.observe_steps,
              noise_std=config.noise_std)

        total_bound += fast_hebbian_bind(
            net, eta=config.nursery_binding_eta, w_max=config.w_max,
        )

        np.maximum(hourly_peak_r, net.r, out=hourly_peak_r)

        # ── Homeostatic excitability (every grid) ──────────────────
        homeostatic_excitability_update(
            net, eta_b=infant_homeo_eta, a_target=config.a_target, b_max=max_exc,
        )

        # ── Hourly tick: synaptogenesis + scaling ──────────────────
        if (i + 1) % HOUR_SIZE == 0:
            synaptic_scaling(net, target_total=2.0, region_filter=abstract_region)

            # Use time-averaged activity for synaptogenesis so it sees
            # all neurons that fired during the hour, not just the few
            # surviving at the final snapshot (pulse-crash dynamics mean
            # the snapshot captures <5% of actual participants).
            saved_r = net.r.copy()
            net.r[:] = hourly_peak_r

            created, last_stats = synaptogenesis(
                net,
                growth_rate=growth_per_hour,
                n_candidates=cands_per_hour,
                rng=rng,
                return_stats=True,
            )

            net.r[:] = saved_r
            n_grown += created
            hourly_peak_r[:] = 0.0

    synaptic_scaling(net, target_total=2.0, region_filter=abstract_region)

    rest(net, config)

    # ── Child sleep: gentle downscaling + very slow pruning ─────────
    # Synapses weaken during sleep (SHY hypothesis) and structural
    # elimination does happen, but very slowly. At removal_rate=0.01,
    # the weakest edge has ~1% chance of removal per night, meaning
    # it survives ~100 nights on average before being eliminated.
    child_downscale = 1.0 - (1.0 - config.sleep_downscale) * 0.2
    sleep_selective(net, downscale=child_downscale,
                    tag_threshold=config.sleep_tag_threshold)
    n_pruned = prune_competitive(
        net, weak_threshold=config.prune_weak_threshold,
        removal_rate=0.01, rng=rng,
    )

    ema_r = homeostatic_excitability_update(
        net, eta_b=config.eta_b, a_target=config.a_target, ema_r=ema_r,
    )

    return NurseryResult(
        grids_seen=len(grids),
        edges_bound=total_bound,
        edges_grown=n_grown,
        edges_pruned=n_pruned,
        edges_after=net.edge_count(),
        synaptogenesis_stats=last_stats if 'last_stats' in dir() else None,
    ), ema_r


# ── Study examples: observe + learn from worked examples (CHL) ───────

def study_examples(
    net: DNG,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    config: LifecycleConfig | None = None,
) -> float:
    """Legacy wrapper -- delegates to observe_examples. No weight changes."""
    observe_examples(net, train_pairs, config)
    return 0.0


def observe_examples(
    net: DNG,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    config: LifecycleConfig | None = None,
    episodic: EpisodicMemory | None = None,
    binding_eta: float = 0.0,
) -> int:
    """
    Present training examples. Facilitation builds as pathways activate.
    If binding_eta > 0, applies fast Hebbian binding after each example
    (hippocampal one-shot learning on internal/motor edges).

    Returns total number of edges modified by binding (0 if binding off).
    """
    if config is None:
        config = LifecycleConfig()

    mh, mw = net.max_h, net.max_w
    n_total = net.n_nodes
    motor_offset = int(net.output_nodes[0])
    total_bound = 0

    for inp, out in train_pairs:
        inp, out = np.asarray(inp), np.asarray(out)
        in_h, in_w = inp.shape
        out_h, out_w = out.shape

        if episodic is not None:
            episodic.store(inp, out)

        net.V[net.output_nodes] = 0.0
        net.r[net.output_nodes] = 0.0
        net.prev_r[net.output_nodes] = 0.0

        focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                            config.focus_strength)

        sig = (grid_to_signal(inp, 0, n_total, max_h=mh, max_w=mw) +
               grid_to_signal(out, motor_offset, n_total, max_h=mh, max_w=mw) +
               focus)

        think(net, signal=sig, steps=config.observe_steps,
              noise_std=config.noise_std)

        if binding_eta > 0:
            total_bound += fast_hebbian_bind(
                net, eta=binding_eta, w_max=config.w_max,
            )

    think(net, signal=None, steps=config.observe_steps // 2,
          noise_std=config.noise_std)

    return total_bound


# ── Study task: learn from examples, then test without correction ────

def study_task_round(
    net: DNG,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    test_input: np.ndarray,
    test_output: np.ndarray,
    config: LifecycleConfig | None = None,
    prev_best_reward: float = 0.0,
    is_first_visit: bool = True,
    episodic: EpisodicMemory | None = None,
) -> AttemptResult:
    """
    One round of studying a task during childhood.

    Learning loop (as described in module docstring, now implemented):
      1. Observe examples with fast Hebbian binding (hippocampal)
      2. Free phase: think with test input, record correlations
      3. Read guess, compute reward → DA = reward prediction error
      4. Clamped phase: inject correct output, record correlations
      5. CHL update: dw = eta * DA * ACh * (clamped - free)
      6. Retry with refreshed observation if wrong
      7. Restore Hebbian weight changes if task attempt made no progress
    """
    if config is None:
        config = LifecycleConfig()

    test_input = np.asarray(test_input)
    test_output = np.asarray(test_output)
    out_h, out_w = test_output.shape
    mh, mw = net.max_h, net.max_w
    n_total = net.n_nodes
    motor_offset = int(net.output_nodes[0])

    if is_first_visit:
        _soft_reset(net)

    # Snapshot weights before Hebbian binding (task-local learning)
    binding_eta = config.eta * 2.0 if config.eta > 0 else 0.0
    w_snapshot = get_weight_snapshot(net) if binding_eta > 0 else None

    # 1. Observe examples with fast Hebbian binding
    observe_examples(net, train_pairs, config, episodic=episodic,
                     binding_eta=binding_eta)

    mem_mask = net.regions == list(Region).index(Region.MEMORY)
    mem_snapshot = net.r[mem_mask].copy()

    in_h, in_w = test_input.shape
    replay_focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                               config.focus_strength)
    input_signal = (grid_to_signal(test_input, 0, n_total, max_h=mh, max_w=mw) +
                    replay_focus)
    full_signal = (input_signal +
                   grid_to_signal(test_output, motor_offset, n_total, max_h=mh, max_w=mw))

    recall_hint = np.zeros(n_total)
    if episodic is not None:
        recall_hint = episodic.recall_signal(
            test_input, motor_offset, n_total,
            strength=config.memory_hint_strength,
        )

    best_guess = None
    best_reward = prev_best_reward
    total_attempts = 0

    for attempt in range(1, config.attempts_per_round + 1):
        total_attempts = attempt

        focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                            config.focus_strength)
        test_signal = (grid_to_signal(test_input, 0, n_total,
                                      max_h=mh, max_w=mw) + focus + recall_hint)

        # 2. Free phase: suppress copy pathway so internal neurons
        #    actually have to produce the answer (not just copy input).
        #    This gives CHL a meaningful learning signal.
        if config.eta > 0:
            net.V[net.output_nodes] = 0.0
            net.r[net.output_nodes] = 0.0
            net.prev_r[net.output_nodes] = 0.0

            copy_mask, copy_saved = _suppress_sensory_to_motor(net)
            free_corr = record_think(
                net, signal=test_signal,
                steps=config.free_phase_steps,
                noise_std=config.noise_std,
            )
            _restore_sensory_to_motor(net, copy_mask, copy_saved)
        else:
            think(net, signal=test_signal, steps=config.think_steps,
                  noise_std=config.noise_std)

        guess = signal_to_grid(net.r, out_h, out_w,
                               node_offset=motor_offset, max_h=mh, max_w=mw)
        fg = test_output != 0
        reward = float(np.mean(guess[fg] == test_output[fg])) if fg.any() else 1.0

        # 3-5. CHL update: clamped phase (with copy restored) + reward-modulated weight change
        if config.eta > 0:
            rpe = reward - net.da_baseline
            net.da = max(abs(rpe), 0.1)
            net.da_baseline = 0.9 * net.da_baseline + 0.1 * reward

            net.V[net.output_nodes] = 0.0
            net.r[net.output_nodes] = 0.0
            net.prev_r[net.output_nodes] = 0.0

            clamped_corr = record_think(
                net, signal=full_signal,
                steps=config.clamped_phase_steps,
                noise_std=config.noise_std * 0.5,
            )

            contrastive_hebbian_update(
                net, free_corr, clamped_corr,
                eta=config.eta, w_max=config.w_max,
            )

        if reward > best_reward:
            best_reward = reward
            best_guess = guess.copy()

        if reward >= 1.0:
            break

        if attempt < config.attempts_per_round:
            observe_examples(net, train_pairs, config, episodic=episodic,
                             binding_eta=binding_eta)

    # Consolidate weight changes if reward improved; restore on regression.
    # Biologically: you can't un-learn seeing the answer, but selective
    # sleep downscaling handles removing unused associations over time.
    # Only revert if the task was genuinely worse than the previous best
    # (which can happen if prev_best was set by a prior round).
    if w_snapshot is not None and best_reward == 0.0 and prev_best_reward > 0.0:
        n = min(len(w_snapshot), net._edge_count)
        net._edge_w[:n] = w_snapshot[:n]
        net._csr_dirty = True

    net.ne *= 0.3

    return AttemptResult(
        guess=best_guess if best_guess is not None else guess,
        reward=best_reward,
        n_attempts=total_attempts,
        memory_snapshot=mem_snapshot,
        input_signal=input_signal,
        full_signal=full_signal,
    )


def _soft_reset(net: DNG) -> None:
    """Reset transient state but preserve long-term memory activity."""
    mem_mask = net.regions == list(Region).index(Region.MEMORY)
    mem_V = net.V[mem_mask].copy()
    mem_r = net.r[mem_mask].copy()

    net.V[:] *= 0.1
    net.r[:] *= 0.1
    net.prev_r[:] *= 0.1
    net.adaptation[:] *= 0.1
    net.reset_facilitation()

    net.V[mem_mask] = mem_V
    net.r[mem_mask] = mem_r


# ── Day of study ─────────────────────────────────────────────────────

@dataclass
class DayResult:
    attempts: List[AttemptResult]
    mean_reward: float
    n_pruned: int
    n_grown: int
    edges_after: int


def study_day(
    net: DNG,
    tasks: List[Tuple[
        List[Tuple[np.ndarray, np.ndarray]],
        np.ndarray,
        np.ndarray,
    ]],
    config: LifecycleConfig | None = None,
    ema_r: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    episodic: EpisodicMemory | None = None,
    replay_buffer: Dict | None = None,
) -> Tuple[DayResult, np.ndarray]:
    """
    One day: study tasks in round-robin, rest, sleep.

    replay_buffer: dict mapping task_index -> (mem_snapshot, input_signal, full_signal).
    Persists across days so sleep replays ALL learned tasks, not just today's.
    Updated in-place with today's memories.
    """
    if config is None:
        config = LifecycleConfig()

    n_tasks = len(tasks)
    best_rewards = [0.0] * n_tasks
    total_attempts = [0] * n_tasks
    best_guesses = [None] * n_tasks
    solved = [False] * n_tasks
    stuck = [False] * n_tasks

    daily_peak_r = np.zeros(net.n_nodes, dtype=np.float64)

    for rnd in range(config.n_rounds):
        any_active = False
        for i, (train_pairs, test_in, test_out) in enumerate(tasks):
            if solved[i] or stuck[i]:
                continue
            any_active = True

            result = study_task_round(
                net, train_pairs, test_in, test_out, config,
                prev_best_reward=best_rewards[i],
                is_first_visit=(rnd == 0),
                episodic=episodic,
            )

            np.maximum(daily_peak_r, net.r, out=daily_peak_r)

            total_attempts[i] += result.n_attempts

            if result.reward > best_rewards[i]:
                best_rewards[i] = result.reward
                best_guesses[i] = result.guess
            elif result.reward <= best_rewards[i]:
                stuck[i] = True

            if result.reward >= 1.0:
                solved[i] = True

        if not any_active:
            break

        think(net, signal=None, steps=config.consolidation_steps,
              noise_std=config.noise_std)

        stuck = [False] * n_tasks

    attempts = [
        AttemptResult(
            guess=best_guesses[i] if best_guesses[i] is not None
                  else np.zeros_like(tasks[i][2]),
            reward=best_rewards[i],
            n_attempts=total_attempts[i],
        )
        for i in range(n_tasks)
    ]

    saved_r = net.r.copy()
    net.r[:] = daily_peak_r
    n_grown = synaptogenesis(
        net,
        growth_rate=config.growth_rate,
        n_candidates=config.growth_candidates,
        rng=rng,
    )
    net.r[:] = saved_r

    rest(net, config)
    n_pruned, ema_r = sleep(net, config, ema_r, rng, episodic=episodic)

    mean_reward = float(np.mean([a.reward for a in attempts]))
    return DayResult(
        attempts=attempts, mean_reward=mean_reward,
        n_pruned=n_pruned, n_grown=n_grown, edges_after=net.edge_count(),
    ), ema_r


def rest(net: DNG, config: LifecycleConfig | None = None) -> None:
    if config is None:
        config = LifecycleConfig()
    # NE drops during rest (calm down)
    net.ne *= 0.1
    think(net, signal=None, steps=config.rest_steps,
          noise_std=config.rest_noise_std)


def sleep(
    net: DNG,
    config: LifecycleConfig | None = None,
    ema_r: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    day_memories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    episodic: EpisodicMemory | None = None,
    n_cycles: int = 4,
) -> Tuple[int, np.ndarray]:
    """
    Sleep phase: interleaved NREM/REM cycles, like a real night's sleep.

    Real sleep alternates ~4-5 times between:
      NREM: synaptic downscaling, gradual pruning, consolidation
      REM:  episodic replay with gentle CHL learning

    Each cycle does a fraction of the total work, so changes are
    gradual and interleaved -- not one big batch.
    """
    if config is None:
        config = LifecycleConfig()
    if rng is None:
        rng = np.random.default_rng()

    saved_ach = net.ach
    saved_da = net.da
    net.ne = 0.0

    mh, mw = net.max_h, net.max_w
    n_total = net.n_nodes
    motor_offset = int(net.output_nodes[0])

    has_memories = (episodic is not None and len(episodic.episodes) > 0
                    and config.replay_eta > 0)

    # Per-cycle downscale: total effect across all cycles equals
    # the original single-shot downscale.
    # downscale^1 = per_cycle^n_cycles  =>  per_cycle = downscale^(1/n_cycles)
    per_cycle_downscale = config.sleep_downscale ** (1.0 / n_cycles)
    per_cycle_consolidation = config.consolidation_decay ** (1.0 / n_cycles)

    total_pruned = 0

    for cycle in range(n_cycles):
        # ── NREM phase: downscale + prune (gradual) ────────────────
        net.ach = 0.1
        net.da = 0.0

        sleep_selective(net, downscale=per_cycle_downscale,
                        tag_threshold=config.sleep_tag_threshold)
        n_pruned = prune_competitive(
            net, weak_threshold=config.prune_weak_threshold, rng=rng,
        )
        total_pruned += n_pruned

        n = net._edge_count
        net._edge_consolidation[:n] *= per_cycle_consolidation

        # ── REM phase: replay a subset of memories ─────────────────
        if has_memories:
            net.ach = 0.5
            net.da = 0.5

            episodes = episodic.episodes
            n_eps = len(episodes)
            # Each cycle replays a fraction of episodes (shuffled)
            order = rng.permutation(n_eps)
            n_replay = max(1, n_eps // n_cycles)
            for idx in order[:n_replay]:
                ep = episodes[idx]
                inp, out = ep.input_grid, ep.output_grid
                in_h, in_w = inp.shape
                out_h, out_w = out.shape

                focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                                    config.focus_strength)
                input_signal = (
                    grid_to_signal(inp, 0, n_total, max_h=mh, max_w=mw) + focus
                )
                full_signal = (
                    input_signal +
                    grid_to_signal(out, motor_offset, n_total, max_h=mh, max_w=mw)
                )

                free_corr = record_think(
                    net, signal=input_signal,
                    steps=config.replay_steps,
                    noise_std=config.noise_std * 0.5,
                )
                clamped_corr = record_think(
                    net, signal=full_signal,
                    steps=config.replay_steps,
                    noise_std=config.noise_std * 0.3,
                )
                contrastive_hebbian_update(
                    net, free_corr, clamped_corr,
                    eta=config.replay_eta,
                    w_max=config.w_max,
                )

        # Brief homeostatic adjustment between cycles
        ema_r = homeostatic_excitability_update(
            net, eta_b=config.eta_b, a_target=config.a_target, ema_r=ema_r,
        )

    net.ach = saved_ach
    return total_pruned, ema_r


# ── Adult solve: NO weight changes ──────────────────────────────────

@dataclass
class ReadoutResult:
    grid: np.ndarray
    decided: bool
    steps_taken: int
    per_cell_confidence: np.ndarray


def solve_task(
    net: DNG,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    test_input: np.ndarray,
    config: LifecycleConfig | None = None,
    output_h: int | None = None,
    output_w: int | None = None,
    episodic: EpisodicMemory | None = None,
) -> ReadoutResult:
    """
    Solve a task with the mature (adult) network.
    NO weight changes. Low ACh (recall mode).
    Episodic memory provides recall hints + spontaneous associations.
    """
    if config is None:
        config = LifecycleConfig()

    test_input = np.asarray(test_input)
    mh, mw = net.max_h, net.max_w
    motor_offset = int(net.output_nodes[0])

    if output_h is None:
        output_h = test_input.shape[0]
    if output_w is None:
        output_w = test_input.shape[1]

    net.ach = 0.2
    net.ne = 0.0
    net.da = 0.0

    _soft_reset(net)

    # Store this task's examples (episodic memory accumulates across tasks).
    # Recall is similarity-based so only relevant examples contribute.
    if episodic is not None:
        episodic.store_pairs(train_pairs)
    observe_examples(net, train_pairs, config, episodic=episodic)

    n_total = net.n_nodes
    in_h, in_w = test_input.shape
    focus = _focus_mask(net, max(in_h, output_h), max(in_w, output_w),
                        config.focus_strength)

    # Deliberate recall: "I've seen something similar, the answer looked like..."
    recall_hint = np.zeros(n_total)
    if episodic is not None:
        recall_hint = episodic.recall_signal(
            test_input, motor_offset, n_total,
            strength=config.memory_hint_strength,
        )

    test_signal = (grid_to_signal(test_input, 0, n_total,
                                  max_h=mh, max_w=mw) + focus + recall_hint)

    prev_r = net.r.copy()
    rng = np.random.default_rng()
    for s in range(1, config.t_max + 1):
        # Spontaneous recall every ~10 steps: a faint association surfaces
        if episodic is not None and s % 10 == 0:
            spont = episodic.spontaneous_signal(
                net.r, net.input_nodes, motor_offset, n_total,
                strength=config.spontaneous_strength, rng=rng,
            )
            step(net, signal=test_signal + spont, noise_std=config.noise_std * 0.5)
        else:
            step(net, signal=test_signal, noise_std=config.noise_std * 0.5)

        conf = _cell_confidence(net.r, motor_offset, output_h, output_w, mh, mw)
        all_decided = bool(np.all(conf > config.theta_conf))
        rate_change = np.max(np.abs(net.r - prev_r))

        if all_decided and rate_change < 0.01:
            grid = signal_to_grid(net.r, output_h, output_w,
                                  node_offset=motor_offset, max_h=mh, max_w=mw)
            return ReadoutResult(
                grid=grid, decided=True, steps_taken=s,
                per_cell_confidence=conf,
            )
        prev_r = net.r.copy()

    conf = _cell_confidence(net.r, motor_offset, output_h, output_w, mh, mw)
    grid = signal_to_grid(net.r, output_h, output_w,
                          node_offset=motor_offset, max_h=mh, max_w=mw)
    return ReadoutResult(
        grid=grid, decided=False, steps_taken=config.t_max,
        per_cell_confidence=conf,
    )
