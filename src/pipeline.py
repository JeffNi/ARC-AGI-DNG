"""
DNG lifecycle: error-corrective three-factor learning.

Two modes:
  CHILDHOOD (error-corrective learning):
    1. Observe training examples (priming only — facilitation, episodic store)
    2. Attempt: think with test input, record activity, read guess
    3. Compute signed motor error (target - guess activity)
    4. Inject error at motor, propagate through feedback connections
    5. Three-factor update: dw = eta * pre * delta_post * DA
    6. Repeat if not solved; sleep at end of day

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
    error_corrective_update,
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

    # Error-corrective learning (three-factor, cerebellar model)
    eta: float = 0.05
    w_max: float = 2.5
    error_prop_steps: int = 15   # steps to propagate error signal through feedback

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

    # Pruning (competitive probabilistic)
    prune_weak_threshold: float = 0.01
    prune_removal_rate: float = 0.3     # probability scale: 0.3=adult, 0.01=child
    prune_cycles_required: int = 5      # legacy, unused by prune_competitive

    # Synaptogenesis (edge growth -- probability-based, modulated by ACh)
    growth_rate: float = 0.3          # base probability scale for new connections
    growth_candidates: int = 50000    # how many random pairs to evaluate per day

    # Readout
    theta_conf: float = 0.3
    t_max: int = 300

    # Sleep replay (consolidation)
    replay_eta: float = 0.003       # gentle contrastive update during sleep replay
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
    values: np.ndarray,
    motor_offset: int,
    h: int,
    w: int,
    max_h: int,
    max_w: int,
) -> np.ndarray:
    """Gap between top-2 motor neuron values per cell (works with V or r)."""
    n_cells = max_h * max_w
    motor_r = values[motor_offset : motor_offset + n_cells * NUM_COLORS]
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
    extra_signal: np.ndarray | None = None,
) -> int:
    """
    Present training examples. Facilitation builds as pathways activate.

    During childhood, binding_eta=0 (priming only). During infancy,
    binding_eta > 0 applies fast Hebbian binding after each example.

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
        if extra_signal is not None:
            sig = sig + extra_signal

        think(net, signal=sig, steps=config.observe_steps,
              noise_std=config.noise_std)

        if binding_eta > 0:
            total_bound += fast_hebbian_bind(
                net, eta=binding_eta, w_max=config.w_max,
            )

    think(net, signal=None, steps=config.observe_steps // 2,
          noise_std=config.noise_std)

    return total_bound


# ── Study task: error-corrective three-factor learning ────────────

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

    Learning flow (cerebellar error-corrective model):
      1. Observe examples — priming only, no weight changes
      2. Attempt: think with test input, record activity, read guess
      3. Compute signed motor error (target - guess activity)
      4. Inject error at motor, propagate through feedback connections
      5. Three-factor update: dw = eta * pre * delta_post * DA
      6. Update DA baseline from reward prediction error
      7. Repeat if not solved
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

    # Snapshot weights so we can revert if the task goes badly
    w_snapshot = get_weight_snapshot(net) if config.eta > 0 else None

    # 1. Observe examples — priming only (facilitation, episodic store)
    observe_examples(net, train_pairs, config, episodic=episodic,
                     binding_eta=0.0)

    mem_mask = net.regions == list(Region).index(Region.MEMORY)
    mem_snapshot = net.r[mem_mask].copy()

    in_h, in_w = test_input.shape
    replay_focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                               config.focus_strength)
    input_signal = (grid_to_signal(test_input, 0, n_total, max_h=mh, max_w=mw) +
                    replay_focus)
    full_signal = (input_signal +
                   grid_to_signal(test_output, motor_offset, n_total, max_h=mh, max_w=mw))

    # Target signal at motor neurons (what correct output looks like)
    target_motor = grid_to_signal(test_output, motor_offset, n_total,
                                  max_h=mh, max_w=mw)

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

        # 2. Attempt: think with test input only
        think(net, signal=test_signal, steps=config.think_steps,
              noise_std=config.noise_std)

        r_guess = net.r.copy()

        guess = signal_to_grid(net.V, out_h, out_w,
                               node_offset=motor_offset, max_h=mh, max_w=mw)
        fg = test_output != 0
        reward = float(np.mean(guess[fg] == test_output[fg])) if fg.any() else 1.0

        # 3. Update DA from reward prediction error
        rpe = reward - net.da_baseline
        net.da = max(abs(rpe), 0.1)
        net.da_baseline = 0.9 * net.da_baseline + 0.1 * reward

        if reward > best_reward:
            best_reward = reward
            best_guess = guess.copy()

        if reward >= 1.0:
            break

        # 4. Error injection: signed correction at motor neurons.
        #    Motor WTA in dynamics gives differentiated rates, so error
        #    has both + (strengthen correct) and - (weaken wrong) components.
        motor_r = net.r[net.output_nodes]
        target_r = target_motor[net.output_nodes]
        error_signal = np.zeros(n_total)
        error_signal[net.output_nodes] = target_r - motor_r

        # 5. Propagate error through feedback connections.
        #    Low noise so the error signal dominates over stochastic activity.
        think(net, signal=error_signal, steps=config.error_prop_steps,
              noise_std=config.noise_std * 0.25)

        r_corrected = net.r.copy()

        # 6. Three-factor update: pre * delta_post * DA
        #    Uses r_guess with sharpened motor (decision signal) but raw
        #    abstract/sensory rates (pre-synaptic activity from full dynamics).
        if config.eta > 0:
            error_corrective_update(
                net, r_guess, r_corrected,
                eta=config.eta, w_max=config.w_max,
            )

    # Revert weights if the task didn't improve. Failed learning attempts
    # on one task corrupt weights that other tasks depend on. Rolling back
    # when there's no progress prevents catastrophic interference while
    # still allowing exploration on the attempts that did help.
    if w_snapshot is not None and best_reward <= prev_best_reward:
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


def _sharpen_motor(
    r: np.ndarray,
    output_nodes: np.ndarray,
    out_h: int, out_w: int,
    max_h: int, max_w: int,
) -> np.ndarray:
    """Per-cell WTA on motor neurons: keep only the winning color per position.

    Applied as post-processing for the learning rule, NOT during dynamics.
    This gives the three-factor update a clean decision signal (the network's
    "guess") while the actual dynamics keep motor neurons unsuppressed to
    maintain abstract layer excitation via recurrence.

    Sharpens ALL motor cells so unused positions don't leak saturated rates
    into the three-factor update's pre-synaptic terms.
    """
    r_sharp = r.copy()
    n_cells = max_h * max_w
    n_colors = NUM_COLORS
    motor_start = int(output_nodes[0])
    for cell in range(n_cells):
        base = motor_start + cell * n_colors
        if base + n_colors > len(r_sharp):
            break
        block = r_sharp[base:base + n_colors]
        winner = np.argmax(block)
        for c in range(n_colors):
            if c != winner:
                block[c] = 0.0
    return r_sharp


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

    # Synaptogenesis after each task study, matching infancy's per-grid
    # pattern. Peak activity accumulated within each task's study round
    # provides the co-activation signal for "fire together, wire together."
    n_active_tasks = sum(1 for _ in tasks)
    n_grow_calls = max(1, n_active_tasks * config.n_rounds)
    cands_per_call = max(1, config.growth_candidates // n_grow_calls)
    n_grown = 0

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

            total_attempts[i] += result.n_attempts

            if result.reward > best_rewards[i]:
                best_rewards[i] = result.reward
                best_guesses[i] = result.guess
            elif result.reward <= best_rewards[i]:
                stuck[i] = True

            if result.reward >= 1.0:
                solved[i] = True

            # Synaptogenesis after each task using current activity.
            # Each task activates different neurons, providing the
            # diverse co-activation patterns growth needs.
            n_grown += synaptogenesis(
                net,
                growth_rate=config.growth_rate,
                n_candidates=cands_per_call,
                rng=rng,
            )

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

    # Synaptic scaling (same as infancy)
    abstract_region = list(Region).index(Region.ABSTRACT)
    synaptic_scaling(net, target_total=2.0, region_filter=abstract_region)

    rest(net, config)

    # Structural maintenance: identical to infancy's nursery_exposure.
    # Single-pass sleep_selective + single prune_competitive.
    # Hyperparams control intensity; mechanism stays the same.
    sleep_selective(net, downscale=config.sleep_downscale,
                    tag_threshold=config.sleep_tag_threshold)
    n_pruned = prune_competitive(
        net, weak_threshold=config.prune_weak_threshold,
        removal_rate=config.prune_removal_rate, rng=rng,
    )

    ema_r = homeostatic_excitability_update(
        net, eta_b=config.eta_b, a_target=config.a_target, ema_r=ema_r,
    )

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

    # Per-cycle parameters: total effect across all cycles equals
    # the single-shot config value. Same approach for downscale,
    # consolidation decay, and pruning removal rate.
    # downscale^1 = per_cycle^n  =>  per_cycle = downscale^(1/n)
    # survival: (1-r_total) = (1-r_per)^n  =>  r_per = 1-(1-r_total)^(1/n)
    per_cycle_downscale = config.sleep_downscale ** (1.0 / n_cycles)
    per_cycle_consolidation = config.consolidation_decay ** (1.0 / n_cycles)
    per_cycle_removal = 1.0 - (1.0 - config.prune_removal_rate) ** (1.0 / n_cycles)

    total_pruned = 0

    for cycle in range(n_cycles):
        # ── NREM phase: downscale + prune (gradual) ────────────────
        net.ach = 0.1
        net.da = 0.0

        sleep_selective(net, downscale=per_cycle_downscale,
                        tag_threshold=config.sleep_tag_threshold)
        n_pruned = prune_competitive(
            net, weak_threshold=config.prune_weak_threshold,
            removal_rate=per_cycle_removal, rng=rng,
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

        conf = _cell_confidence(net.V, motor_offset, output_h, output_w, mh, mw)
        all_decided = bool(np.all(conf > config.theta_conf))
        rate_change = np.max(np.abs(net.r - prev_r))

        if all_decided and rate_change < 0.01:
            grid = signal_to_grid(net.V, output_h, output_w,
                                  node_offset=motor_offset, max_h=mh, max_w=mw)
            return ReadoutResult(
                grid=grid, decided=True, steps_taken=s,
                per_cell_confidence=conf,
            )
        prev_r = net.r.copy()

    conf = _cell_confidence(net.V, motor_offset, output_h, output_w, mh, mw)
    grid = signal_to_grid(net.V, output_h, output_w,
                          node_offset=motor_offset, max_h=mh, max_w=mw)
    return ReadoutResult(
        grid=grid, decided=False, steps_taken=config.t_max,
        per_cell_confidence=conf,
    )
