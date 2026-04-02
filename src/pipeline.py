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
    homeostatic_excitability_update,
    prune_sustained,
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

    # Episodic memory (tensor buffer)
    memory_hint_strength: float = 1.0   # recalled outputs bias motor neurons (same scale as input)
    spontaneous_strength: float = 0.15  # faint associative signals during thinking

    # Focus signal
    focus_strength: float = 0.5

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
) -> None:
    """
    Present training examples passively. NO weight changes.
    Facilitation builds as pathways activate.
    If episodic memory is provided, stores the examples for later recall.
    """
    if config is None:
        config = LifecycleConfig()

    mh, mw = net.max_h, net.max_w
    n_total = net.n_nodes
    motor_offset = int(net.output_nodes[0])

    for inp, out in train_pairs:
        inp, out = np.asarray(inp), np.asarray(out)
        in_h, in_w = inp.shape
        out_h, out_w = out.shape

        if episodic is not None:
            episodic.store(inp, out)

        focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                            config.focus_strength)

        sig = (grid_to_signal(inp, 0, n_total, max_h=mh, max_w=mw) +
               grid_to_signal(out, motor_offset, n_total, max_h=mh, max_w=mw) +
               focus)

        think(net, signal=sig, steps=config.observe_steps,
              noise_std=config.noise_std)

    think(net, signal=None, steps=config.observe_steps // 2,
          noise_std=config.noise_std)


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

    Episodic memory provides a recall hint during the free phase --
    the network gets a nudge toward what stored examples suggest
    the answer might look like.
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

    # Task-scope the episodic buffer so recall only uses THIS task's examples
    if episodic is not None:
        episodic.clear()

    # 1. Observe examples (builds facilitation, stores in episodic memory)
    observe_examples(net, train_pairs, config, episodic=episodic)

    # Snapshot for sleep replay
    mem_mask = net.regions == list(Region).index(Region.MEMORY)
    mem_snapshot = net.r[mem_mask].copy()

    in_h, in_w = test_input.shape
    replay_focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                               config.focus_strength)
    input_signal = (grid_to_signal(test_input, 0, n_total, max_h=mh, max_w=mw) +
                    replay_focus)
    full_signal = (input_signal +
                   grid_to_signal(test_output, motor_offset, n_total, max_h=mh, max_w=mw))

    # Compute recall hint from episodic memory
    recall_hint = np.zeros(n_total)
    if episodic is not None:
        recall_hint = episodic.recall_signal(
            test_input, motor_offset, n_total,
            strength=config.memory_hint_strength,
        )

    best_guess = None
    best_reward = prev_best_reward
    total_attempts = 0

    w_snapshot = get_weight_snapshot(net)

    for attempt in range(1, config.attempts_per_round + 1):
        total_attempts = attempt

        # 2. Think: test input + episodic recall hint
        focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                            config.focus_strength)
        test_signal = (grid_to_signal(test_input, 0, n_total,
                                      max_h=mh, max_w=mw) + focus + recall_hint)

        free_corr = record_phase(
            net, step, test_signal,
            config.free_phase_steps, config.noise_std,
        )

        guess = signal_to_grid(net.r, out_h, out_w,
                               node_offset=motor_offset, max_h=mh, max_w=mw)
        fg = test_output != 0
        reward = float(np.mean(guess[fg] == test_output[fg])) if fg.any() else 1.0

        # 3. CHL: show the correct answer
        clamped_signal = (test_signal +
                          grid_to_signal(test_output, motor_offset, n_total,
                                         max_h=mh, max_w=mw))
        clamped_corr = record_phase(
            net, step, clamped_signal,
            config.clamped_phase_steps, config.noise_std * 0.5,
        )

        net.da = max(0.1, 1.0 - reward)
        net.ne = min(1.0, (1.0 - reward) * 1.5)

        contrastive_hebbian_update(
            net, free_corr, clamped_corr,
            eta=config.eta,
            w_max=config.w_max,
        )

        if reward > best_reward:
            best_reward = reward
            best_guess = guess.copy()

        if reward >= 1.0:
            break

        if attempt < config.attempts_per_round:
            observe_examples(net, train_pairs, config, episodic=episodic)

    # Consolidate synapses that changed during successful learning.
    # Only triggers on high performance; accumulates gradually so
    # synapses confirmed important by repeated success become very stable.
    consolidate_synapses(net, w_snapshot, best_reward,
                         reward_threshold=0.8, consolidation_strength=0.5)

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

            if rnd == 0 and result.memory_snapshot is not None:
                if replay_buffer is not None:
                    task_key = hash(test_in.tobytes())
                    replay_buffer[task_key] = (result.memory_snapshot,
                                               result.input_signal,
                                               result.full_signal)

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

    # Sleep replays ALL accumulated memories, not just today's.
    # Cap buffer to prevent unbounded growth with many tasks.
    MAX_REPLAY_MEMORIES = 50
    if replay_buffer is not None and len(replay_buffer) > MAX_REPLAY_MEMORIES:
        keys = list(replay_buffer.keys())
        for k in keys[:len(keys) - MAX_REPLAY_MEMORIES]:
            del replay_buffer[k]
    all_memories = list((replay_buffer or {}).values())

    rest(net, config)
    n_pruned, ema_r = sleep(net, config, ema_r, rng, day_memories=all_memories)

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
) -> Tuple[int, np.ndarray]:
    """
    Sleep phase: replay, consolidation, pruning, homeostasis.

    Replay: reactivate each stored memory pattern, let it drive the
    network (free phase), then clamp the correct signal (clamped phase),
    and apply gentle CHL. Interleaved across all memories to prevent
    any single task from dominating weight changes.
    """
    if config is None:
        config = LifecycleConfig()

    saved_ach = net.ach
    saved_da = net.da
    net.ne = 0.0

    mem_mask = net.regions == list(Region).index(Region.MEMORY)

    # ── Sleep replay: consolidate day's memories ──────────────────
    # ACh and DA are elevated during REM sleep to enable consolidation.
    if day_memories and config.replay_eta > 0:
        net.ach = 1.0
        net.da = 1.0
        for _pass in range(config.replay_passes):
            for mem_snapshot, input_signal, full_signal in day_memories:
                # Inject stored memory pattern
                net.r[mem_mask] = mem_snapshot
                net.V[mem_mask] = mem_snapshot + net.threshold[mem_mask]

                # Free replay: memory + sensory input → what does the network produce?
                free_corr = record_think(
                    net, signal=input_signal,
                    steps=config.replay_steps,
                    noise_std=config.noise_std * 0.5,
                )

                # Re-inject memory before clamped phase
                net.r[mem_mask] = mem_snapshot
                net.V[mem_mask] = mem_snapshot + net.threshold[mem_mask]

                # Clamped replay: memory + input + correct output
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

    # ── Non-REM: quiet phase for downscaling/pruning ─────────────
    net.ach = 0.1
    net.da = 0.0

    # ── Synaptic downscaling + pruning ────────────────────────────
    sleep_selective(net, downscale=config.sleep_downscale,
                    tag_threshold=config.sleep_tag_threshold)
    n_pruned = prune_sustained(net, weak_threshold=config.prune_weak_threshold,
                               cycles_required=config.prune_cycles_required)

    ema_r = homeostatic_excitability_update(
        net, eta_b=config.eta_b, a_target=config.a_target, ema_r=ema_r,
    )

    net.ach = saved_ach
    return n_pruned, ema_r


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

    # Task-scope the episodic buffer so recall only uses THIS task's examples
    if episodic is not None:
        episodic.clear()
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
