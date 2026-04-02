"""
Childhood: raise a DNG through developmental phases.

Four phases modeled on human brain development:

  CHILDHOOD (0-40% of days):
    High ACh, high learning rate, NO pruning.
    Network explores freely, overproduced connections get tagged by CHL.

  ADOLESCENCE (40-70% of days):
    ACh declining, learning rate declining, AGGRESSIVE pruning.
    Untagged connections removed. Circuits sharpen.

  STABILIZATION (70-90% of days):
    Low ACh, low learning rate, gentle pruning.
    Fine-tuning. Surviving circuits lock in.

  YOUNG ADULT (90-100% of days):
    Very low ACh, minimal learning, no pruning.
    Final consolidation before maturity.

Checkpoints saved with adaptive frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .graph import DNG
from .pipeline import LifecycleConfig, study_day, DayResult


@dataclass
class ChildhoodConfig:
    n_days: int = 50
    tasks_per_day: int = 8
    checkpoint_path: str | None = None
    verbose: bool = True

    # Phase boundaries (fractions of n_days)
    childhood_end: float = 0.40
    adolescence_end: float = 0.70
    stabilization_end: float = 0.90
    # rest is young adult until 1.0


@dataclass
class ChildhoodResult:
    day_rewards: List[float]
    day_edges: List[int]
    final_edges: int


TaskTuple = Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
]


def _should_checkpoint(day: int) -> bool:
    if day <= 10:
        return True
    if day <= 50:
        return day % 10 == 0
    return day % 50 == 0


def extract_tasks(arc_tasks) -> List[TaskTuple]:
    """Convert arckit task objects into (train_pairs, test_in, test_out)."""
    tasks = []
    for task in arc_tasks:
        train_pairs = [(np.array(i), np.array(o)) for i, o in task.train]
        if task.test:
            test_in = np.array(task.test[0][0])
            test_out = np.array(task.test[0][1])
            tasks.append((train_pairs, test_in, test_out))
    return tasks


def _phase_name(progress: float, config: ChildhoodConfig) -> str:
    if progress < config.childhood_end:
        return "child"
    elif progress < config.adolescence_end:
        return "adoles"
    elif progress < config.stabilization_end:
        return "stable"
    else:
        return "y.adult"


def _schedule(
    day: int,
    n_days: int,
    config: ChildhoodConfig,
    base_lifecycle: LifecycleConfig,
) -> LifecycleConfig:
    """
    Return a LifecycleConfig with parameters adjusted for the current
    developmental phase. Creates a modified copy each day.
    """
    progress = day / n_days  # 0.0 to 1.0
    lc = LifecycleConfig(
        observe_steps=base_lifecycle.observe_steps,
        think_steps=base_lifecycle.think_steps,
        consolidation_steps=base_lifecycle.consolidation_steps,
        w_max=base_lifecycle.w_max,
        free_phase_steps=base_lifecycle.free_phase_steps,
        clamped_phase_steps=base_lifecycle.clamped_phase_steps,
        attempts_per_round=base_lifecycle.attempts_per_round,
        n_rounds=base_lifecycle.n_rounds,
        stuck_patience=base_lifecycle.stuck_patience,
        noise_std=base_lifecycle.noise_std,
        eta_b=base_lifecycle.eta_b,
        a_target=base_lifecycle.a_target,
        rest_steps=base_lifecycle.rest_steps,
        rest_noise_std=base_lifecycle.rest_noise_std,
        theta_conf=base_lifecycle.theta_conf,
        t_max=base_lifecycle.t_max,
        focus_strength=base_lifecycle.focus_strength,
    )

    base_eta = base_lifecycle.eta

    # Shared: no weight decay (consolidation protects important synapses).
    lc.sleep_downscale = 1.0
    lc.sleep_tag_threshold = 0.001
    lc.replay_eta = base_lifecycle.replay_eta
    lc.replay_passes = base_lifecycle.replay_passes
    lc.replay_steps = base_lifecycle.replay_steps

    if progress < config.childhood_end:
        # CHILDHOOD: high growth, no pruning -- overshoot phase.
        lc.eta = base_eta
        lc.n_rounds = 2
        lc.attempts_per_round = 4
        lc.growth_rate = 0.3
        lc.growth_candidates = 50000
        lc.prune_weak_threshold = 0.0
        lc.prune_cycles_required = 999

    elif progress < config.adolescence_end:
        # ADOLESCENCE: moderate growth, gentle pruning of unused edges.
        phase_progress = ((progress - config.childhood_end)
                          / (config.adolescence_end - config.childhood_end))
        lc.eta = base_eta * (1.0 - 0.4 * phase_progress)
        lc.n_rounds = 2
        lc.attempts_per_round = 3
        lc.growth_rate = 0.15
        lc.growth_candidates = 30000
        lc.prune_weak_threshold = 0.002
        lc.prune_cycles_required = 5

    elif progress < config.stabilization_end:
        # STABILIZATION: low growth, moderate pruning -- circuits sharpen.
        phase_progress = ((progress - config.adolescence_end)
                          / (config.stabilization_end - config.adolescence_end))
        lc.eta = base_eta * (0.6 - 0.3 * phase_progress)
        lc.n_rounds = 2
        lc.attempts_per_round = 3
        lc.growth_rate = 0.05
        lc.growth_candidates = 10000
        lc.prune_weak_threshold = 0.005
        lc.prune_cycles_required = 3

    else:
        # YOUNG ADULT: rare growth, gentle pruning -- mature circuits.
        lc.eta = base_eta * 0.2
        lc.n_rounds = 2
        lc.attempts_per_round = 2
        lc.growth_rate = 0.02
        lc.growth_candidates = 5000
        lc.prune_weak_threshold = 0.003
        lc.prune_cycles_required = 5

    return lc


def run_childhood(
    net: DNG,
    all_tasks: List[TaskTuple],
    config: ChildhoodConfig | None = None,
    lifecycle: LifecycleConfig | None = None,
    rng: np.random.Generator | None = None,
) -> ChildhoodResult:
    if config is None:
        config = ChildhoodConfig()
    if lifecycle is None:
        lifecycle = LifecycleConfig()
    if rng is None:
        rng = np.random.default_rng()

    day_rewards = []
    day_edges = []
    ema_r = None
    ema_reward = None
    replay_buffer: dict = {}

    for day in range(1, config.n_days + 1):
        progress = day / config.n_days
        phase = _phase_name(progress, config)
        day_lc = _schedule(day, config.n_days, config, lifecycle)

        if progress < config.childhood_end:
            net.ach = 1.0
        elif progress < config.adolescence_end:
            phase_p = ((progress - config.childhood_end)
                       / (config.adolescence_end - config.childhood_end))
            net.ach = 1.0 - 0.5 * phase_p
        elif progress < config.stabilization_end:
            phase_p = ((progress - config.adolescence_end)
                       / (config.stabilization_end - config.adolescence_end))
            net.ach = 0.5 - 0.2 * phase_p
        else:
            net.ach = 0.2

        n_pick = min(config.tasks_per_day, len(all_tasks))
        indices = rng.choice(len(all_tasks), size=n_pick, replace=False)
        today = [all_tasks[i] for i in indices]

        day_result, ema_r = study_day(net, today, day_lc, ema_r, rng,
                                      replay_buffer=replay_buffer)
        day_rewards.append(day_result.mean_reward)
        day_edges.append(day_result.edges_after)

        if ema_reward is None:
            ema_reward = day_result.mean_reward
        else:
            ema_reward = 0.9 * ema_reward + 0.1 * day_result.mean_reward

        if config.verbose:
            solves = sum(1 for a in day_result.attempts if a.reward >= 1.0)
            rewards = [f"{a.reward:.0%}({a.n_attempts})" for a in day_result.attempts]
            n_cons = int((net._edge_consolidation[:net._edge_count] > 0.1).sum())
            print(f"  Day {day:3d} [{phase:>7s}] ACh={net.ach:.2f} "
                  f"eta={day_lc.eta:.3f}: "
                  f"reward={day_result.mean_reward:.2f} "
                  f"ema={ema_reward:.2f} "
                  f"solves={solves}/{len(day_result.attempts)} "
                  f"edges={day_result.edges_after} "
                  f"+{day_result.n_grown}/-{day_result.n_pruned} "
                  f"cons={n_cons} replay={len(replay_buffer)} "
                  f"[{', '.join(rewards)}]", flush=True)

        if config.checkpoint_path and _should_checkpoint(day):
            path = f"{config.checkpoint_path}_day{day}.npz"
            net.save(path)
            if config.verbose:
                print(f"         -> checkpoint: {path}", flush=True)

    return ChildhoodResult(
        day_rewards=day_rewards,
        day_edges=day_edges,
        final_edges=net.edge_count(),
    )
