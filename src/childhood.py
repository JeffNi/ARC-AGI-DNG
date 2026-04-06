"""
Childhood: raise a DNG through developmental phases.

Learning uses error-corrective three-factor plasticity (cerebellar model):
the network attempts a task, receives signed error at motor neurons, and
error propagates through feedback connections. Weights adjust where
pre-activity, post-error, and dopamine converge.

Four phases modeled on human brain development:

  CHILDHOOD (0-40% of days):
    High ACh, high growth, NO pruning.
    Gentle learning rate, many attempts per task.

  ADOLESCENCE (40-70% of days):
    ACh declining, peak learning rate, gentle pruning begins.
    Untagged connections weakened. Circuits sharpen.

  STABILIZATION (70-90% of days):
    Low ACh, declining learning rate, moderate pruning.
    Fine-tuning. Surviving circuits lock in.

  YOUNG ADULT (90-100% of days):
    Very low ACh, minimal learning, strong replay consolidation.
    Final consolidation before maturity.

Checkpoints saved with adaptive frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .graph import DNG, Region
from .pipeline import LifecycleConfig, study_day, DayResult

ADULT_INH = 1.0
ADULT_WTA = 0.15
ADULT_EXC_CAP = 5.0


@dataclass
class ChildhoodConfig:
    n_days: int = 50
    tasks_per_day: int = 3
    checkpoint_path: str | None = None
    verbose: bool = True

    # Mastery-based curriculum: only advance when ALL current tasks are solved
    mastery_patience: int = 5        # days before advancing anyway if stuck

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


def extract_tasks(arc_tasks, max_h: int = 30, max_w: int = 30) -> List[TaskTuple]:
    """Convert arckit task objects into (train_pairs, test_in, test_out).

    Skips tasks where any grid exceeds max_h x max_w.
    """
    tasks = []
    for task in arc_tasks:
        train_pairs = [(np.array(i), np.array(o)) for i, o in task.train]
        if not task.test:
            continue
        test_in = np.array(task.test[0][0])
        test_out = np.array(task.test[0][1])
        all_grids = [g for pair in train_pairs for g in pair] + [test_in, test_out]
        if any(g.shape[0] > max_h or g.shape[1] > max_w for g in all_grids):
            continue
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


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a at t=0, b at t=1."""
    return a + (b - a) * t


def _schedule(
    day: int,
    n_days: int,
    config: ChildhoodConfig,
    base_lifecycle: LifecycleConfig,
) -> LifecycleConfig:
    """
    Return a LifecycleConfig with parameters interpolated smoothly
    through developmental phases. base_lifecycle carries infancy-exit
    values for structural plasticity, so childhood picks up exactly
    where infancy left off — no sudden jumps.

    Phase targets (what each phase ramps toward):
      CHILDHOOD  -> moderate growth, gentle pruning, gentle sleep
      ADOLESCENCE -> declining growth, rising pruning (synaptic elimination)
      STABILIZATION -> minimal growth, moderate pruning, strong consolidation
      YOUNG ADULT -> mature: minimal growth/pruning, strong replay
    """
    progress = day / n_days  # 0.0 to 1.0

    lc = LifecycleConfig(
        observe_steps=base_lifecycle.observe_steps,
        think_steps=base_lifecycle.think_steps,
        consolidation_steps=base_lifecycle.consolidation_steps,
        w_max=base_lifecycle.w_max,
        error_prop_steps=base_lifecycle.error_prop_steps,
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

    # ── Infancy-exit starting points (from base_lifecycle) ────────
    inf_gr = base_lifecycle.growth_rate
    inf_gc = base_lifecycle.growth_candidates
    inf_pr = base_lifecycle.prune_removal_rate
    inf_pw = base_lifecycle.prune_weak_threshold
    inf_sd = base_lifecycle.sleep_downscale
    inf_st = base_lifecycle.sleep_tag_threshold

    # ── Phase targets: end-of-phase values ────────────────────────
    # Biology: synaptogenesis peaks in early childhood and stays high
    # until adolescence, when pruning overtakes growth (synaptic
    # elimination wave). Growth shouldn't crater during childhood.
    #                       gr    gc      pr    pw     sd     st
    childhood_tgt =       (0.70, 400000, 0.01, 0.003, 0.998, 0.003)
    adolescence_tgt =     (0.25, 150000, 0.08, 0.005, 0.97,  0.01)
    stabilization_tgt =   (0.08,  50000, 0.15, 0.008, 0.95,  0.02)
    adult_tgt =           (0.03,  20000, 0.20, 0.005, 0.96,  0.02)

    ce = config.childhood_end
    ae = config.adolescence_end
    se = config.stabilization_end

    if progress < ce:
        # CHILDHOOD: ramp from infancy-exit -> childhood target
        t = progress / ce
        start = (inf_gr, inf_gc, inf_pr, inf_pw, inf_sd, inf_st)
        end = childhood_tgt
        lc.eta = 0.05
        lc.n_rounds = 4
        lc.attempts_per_round = 5
        lc.consolidation_strength = _lerp(0.1, 0.3, t)
        lc.replay_eta = base_lifecycle.replay_eta
        lc.replay_passes = base_lifecycle.replay_passes
        lc.replay_steps = base_lifecycle.replay_steps

    elif progress < ae:
        # ADOLESCENCE: peak learning. Growth declining, pruning rising.
        t = (progress - ce) / (ae - ce)
        start = childhood_tgt
        end = adolescence_tgt
        lc.eta = _lerp(0.05, 0.08, t)
        lc.n_rounds = 4
        lc.attempts_per_round = 5
        lc.consolidation_strength = _lerp(0.3, 0.5, t)
        lc.replay_eta = base_lifecycle.replay_eta * _lerp(1.0, 2.0, t)
        lc.replay_passes = base_lifecycle.replay_passes + int(2 * t)
        lc.replay_steps = base_lifecycle.replay_steps

    elif progress < se:
        # STABILIZATION: winding down. Circuits locking in.
        t = (progress - ae) / (se - ae)
        start = adolescence_tgt
        end = stabilization_tgt
        lc.eta = _lerp(0.08, 0.03, t)
        lc.n_rounds = 2
        lc.attempts_per_round = 3
        lc.consolidation_strength = _lerp(0.5, 0.7, t)
        lc.replay_eta = base_lifecycle.replay_eta * _lerp(2.0, 3.0, t)
        lc.replay_passes = base_lifecycle.replay_passes + int(_lerp(2, 4, t))
        lc.replay_steps = base_lifecycle.replay_steps

    else:
        # YOUNG ADULT: mature circuits. Strong consolidation/replay.
        t = (progress - se) / (1.0 - se)
        start = stabilization_tgt
        end = adult_tgt
        lc.eta = _lerp(0.03, 0.01, t)
        lc.n_rounds = 2
        lc.attempts_per_round = 2
        lc.consolidation_strength = _lerp(0.7, 0.8, t)
        lc.replay_eta = base_lifecycle.replay_eta * _lerp(3.0, 5.0, t)
        lc.replay_passes = base_lifecycle.replay_passes + int(_lerp(4, 6, t))
        lc.replay_steps = base_lifecycle.replay_steps

    # Interpolate structural plasticity parameters
    lc.growth_rate = _lerp(start[0], end[0], t)
    lc.growth_candidates = int(_lerp(start[1], end[1], t))
    lc.prune_removal_rate = _lerp(start[2], end[2], t)
    lc.prune_weak_threshold = _lerp(start[3], end[3], t)
    lc.sleep_downscale = _lerp(start[4], end[4], t)
    lc.sleep_tag_threshold = _lerp(start[5], end[5], t)

    return lc


def run_childhood(
    net: DNG,
    all_tasks: List[TaskTuple],
    config: ChildhoodConfig | None = None,
    lifecycle: LifecycleConfig | None = None,
    rng: np.random.Generator | None = None,
    episodic=None,
    start_maturity: float = 0.0,
) -> ChildhoodResult:
    if config is None:
        config = ChildhoodConfig()
    if lifecycle is None:
        lifecycle = LifecycleConfig()
    if rng is None:
        rng = np.random.default_rng()

    # Snapshot starting developmental parameters from the loaded checkpoint.
    # These are whatever infancy left on the net object -- no hardcoding.
    start_inh = net.inh_scale
    start_wta = net.wta_k_frac
    start_exc_cap = 1.5 + 3.5 * start_maturity

    # Identify nodes that undergo developmental excitability capping
    abstract_idx = list(Region).index(Region.ABSTRACT)
    memory_idx = list(Region).index(Region.MEMORY)
    developing_nodes = np.where(
        (net.regions == abstract_idx) | (net.regions == memory_idx)
    )[0]

    if config.verbose:
        print(f"  Childhood starting from maturity={start_maturity:.4f}")
        print(f"    inh_scale={start_inh:.3f} -> {ADULT_INH:.3f}")
        print(f"    wta_k_frac={start_wta:.4f} -> {ADULT_WTA:.4f}")
        print(f"    exc_cap={start_exc_cap:.2f} -> {ADULT_EXC_CAP:.2f}")
        print(f"    eta schedule: child=0.05 -> adoles=0.08 -> stable=0.03 -> adult=0.01")

    day_rewards = []
    day_edges = []
    ema_r = None
    ema_reward = None

    # Mastery-based curriculum state
    n_pick = min(config.tasks_per_day, len(all_tasks))
    cursor = 0                     # index into difficulty-sorted tasks
    days_on_current = 0            # how many days we've spent on current batch
    current_best = [0.0] * n_pick  # best reward per task in current batch

    for day in range(1, config.n_days + 1):
        progress = day / config.n_days
        phase = _phase_name(progress, config)
        day_lc = _schedule(day, config.n_days, config, lifecycle)

        # Ramp developmental parameters from checkpoint values to adult
        net.inh_scale = start_inh + (ADULT_INH - start_inh) * progress
        net.wta_k_frac = start_wta + (ADULT_WTA - start_wta) * progress
        max_exc = start_exc_cap + (ADULT_EXC_CAP - start_exc_cap) * progress
        net.excitability[developing_nodes] = np.clip(
            net.excitability[developing_nodes], 0.1, max_exc,
        )

        # Neuromodulator schedule (ACh)
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

        # Mastery-based curriculum: stay on the same tasks until learned,
        # then advance. Like a child practicing until proficient.
        indices = [(cursor + i) % len(all_tasks) for i in range(n_pick)]
        today = [all_tasks[i] for i in indices]

        day_result, ema_r = study_day(net, today, day_lc, ema_r, rng,
                                      episodic=episodic)
        day_rewards.append(day_result.mean_reward)
        day_edges.append(day_result.edges_after)
        days_on_current += 1

        # Track best reward per task across days on this batch
        for i, att in enumerate(day_result.attempts):
            current_best[i] = max(current_best[i], att.reward)

        # Advance only when ALL tasks in the batch are solved, or patience expires
        all_solved = all(r >= 1.0 for r in current_best)
        if all_solved or days_on_current >= config.mastery_patience:
            cursor = (cursor + n_pick) % len(all_tasks)
            days_on_current = 0
            current_best = [0.0] * n_pick

        if ema_reward is None:
            ema_reward = day_result.mean_reward
        else:
            ema_reward = 0.9 * ema_reward + 0.1 * day_result.mean_reward

        if config.verbose:
            solves = sum(1 for a in day_result.attempts if a.reward >= 1.0)
            rewards = [f"{a.reward:.0%}({a.n_attempts})" for a in day_result.attempts]
            n_eps = len(episodic.episodes) if episodic is not None else 0
            n_mastered = sum(1 for r in current_best if r >= 1.0)
            batch_label = f"batch={cursor//n_pick+1}/{(len(all_tasks)+n_pick-1)//n_pick}"
            mastery_label = f"mastered={n_mastered}/{n_pick}"
            print(f"  Day {day:3d} [{phase:>7s}] "
                  f"inh={net.inh_scale:.2f} wta={net.wta_k_frac:.3f} "
                  f"ACh={net.ach:.2f} eta={day_lc.eta:.3f}: "
                  f"reward={day_result.mean_reward:.2f} "
                  f"ema={ema_reward:.2f} "
                  f"solves={solves}/{len(day_result.attempts)} "
                  f"edges={day_result.edges_after} "
                  f"+{day_result.n_grown}/-{day_result.n_pruned} "
                  f"episodic={n_eps} "
                  f"{batch_label} {mastery_label} "
                  f"d{days_on_current}/{config.mastery_patience} "
                  f"[{', '.join(rewards)}]")
            print(f"         gr={day_lc.growth_rate:.3f} "
                  f"gc={day_lc.growth_candidates} "
                  f"pr={day_lc.prune_removal_rate:.4f} "
                  f"sd={day_lc.sleep_downscale:.4f}", flush=True)

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
