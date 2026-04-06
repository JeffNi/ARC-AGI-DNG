"""Test whether infancy training helps with ARC tasks.

Compares:
  A) Fresh untrained network (same architecture, no infancy)
  B) Day 90 infant (90 days of unsupervised exposure)

Both tested with:
  1. Direct solve (observe examples, read motor output, no weight changes)
  2. Brief CHL training (5 rounds of study_task_round per task)
"""
import sys, os, time
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.childhood import extract_tasks
from src.curriculum import sort_by_difficulty
from src.pipeline import (
    LifecycleConfig, solve_task, study_task_round, _soft_reset,
)
from src.graph import DNG, Region
from src.episodic_memory import EpisodicMemory

MAX_H = MAX_W = 10
N_TASKS = 20

GENOME = Genome(
    max_h=MAX_H, max_w=MAX_W,
    n_internal=1500, n_concept=200, n_memory=300,
    max_fan_in=200, weight_scale=0.01,
)

CONFIG = LifecycleConfig(
    observe_steps=30,
    think_steps=40,
    free_phase_steps=40,
    clamped_phase_steps=40,
    eta=0.05,
    w_max=2.5,
    noise_std=0.02,
    focus_strength=0.5,
    attempts_per_round=3,
    n_rounds=1,
    rest_steps=15,
    rest_noise_std=0.03,
    memory_hint_strength=1.5,
    spontaneous_strength=0.2,
)


def max_dim(task):
    d = 0
    for inp, out in task.train:
        inp, out = np.array(inp), np.array(out)
        d = max(d, inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
    for inp, out in task.test:
        inp, out = np.array(inp), np.array(out)
        d = max(d, inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
    return d


def evaluate_network(net, tasks, label):
    """Run direct solve + trained solve on a set of tasks."""
    episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=200)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {net.n_nodes} nodes, {net.edge_count():,} edges")
    print(f"  WTA: {net.wta_k_frac:.3f}")
    print(f"{'='*60}")

    # A) Direct solve
    print(f"\n  A) Direct solve (no task training):")
    direct_scores = []
    for i, (pairs, ti, to) in enumerate(tasks):
        _soft_reset(net)
        out_h, out_w = to.shape
        r = solve_task(net, pairs, ti, CONFIG, output_h=out_h, output_w=out_w,
                       episodic=episodic)
        perfect = np.array_equal(r.grid, to)
        fg = to != 0
        fg_acc = (float(np.mean(r.grid[fg] == to[fg])) if fg.any()
                  else (1.0 if np.all(r.grid == 0) else 0.0))
        total_acc = float(np.mean(r.grid == to))
        direct_scores.append({
            'perfect': perfect, 'fg_acc': fg_acc, 'total_acc': total_acc,
            'decided': r.decided, 'steps': r.steps_taken,
        })
        status = "SOLVED" if perfect else f"fg={fg_acc:.2f} total={total_acc:.2f}"
        if i < 10 or perfect:
            print(f"    [{i:2d}] {status}")

    n_solved_direct = sum(1 for s in direct_scores if s['perfect'])
    mean_fg_direct = np.mean([s['fg_acc'] for s in direct_scores])
    mean_total_direct = np.mean([s['total_acc'] for s in direct_scores])
    print(f"  Direct: {n_solved_direct}/{len(tasks)} solved, "
          f"mean_fg={mean_fg_direct:.3f}, mean_total={mean_total_direct:.3f}")

    # B) Trained solve (CHL learning)
    print(f"\n  B) With CHL training (5 rounds per task):")
    trained_scores = []
    for i, (pairs, ti, to) in enumerate(tasks):
        _soft_reset(net)
        episodic_task = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=50)
        best_r = 0.0
        for rnd in range(5):
            result = study_task_round(
                net, pairs, ti, to, CONFIG,
                prev_best_reward=best_r,
                is_first_visit=(rnd == 0),
                episodic=episodic_task,
            )
            best_r = max(best_r, result.reward)
            if result.reward >= 1.0:
                break

        out_h, out_w = to.shape
        r = solve_task(net, pairs, ti, CONFIG, output_h=out_h, output_w=out_w,
                       episodic=episodic_task)
        perfect = np.array_equal(r.grid, to)
        fg = to != 0
        fg_acc = (float(np.mean(r.grid[fg] == to[fg])) if fg.any()
                  else (1.0 if np.all(r.grid == 0) else 0.0))
        total_acc = float(np.mean(r.grid == to))
        trained_scores.append({
            'perfect': perfect, 'fg_acc': fg_acc, 'total_acc': total_acc,
            'train_best': best_r,
        })
        status = "SOLVED" if perfect else f"fg={fg_acc:.2f} total={total_acc:.2f} train_best={best_r:.2f}"
        if i < 10 or perfect:
            print(f"    [{i:2d}] {status}")

    n_solved_trained = sum(1 for s in trained_scores if s['perfect'])
    mean_fg_trained = np.mean([s['fg_acc'] for s in trained_scores])
    mean_total_trained = np.mean([s['total_acc'] for s in trained_scores])
    mean_train_best = np.mean([s['train_best'] for s in trained_scores])
    print(f"  Trained: {n_solved_trained}/{len(tasks)} solved, "
          f"mean_fg={mean_fg_trained:.3f}, mean_total={mean_total_trained:.3f}, "
          f"mean_train_reward={mean_train_best:.3f}")

    return {
        'direct_solved': n_solved_direct,
        'direct_fg': mean_fg_direct,
        'direct_total': mean_total_direct,
        'trained_solved': n_solved_trained,
        'trained_fg': mean_fg_trained,
        'trained_total': mean_total_trained,
        'trained_reward': mean_train_best,
    }


def main():
    rng = np.random.default_rng(42)

    # Load ARC tasks
    train_set, _ = arckit.load_data()
    small_tasks = [t for t in train_set if max_dim(t) <= MAX_H]
    tasks = extract_tasks(small_tasks)
    tasks = sort_by_difficulty(tasks)[:N_TASKS]
    print(f"Testing on {len(tasks)} easiest ARC tasks (max dim {MAX_H})")

    # ── Network A: Fresh untrained ──
    print("\nCreating fresh network...")
    fresh = create_dng(GENOME, MAX_H, MAX_W, rng=np.random.default_rng(42))
    fresh.adapt_rate = 0.01
    fresh.wta_k_frac = 0.05  # adult WTA for task solving
    fresh.threshold[:] = 0.15
    motor_mask = fresh.regions == list(Region).index(Region.MOTOR)
    fresh.threshold[motor_mask] = 0.05

    fresh_results = evaluate_network(fresh, tasks, "FRESH UNTRAINED NETWORK")

    # ── Network B: Day 90 infant ──
    print("\nLoading day 90 infant checkpoint...")
    infant = DNG.load('models/infancy_checkpoints/day_0090.net.npz')
    # Set adult-like parameters for task solving
    infant.wta_k_frac = 0.05
    infant.inh_scale = 1.0  # adult E/I balance for task solving
    infant.ach = 1.0  # learning mode for CHL

    infant_results = evaluate_network(infant, tasks, "DAY 90 INFANT NETWORK")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Fresh':>10} {'Infant':>10} {'Delta':>10}")
    print(f"{'-'*55}")
    for key, label in [
        ('direct_solved', 'Direct solved'),
        ('direct_fg', 'Direct fg_acc'),
        ('direct_total', 'Direct total_acc'),
        ('trained_solved', 'Trained solved'),
        ('trained_fg', 'Trained fg_acc'),
        ('trained_total', 'Trained total_acc'),
        ('trained_reward', 'Train reward'),
    ]:
        f = fresh_results[key]
        i = infant_results[key]
        d = i - f
        sign = '+' if d > 0 else ''
        print(f"  {label:<23} {f:>10.3f} {i:>10.3f} {sign}{d:>9.3f}")

    print(f"\nConclusion: ", end="")
    if infant_results['trained_solved'] > fresh_results['trained_solved']:
        print("Infancy training HELPS task solving.")
    elif infant_results['trained_fg'] > fresh_results['trained_fg'] + 0.02:
        print("Infancy training shows MARGINAL improvement in accuracy.")
    else:
        print("Infancy training shows NO clear benefit for task solving.")


if __name__ == "__main__":
    main()
