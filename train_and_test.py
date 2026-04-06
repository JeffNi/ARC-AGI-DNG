"""
Train on simple ARC tasks, then test on unseen tasks.

The question: can the network learn general features during training
that transfer to solve new tasks it hasn't practiced on?

Approach:
  1. Pick the 20 easiest ARC tasks (small grids, simple rules)
  2. Split: 12 for training, 8 held-out for testing
  3. Train for N days: study_task_round (Hebbian + CHL) + sleep
  4. After training, test on ALL 20 tasks (train + unseen)
  5. Report which tasks are solved, and whether any unseen ones are
"""
import sys, time, os
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.childhood import extract_tasks
from src.curriculum import sort_by_difficulty
from src.pipeline import (
    LifecycleConfig, study_task_round, solve_task, sleep, rest,
    _soft_reset,
)
from src.encoding import NUM_COLORS
from src.graph import DNG, Region
from src.episodic_memory import EpisodicMemory
from src.plasticity import synaptogenesis, get_weight_snapshot

# ── Parameters ─────────────────────────────────────────────────────

MAX_H = MAX_W = 10
N_TRAIN = 8
N_TEST = 8
N_DAYS = 40
ROUNDS_PER_TASK = 5

# ── Load tasks ─────────────────────────────────────────────────────

train_set, _ = arckit.load_data()

def max_dim(task):
    d = 0
    for inp, out in task.train:
        inp, out = np.array(inp), np.array(out)
        d = max(d, inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
    for inp, out in task.test:
        inp, out = np.array(inp), np.array(out)
        d = max(d, inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
    return d

small_tasks = [t for t in train_set if max_dim(t) <= MAX_H]
all_tasks = extract_tasks(small_tasks)
all_tasks = sort_by_difficulty(all_tasks)

train_tasks = all_tasks[:N_TRAIN]
test_tasks = all_tasks[N_TRAIN:N_TRAIN + N_TEST]
all_eval = all_tasks[:N_TRAIN + N_TEST]

print(f"Tasks: {len(all_tasks)} total, {N_TRAIN} train, {N_TEST} test (unseen)")

# Show what the tasks look like
for i, (pairs, ti, to) in enumerate(all_eval):
    tag = "TRAIN" if i < N_TRAIN else "TEST "
    shapes = f"{ti.shape}->{to.shape}"
    colors_in = set(ti.ravel().tolist())
    colors_out = set(to.ravel().tolist())
    identity = np.array_equal(ti, to)
    print(f"  {tag} {i:2d}: {shapes:15s} colors_in={colors_in} "
          f"colors_out={colors_out} {'(identity)' if identity else ''}")

# ── Create network ─────────────────────────────────────────────────

rng = np.random.default_rng(42)
genome = Genome(
    max_h=MAX_H, max_w=MAX_W,
    n_internal=400, n_concept=80, n_memory=120,
    max_fan_in=100, weight_scale=0.01,
)
net = create_dng(genome, MAX_H, MAX_W, rng=rng)
net.adapt_rate = 0.01
net.wta_k_frac = 0.05
net.threshold[:] = 0.15
motor_mask = net.regions == list(Region).index(Region.MOTOR)
net.threshold[motor_mask] = 0.05

print(f"\nNetwork: {net.n_nodes} nodes, {net.edge_count()} edges")

config = LifecycleConfig(
    observe_steps=40,
    think_steps=40,
    eta=0.05,
    w_max=2.5,
    noise_std=0.02,
    focus_strength=0.5,
    attempts_per_round=3,
    n_rounds=1,
    rest_steps=20,
    rest_noise_std=0.03,
    sleep_downscale=0.995,
    sleep_tag_threshold=0.001,
    prune_weak_threshold=0.003,
    prune_cycles_required=20,
    replay_eta=0.005,
    replay_passes=3,
    replay_steps=25,
    memory_hint_strength=1.5,
    spontaneous_strength=0.2,
    growth_rate=0.15,
    growth_candidates=15000,
)

episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=200)

# ── Evaluation function ────────────────────────────────────────────

def evaluate_all(net, tasks, episodic, config, label=""):
    """Test the network on a list of tasks (no weight changes)."""
    saved_ach = net.ach
    results = []
    for i, (pairs, ti, to) in enumerate(tasks):
        out_h, out_w = to.shape
        r = solve_task(net, pairs, ti, config,
                       output_h=out_h, output_w=out_w,
                       episodic=episodic)
        perfect = np.array_equal(r.grid, to)
        fg = to != 0
        fg_acc = float(np.mean(r.grid[fg] == to[fg])) if fg.any() else (1.0 if np.all(r.grid == 0) else 0.0)
        is_copy = np.array_equal(r.grid, ti[:out_h, :out_w]) if ti.shape[0] >= out_h and ti.shape[1] >= out_w else False
        results.append({
            'perfect': perfect,
            'fg_acc': fg_acc,
            'is_copy': is_copy,
            'decided': r.decided,
            'steps': r.steps_taken,
        })
    net.ach = saved_ach
    return results

def print_eval(results, label, offset=0):
    n_solved = sum(r['perfect'] for r in results)
    n_copy = sum(r['is_copy'] for r in results)
    mean_acc = np.mean([r['fg_acc'] for r in results])
    print(f"\n  {label}: {n_solved}/{len(results)} solved, "
          f"mean_fg_acc={mean_acc:.3f}, copies={n_copy}")
    for i, r in enumerate(results):
        status = "SOLVED" if r['perfect'] else f"fg={r['fg_acc']:.2f}"
        extra = " (copy)" if r['is_copy'] else ""
        print(f"    [{offset+i:2d}] {status}{extra}")

# ── Baseline evaluation (before any training) ──────────────────────

print("\n" + "=" * 60)
print("BASELINE (no training)")
print("=" * 60)

train_results = evaluate_all(net, train_tasks, episodic, config)
test_results = evaluate_all(net, test_tasks, episodic, config)
print_eval(train_results, "Train set")
print_eval(test_results, "Test set (unseen)", offset=N_TRAIN)

# ── Training loop ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"TRAINING ({N_DAYS} days)")
print("=" * 60)

ema_r = None
best_rewards = [0.0] * N_TRAIN
day_start = time.time()

for day in range(1, N_DAYS + 1):
    net.ach = 1.0
    net.ne = 0.0
    day_rewards = []
    day_improved = 0

    task_order = rng.permutation(N_TRAIN)

    for t_idx in task_order:
        pairs, ti, to = train_tasks[t_idx]

        for rnd in range(ROUNDS_PER_TASK):
            result = study_task_round(
                net, pairs, ti, to, config,
                prev_best_reward=best_rewards[t_idx],
                is_first_visit=(rnd == 0),
                episodic=episodic,
            )

        day_rewards.append(result.reward)
        if result.reward > best_rewards[t_idx]:
            best_rewards[t_idx] = result.reward
            day_improved += 1

    rest(net, config)

    n_grown = synaptogenesis(net, growth_rate=config.growth_rate,
                             n_candidates=config.growth_candidates, rng=rng)
    n_pruned, ema_r = sleep(net, config, ema_r=ema_r, rng=rng, episodic=episodic)

    mean_r = np.mean(day_rewards)
    n_best_nonzero = sum(1 for r in best_rewards if r > 0)
    best_max = max(best_rewards)
    elapsed = time.time() - day_start

    if day <= 5 or day % 5 == 0 or day == N_DAYS:
        best_str = " ".join(f"{r:.2f}" for r in best_rewards)
        print(f"  Day {day:3d}: mean={mean_r:.3f} best_max={best_max:.2f} "
              f"improved={day_improved}/{N_TRAIN} "
              f"edges={net.edge_count()} +{n_grown}/-{n_pruned} "
              f"({elapsed:.0f}s)", flush=True)
        print(f"           per-task best: [{best_str}]", flush=True)

# ── Post-training evaluation ───────────────────────────────────────

print("\n" + "=" * 60)
print("AFTER TRAINING")
print("=" * 60)

train_results = evaluate_all(net, train_tasks, episodic, config)
test_results = evaluate_all(net, test_tasks, episodic, config)
print_eval(train_results, "Train set (practiced)")
print_eval(test_results, "Test set (UNSEEN)", offset=N_TRAIN)

total_solved = sum(r['perfect'] for r in train_results + test_results)
train_solved = sum(r['perfect'] for r in train_results)
test_solved = sum(r['perfect'] for r in test_results)
print(f"\n  TOTAL: {total_solved}/{N_TRAIN + N_TEST} solved "
      f"(train={train_solved}, unseen={test_solved})")

elapsed_total = time.time() - day_start
print(f"  Time: {elapsed_total:.0f}s")

# ── Save checkpoint ────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
ckpt_path = os.path.join("models", "trained_infant.npz")
net.save(ckpt_path)
print(f"\n  Checkpoint saved: {ckpt_path}")
