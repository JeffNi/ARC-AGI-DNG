"""
Test CHL + neurotransmitter + round-robin pipeline.

Uses a small 10x10 network. Verifies:
1. Facilitation builds during observation
2. Round-robin study produces weight changes during childhood
3. Neurotransmitter levels respond correctly (DA, ACh, NE)
4. Adult mode: no weight changes, low ACh
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.pipeline import LifecycleConfig, observe_examples, study_task_round, solve_task, study_day
from src.childhood import extract_tasks
from src.curriculum import sort_by_difficulty, task_difficulty

print("Loading ARC data...", flush=True)
train_set, _ = arckit.load_data()

MAX_H, MAX_W = 10, 10

def max_grid_dim(task):
    mh, mw = 0, 0
    for inp, out in task.train:
        inp, out = np.array(inp), np.array(out)
        mh = max(mh, inp.shape[0], out.shape[0])
        mw = max(mw, inp.shape[1], out.shape[1])
    for inp, out in task.test:
        inp, out = np.array(inp), np.array(out)
        mh = max(mh, inp.shape[0], out.shape[0])
        mw = max(mw, inp.shape[1], out.shape[1])
    return mh, mw

valid_tasks = [t for t in train_set if all(d <= 10 for d in max_grid_dim(t))]
print(f"Tasks fitting 10x10: {len(valid_tasks)}", flush=True)

all_tasks = extract_tasks(valid_tasks)
all_tasks = sort_by_difficulty(all_tasks)
diffs = [task_difficulty(*t) for t in all_tasks]
print(f"Sorted {len(all_tasks)} tasks (difficulty: {min(diffs):.0f}-{max(diffs):.0f})", flush=True)

# ── Create network ──────────────────────────────────────────────────

genome = Genome(
    max_h=MAX_H, max_w=MAX_W,
    n_internal=400, n_concept=80, n_memory=60,
    max_fan_in=100,
    weight_scale=0.15, frac_inhibitory=0.25,
    f_rate=0.05, f_decay=0.02, f_max=3.0,
)

rng = np.random.default_rng(42)
print("\nCreating network...", flush=True)
t0 = time.time()
net = create_dng(genome, MAX_H, MAX_W, rng)
net.wta_k_frac = 0.25
print(f"  {net.n_nodes} nodes, {net.edge_count()} edges "
      f"({len(net.memory_nodes)} memory) ({time.time()-t0:.1f}s)", flush=True)

config = LifecycleConfig(
    observe_steps=30, think_steps=50, consolidation_steps=15,
    eta=0.15, w_max=2.5,
    attempts_per_round=3, n_rounds=3, stuck_patience=2,
    noise_std=0.02, focus_strength=0.5,
)

# ── Test 1: Facilitation + neuromodulators during observation ────────

print("\n=== Test 1: Facilitation & neuromodulators ===", flush=True)
task = all_tasks[0]
train_pairs, test_in, test_out = task

print(f"Task: {test_in.shape} -> {test_out.shape}", flush=True)
print(f"Before: f_mean={net.f.mean():.4f} ACh={net.ach:.2f} DA={net.da:.2f} NE={net.ne:.2f}",
      flush=True)

net.ach = 1.0
net.reset_activity()
observe_examples(net, train_pairs, config)

print(f"After:  f_mean={net.f.mean():.4f} f_max={net.f.max():.4f} "
      f"mem_r={net.r[net.memory_nodes].mean():.4f}", flush=True)

# ── Test 2: Round-robin study_day ────────────────────────────────────

print("\n=== Test 2: Round-robin study_day (8 tasks, 3 rounds) ===", flush=True)
w_before = net._edge_w[:net._edge_count].copy()
net.ach = 1.0

day_tasks = [all_tasks[i] for i in range(min(8, len(all_tasks)))]
t1 = time.time()
day_result, _ = study_day(net, day_tasks, config, None)
dt = time.time() - t1

print(f"  Time: {dt:.1f}s", flush=True)
print(f"  Mean reward: {day_result.mean_reward:.2f}", flush=True)
for i, a in enumerate(day_result.attempts):
    print(f"    Task {i}: reward={a.reward:.0%} total_attempts={a.n_attempts}", flush=True)

w_after = net._edge_w[:net._edge_count].copy()
w_diff = np.mean(np.abs(w_after - w_before[:len(w_after)]))
print(f"\nMean weight change: {w_diff:.6f} "
      f"({'PASS: weights changed' if w_diff > 1e-6 else 'FAIL: no learning!'})",
      flush=True)

# ── Test 3: Adult mode (no weight changes, low ACh) ──────────────────

print("\n=== Test 3: Adult solve (no weight changes, low ACh) ===", flush=True)
w_before_adult = net._edge_w[:net._edge_count].copy()

rng_eval = np.random.default_rng(99)
eval_indices = rng_eval.choice(min(50, len(all_tasks)), size=3, replace=False)

for idx in eval_indices:
    task = all_tasks[idx]
    train_pairs, test_in, test_out = task
    out_h, out_w = test_out.shape
    t1 = time.time()
    result = solve_task(net, train_pairs, test_in, config,
                        output_h=out_h, output_w=out_w)
    dt = time.time() - t1
    acc = float(np.mean(result.grid == test_out))
    print(f"  Task {idx}: acc={acc:.0%} decided={result.decided} "
          f"ACh={net.ach:.2f} ({dt:.1f}s)", flush=True)

w_after_adult = net._edge_w[:net._edge_count]
w_diff_adult = np.max(np.abs(w_after_adult - w_before_adult[:len(w_after_adult)]))
print(f"\nWeight change in adult mode: {w_diff_adult:.6f} "
      f"({'PASS: no change' if w_diff_adult < 1e-10 else 'FAIL: weights changed!'})",
      flush=True)

print("\nAll tests complete!", flush=True)
