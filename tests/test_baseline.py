"""
Diagnostic: measure what the fixed architecture can do WITHOUT any learning.
Runs three baselines on the same tasks:
  A) Fixed weights + episodic recall (no CHL anywhere)
  B) Fixed weights only (no episodic, no CHL) -- pure copy pathway
  C) Copy baseline (output = input)

This tells us whether "learning" is actually happening or if
the ~15% reward is just architectural artifacts.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.childhood import extract_tasks
from src.curriculum import sort_by_difficulty
from src.pipeline import LifecycleConfig, study_task_round, solve_task
from src.graph import DNG, Region
from src.episodic_memory import EpisodicMemory
from src.encoding import pad_grid

MAX_H = MAX_W = 10
rng = np.random.default_rng(42)

# Load tasks
train_set, _ = arckit.load_data()

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

valid = [t for t in train_set if all(d <= MAX_H for d in max_grid_dim(t))]
rng.shuffle(valid)
test_tasks = valid[:30]  # 30 diverse tasks

print(f"Testing on {len(test_tasks)} tasks\n", flush=True)

# Create network
genome = Genome(
    max_h=MAX_H, max_w=MAX_W,
    n_internal=400, n_concept=80, n_memory=120,
    max_fan_in=100,
    weight_scale=0.02, frac_inhibitory=0.25,
    f_rate=0.05, f_decay=0.02, f_max=3.0,
)
net = create_dng(genome, MAX_H, MAX_W, rng)
net.wta_k_frac = 0.05
net.threshold[:] = 0.15
motor_mask = net.regions == list(Region).index(Region.MOTOR)
net.threshold[motor_mask] = 0.05
net.adapt_rate = 0.05

config = LifecycleConfig(
    observe_steps=40,
    think_steps=60,
    eta=0.0,  # NO CHL
    noise_std=0.02,
    focus_strength=0.5,
    t_max=200,
    memory_hint_strength=1.5,
    spontaneous_strength=0.3,
)

# Save initial weights for reset
initial_w = net._edge_w[:net._edge_count].copy()

def reset_net():
    """Reset to initial state without rebuilding."""
    net._edge_w[:net._edge_count] = initial_w.copy()
    net.V[:] = 0.0
    net.r[:] = 0.0
    net.f[:] = 0.0
    net.adaptation[:] = 0.0

def eval_task(task, episodic=None):
    """Evaluate on one task, return (fg_accuracy, copy_accuracy, n_fg_cells)."""
    train_pairs = [(np.array(i), np.array(o)) for i, o in task.train]
    results = []
    for test_inp, test_out in task.test:
        test_inp, test_out = np.array(test_inp), np.array(test_out)
        out_h, out_w = test_out.shape

        result = solve_task(net, train_pairs, test_inp, config,
                            output_h=out_h, output_w=out_w,
                            episodic=episodic)

        fg = test_out != 0
        if fg.any():
            acc = float(np.mean(result.grid[fg] == test_out[fg]))
            copy_h = min(test_inp.shape[0], out_h)
            copy_w = min(test_inp.shape[1], out_w)
            copy_grid = np.zeros_like(test_out)
            copy_grid[:copy_h, :copy_w] = test_inp[:copy_h, :copy_w]
            copy_acc = float(np.mean(copy_grid[fg] == test_out[fg]))
            results.append((acc, copy_acc, int(fg.sum())))
        else:
            results.append((1.0, 1.0, 0))
    return results

# ── Test A: Fixed weights + episodic recall ──────────────────────
print("=" * 60, flush=True)
print("TEST A: Fixed weights + episodic recall (no CHL)", flush=True)
print("=" * 60, flush=True)
episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=500)
reset_net()

a_scores = []
a_copy = []
for task in test_tasks:
    results = eval_task(task, episodic=episodic)
    for acc, cp, nfg in results:
        a_scores.append(acc)
        a_copy.append(cp)
        better = "+" if acc > cp else ("=" if acc == cp else "-")
        print(f"  {task.id}: net={acc:.0%} copy={cp:.0%} {better}", flush=True)

mean_a = np.mean(a_scores)
mean_copy = np.mean(a_copy)
print(f"\nTest A mean: network={mean_a:.1%}  copy={mean_copy:.1%}  "
      f"delta={mean_a - mean_copy:+.1%}", flush=True)

# ── Test B: Fixed weights, NO episodic memory ────────────────────
print(f"\n{'=' * 60}", flush=True)
print("TEST B: Fixed weights only (no episodic, no CHL)", flush=True)
print("=" * 60, flush=True)
reset_net()

b_scores = []
for task in test_tasks:
    results = eval_task(task, episodic=None)
    for acc, cp, nfg in results:
        b_scores.append(acc)
        print(f"  {task.id}: net={acc:.0%} copy={cp:.0%}", flush=True)

mean_b = np.mean(b_scores)
print(f"\nTest B mean: network={mean_b:.1%}", flush=True)

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'=' * 60}", flush=True)
print(f"  Copy baseline:                    {mean_copy:.1%}", flush=True)
print(f"  Fixed weights only (no episodic): {mean_b:.1%}", flush=True)
print(f"  Fixed weights + episodic recall:  {mean_a:.1%}", flush=True)
print(f"  Episodic contribution:            {mean_a - mean_b:+.1%}", flush=True)
print(f"  Network vs copy:                  {mean_a - mean_copy:+.1%}", flush=True)
n_better = sum(1 for a, c in zip(a_scores, a_copy) if a > c)
n_same = sum(1 for a, c in zip(a_scores, a_copy) if a == c)
n_worse = sum(1 for a, c in zip(a_scores, a_copy) if a < c)
print(f"  Tasks: {n_better} better than copy, {n_same} same, {n_worse} worse", flush=True)
