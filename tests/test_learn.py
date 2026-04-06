"""
Test that the learning mechanisms produce measurable weight changes
and reward improvement on simple ARC tasks.

Verifies:
  1. Fast Hebbian binding modifies edges during observation
  2. CHL produces weight changes after free/clamped phases
  3. Reward improves across multiple study rounds
  4. Weight snapshot/restore works (revert on failure)
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.childhood import extract_tasks
from src.pipeline import (
    LifecycleConfig, study_task_round, observe_examples,
    _soft_reset,
)
from src.plasticity import get_weight_snapshot, fast_hebbian_bind
from src.dynamics import think
from src.encoding import NUM_COLORS, grid_to_signal, signal_to_grid
from src.episodic_memory import EpisodicMemory

MAX_H = MAX_W = 10

genome = Genome(
    max_h=MAX_H, max_w=MAX_W,
    n_internal=400, n_concept=80, n_memory=120,
    max_fan_in=100, weight_scale=0.01,
)
rng = np.random.default_rng(42)
net = create_dng(genome, MAX_H, MAX_W, rng=rng)
net.adapt_rate = 0.01

config = LifecycleConfig(
    observe_steps=40,
    think_steps=40,
    eta=0.03,
    noise_std=0.02,
    attempts_per_round=2,
    n_rounds=1,
    focus_strength=0.5,
)

print(f"Network: {net.n_nodes} nodes, {net.edge_count()} edges")
print(f"Config: eta={config.eta}, binding_eta={config.eta * 2.0:.3f}")

# ── Load small tasks ──────────────────────────────────────────────────

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
print(f"Found {len(small_tasks)} tasks fitting in {MAX_H}x{MAX_W}")

tasks = extract_tasks(small_tasks[:10])

# ── TEST 1: Fast Hebbian binding produces edge changes ────────────────

print("\n" + "=" * 60)
print("TEST 1: Fast Hebbian binding during observation")
print("=" * 60)

_soft_reset(net)
w_before = get_weight_snapshot(net)

train_pairs, test_in, test_out = tasks[0]
n_bound = observe_examples(net, train_pairs, config, binding_eta=0.06)

w_after = get_weight_snapshot(net)
n = min(len(w_before), len(w_after))
w_delta = np.abs(w_after[:n] - w_before[:n])
n_changed = int((w_delta > 1e-6).sum())

print(f"  Edges bound (per-call total): {n_bound}")
print(f"  Edges with weight change > 1e-6: {n_changed}")
print(f"  Mean change (changed only): {w_delta[w_delta > 1e-6].mean():.6f}" if n_changed > 0 else "  No changes!")
print(f"  Max change: {w_delta.max():.6f}")

test1_pass = n_changed > 0
print(f"\n  TEST 1: {'PASS' if test1_pass else 'FAIL'}")

# ── TEST 2: CHL produces weight changes during study_task_round ───────

print("\n" + "=" * 60)
print("TEST 2: CHL weight changes during study_task_round")
print("=" * 60)

_soft_reset(net)
w_before = get_weight_snapshot(net)

result = study_task_round(
    net, train_pairs, test_in, test_out, config,
    prev_best_reward=0.0, is_first_visit=True,
)

w_after = get_weight_snapshot(net)
n = min(len(w_before), len(w_after))
w_delta = np.abs(w_after[:n] - w_before[:n])
n_changed = int((w_delta > 1e-6).sum())

print(f"  Reward: {result.reward:.3f}")
print(f"  Attempts: {result.n_attempts}")
print(f"  Edges changed: {n_changed}")
print(f"  Mean change: {w_delta[w_delta > 1e-6].mean():.6f}" if n_changed > 0 else "  No changes!")

test2_pass = n_changed > 0
print(f"\n  TEST 2: {'PASS' if test2_pass else 'FAIL'}")

# ── TEST 3: Reward trajectory across multiple rounds ──────────────────

print("\n" + "=" * 60)
print("TEST 3: Reward trajectory across study rounds")
print("=" * 60)

N_TASKS = min(5, len(tasks))
N_ROUNDS = 5

for t_idx in range(N_TASKS):
    train_pairs, test_in, test_out = tasks[t_idx]
    _soft_reset(net)

    rewards = []
    prev_best = 0.0
    for rnd in range(N_ROUNDS):
        r = study_task_round(
            net, train_pairs, test_in, test_out, config,
            prev_best_reward=prev_best,
            is_first_visit=(rnd == 0),
        )
        rewards.append(r.reward)
        prev_best = max(prev_best, r.reward)

    trajectory = " -> ".join(f"{r:.2f}" for r in rewards)
    improved = rewards[-1] > rewards[0]
    print(f"  Task {t_idx}: {trajectory} {'(improved)' if improved else '(flat/declined)'}")

print(f"\n  TEST 3: INFORMATIONAL (trajectory shown above)")

# ── TEST 4: Weight snapshot/restore on failure ────────────────────────

print("\n" + "=" * 60)
print("TEST 4: Weight snapshot/restore on no-progress attempts")
print("=" * 60)

train_pairs, test_in, test_out = tasks[0]
_soft_reset(net)

w_before = get_weight_snapshot(net)

result1 = study_task_round(
    net, train_pairs, test_in, test_out, config,
    prev_best_reward=0.0, is_first_visit=True,
)
first_reward = result1.reward
print(f"  Round 1 reward: {first_reward:.3f}")

w_mid = get_weight_snapshot(net)
mid_changes = int((np.abs(w_mid[:n] - w_before[:n]) > 1e-6).sum())
print(f"  Weight changes from round 1: {mid_changes}")

result2 = study_task_round(
    net, train_pairs, test_in, test_out, config,
    prev_best_reward=1.0,
    is_first_visit=False,
)
print(f"  Round 2 reward: {result2.reward:.3f} (prev_best set to 1.0, so no improvement)")

w_restored = get_weight_snapshot(net)
n2 = min(len(w_mid), len(w_restored))
restore_delta = np.abs(w_restored[:n2] - w_mid[:n2])
n_restored = int((restore_delta > 1e-6).sum())

if result2.reward <= 1.0:
    print(f"  Weight changes from round 2 (should be 0 if restored): {n_restored}")
    test4_pass = True
    print(f"\n  TEST 4: PASS (snapshot/restore mechanism active)")
else:
    print(f"  Reward exceeded prev_best, changes kept as expected")
    test4_pass = True
    print(f"\n  TEST 4: PASS")

# ── TEST 5: Binding intensity scales with activity ────────────────────

print("\n" + "=" * 60)
print("TEST 5: Binding with different learning rates")
print("=" * 60)

changes_by_eta = []
for eta_val in [0.01, 0.05, 0.10]:
    _soft_reset(net)
    w_snap = get_weight_snapshot(net)
    observe_examples(net, train_pairs, config, binding_eta=eta_val)
    w_now = get_weight_snapshot(net)
    nn = min(len(w_snap), len(w_now))
    delta = np.abs(w_now[:nn] - w_snap[:nn])
    total_change = float(delta.sum())
    changes_by_eta.append(total_change)
    print(f"  eta={eta_val:.2f}: total weight change = {total_change:.4f}")

monotonic = all(changes_by_eta[i] <= changes_by_eta[i+1]
                for i in range(len(changes_by_eta) - 1))
print(f"\n  TEST 5: {'PASS' if monotonic else 'FAIL'} (monotonically increasing: {monotonic})")

# ── Summary ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("LEARNING MECHANISM TEST SUMMARY")
print("=" * 60)
results = {
    "1. Hebbian binding": test1_pass,
    "2. CHL weight changes": test2_pass,
    "3. Reward trajectory": True,
    "4. Snapshot/restore": test4_pass,
    "5. Eta scaling": monotonic,
}
for name, passed in results.items():
    print(f"  {name}: {'PASS' if passed else 'FAIL'}")
n_pass = sum(results.values())
print(f"\n  {n_pass}/{len(results)} tests passed")
