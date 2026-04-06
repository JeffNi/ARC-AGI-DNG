"""Test error-corrective three-factor learning on simple tasks.

Verifies:
  (a) Weights actually change during study_task_round
  (b) Reward improves over repeated rounds on identity and fill tasks
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.graph import DNG
from src.pipeline import LifecycleConfig, study_task_round
from src.episodic_memory import EpisodicMemory

MAX_H = MAX_W = 10

net = DNG.load('models/infancy_checkpoints/day_0110.net.npz')
print(f"Loaded: {net._edge_count:,} edges, inh={net.inh_scale:.3f}, wta={net.wta_k_frac:.4f}")

config = LifecycleConfig(
    observe_steps=30,
    think_steps=80,
    eta=0.02,
    w_max=2.5,
    error_prop_steps=15,
    noise_std=0.02,
    focus_strength=0.5,
    attempts_per_round=5,
    n_rounds=1,
)

episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=50)
rng = np.random.default_rng(42)

# === TASK 1: Identity (3x3, 4 colors) ===
print(f"\n{'='*60}")
print("TASK 1: Identity (copy input -> output)")
print(f"{'='*60}")

train_pairs = []
for _ in range(3):
    g = rng.integers(0, 4, size=(3, 3))
    train_pairs.append((g, g.copy()))
test_in = rng.integers(0, 4, size=(3, 3))
test_out = test_in.copy()

print(f"Test input:\n{test_in}\n")

w_before = net._edge_w[:net._edge_count].copy()

rewards = []
for rnd in range(30):
    result = study_task_round(
        net, train_pairs, test_in, test_out, config,
        prev_best_reward=max(rewards) if rewards else 0.0,
        is_first_visit=(rnd == 0),
        episodic=episodic,
    )
    rewards.append(result.reward)

    w_now = net._edge_w[:net._edge_count]
    n_changed = int(np.sum(np.abs(w_now - w_before) > 1e-10))
    max_dw = float(np.max(np.abs(w_now - w_before)))

    print(f"  Round {rnd+1:3d}: reward={result.reward:.3f}  "
          f"attempts={result.n_attempts}  "
          f"w_changed={n_changed:,}  max_dw={max_dw:.6f}  "
          f"guess_colors={np.unique(result.guess).tolist()}", flush=True)

print(f"\n  First 10 avg: {np.mean(rewards[:10]):.3f}")
print(f"  Last 10 avg:  {np.mean(rewards[-10:]):.3f}")
print(f"  Best:         {max(rewards):.3f}")
w_final = net._edge_w[:net._edge_count]
total_changed = int(np.sum(np.abs(w_final - w_before) > 1e-10))
print(f"  Total weights changed: {total_changed:,}")
print(f"  Trend: {'IMPROVING' if np.mean(rewards[-10:]) > np.mean(rewards[:10]) + 0.02 else 'FLAT/DECLINING'}")

# === TASK 2: Fill (all output = color 2) ===
print(f"\n{'='*60}")
print("TASK 2: Fill (output = all color 2)")
print(f"{'='*60}")

net2 = DNG.load('models/infancy_checkpoints/day_0110.net.npz')
episodic2 = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=50)

target = np.full((3, 3), 2, dtype=int)
train_pairs2 = []
for _ in range(3):
    g = rng.integers(0, 4, size=(3, 3))
    train_pairs2.append((g, target.copy()))
test_in2 = rng.integers(0, 4, size=(3, 3))

w_before2 = net2._edge_w[:net2._edge_count].copy()

rewards2 = []
for rnd in range(30):
    result = study_task_round(
        net2, train_pairs2, test_in2, target, config,
        prev_best_reward=max(rewards2) if rewards2 else 0.0,
        is_first_visit=(rnd == 0),
        episodic=episodic2,
    )
    rewards2.append(result.reward)

    w_now2 = net2._edge_w[:net2._edge_count]
    n_changed2 = int(np.sum(np.abs(w_now2 - w_before2) > 1e-10))
    max_dw2 = float(np.max(np.abs(w_now2 - w_before2)))

    print(f"  Round {rnd+1:3d}: reward={result.reward:.3f}  "
          f"attempts={result.n_attempts}  "
          f"w_changed={n_changed2:,}  max_dw={max_dw2:.6f}  "
          f"guess_colors={np.unique(result.guess).tolist()}", flush=True)

print(f"\n  First 10 avg: {np.mean(rewards2[:10]):.3f}")
print(f"  Last 10 avg:  {np.mean(rewards2[-10:]):.3f}")
print(f"  Best:         {max(rewards2):.3f}")
w_final2 = net2._edge_w[:net2._edge_count]
total_changed2 = int(np.sum(np.abs(w_final2 - w_before2) > 1e-10))
print(f"  Total weights changed: {total_changed2:,}")
print(f"  Trend: {'IMPROVING' if np.mean(rewards2[-10:]) > np.mean(rewards2[:10]) + 0.02 else 'FLAT/DECLINING'}")
