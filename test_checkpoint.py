"""Quick task-solving test against a checkpoint."""
import sys; sys.path.insert(0, '.')
import numpy as np
import arckit
from src.graph import DNG, Region
from src.pipeline import LifecycleConfig, solve_task, study_task_round, _soft_reset
from src.childhood import extract_tasks
from src.curriculum import sort_by_difficulty
from src.episodic_memory import EpisodicMemory

net = DNG.load('models/infancy_checkpoints/day_0030.net.npz')
print(f'Loaded: {net.n_nodes} nodes, {net.edge_count():,} edges')
print(f'WTA: {net.wta_k_frac:.3f}')

CONFIG = LifecycleConfig(
    observe_steps=30, think_steps=40,
    eta=0.05, w_max=2.5, noise_std=0.02, focus_strength=0.5,
    attempts_per_round=3, n_rounds=1,
    rest_steps=15, rest_noise_std=0.03,
    sleep_downscale=0.998, sleep_tag_threshold=0.001,
    prune_weak_threshold=0.003, prune_cycles_required=20,
    replay_eta=0.005, replay_passes=2, replay_steps=20,
    memory_hint_strength=1.5, spontaneous_strength=0.2,
    nursery_binding_eta=0.08,
    nursery_growth_rate=0.8, nursery_growth_candidates=500000,
)

train_set, _ = arckit.load_data()

def max_dim(task):
    d = 0
    for inp, out in task.train:
        d = max(d, np.array(inp).shape[0], np.array(inp).shape[1],
                np.array(out).shape[0], np.array(out).shape[1])
    for inp, out in task.test:
        d = max(d, np.array(inp).shape[0], np.array(inp).shape[1],
                np.array(out).shape[0], np.array(out).shape[1])
    return d

small = [t for t in train_set if max_dim(t) <= 10]
tasks = extract_tasks(small)
tasks = sort_by_difficulty(tasks)[:15]
episodic = EpisodicMemory(max_h=10, max_w=10, capacity=200)

print('\nA) Direct solve (no training):')
n_solved = 0
for i, (pairs, ti, to) in enumerate(tasks):
    oh, ow = to.shape
    r = solve_task(net, pairs, ti, CONFIG, output_h=oh, output_w=ow, episodic=episodic)
    perfect = np.array_equal(r.grid, to)
    fg = to != 0
    fg_acc = float(np.mean(r.grid[fg] == to[fg])) if fg.any() else (1.0 if np.all(r.grid == 0) else 0.0)
    if perfect:
        n_solved += 1
    print(f'  [{i:2d}] {"SOLVED" if perfect else "fg=" + f"{fg_acc:.2f}"}')
print(f'Direct: {n_solved}/15')

print('\nB) With 5 rounds training:')
n_solved2 = 0
for i, (pairs, ti, to) in enumerate(tasks):
    _soft_reset(net)
    best_r = 0.0
    for rnd in range(5):
        result = study_task_round(net, pairs, ti, to, CONFIG,
            prev_best_reward=best_r, is_first_visit=(rnd == 0), episodic=episodic)
        best_r = max(best_r, result.reward)
        if result.reward >= 1.0:
            break
    oh, ow = to.shape
    r = solve_task(net, pairs, ti, CONFIG, output_h=oh, output_w=ow, episodic=episodic)
    perfect = np.array_equal(r.grid, to)
    fg = to != 0
    fg_acc = float(np.mean(r.grid[fg] == to[fg])) if fg.any() else (1.0 if np.all(r.grid == 0) else 0.0)
    if perfect:
        n_solved2 += 1
    status = "SOLVED" if perfect else f"fg={fg_acc:.2f} (train_best={best_r:.2f})"
    print(f'  [{i:2d}] {status}')
print(f'Trained: {n_solved2}/15')
