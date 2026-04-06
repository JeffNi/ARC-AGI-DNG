"""
Test: can a richly recurrent DNG learn from interactive childhood?

Uses a small genome (200 internal nodes) and trains on a few simple
ARC tasks. The key question: does the reward signal increase over days?
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.pipeline import LifecycleConfig, solve_task
from src.childhood import run_childhood, extract_problems, ChildhoodConfig

# ── Load ARC data ─────────────────────────────────────────────────────
train_set, _ = arckit.load_data()

# Filter to small uniform-grid tasks for speed
small_tasks = []
for task in train_set:
    shapes = set()
    for inp, out in task.train + task.test:
        shapes.add(np.array(inp).shape)
        shapes.add(np.array(out).shape)
    if len(shapes) == 1:
        h, w = list(shapes)[0]
        if h <= 3 and w <= 3:
            small_tasks.append(task)

print(f"Small tasks (<=3x3, uniform): {len(small_tasks)}")

# Use a fixed grid size for the network
GRID_H, GRID_W = 3, 3
problems = extract_problems(small_tasks)
print(f"Total training problems: {len(problems)}")

# ── Create network from genome ────────────────────────────────────────
genome = Genome(
    n_internal=100,
    frac_inhibitory=0.2,
    frac_modulatory=0.05,
    frac_memory=0.1,
    density_sensory_to_internal=0.1,
    density_internal_to_motor=0.1,
    density_internal_to_internal=0.2,
    density_motor_to_internal=0.05,
    density_sensory_to_motor=0.02,
    density_internal_to_sensory=0.02,
    density_sensory_neighbors=0.05,
    density_motor_neighbors=0.05,
    weight_scale=0.01,
    eta_w=0.05,
    think_steps=30,
    learn_steps=20,
    rest_steps=15,
)

rng = np.random.default_rng(42)
net = create_dng(genome, GRID_H, GRID_W, rng)
config = LifecycleConfig.from_genome(genome)

print(f"\nNetwork: {net.n_nodes} nodes, {net.edge_count()} edges")
print(f"  Sensory: {GRID_H*GRID_W*10}, Internal: {genome.n_internal}, "
      f"Motor: {GRID_H*GRID_W*10}")

# ── Run childhood ─────────────────────────────────────────────────────
print(f"\n=== Childhood ({20} days) ===")
childhood_config = ChildhoodConfig(
    n_days=20,
    problems_per_day=4,
    verbose=True,
)

result = run_childhood(net, problems, childhood_config, config, rng)

# ── Test on a specific task ───────────────────────────────────────────
print(f"\n=== Test on task 0d3d703e (color swap) ===")
task_map = {t.id: t for t in small_tasks}
if '0d3d703e' in task_map:
    task = task_map['0d3d703e']
    test_in, test_out = np.array(task.test[0][0]), np.array(task.test[0][1])
    readout = solve_task(net, [(np.array(i), np.array(o)) for i, o in task.train],
                         test_in, config)
    acc = np.mean(readout.grid == test_out)
    print(f"Decided: {readout.decided} ({readout.steps_taken} steps)")
    print(f"Accuracy: {acc:.0%}")
    print(f"Predicted:\n{readout.grid}")
    print(f"Expected:\n{test_out}")
else:
    print("Task not in small set")

print(f"\n=== Summary ===")
print(f"Day 1 reward:  {result.day_rewards[0]:.2f}")
print(f"Day 20 reward: {result.day_rewards[-1]:.2f}")
print(f"Edges: {net.edge_count()}")
improving = result.day_rewards[-1] > result.day_rewards[0]
print(f"Improving: {'YES' if improving else 'NO'}")
