"""
Test the micro-DNG on a batch of ARC-AGI-1 tasks.

Identifies which tasks our current architecture can handle and which it can't.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import arckit
from src.template import create_micro_dng
from src.pipeline import perceive, rest, sleep, read_output, LifecycleConfig

train_set, eval_set = arckit.load_data()
print(f"Total tasks: {len(train_set)} training, {len(eval_set)} eval\n")

config = LifecycleConfig()

# Find tasks with small grids (our micro-DNG is hardcoded to a fixed grid size)
small_tasks = []
for task in train_set:
    pairs = task.train
    test_pairs = task.test
    shapes = set()
    for inp, out in pairs:
        shapes.add(inp.shape)
        shapes.add(out.shape)
    for inp, out in test_pairs:
        shapes.add(inp.shape)
        shapes.add(out.shape)

    # All grids must be the same shape and small
    if len(shapes) == 1:
        h, w = list(shapes)[0]
        if h <= 5 and w <= 5:
            small_tasks.append((task, h, w))

print(f"Tasks with uniform grids <= 5x5: {len(small_tasks)}\n")

# Test each one
results = []
for task, h, w in small_tasks[:30]:  # cap at 30 for speed
    net = create_micro_dng(grid_h=h, grid_w=w, rng=np.random.default_rng(42))

    ema = None
    for epoch in range(5):
        for inp, out in task.train:
            perceive(net, np.array(inp), np.array(out), config)
            rest(net, config)
        _, ema = sleep(net, config, ema)

    # Test on first test pair
    test_in, test_out = task.test[0]
    test_in, test_out = np.array(test_in), np.array(test_out)
    result = read_output(net, test_in, config)

    correct = np.array_equal(result.grid, test_out)
    cell_acc = np.mean(result.grid == test_out)
    results.append((task.id, correct, cell_acc, result.decided, h, w))

    status = "PASS" if correct else "FAIL"
    dec = "decided" if result.decided else "undecided"
    print(f"  {task.id}: {status} acc={cell_acc:.0%} {dec} ({h}x{w})")

# Summary
n_pass = sum(1 for _, c, *_ in results if c)
n_total = len(results)
print(f"\n{'='*50}")
print(f"Passed: {n_pass}/{n_total} ({n_pass/n_total:.0%})")
print(f"\nPassing tasks:")
for tid, correct, acc, decided, h, w in results:
    if correct:
        print(f"  {tid} ({h}x{w})")
