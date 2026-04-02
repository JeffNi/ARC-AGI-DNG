"""Inspect a few failing tasks to understand what rules they involve."""
import sys
sys.path.insert(0, '.')
import numpy as np
import arckit

train_set, _ = arckit.load_data()
task_map = {t.id: t for t in train_set}

# Pick the tasks with highest partial accuracy (most promising)
inspect = ['3618c87e', '87ab05b8', '25d8a9c8', '4cd1b7b2',
           '6150a2bd', '5582e5ca', 'a85d4709']

for tid in inspect:
    if tid not in task_map:
        continue
    task = task_map[tid]
    print(f"\n{'='*60}")
    print(f"Task: {tid}")
    print(f"Training pairs: {len(task.train)}")
    for i, (inp, out) in enumerate(task.train):
        inp, out = np.array(inp), np.array(out)
        print(f"\n  Pair {i}: input {inp.shape} -> output {out.shape}")
        print(f"  Input:\n{inp}")
        print(f"  Output:\n{out}")

        # Check if it's a per-cell transformation
        same_shape = inp.shape == out.shape
        if same_shape:
            diff_mask = inp != out
            n_changed = diff_mask.sum()
            n_total = inp.size
            print(f"  Changed cells: {n_changed}/{n_total}")

    # Show test
    for i, (inp, out) in enumerate(task.test):
        inp, out = np.array(inp), np.array(out)
        print(f"\n  Test {i}: input {inp.shape} -> output {out.shape}")
        print(f"  Input:\n{inp}")
        print(f"  Output:\n{out}")
