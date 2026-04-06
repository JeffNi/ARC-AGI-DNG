"""
Load pre-generated micro-tasks from the micro_tasks/ folder.

Returns tasks in the same (train_pairs, test_in, test_out) format
used by extract_tasks() in childhood.py, sorted by tier so identity
tasks come first and harder tasks come later.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np


TaskTuple = Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
]


def load_micro_tasks(
    base_dir: str = "micro_tasks",
    max_h: int = 10,
    max_w: int = 10,
) -> List[TaskTuple]:
    """
    Read all JSON task files from base_dir, convert to TaskTuples,
    and return sorted by tier (easiest first).

    Skips any task where a grid exceeds max_h x max_w.
    """
    tasks: List[Tuple[int, str, TaskTuple]] = []

    if not os.path.isdir(base_dir):
        print(f"Warning: micro_tasks dir '{base_dir}' not found. "
              "Run scripts/generate_micro_tasks.py first.")
        return []

    for tier_dir in sorted(os.listdir(base_dir)):
        tier_path = os.path.join(base_dir, tier_dir)
        if not os.path.isdir(tier_path):
            continue

        for fname in sorted(os.listdir(tier_path)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(tier_path, fname)
            with open(fpath) as f:
                obj = json.load(f)

            tier = obj.get("tier", 0)
            task_type = obj.get("type", "unknown")

            train_pairs = []
            for ex in obj["train"]:
                inp = np.array(ex["input"], dtype=int)
                out = np.array(ex["output"], dtype=int)
                train_pairs.append((inp, out))

            test_in = np.array(obj["test"][0]["input"], dtype=int)
            test_out = np.array(obj["test"][0]["output"], dtype=int)

            all_grids = [g for pair in train_pairs for g in pair]
            all_grids += [test_in, test_out]
            if any(g.shape[0] > max_h or g.shape[1] > max_w for g in all_grids):
                continue

            tasks.append((tier, task_type, (train_pairs, test_in, test_out)))

    tasks.sort(key=lambda x: (x[0], x[1]))
    result = [t for _, _, t in tasks]
    return result
