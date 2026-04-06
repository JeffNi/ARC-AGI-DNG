"""
Load pre-generated micro-tasks from the micro_tasks/ folder.

Two loading modes:
  load_micro_tasks()        -- flat list of TaskTuples (backward compat)
  load_micro_tasks_tagged() -- dict[type_name -> list of TaggedTask],
                               preserving tier and type metadata for
                               the mastery curriculum.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


TaskTuple = Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
]


@dataclass
class TaggedTask:
    tier: int
    task_type: str
    task: TaskTuple


def _load_all(
    base_dir: str,
    max_h: int,
    max_w: int,
) -> List[Tuple[int, str, TaskTuple]]:
    """Shared loader: returns (tier, type, task) triples sorted by tier."""
    raw: List[Tuple[int, str, TaskTuple]] = []

    if not os.path.isdir(base_dir):
        print(f"Warning: micro_tasks dir '{base_dir}' not found. "
              "Run scripts/generate_micro_tasks.py first.")
        return raw

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

            raw.append((tier, task_type, (train_pairs, test_in, test_out)))

    raw.sort(key=lambda x: (x[0], x[1]))
    return raw


def load_micro_tasks(
    base_dir: str = "micro_tasks",
    max_h: int = 10,
    max_w: int = 10,
) -> List[TaskTuple]:
    """Flat list of tasks sorted by tier (backward compatible)."""
    return [t for _, _, t in _load_all(base_dir, max_h, max_w)]


def load_micro_tasks_tagged(
    base_dir: str = "micro_tasks",
    max_h: int = 10,
    max_w: int = 10,
) -> OrderedDict[str, List[TaggedTask]]:
    """Tasks grouped by type, ordered by tier then type name.

    Returns OrderedDict so the curriculum can iterate types in
    difficulty order (tier 0 identity first, tier 9 pattern last).
    """
    raw = _load_all(base_dir, max_h, max_w)

    by_type: OrderedDict[str, List[TaggedTask]] = OrderedDict()
    for tier, task_type, task in raw:
        if task_type not in by_type:
            by_type[task_type] = []
        by_type[task_type].append(TaggedTask(tier=tier, task_type=task_type, task=task))

    return by_type
