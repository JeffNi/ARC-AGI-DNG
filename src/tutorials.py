"""
Tutorial loader: hand-curated step-by-step reasoning sequences for ARC tasks.

Each tutorial is a folder inside tutorials/ containing a steps.json with:
  - name, description
  - input/output grids (the task being demonstrated)
  - steps: a list of intermediate grids showing progressive construction

The network observes these steps during childhood to learn reasoning patterns.
Progressive revelation: more steps shown on successive failed attempts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Tutorial:
    name: str
    description: str
    input_grid: np.ndarray
    output_grid: np.ndarray
    steps: List[np.ndarray] = field(default_factory=list)

    def steps_for_attempt(self, attempt: int) -> List[np.ndarray]:
        """
        Progressive revelation: return a subset of reasoning steps based
        on the attempt number.

        attempt 1: 0 steps (try on your own)
        attempt 2: first ~1/3 of steps
        attempt 3: first ~2/3 of steps
        attempt 4+: all steps
        """
        n = len(self.steps)
        if n == 0 or attempt <= 1:
            return []
        if attempt == 2:
            k = max(1, n // 3)
        elif attempt == 3:
            k = max(1, 2 * n // 3)
        else:
            k = n
        return self.steps[:k]


def load_tutorial(path: Path) -> Optional[Tutorial]:
    """Load a single tutorial from a steps.json file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

    return Tutorial(
        name=data.get("name", path.parent.name),
        description=data.get("description", ""),
        input_grid=np.array(data["input"], dtype=int),
        output_grid=np.array(data["output"], dtype=int),
        steps=[np.array(s, dtype=int) for s in data.get("steps", [])],
    )


def load_tutorials(tutorials_dir: str | Path = "tutorials") -> List[Tutorial]:
    """
    Scan the tutorials directory for all steps.json files and load them.
    Returns a list of Tutorial objects sorted by name.
    """
    root = Path(tutorials_dir)
    if not root.exists():
        return []

    tutorials = []
    for steps_file in sorted(root.glob("*/steps.json")):
        tut = load_tutorial(steps_file)
        if tut is not None:
            tutorials.append(tut)

    return tutorials
