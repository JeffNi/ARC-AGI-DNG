"""
Brain checkpoint management — autosave, milestone saves, resume.

Saves full brain state so we can pick up exactly where we left off.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .engine import Brain


class Checkpointer:
    """Manages brain checkpoint saves and loads."""

    def __init__(self, checkpoint_dir: str = "life", max_rolling: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_rolling = max_rolling
        self._rolling_saves: list[Path] = []
        self._last_save_time = time.time()
        self.autosave_interval_sec = 300  # 5 minutes

    def should_autosave(self) -> bool:
        return (time.time() - self._last_save_time) >= self.autosave_interval_sec

    def save(self, brain: "Brain", tag: str = "auto") -> Path:
        """Save full brain state."""
        save_dir = self.checkpoint_dir / f"age_{brain.age}_{tag}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save DNG arrays
        brain.net.save(str(save_dir / "network.npz"))

        # Save brain metadata
        meta = {
            "age": brain.age,
            "birth_edges": brain.birth_edges,
            "neuromod": brain.neuromod.to_dict(),
            "fatigue": brain.fatigue.to_dict(),
            "rng_state": brain.rng.bit_generator.state,
            "homeostasis": brain.homeostasis.state_dict(),
            "stage_manager": brain.stage_manager.state_dict(),
        }
        with open(save_dir / "brain_meta.json", "w") as f:
            json.dump(meta, f, default=_json_default)

        self._last_save_time = time.time()

        # Rolling window: keep only last N autosaves
        if tag == "auto":
            self._rolling_saves.append(save_dir)
            while len(self._rolling_saves) > self.max_rolling:
                old = self._rolling_saves.pop(0)
                _safe_rmtree(old)

        return save_dir

    def save_milestone(self, brain: "Brain", label: str) -> Path:
        """Permanent milestone save (never auto-deleted)."""
        return self.save(brain, tag=f"milestone_{label}")

    def find_latest(self) -> Path | None:
        """Find the most recent checkpoint."""
        if not self.checkpoint_dir.exists():
            return None
        checkpoints = sorted(self.checkpoint_dir.iterdir(), key=_ckpt_age, reverse=True)
        for ckpt in checkpoints:
            if (ckpt / "network.npz").exists() and (ckpt / "brain_meta.json").exists():
                return ckpt
        return None


def _ckpt_age(p: Path) -> int:
    """Extract age from checkpoint directory name."""
    name = p.name
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _safe_rmtree(p: Path):
    """Remove a checkpoint directory safely."""
    import shutil
    try:
        shutil.rmtree(p)
    except OSError:
        pass


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
