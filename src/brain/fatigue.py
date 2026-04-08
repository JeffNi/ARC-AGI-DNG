"""
Fatigue / sleep pressure tracking.

Accumulates proportional to mean population firing rate.
When threshold exceeded, the brain needs sleep.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FatigueTracker:
    """Tracks sleep pressure accumulation."""
    level: float = 0.0
    rate: float = 0.001
    threshold: float = 50.0
    sleep_reset: float = 0.1

    def accumulate(self, mean_rate: float, n_steps: int = 1):
        self.level += self.rate * mean_rate * n_steps

    def needs_sleep(self) -> bool:
        return self.level >= self.threshold

    def reset(self):
        self.level *= self.sleep_reset

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "rate": self.rate,
            "threshold": self.threshold,
            "sleep_reset": self.sleep_reset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FatigueTracker":
        return cls(**{k: d[k] for k in ("level", "rate", "threshold", "sleep_reset") if k in d})
