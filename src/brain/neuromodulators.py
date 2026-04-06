"""
Neuromodulator state — Phase 1: DA only.

DA is the sole neuromodulator. It's computed externally (Teacher sets it
based on RPE from the basal ganglia shortcut box) and decays toward baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NeuromodState:
    """Global neuromodulator levels."""
    da: float = 0.0
    da_baseline: float = 0.05
    da_decay: float = 0.05

    # Phase 2 placeholders
    ne: float = 0.0
    ach: float = 1.0

    def set_da(self, level: float):
        self.da = level

    def decay(self):
        """DA decays toward baseline each step."""
        self.da += self.da_decay * (self.da_baseline - self.da)

    def to_dict(self) -> dict:
        return {"da": self.da, "da_baseline": self.da_baseline, "ne": self.ne, "ach": self.ach}

    @classmethod
    def from_dict(cls, d: dict) -> "NeuromodState":
        return cls(
            da=d.get("da", 0.0),
            da_baseline=d.get("da_baseline", 0.05),
            da_decay=d.get("da_decay", 0.05),
            ne=d.get("ne", 0.0),
            ach=d.get("ach", 1.0),
        )
