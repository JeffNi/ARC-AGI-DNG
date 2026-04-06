"""
Homeostatic setpoints — the tuning knobs.

Each developmental stage shifts these to change what "stable" means.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HomeostasisSetpoints:
    target_rate: float = 0.15
    scaling_gain: float = 0.01
    scaling_protect_consolidated: bool = True
    bcm_tau: float = 0.01
    intrinsic_eta: float = 0.005
    intrinsic_min: float = 0.1
    intrinsic_max: float = 5.0
    ei_target_ratio: float = 0.8
    ei_adjustment_rate: float = 0.005
    ema_tau: float = 0.02

    def interpolate(self, other: "HomeostasisSetpoints", t: float) -> "HomeostasisSetpoints":
        """Linearly interpolate between self and other at fraction t in [0, 1]."""
        t = max(0.0, min(1.0, t))
        s = 1.0 - t
        return HomeostasisSetpoints(
            target_rate=s * self.target_rate + t * other.target_rate,
            scaling_gain=s * self.scaling_gain + t * other.scaling_gain,
            scaling_protect_consolidated=self.scaling_protect_consolidated,
            bcm_tau=s * self.bcm_tau + t * other.bcm_tau,
            intrinsic_eta=s * self.intrinsic_eta + t * other.intrinsic_eta,
            intrinsic_min=s * self.intrinsic_min + t * other.intrinsic_min,
            intrinsic_max=s * self.intrinsic_max + t * other.intrinsic_max,
            ei_target_ratio=s * self.ei_target_ratio + t * other.ei_target_ratio,
            ei_adjustment_rate=s * self.ei_adjustment_rate + t * other.ei_adjustment_rate,
            ema_tau=s * self.ema_tau + t * other.ema_tau,
        )
