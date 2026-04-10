"""
Homeostatic setpoints — the tuning knobs.

Each developmental stage shifts these to change what "stable" means.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HomeostasisSetpoints:
    # --- Core homeostatic parameters ---
    target_rate: float = 0.15
    scaling_gain: float = 0.01
    scaling_protect_consolidated: bool = True
    intrinsic_eta: float = 0.005
    intrinsic_min: float = 0.1
    intrinsic_max: float = 5.0
    ei_target_ratio: float = 0.8
    ei_adjustment_rate: float = 0.005
    ema_tau: float = 0.02

    # --- Developmental parameters (set by genetic clock) ---
    synaptogenesis_rate: float = 0.3       # acceptance probability for neurotrophin pool
    synaptogenesis_candidates: int = 2_000    # candidate pairs sampled per step (continuous)
    health_decay_rate: float = 0.01        # per-sleep health decay (low=gentle, high=aggressive)
    da_baseline: float = 0.05              # resting dopamine level
    da_sensitivity: float = 0.5            # novelty detector gain
    plasticity_rate: float = 1.0           # multiplier on all learning rates
    leak_rate_target: float = 0.12         # myelination analog; lower = faster signals
    n_demos: int = 2                       # demo pairs shown before attempt
    spotlight_strength: float = 1.5        # attention boost on changed cells
    fatigue_threshold: float = 10.0        # sleep trigger level
    wta_active_frac: float = 0.2            # fraction of internal pool that fires at full rate
    noise_std: float = 0.02                  # membrane noise (spontaneous activity driver)
    task_mix_ratio: float = 0.0              # 0.0=all stimuli, 1.0=all tasks (interpolated across stages)
    babble_ratio: float = 0.0               # fraction of waking spent on motor babble + mimicry
    copy_strength: float = 1.0               # multiplier on copy pathway weights (instinct decay)

    def interpolate(self, other: "HomeostasisSetpoints", t: float) -> "HomeostasisSetpoints":
        """Linearly interpolate between self and other at fraction t in [0, 1]."""
        t = max(0.0, min(1.0, t))
        s = 1.0 - t
        return HomeostasisSetpoints(
            target_rate=s * self.target_rate + t * other.target_rate,
            scaling_gain=s * self.scaling_gain + t * other.scaling_gain,
            scaling_protect_consolidated=self.scaling_protect_consolidated,
            intrinsic_eta=s * self.intrinsic_eta + t * other.intrinsic_eta,
            intrinsic_min=s * self.intrinsic_min + t * other.intrinsic_min,
            intrinsic_max=s * self.intrinsic_max + t * other.intrinsic_max,
            ei_target_ratio=s * self.ei_target_ratio + t * other.ei_target_ratio,
            ei_adjustment_rate=s * self.ei_adjustment_rate + t * other.ei_adjustment_rate,
            ema_tau=s * self.ema_tau + t * other.ema_tau,
            # Developmental
            synaptogenesis_rate=s * self.synaptogenesis_rate + t * other.synaptogenesis_rate,
            synaptogenesis_candidates=round(s * self.synaptogenesis_candidates + t * other.synaptogenesis_candidates),
            health_decay_rate=s * self.health_decay_rate + t * other.health_decay_rate,
            da_baseline=s * self.da_baseline + t * other.da_baseline,
            da_sensitivity=s * self.da_sensitivity + t * other.da_sensitivity,
            plasticity_rate=s * self.plasticity_rate + t * other.plasticity_rate,
            leak_rate_target=s * self.leak_rate_target + t * other.leak_rate_target,
            n_demos=round(s * self.n_demos + t * other.n_demos),
            spotlight_strength=s * self.spotlight_strength + t * other.spotlight_strength,
            fatigue_threshold=s * self.fatigue_threshold + t * other.fatigue_threshold,
            wta_active_frac=s * self.wta_active_frac + t * other.wta_active_frac,
            noise_std=s * self.noise_std + t * other.noise_std,
            task_mix_ratio=s * self.task_mix_ratio + t * other.task_mix_ratio,
            babble_ratio=s * self.babble_ratio + t * other.babble_ratio,
            copy_strength=s * self.copy_strength + t * other.copy_strength,
        )
