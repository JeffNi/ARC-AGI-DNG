"""
Homeostasis orchestrator — runs all mechanisms on a schedule.

Called every Brain.step(). Only applies corrections every `interval`
steps, but always tracks the firing rate EMA.

Region-aware: only INTERNAL and MEMORY neurons are regulated.
Sensory neurons are externally driven; motor neurons need to fire
strongly to signal outputs. Applying homeostasis to them fights
the network's function rather than stabilizing it.
"""

from __future__ import annotations

import numpy as np

from ..graph import DNG, Region
from .setpoints import HomeostasisSetpoints
from .scaling import synaptic_scaling
from .intrinsic import intrinsic_plasticity
from .ei_balance import ei_balance_update


class Homeostasis:
    def __init__(self, net: DNG, setpoints: HomeostasisSetpoints, interval: int = 50,
                 r_pre_wta: np.ndarray | None = None):
        self.net = net
        self.setpoints = setpoints
        self.interval = interval
        self._r_pre_wta = r_pre_wta

        self.ema_rate = np.zeros(net.n_nodes, dtype=np.float64)
        self.ema_rate_dendritic = np.zeros(net.n_nodes, dtype=np.float64)
        self._step_counter = 0

        # Regulate all cortical layers + memory neurons
        from ..graph import internal_mask as _internal_mask
        _reg = list(Region)
        memory_idx = _reg.index(Region.MEMORY)
        self.regulated_mask = (
            _internal_mask(net.regions) | (net.regions == memory_idx)
        )

    def step(self):
        """Called every Brain.step(). Tracks firing rate EMA only.

        Synaptic scaling and intrinsic plasticity are deferred to sleep
        (see sleep_correction). Biology: Turrigiano (2011) showed synaptic
        scaling requires hours of altered activity + protein synthesis,
        not millisecond-scale feedback. Running it continuously fights
        competitive Hebbian differentiation.
        """
        self._step_counter += 1
        self._update_ema()

    def sleep_correction(self):
        """Apply homeostatic corrections during sleep.

        This is when synaptic scaling + intrinsic plasticity run — after
        a full wake cycle of accumulated EMA statistics. Biologically
        grounded: Tononi & Cirelli (2014) SHY hypothesis places synaptic
        renormalization during sleep, not waking.
        """
        rate_for_homeo = (self.ema_rate_dendritic
                          if self._r_pre_wta is not None
                          else self.ema_rate)
        synaptic_scaling(self.net, rate_for_homeo, self.setpoints,
                         self.regulated_mask)
        intrinsic_plasticity(self.net, rate_for_homeo, self.setpoints,
                             self.regulated_mask)
        ei_balance_update(self.net, self.setpoints)

    def _update_ema(self):
        """Exponential moving average of firing rates.

        Two tracks: somatic (post-WTA) and dendritic (pre-WTA).
        Dendritic rates reflect actual input drive, somatic reflects
        competitive outcome. Homeostasis uses dendritic to avoid
        fighting WTA suppression.
        """
        tau = self.setpoints.ema_tau
        self.ema_rate *= (1.0 - tau)
        self.ema_rate += tau * self.net.r
        if self._r_pre_wta is not None:
            self.ema_rate_dendritic *= (1.0 - tau)
            self.ema_rate_dendritic += tau * self._r_pre_wta

    def calibrate_from_rates(self, r_pre_wta: np.ndarray | None = None):
        """Set EMA to match current firing rates.

        Called once at birth so homeostasis starts from realistic values.
        """
        rates = r_pre_wta if r_pre_wta is not None else self.net.r
        self.ema_rate[:] = self.net.r
        self.ema_rate_dendritic[:] = rates

    def state_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "ema_rate": self.ema_rate.tolist(),
            "ema_rate_dendritic": self.ema_rate_dendritic.tolist(),
            "step_counter": self._step_counter,
        }

    def load_state_dict(self, d: dict):
        """Restore from checkpoint."""
        if "ema_rate" in d:
            arr = np.array(d["ema_rate"], dtype=np.float64)
            n = min(len(arr), len(self.ema_rate))
            self.ema_rate[:n] = arr[:n]
        if "ema_rate_dendritic" in d:
            arr = np.array(d["ema_rate_dendritic"], dtype=np.float64)
            n = min(len(arr), len(self.ema_rate_dendritic))
            self.ema_rate_dendritic[:n] = arr[:n]
        if "step_counter" in d:
            self._step_counter = d["step_counter"]
