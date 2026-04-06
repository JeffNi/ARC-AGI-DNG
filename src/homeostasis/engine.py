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
from .bcm import update_bcm_theta
from .intrinsic import intrinsic_plasticity
from .ei_balance import ei_balance_update


class Homeostasis:
    def __init__(self, net: DNG, setpoints: HomeostasisSetpoints, interval: int = 50):
        self.net = net
        self.setpoints = setpoints
        self.interval = interval

        self.ema_rate = np.zeros(net.n_nodes, dtype=np.float64)
        self.bcm_theta = np.full(net.n_nodes, 0.1, dtype=np.float64)
        self._step_counter = 0

        # Only regulate internal + memory neurons
        _reg = list(Region)
        internal_idx = _reg.index(Region.INTERNAL)
        memory_idx = _reg.index(Region.MEMORY)
        self.regulated_mask = (
            (net.regions == internal_idx) | (net.regions == memory_idx)
        )

    def step(self):
        """Called every Brain.step(). Applies corrections on schedule."""
        self._step_counter += 1
        self._update_ema()

        if self._step_counter % self.interval == 0:
            synaptic_scaling(self.net, self.ema_rate, self.setpoints,
                             self.regulated_mask)
            intrinsic_plasticity(self.net, self.ema_rate, self.setpoints,
                                 self.regulated_mask)
            ei_balance_update(self.net, self.setpoints)
            update_bcm_theta(self.bcm_theta, self.ema_rate, self.setpoints)

    def _update_ema(self):
        """Exponential moving average of firing rates."""
        tau = self.setpoints.ema_tau
        self.ema_rate *= (1.0 - tau)
        self.ema_rate += tau * self.net.r

    def state_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "ema_rate": self.ema_rate.tolist(),
            "bcm_theta": self.bcm_theta.tolist(),
            "step_counter": self._step_counter,
        }

    def load_state_dict(self, d: dict):
        """Restore from checkpoint."""
        if "ema_rate" in d:
            arr = np.array(d["ema_rate"], dtype=np.float64)
            n = min(len(arr), len(self.ema_rate))
            self.ema_rate[:n] = arr[:n]
        if "bcm_theta" in d:
            arr = np.array(d["bcm_theta"], dtype=np.float64)
            n = min(len(arr), len(self.bcm_theta))
            self.bcm_theta[:n] = arr[:n]
        if "step_counter" in d:
            self._step_counter = d["step_counter"]
