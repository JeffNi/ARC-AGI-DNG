"""
BCM Sliding Threshold — metaplasticity.

Bienenstock, Cooper, Munro (1982): the threshold for LTP vs LTD slides
based on recent average activity. Prevents runaway potentiation in
active neurons and silent death in quiet ones.

theta tracks the SQUARE of recent activity (classic BCM formulation).
"""

from __future__ import annotations

import numpy as np

from .setpoints import HomeostasisSetpoints


def update_bcm_theta(
    bcm_theta: np.ndarray,
    ema_rate: np.ndarray,
    setpoints: HomeostasisSetpoints,
) -> None:
    """
    Update per-neuron BCM sliding threshold in-place.

    bcm_theta[i] += tau * (ema_rate[i]^2 - bcm_theta[i])
    """
    tau = setpoints.bcm_tau
    target_sq = ema_rate ** 2
    bcm_theta += tau * (target_sq - bcm_theta)
    np.clip(bcm_theta, 1e-6, 10.0, out=bcm_theta)
