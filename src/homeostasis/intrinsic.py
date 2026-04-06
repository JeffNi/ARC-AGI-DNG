"""
Intrinsic Plasticity — per-neuron excitability adjustment.

Neurons adjust their own gain/threshold to maintain a target
firing rate. Previously only ran during sleep; now continuous.
"""

from __future__ import annotations

import numpy as np

from ..graph import DNG
from .setpoints import HomeostasisSetpoints


def intrinsic_plasticity(
    net: DNG,
    ema_rate: np.ndarray,
    setpoints: HomeostasisSetpoints,
    regulated_mask: np.ndarray | None = None,
) -> None:
    """
    Adjust excitability to push each neuron's rate toward target.

    excitability[i] += eta * (target_rate - ema_rate[i])

    Only applies to regulated neurons (internal + memory).
    """
    eta = setpoints.intrinsic_eta
    target = setpoints.target_rate

    delta = eta * (target - ema_rate)

    if regulated_mask is not None:
        delta[~regulated_mask] = 0.0

    net.excitability += delta
    np.clip(
        net.excitability,
        setpoints.intrinsic_min,
        setpoints.intrinsic_max,
        out=net.excitability,
    )
