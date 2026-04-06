"""
Synaptic Scaling — multiplicative, per-neuron weight normalization.

Turrigiano (1998): each neuron monitors its own firing rate and
multiplicatively adjusts ALL incoming synaptic weights to keep the
rate near target. Preserves relative weight ratios (learned structure
survives). Consolidated edges get reduced scaling.
"""

from __future__ import annotations

import numpy as np

from ..graph import DNG
from .setpoints import HomeostasisSetpoints


def synaptic_scaling(
    net: DNG,
    ema_rate: np.ndarray,
    setpoints: HomeostasisSetpoints,
    regulated_mask: np.ndarray | None = None,
) -> None:
    """
    Multiplicatively scale incoming weights per neuron to push
    firing rates toward the target.

    scale_factor[i] = 1 - gain * (ema_rate[i] - target)
    Only applied to edges whose DESTINATION is a regulated neuron.
    Sensory and motor neurons are excluded — they have different
    functional requirements.
    """
    n = net._edge_count
    if n == 0:
        return

    gain = setpoints.scaling_gain
    target = setpoints.target_rate

    dst = net._edge_dst[:n]
    w = net._edge_w[:n]

    error = ema_rate[dst] - target
    scale = np.ones(n, dtype=np.float64)

    if setpoints.scaling_protect_consolidated:
        cons = net._edge_consolidation[:n]
        effective_gain = gain / (1.0 + cons)
        scale -= effective_gain * error
    else:
        scale -= gain * error

    # Only scale edges to regulated neurons
    if regulated_mask is not None:
        unregulated = ~regulated_mask[dst]
        scale[unregulated] = 1.0

    np.clip(scale, 0.5, 2.0, out=scale)

    new_w = w * scale

    # Dale's law: never flip sign
    pos = w > 0
    neg = w < 0
    new_w[pos] = np.maximum(new_w[pos], 1e-7)
    new_w[neg] = np.minimum(new_w[neg], -1e-7)

    net._edge_w[:n] = new_w
    net._csr_dirty = True
