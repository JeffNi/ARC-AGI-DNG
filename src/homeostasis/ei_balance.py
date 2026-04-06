"""
E/I Balance Regulation — dynamic inhibitory scaling.

PV+ interneurons track and match local excitatory activity.
We replace the fixed inh_scale with a dynamic one that adjusts
to maintain a target ratio between inhibitory and excitatory drive.
"""

from __future__ import annotations

import numpy as np

from ..graph import DNG
from .setpoints import HomeostasisSetpoints


def ei_balance_update(
    net: DNG,
    setpoints: HomeostasisSetpoints,
) -> None:
    """
    Adjust inh_scale so that mean inhibitory input tracks mean excitatory input.

    current_ratio = mean_inhib / (mean_excit + eps)
    inh_scale += rate * (target_ratio - current_ratio)
    """
    n = net._edge_count
    if n == 0:
        return

    w = net._edge_w[:n]
    dst = net._edge_dst[:n]
    r = net.r

    pre_activity = r[net._edge_src[:n]]
    weighted_input = w * pre_activity

    exc_input = np.zeros(net.n_nodes)
    inh_input = np.zeros(net.n_nodes)

    exc_mask = w > 0
    inh_mask = w < 0

    np.add.at(exc_input, dst[exc_mask], weighted_input[exc_mask])
    np.add.at(inh_input, dst[inh_mask], np.abs(weighted_input[inh_mask]))

    mean_exc = exc_input.mean()
    mean_inh = inh_input.mean()

    eps = 1e-8
    current_ratio = mean_inh / (mean_exc + eps)

    adjustment = setpoints.ei_adjustment_rate * (setpoints.ei_target_ratio - current_ratio)
    net.inh_scale = float(np.clip(net.inh_scale + adjustment, 0.5, 3.0))
