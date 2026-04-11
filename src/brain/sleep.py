"""
Sleep module — offline consolidation.

When fatigue triggers sleep:
  1. NREM-like phase: CHL replay of recent patterns
  2. SHY downscaling: weaken untagged synapses
  3. Competitive pruning: remove chronically weak edges

Homeostatic excitability now runs continuously (src/homeostasis/),
so sleep focuses purely on consolidation and cleanup.
"""

from __future__ import annotations

import numpy as np

from ..graph import DNG
from ..plasticity import (
    sleep_selective,
    update_edge_health,
    prune_sustained,
    contrastive_hebbian_update,
)
from .fatigue import FatigueTracker


def nrem_sleep(
    net: DNG,
    replay_buffer: list,
    chl_eta: float = 0.005,
    n_replay: int = 5,
    shy_downscale: float = 0.97,
    tag_threshold: float = 0.01,
    health_decay_rate: float = 0.01,
    ema_rate: np.ndarray | None = None,
    target_rate: float = 0.15,
    w_max: float = 5.0,
    layer_decay_scales: np.ndarray | None = None,
    **kwargs,
) -> dict:
    """
    Full NREM sleep cycle.

    replay_buffer: list of (free_corr, clamped_corr, da_tag) tuples.
        da_tag records the DA level at storage time — higher = more surprising.
        Replays are sorted by da_tag (highest first) so the most salient
        experiences get consolidated first, mirroring hippocampal sharp-wave
        ripple prioritization.
    layer_decay_scales: per-edge multiplier on health decay (layer-aware pruning).
    """
    stats = {"replays": 0, "pruned": 0}

    # Sort replay buffer by DA tag (most surprising first).
    # Handles both old 2-tuple and new 3-tuple format.
    tagged = []
    for entry in replay_buffer:
        if len(entry) == 3:
            tagged.append(entry)
        else:
            tagged.append((entry[0], entry[1], 0.0))
    tagged.sort(key=lambda x: x[2], reverse=True)

    n_to_replay = min(n_replay, len(tagged))
    for i in range(n_to_replay):
        free_corr, clamped_corr, _da = tagged[i]
        contrastive_hebbian_update(net, free_corr, clamped_corr, eta=chl_eta,
                                   w_max=w_max)
        stats["replays"] += 1

    # SHY: selective downscaling — active edges protected, inactive downscaled
    sleep_selective(
        net, downscale=shy_downscale, tag_threshold=tag_threshold,
        ema_rate=ema_rate, target_rate=target_rate,
    )

    # Health decay then health-based pruning
    update_edge_health(
        net, decay_rate=health_decay_rate,
        ema_rate=ema_rate, target_rate=target_rate,
        layer_decay_scales=layer_decay_scales,
    )
    stats["pruned"] = prune_sustained(net)

    return stats
