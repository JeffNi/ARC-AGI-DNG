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
    prune_threshold: float = 0.005,
    prune_cycles: int = 5,
    **kwargs,
) -> dict:
    """
    Full NREM sleep cycle.

    replay_buffer: list of (free_corr, clamped_corr) tuples from recent tasks.
    """
    stats = {"replays": 0, "pruned": 0}

    # CHL replay: consolidate recent learning
    n_to_replay = min(n_replay, len(replay_buffer))
    for i in range(n_to_replay):
        free_corr, clamped_corr = replay_buffer[-(i + 1)]
        contrastive_hebbian_update(net, free_corr, clamped_corr, eta=chl_eta)
        stats["replays"] += 1

    # SHY: selective downscaling of untagged synapses
    sleep_selective(net, downscale=shy_downscale, tag_threshold=tag_threshold)

    # Pruning
    stats["pruned"] = prune_sustained(
        net, weak_threshold=prune_threshold, cycles_required=prune_cycles
    )

    return stats
