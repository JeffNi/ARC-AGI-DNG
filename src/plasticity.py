"""
Synaptic and structural plasticity.

Phase 1 learning rules:
  - Contrastive Hebbian Learning (CHL): error-corrective
  - Eligibility-modulated update: DA cashes in tagged synapses
  - Consolidation: protects well-learned synapses
  - Structural: synaptogenesis + pruning
  - Homeostatic excitability

Dale's law maintained: weight updates preserve each edge's sign.
ALL edges are plastic (no region-based exclusions — past failure).
"""

from __future__ import annotations

import numpy as np

from .graph import DNG, _NTYPE_I


# ── Correlation recording ─────────────────────────────────────────

def record_phase(
    net: DNG,
    step_fn,
    signal: np.ndarray | None,
    steps: int,
    noise_std: float,
) -> np.ndarray:
    """Run network for N steps, accumulate average pre*post correlations."""
    from .dynamics import record_think
    return record_think(net, signal=signal, steps=steps, noise_std=noise_std)


# ── Contrastive Hebbian Learning ──────────────────────────────────

def contrastive_hebbian_update(
    net: DNG,
    free_corr: np.ndarray,
    clamped_corr: np.ndarray,
    eta: float = 0.01,
    w_max: float = 2.0,
) -> float:
    """
    Update weights using correlation difference between clamped and free phases.

    dw = eta * DA * (clamped_corr - free_corr)

    DA can be negative, which REVERSES the update direction.
    """
    n = net._edge_count
    if n == 0:
        return 0.0

    da = net.da
    modulation = eta * da
    if abs(modulation) < 1e-8:
        return 0.0

    diff = clamped_corr[:n] - free_corr[:n]
    dw = modulation * diff

    # Consolidation: well-established synapses resist modification
    cons = net._edge_consolidation[:n]
    if cons.any():
        plasticity_scale = 1.0 / (1.0 + cons)
        dw *= plasticity_scale

    w = net._edge_w[:n]
    pos_mask = w > 0
    neg_mask = w < 0

    new_w = w + dw
    new_w[pos_mask] = np.clip(new_w[pos_mask], 0.0, w_max)
    new_w[neg_mask] = np.clip(new_w[neg_mask], -w_max, 0.0)

    change = np.abs(new_w - w)
    net._edge_w[:n] = new_w
    net._edge_tag[:n] += change
    net._csr_dirty = True

    return float(np.mean(change))


# ── Eligibility-modulated update ──────────────────────────────────

def eligibility_modulated_update(
    net: DNG,
    DA: float,
    eta: float = 0.02,
    w_max: float = 5.0,
) -> float:
    """
    Apply delayed reward signal to eligible synapses.

    dw = eta * DA * eligibility * consolidation_scale

    CRITICAL BEHAVIOR:
    - Positive DA + positive eligibility -> strengthen (excitatory: increase, inhibitory: more negative)
    - Negative DA + positive eligibility -> WEAKEN (excitatory: decrease, inhibitory: less negative)
    - Zero eligibility -> no change regardless of DA

    After update, eligibility traces are decayed (not zeroed — partial credit).
    """
    n = net._edge_count
    if n == 0:
        return 0.0

    elig = net._edge_eligibility[:n]
    if not np.any(elig > 1e-6):
        return 0.0

    # Base weight change: eta * DA * eligibility
    dw = eta * DA * elig

    # Consolidation protection
    cons = net._edge_consolidation[:n]
    if cons.any():
        plasticity_scale = 1.0 / (1.0 + cons)
        dw *= plasticity_scale

    w = net._edge_w[:n]

    # Dale's law: inhibitory edges flip dw so "strengthen" = more negative
    neg_mask = w < 0
    dw[neg_mask] = -dw[neg_mask]

    new_w = w + dw

    # Clip to maintain sign (Dale's law)
    pos_mask = w > 0
    new_w[pos_mask] = np.clip(new_w[pos_mask], 1e-6, w_max)
    new_w[neg_mask] = np.clip(new_w[neg_mask], -w_max, -1e-6)

    change = np.abs(new_w - w)
    net._edge_w[:n] = new_w
    net._edge_tag[:n] += change
    net._csr_dirty = True

    # Decay eligibility after cashing in (partial, not zeroed)
    net._edge_eligibility[:n] *= 0.5

    return float(np.mean(change))


# ── Consolidation ─────────────────────────────────────────────────

def consolidate_synapses(
    net: DNG,
    w_before: np.ndarray,
    reward: float,
    reward_threshold: float = 0.5,
    consolidation_strength: float = 2.0,
) -> int:
    """
    After successful learning, mark changed synapses as consolidated.

    Consolidated synapses have reduced plasticity: effective learning rate
    is scaled by 1/(1+consolidation). With strength=2.0, a consolidated
    synapse has ~33% of normal plasticity.
    """
    if reward < reward_threshold:
        return 0

    n = net._edge_count
    dw = np.abs(net._edge_w[:n] - w_before[:n])
    changed = dw > 1e-5
    net._edge_consolidation[:n][changed] += consolidation_strength
    return int(changed.sum())


def get_weight_snapshot(net: DNG) -> np.ndarray:
    """Save current edge weights for later consolidation comparison."""
    return net._edge_w[:net._edge_count].copy()


# ── Homeostatic plasticity ─────────────────────────────────────────

def homeostatic_excitability_update(
    net: DNG,
    eta_b: float = 0.005,
    a_target: float = 0.2,
    b_min: float = 0.1,
    b_max: float = 5.0,
    ema_r: np.ndarray | None = None,
    ema_rate: float = 0.1,
) -> np.ndarray:
    """Homeostatic gain: too active -> lower excitability, too quiet -> raise."""
    if ema_r is None:
        ema_r = net.r.copy()
    else:
        ema_r = (1 - ema_rate) * ema_r + ema_rate * net.r
    net.excitability += eta_b * (a_target - ema_r)
    np.clip(net.excitability, b_min, b_max, out=net.excitability)
    return ema_r


# ── Sleep & pruning ────────────────────────────────────────────────

def sleep_selective(
    net: DNG,
    downscale: float = 0.95,
    tag_threshold: float = 0.01,
) -> None:
    """Selective downscaling: only weaken untagged synapses (SHY hypothesis)."""
    n = net._edge_count
    if n == 0:
        return
    tags = net._edge_tag[:n]
    untagged = tags < tag_threshold
    net._edge_w[:n][untagged] *= downscale
    net._edge_tag[:n] = 0.0
    net._csr_dirty = True


def prune_sustained(
    net: DNG,
    weak_threshold: float = 0.01,
    cycles_required: int = 5,
) -> int:
    """Remove edges weak for multiple consecutive sleep cycles."""
    n = net._edge_count
    if n == 0:
        return 0
    w = net._edge_w[:n]
    is_weak = np.abs(w) < weak_threshold
    net._edge_weak_count[:n][is_weak] += 1
    net._edge_weak_count[:n][~is_weak] = 0
    to_remove = net._edge_weak_count[:n] >= cycles_required
    n_pruned = int(np.sum(to_remove))
    if n_pruned > 0:
        net._edge_w[:n][to_remove] = 0.0
        net.compact()
    return n_pruned


# ── Synaptogenesis ────────────────────────────────────────────────

def synaptogenesis(
    net: DNG,
    growth_rate: float = 0.3,
    n_candidates: int = 5000,
    rng: np.random.Generator | None = None,
) -> int:
    """Grow new edges between co-active unconnected node pairs."""
    if rng is None:
        rng = np.random.default_rng()
    if growth_rate <= 0 or net.ach < 0.01:
        return 0

    from .graph import Region
    region_list = list(Region)
    motor_idx = region_list.index(Region.MOTOR)

    active_mask = net.r > 0.02
    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)

    if n_active < 2:
        return 0

    explore_frac = 0.2 * net.ach
    n_explore = int(n_candidates * explore_frac)
    n_coactive = n_candidates - n_explore

    n_cand_active = min(n_coactive, n_active * n_active)
    src_active = active_idx[rng.integers(0, n_active, size=n_cand_active)]
    dst_active = active_idx[rng.integers(0, n_active, size=n_cand_active)]

    all_idx = np.arange(net.n_nodes)
    src_explore = all_idx[rng.integers(0, net.n_nodes, size=n_explore)]
    dst_explore = all_idx[rng.integers(0, net.n_nodes, size=n_explore)]

    src_samples = np.concatenate([src_active, src_explore])
    dst_samples = np.concatenate([dst_active, dst_explore])

    # No self-loops, no edges FROM motor (motor is output only)
    valid = (
        (src_samples != dst_samples) &
        (net.regions[src_samples] != motor_idx)
    )
    src_samples = src_samples[valid]
    dst_samples = dst_samples[valid]

    if len(src_samples) == 0:
        return 0

    n_nodes = net.n_nodes
    n_existing = net._edge_count
    existing_set = set(
        (net._edge_src[:n_existing].astype(np.int64) * n_nodes
         + net._edge_dst[:n_existing].astype(np.int64)).tolist()
    )
    candidate_hash = (src_samples.astype(np.int64) * n_nodes
                      + dst_samples.astype(np.int64))
    novel_mask = np.array([h not in existing_set for h in candidate_hash.tolist()],
                          dtype=bool)
    src_samples = src_samples[novel_mask]
    dst_samples = dst_samples[novel_mask]

    if len(src_samples) == 0:
        return 0

    coact = net.r[src_samples] * net.r[dst_samples]
    baseline_p = 0.001 * net.ach
    prob = growth_rate * net.ach * coact + baseline_p
    np.clip(prob, 0.0, 1.0, out=prob)
    rolls = rng.random(len(prob))
    winners = rolls < prob

    new_src = src_samples[winners]
    new_dst = dst_samples[winners]

    if len(new_src) == 0:
        return 0

    new_coact = coact[winners]
    base_weight = 0.001 + 0.004 * (new_coact / (new_coact.max() + 1e-8))
    is_inhib = net.node_types[new_src] == _NTYPE_I
    weights = base_weight.copy()
    weights[is_inhib] *= -1.0

    net.add_edges_batch(new_src, new_dst, weights)
    return len(new_src)
