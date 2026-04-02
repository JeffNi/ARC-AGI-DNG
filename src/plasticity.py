"""
Synaptic and structural plasticity.

Learning rules operate on firing rates (r >= 0).
Dale's law maintained: weight updates preserve each edge's sign.

Primary rule: Contrastive Hebbian Learning (CHL).
  Free phase:    network thinks with test input, records correlations.
  Clamped phase: correct output injected on motor nodes, records correlations.
  Update:        dw = eta * DA * ACh * (clamped_corr - free_corr)

  Biologically: the difference between "what I thought" and
  "what I see when shown the answer" drives synaptic change.
  Dopamine modulates magnitude (reward prediction error).
  Acetylcholine gates plasticity (high in childhood, low in adult).

Structural plasticity:
  Synaptogenesis: co-active unconnected nodes grow new edges.
  Pruning: sustained weak edges are removed.
  Both happen at every developmental phase, at different rates.

Also: homeostatic excitability, selective sleep.
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
    """
    Run network for N steps, accumulate average pre*post correlations.
    Delegates to the Numba-compiled record_think for speed.
    """
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
    Update weights using the difference in correlations between
    clamped (correct answer shown) and free (network's own guess) phases.

    dw = eta * DA * ACh * (clamped_corr - free_corr)

    DA and ACh are read from net.da, net.ach.
    Returns mean absolute weight change for monitoring.
    """
    n = net._edge_count
    if n == 0:
        return 0.0

    da = max(net.da, 0.0)
    ach = net.ach

    modulation = eta * da * ach
    if modulation < 1e-8:
        return 0.0

    diff = clamped_corr[:n] - free_corr[:n]
    dw = modulation * diff

    # Synaptic consolidation: well-established synapses resist modification.
    # plasticity_scale ∈ (0, 1]: fully plastic when consolidation=0,
    # approaching 0 for highly consolidated synapses.
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


def consolidate_synapses(
    net: DNG,
    w_before: np.ndarray,
    reward: float,
    reward_threshold: float = 0.5,
    consolidation_strength: float = 2.0,
) -> int:
    """
    After successful learning, mark changed synapses as consolidated.

    Biologically: LTP involves structural spine growth and AMPA receptor
    insertion. Strongly modified synapses become physically larger and
    more resistant to future depression -- metaplasticity.

    Any synapse that changed gets consolidation += strength, making it
    ~(1+strength)x less plastic. With strength=2.0, a consolidated
    synapse has only 33% of normal plasticity.
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


# ── Prediction-error learning (legacy, kept for comparison) ───────

def prediction_error_update(
    net: DNG,
    target_r: np.ndarray,
    eta: float = 0.01,
    w_max: float = 2.0,
    error_propagation: float = 0.3,
) -> float:
    """2-hop prediction error from motor nodes. See previous docstring."""
    n = net._edge_count
    if n == 0:
        return 0.0

    src = net._edge_src[:n]
    dst = net._edge_dst[:n]
    w = net._edge_w[:n]

    motor_nodes = net.output_nodes

    error = np.zeros(net.n_nodes)
    error[motor_nodes] = target_r[motor_nodes] - net.r[motor_nodes]
    mean_err = float(np.mean(np.abs(error[motor_nodes])))

    motor_mask = np.isin(dst, motor_nodes)
    if np.any(motor_mask):
        mi = np.where(motor_mask)[0]
        np.add.at(error, src[mi],
                  error_propagation * w[mi] * error[dst[mi]])

    has_error = np.where(np.abs(error) > 0.001)[0]
    if len(has_error) > 0:
        error_mask = np.isin(dst, has_error) & ~motor_mask
        if np.any(error_mask):
            ei = np.where(error_mask)[0]
            np.add.at(error, src[ei],
                      error_propagation * 0.5 * w[ei] * error[dst[ei]])

    pre_rate = net.r[src]
    post_error = error[dst]
    active = pre_rate > 0.01

    dw = np.zeros(n)
    dw[active] = eta * pre_rate[active] * post_error[active]

    pos_mask = w > 0
    neg_mask = w < 0

    new_w = w + dw
    new_w[pos_mask] = np.clip(new_w[pos_mask], 0.0, w_max)
    new_w[neg_mask] = np.clip(new_w[neg_mask], -w_max, 0.0)

    net._edge_w[:n] = new_w
    net._edge_tag[:n] += np.abs(new_w - w)
    net._csr_dirty = True

    return mean_err


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
    """Selective downscaling: only weaken untagged synapses."""
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
    """
    Grow new edges between co-active unconnected node pairs.

    Probability-based, modulated by ACh (acetylcholine):
      For each candidate pair (src, dst):
        p = growth_rate * ACh * r[src] * r[dst]
        if random() < p: create edge

    This means:
      - More co-active pairs → more connections (fire together, wire together)
      - Higher ACh (childhood) → more growth
      - Lower ACh (adult) → rare growth
      - The NUMBER of new edges isn't hard-coded; it emerges from activity

    New edge weight is proportional to co-activation strength.
    Dale's law: excitatory source → positive weight, inhibitory → negative.

    Returns number of edges created.
    """
    if rng is None:
        rng = np.random.default_rng()
    if growth_rate <= 0 or net.ach < 0.01:
        return 0

    from .graph import Region
    region_list = list(Region)
    sensory_idx = region_list.index(Region.SENSORY)
    motor_idx = region_list.index(Region.MOTOR)

    # Two pools: active nodes (co-activation driven) and all nodes (exploratory).
    # High ACh (childhood) allows more exploratory connections.
    active_mask = net.r > 0.02
    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)

    if n_active < 2:
        return 0

    # Split candidates: most from active nodes, some exploratory from ALL nodes
    explore_frac = 0.2 * net.ach  # up to 20% exploratory during high-ACh
    n_explore = int(n_candidates * explore_frac)
    n_coactive = n_candidates - n_explore

    n_cand_active = min(n_coactive, n_active * n_active)
    src_active = active_idx[rng.integers(0, n_active, size=n_cand_active)]
    dst_active = active_idx[rng.integers(0, n_active, size=n_cand_active)]

    # Exploratory: sample from ALL non-sensory/non-motor-source nodes
    all_idx = np.arange(net.n_nodes)
    src_explore = all_idx[rng.integers(0, net.n_nodes, size=n_explore)]
    dst_explore = all_idx[rng.integers(0, net.n_nodes, size=n_explore)]

    src_samples = np.concatenate([src_active, src_explore])
    dst_samples = np.concatenate([dst_active, dst_explore])

    # Filter: no self-loops, no edges TO sensory, no edges FROM motor
    valid = (
        (src_samples != dst_samples) &
        (net.regions[dst_samples] != sensory_idx) &
        (net.regions[src_samples] != motor_idx)
    )
    src_samples = src_samples[valid]
    dst_samples = dst_samples[valid]

    if len(src_samples) == 0:
        return 0

    # Filter out existing edges using hash set (O(1) lookup per candidate)
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

    # Probability: co-activation driven + tiny baseline for exploration
    coact = net.r[src_samples] * net.r[dst_samples]
    baseline_p = 0.001 * net.ach  # very small chance without co-firing
    prob = growth_rate * net.ach * coact + baseline_p
    np.clip(prob, 0.0, 1.0, out=prob)
    rolls = rng.random(len(prob))
    winners = rolls < prob

    new_src = src_samples[winners]
    new_dst = dst_samples[winners]

    if len(new_src) == 0:
        return 0

    # "Silent synapse" initial weights: structurally present but functionally
    # very weak. They must be strengthened through CHL to become functional.
    new_coact = coact[winners]
    base_weight = 0.001 + 0.004 * (new_coact / (new_coact.max() + 1e-8))
    is_inhib = net.node_types[new_src] == _NTYPE_I
    weights = base_weight.copy()
    weights[is_inhib] *= -1.0

    net.add_edges_batch(new_src, new_dst, weights)
    return len(new_src)
