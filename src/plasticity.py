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
    activity_gate: float = 0.02,
) -> float:
    """
    Update weights using the difference in correlations between
    clamped (correct answer shown) and free (network's own guess) phases.

    dw = eta * DA * ACh * (clamped_corr - free_corr) * gate

    Two selectivity mechanisms prevent catastrophic interference:
      1. Activity gating: only edges where BOTH endpoints had significant
         activity (max of free/clamped r > activity_gate) get updated.
         This makes CHL task-local.
      2. Pathway freezing: edges FROM sensory neurons are never modified.
         Sensory→abstract projections are general-purpose and fixed,
         like retina→V1 in the brain. Only internal processing is plastic.
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

    # ── Pathway freeze: never modify edges originating from sensory nodes ──
    from .graph import Region
    _sensory_idx = list(Region).index(Region.SENSORY)
    src_regions = net.regions[net._edge_src[:n]]
    frozen = src_regions == _sensory_idx
    dw[frozen] = 0.0

    # ── Activity gating: only update edges where both neurons were active ──
    if activity_gate > 0:
        src_active = net.r[net._edge_src[:n]] > activity_gate
        dst_active = net.r[net._edge_dst[:n]] > activity_gate
        inactive = ~(src_active & dst_active)
        dw[inactive] = 0.0

    # Synaptic consolidation: well-established synapses resist modification.
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


# ── Fast Hebbian binding (hippocampal one-shot learning) ──────────

def fast_hebbian_bind(
    net: DNG,
    eta: float = 0.05,
    w_max: float = 2.5,
    activity_threshold: float = 0.02,
) -> int:
    """
    Strengthen edges between co-active internal neurons.

    Hippocampal fast learning: when the network observes an input-output
    pair, neurons that fire together get wired together in one shot.
    Only modifies internal-to-internal and internal-to-motor edges
    where both endpoints are active. Sensory edges and copy pathway
    are never touched.

    Returns number of edges modified.
    """
    n = net._edge_count
    if n == 0:
        return 0

    from .graph import Region
    _region_list = list(Region)
    _abstract = _region_list.index(Region.ABSTRACT)
    _motor = _region_list.index(Region.MOTOR)
    _memory = _region_list.index(Region.MEMORY)

    src = net._edge_src[:n]
    dst = net._edge_dst[:n]
    src_reg = net.regions[src]
    dst_reg = net.regions[dst]

    # Eligible: abstract->abstract, abstract->motor, memory->abstract, memory->motor
    src_ok = (src_reg == _abstract) | (src_reg == _memory)
    dst_ok = (dst_reg == _abstract) | (dst_reg == _motor)
    eligible = src_ok & dst_ok

    src_active = net.r[src] > activity_threshold
    dst_active = net.r[dst] > activity_threshold
    active = src_active & dst_active & eligible

    n_active = int(active.sum())
    if n_active == 0:
        return 0

    pre = net.r[src]
    post = net.r[dst]

    # BCM-style sliding threshold: very active neurons raise their
    # potentiation threshold, making it harder to strengthen their inputs
    # further. Less active neurons lower it, making potentiation easier.
    # This self-corrects the rich-get-richer problem.
    bcm_theta = net.adaptation[dst] * 10.0
    bcm_mod = post - bcm_theta  # positive = potentiate, negative = depress
    dw_raw = eta * pre * np.maximum(bcm_mod, 0.0)

    w = net._edge_w[:n].copy()
    dw = np.where(active, dw_raw, 0.0)
    inhib_src = net.node_types[src] == _NTYPE_I
    dw[inhib_src & active] *= -1.0

    new_w = w + dw
    pos = w > 0
    neg = w < 0
    new_w[pos] = np.clip(new_w[pos], 0.0, w_max)
    new_w[neg] = np.clip(new_w[neg], -w_max, 0.0)

    # ── Heterosynaptic LTD: redistribute, don't just add ──────────
    # For each postsynaptic neuron that gained weight, proportionally
    # weaken its OTHER excitatory inputs. Total input stays roughly
    # constant, but active pathways capture a larger share.
    # Sensory (thalamocortical) edges are exempt -- biology maintains
    # these via a separate mechanism to preserve input drive.
    from .graph import Region
    _sensory_idx = list(Region).index(Region.SENSORY)
    from_sensory = net.regions[src] == _sensory_idx

    total_gained = np.bincount(dst, weights=np.maximum(dw, 0.0),
                               minlength=net.n_nodes)
    neurons_with_gain = total_gained > 0
    if neurons_with_gain.any():
        # non-potentiated, non-sensory excitatory edges
        exc_edges = (w > 0) & (~active) & (~from_sensory)
        dst_gains = total_gained[dst]
        n_exc_per = np.bincount(dst[exc_edges], minlength=net.n_nodes)
        n_exc_at_dst = np.maximum(n_exc_per[dst], 1).astype(np.float64)
        hetero_ltd = np.where(
            exc_edges & (dst_gains > 0),
            dst_gains / n_exc_at_dst * 0.5,
            0.0,
        )
        new_w -= hetero_ltd
        SILENT_FLOOR = 1e-4
        new_w[pos] = np.maximum(new_w[pos], SILENT_FLOOR)

    change = np.abs(new_w - w)
    net._edge_w[:n] = new_w
    net._edge_tag[:n] += change
    net._csr_dirty = True

    return n_active


# ── Synaptic scaling (homeostatic weight normalization) ────────────

def synaptic_scaling(
    net: DNG,
    target_total: float = 2.0,
    region_filter: int | None = None,
) -> int:
    """
    Normalize total incoming excitatory weight to each neuron,
    excluding edges from sensory neurons (thalamocortical inputs).

    In biology, thalamocortical synapses are maintained by a separate
    homeostatic mechanism and are NOT downscaled when cortical neurons
    strengthen their lateral connections. This preserves the sensory
    pathway that keeps neurons responsive to input.

    Only scales neurons in region_filter (if given), otherwise all.
    Returns number of neurons scaled.
    """
    from .graph import Region
    n = net._edge_count
    if n == 0:
        return 0

    src = net._edge_src[:n]
    dst = net._edge_dst[:n]
    w = net._edge_w[:n]

    if region_filter is not None:
        target_nodes = np.where(net.regions == region_filter)[0]
    else:
        target_nodes = np.arange(net.n_nodes)

    sensory_idx = list(Region).index(Region.SENSORY)
    from_sensory = net.regions[src] == sensory_idx

    # Only scale non-sensory excitatory edges
    exc_mask = (w > 0) & (~from_sensory)
    exc_dst = dst[exc_mask]
    exc_w = w[exc_mask]
    exc_edge_idx = np.where(exc_mask)[0]

    nn = net.n_nodes
    total_exc = np.bincount(exc_dst, weights=exc_w, minlength=nn)

    high = target_nodes[(total_exc[target_nodes] > target_total * 1.2)]
    low = target_nodes[(total_exc[target_nodes] < target_total * 0.5) &
                        (total_exc[target_nodes] > 0.01)]

    # Biological synaptic scaling is gradual (~hours to days). Cap the
    # per-step adjustment so new weak edges aren't crushed in one pass.
    # Max 20% down, max 50% up per call.
    n_scaled = 0
    if len(high) > 0:
        high_set = np.zeros(nn, dtype=bool)
        high_set[high] = True
        scale_map = np.ones(nn)
        raw_scale = target_total / total_exc[high]
        scale_map[high] = np.maximum(raw_scale, 0.8)  # at most 20% reduction
        mask = high_set[exc_dst]
        net._edge_w[exc_edge_idx[mask]] *= scale_map[exc_dst[mask]]
        n_scaled += len(high)

    if len(low) > 0:
        low_set = np.zeros(nn, dtype=bool)
        low_set[low] = True
        scale_map = np.ones(nn)
        raw_scale = target_total / total_exc[low]
        scale_map[low] = np.minimum(raw_scale, 1.5)  # at most 50% increase
        mask = low_set[exc_dst]
        net._edge_w[exc_edge_idx[mask]] *= scale_map[exc_dst[mask]]
        n_scaled += len(low)

    if n_scaled > 0:
        net._csr_dirty = True
    return n_scaled


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


def prune_competitive(
    net: DNG,
    weak_threshold: float = 0.003,
    removal_rate: float = 0.3,
    rng: np.random.Generator | None = None,
) -> int:
    """Competitive probabilistic pruning -- mimics neurotrophic competition.

    Each weak synapse has a probability of removal that depends on how
    much stronger the OTHER synapses on the same postsynaptic neuron are.

    removal_rate controls how fast pruning happens:
      0.3  = adult (aggressive, ~30% max chance per night)
      0.01 = child (~1% max, edge survives ~100 nights on average)
    """
    if rng is None:
        rng = np.random.default_rng()
    n = net._edge_count
    if n == 0:
        return 0

    dst = net._edge_dst[:n]
    w = net._edge_w[:n]
    abs_w = np.abs(w)

    total_in = np.bincount(dst, weights=abs_w, minlength=net.n_nodes)
    edge_count_per = np.bincount(dst, minlength=net.n_nodes)

    is_weak = abs_w < weak_threshold
    if not is_weak.any():
        return 0

    dst_weak = dst[is_weak]
    total_at_dst = total_in[dst_weak]
    count_at_dst = edge_count_per[dst_weak].astype(np.float64)

    mean_w_at_dst = np.where(count_at_dst > 0, total_at_dst / count_at_dst, 0.0)

    weakness = 1.0 - abs_w[is_weak] / np.maximum(weak_threshold, 1e-8)
    pressure = np.clip(mean_w_at_dst / 0.1, 0.0, 1.0)
    p_remove = removal_rate * weakness * pressure

    rolls = rng.random(len(p_remove))
    remove_weak = rolls < p_remove

    # Map back to full edge indices
    weak_indices = np.where(is_weak)[0]
    to_remove_idx = weak_indices[remove_weak]

    n_pruned = len(to_remove_idx)
    if n_pruned > 0:
        net._edge_w[to_remove_idx] = 0.0
        net.compact()
    return n_pruned


# ── Synaptogenesis ────────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class SynaptogenesisStats:
    n_created: int
    n_candidates: int
    n_valid: int        # passed structural filters (no self-loops etc)
    n_novel: int        # not already connected
    n_prob_pass: int    # passed the probability roll
    n_active: int       # neurons with r > 0.02 at call time
    n_silent: int       # non-IO neurons with r <= 0.02
    mean_r_active: float
    mean_prob: float    # mean probability for novel candidates
    reject_existing_frac: float  # fraction rejected by novelty filter


def synaptogenesis(
    net: DNG,
    growth_rate: float = 0.3,
    n_candidates: int = 5000,
    rng: np.random.Generator | None = None,
    return_stats: bool = False,
) -> "int | tuple[int, SynaptogenesisStats]":
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

    # Three pools of candidate pairs:
    #   1. Co-active: both neurons firing -> "fire together wire together"
    #   2. Recruitment: active neuron -> silent neuron (neurotrophic attraction)
    #      Active neurons release BDNF-like signals that attract growth from
    #      nearby silent neurons. This is how real brains wire up ALL neurons,
    #      not just the ones that are already firing.
    #   3. Exploratory: random pairs for global connectivity

    active_mask = net.r > 0.02
    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)

    silent_mask = (~active_mask) & (net.regions != sensory_idx) & (net.regions != motor_idx)
    silent_idx = np.where(silent_mask)[0]
    n_silent = len(silent_idx)

    if n_active < 2:
        return 0

    recruit_frac = 0.3 * net.ach  # 30% of candidates recruit silent neurons
    explore_frac = 0.1 * net.ach
    n_recruit = int(n_candidates * recruit_frac) if n_silent > 0 else 0
    n_explore = int(n_candidates * explore_frac)
    n_coactive = n_candidates - n_recruit - n_explore

    # Pool 1: co-active pairs
    n_cand_active = min(n_coactive, n_active * n_active)
    src_active = active_idx[rng.integers(0, n_active, size=n_cand_active)]
    dst_active = active_idx[rng.integers(0, n_active, size=n_cand_active)]

    # Pool 2: active -> silent (neurotrophic recruitment)
    if n_recruit > 0 and n_silent > 0:
        src_recruit = active_idx[rng.integers(0, n_active, size=n_recruit)]
        dst_recruit = silent_idx[rng.integers(0, n_silent, size=n_recruit)]
    else:
        src_recruit = np.array([], dtype=np.int64)
        dst_recruit = np.array([], dtype=np.int64)

    # Pool 3: exploratory random
    all_idx = np.arange(net.n_nodes)
    src_explore = all_idx[rng.integers(0, net.n_nodes, size=n_explore)]
    dst_explore = all_idx[rng.integers(0, net.n_nodes, size=n_explore)]

    src_samples = np.concatenate([src_active, src_recruit, src_explore])
    dst_samples = np.concatenate([dst_active, dst_recruit, dst_explore])

    # Filter: no self-loops, no edges TO sensory, no edges FROM motor
    valid = (
        (src_samples != dst_samples) &
        (net.regions[dst_samples] != sensory_idx) &
        (net.regions[src_samples] != motor_idx)
    )
    n_total_cands = len(src_samples)
    src_samples = src_samples[valid]
    dst_samples = dst_samples[valid]
    n_valid = len(src_samples)

    _empty_stats = SynaptogenesisStats(
        0, n_total_cands, n_valid, 0, 0, n_active, n_silent,
        float(net.r[active_idx].mean()) if n_active > 0 else 0.0,
        0.0, 0.0,
    )
    if n_valid == 0:
        return (0, _empty_stats) if return_stats else 0

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
    n_before_novel = len(src_samples)
    src_samples = src_samples[novel_mask]
    dst_samples = dst_samples[novel_mask]
    n_novel = len(src_samples)
    reject_existing_frac = 1.0 - (n_novel / max(1, n_before_novel))

    if n_novel == 0:
        _empty_stats.n_valid = n_valid
        _empty_stats.n_novel = 0
        _empty_stats.reject_existing_frac = reject_existing_frac
        return (0, _empty_stats) if return_stats else 0

    r_src = net.r[src_samples]
    r_dst = net.r[dst_samples]
    coact = r_src * r_dst
    recruit_boost = 0.1 * r_src * (r_dst < 0.02).astype(np.float64)
    baseline_p = 0.001 * net.ach
    prob = growth_rate * net.ach * (coact + recruit_boost) + baseline_p
    np.clip(prob, 0.0, 1.0, out=prob)
    mean_prob = float(prob.mean())
    rolls = rng.random(len(prob))
    winners = rolls < prob

    new_src = src_samples[winners]
    new_dst = dst_samples[winners]
    n_created = len(new_src)

    stats = SynaptogenesisStats(
        n_created=n_created,
        n_candidates=n_total_cands,
        n_valid=n_valid,
        n_novel=n_novel,
        n_prob_pass=n_created,
        n_active=n_active,
        n_silent=n_silent,
        mean_r_active=float(net.r[active_idx].mean()) if n_active > 0 else 0.0,
        mean_prob=mean_prob,
        reject_existing_frac=reject_existing_frac,
    )

    if n_created == 0:
        return (0, stats) if return_stats else 0

    new_coact = coact[winners]
    base_weight = 0.001 + 0.004 * (new_coact / (new_coact.max() + 1e-8))
    is_inhib = net.node_types[new_src] == _NTYPE_I
    weights = base_weight.copy()
    weights[is_inhib] *= -1.0

    net.add_edges_batch(new_src, new_dst, weights)
    return (n_created, stats) if return_stats else n_created
