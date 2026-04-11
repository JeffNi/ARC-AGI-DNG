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

from .graph import DNG, _NTYPE_I, layer_index


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

    dw = eta * (clamped_corr - free_corr)

    The correlation difference itself is the error signal — no DA gating.
    DA's role with CHL is to tag experiences for replay priority, not to
    gate the update. This mirrors cortical error-driven learning where the
    mismatch between predicted and actual activity drives synaptic change
    directly, while DA modulates consolidation priority.
    """
    n = net._edge_count
    if n == 0:
        return 0.0

    if abs(eta) < 1e-8:
        return 0.0

    fc = free_corr[:n] if len(free_corr) >= n else np.pad(free_corr, (0, n - len(free_corr)))
    cc = clamped_corr[:n] if len(clamped_corr) >= n else np.pad(clamped_corr, (0, n - len(clamped_corr)))
    diff = cc - fc
    dw = eta * diff

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


def eligibility_modulated_update_percell(
    net: DNG,
    da_per_node: np.ndarray,
    eta: float = 0.02,
    w_max: float = 5.0,
) -> float:
    """
    Per-cell reward: each edge gets DA from its destination node.

    Topographic DA analog — striatal patches receive different dopamine
    based on which cortical motor area projects to them. Correct motor
    cells get positive DA, wrong cells get negative DA, non-motor nodes
    get zero (their learning happens via novelty DA in the main loop).

    Same mechanics as eligibility_modulated_update (consolidation
    protection, Dale's law, clipping) but with a per-edge DA vector.
    """
    n = net._edge_count
    if n == 0:
        return 0.0

    elig = net._edge_eligibility[:n]
    if not np.any(elig > 1e-6):
        return 0.0

    dst = net._edge_dst[:n]
    da_per_edge = da_per_node[dst]

    active = np.abs(da_per_edge) > 1e-8
    if not active.any():
        return 0.0

    dw = np.zeros(n, dtype=np.float64)
    dw[active] = eta * da_per_edge[active] * elig[active]

    cons = net._edge_consolidation[:n]
    if cons.any():
        plasticity_scale = 1.0 / (1.0 + cons)
        dw *= plasticity_scale

    # Copy edges get re-pinned every step — don't waste reward signal on them
    copy_edges = cons >= 20.0
    dw[copy_edges] = 0.0

    w = net._edge_w[:n]

    neg_mask = w < 0
    dw[neg_mask] = -dw[neg_mask]

    new_w = w + dw

    pos_mask = w > 0
    new_w[pos_mask] = np.clip(new_w[pos_mask], 1e-6, w_max)
    new_w[neg_mask] = np.clip(new_w[neg_mask], -w_max, -1e-6)

    change = np.abs(new_w - w)
    net._edge_w[:n] = new_w
    net._edge_tag[:n] += change
    net._csr_dirty = True

    net._edge_eligibility[:n] *= 0.5

    return float(np.mean(change[active]))


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
    ema_rate: np.ndarray | None = None,
    target_rate: float = 0.15,
) -> None:
    """Selective downscaling (SHY hypothesis).

    Synapses that were recently active (high co-activity or tagged)
    are protected. Inactive synapses get downscaled. Consolidated
    (instinct) edges are exempt.

    Uses ema_rate co-activity when available (works without Teacher
    rewards), falls back to tag-only gating for backward compat.
    """
    n = net._edge_count
    if n == 0:
        return
    tags = net._edge_tag[:n]
    cons = net._edge_consolidation[:n]

    # Protected = tagged OR consolidated OR actively carrying signal
    protected = (tags >= tag_threshold) | (cons >= 1.0)

    if ema_rate is not None:
        src_rate = ema_rate[net._edge_src[:n]]
        dst_rate = ema_rate[net._edge_dst[:n]]
        co_activity = np.sqrt(src_rate * dst_rate) / max(target_rate, 1e-6)
        # Scale downscaling by inactivity: active edges get less downscaling
        # co_activity ~1.0 → scale ≈ 1.0 (no downscale)
        # co_activity ~0.0 → scale = downscale (full downscale)
        scale = np.where(
            protected, 1.0,
            downscale + (1.0 - downscale) * np.minimum(co_activity, 1.0),
        )
        w = net._edge_w[:n]
        new_w = w * scale
        pos = w > 0
        neg = w < 0
        new_w[pos] = np.maximum(new_w[pos], 1e-7)
        new_w[neg] = np.minimum(new_w[neg], -1e-7)
        net._edge_w[:n] = new_w
    else:
        untagged = ~protected
        net._edge_w[:n][untagged] *= downscale

    net._edge_tag[:n] = 0.0
    net._csr_dirty = True


def update_edge_health(
    net: DNG,
    decay_rate: float = 0.01,
    ema_rate: np.ndarray | None = None,
    target_rate: float = 0.15,
    layer_decay_scales: np.ndarray | None = None,
) -> None:
    """
    Update per-synapse health each sleep cycle. Biological model:
    inactive synapses lose structural proteins and shrink; active
    ones get maintained by activity-dependent trophic signaling.

    Activity is measured via neuron-level firing rate EMA (ema_rate)
    rather than per-edge eligibility traces, which decay too fast
    (0.85/step) to survive until sleep. A synapse is "active" if
    both its pre and post neurons have been firing.

    layer_decay_scales: optional per-edge multiplier on decay_rate.
    Higher cortical layers get lower multipliers during early
    development (L3/PFC matures last — Huttenlocher 1997).

    Falls back to eligibility + tag if ema_rate is not provided
    (backward compat for non-Brain callers).
    """
    n = net._edge_count
    if n == 0:
        return

    cons = net._edge_consolidation[:n]
    health = net._edge_health[:n]

    if ema_rate is not None:
        src_rate = ema_rate[net._edge_src[:n]]
        dst_rate = ema_rate[net._edge_dst[:n]]
        co_activity = np.sqrt(src_rate * dst_rate) / max(target_rate, 1e-6)
        np.minimum(co_activity, 1.0, out=co_activity)
        activity_factor = co_activity
    else:
        elig = net._edge_eligibility[:n]
        tag = net._edge_tag[:n]
        activity_factor = np.minimum(1.0, elig + tag)

    if layer_decay_scales is not None:
        health -= (decay_rate * layer_decay_scales) * (1.0 - activity_factor)
    else:
        health -= decay_rate * (1.0 - activity_factor)

    boost_mask = activity_factor > 0.3
    health[boost_mask] += 0.05 * activity_factor[boost_mask]
    np.minimum(health, 1.0, out=health)

    cons_floor = cons / 20.0
    np.maximum(health, cons_floor, out=health)

    net._edge_health[:n] = health


def prune_sustained(net: DNG, **kwargs) -> int:
    """
    Health-based pruning: remove edges whose health has decayed to zero.
    Each synapse dies on its own timeline based on its individual activity
    history — no synchronized counters, no cliff-edge mass extinction.
    """
    n = net._edge_count
    if n == 0:
        return 0

    to_remove = net._edge_health[:n] <= 0.0
    n_pruned = int(np.sum(to_remove))
    if n_pruned > 0:
        net._edge_w[:n][to_remove] = 0.0
        net.compact()
    return n_pruned


# ── Synaptogenesis ────────────────────────────────────────────────

def synaptogenesis(
    net: DNG,
    growth_rate: float = 0.3,
    n_candidates: int = 5_000,
    activity_ema: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    allow_motor_target: bool = True,
    layer_demand_boost: dict[int, float] | None = None,
) -> int:
    """
    Co-activity based synaptogenesis: fire together, wire together.

    The same universal rule produces different connectivity patterns in
    different pathways because the *input statistics* differ:
    - Sensory->internal: spatially correlated inputs -> topographic maps
    - Internal->internal: co-represented concepts -> associative connections
    - Internal->motor: task-relevant mappings -> action circuits

    Growth is gated by ACh (high during development, low in adulthood).
    New synapses start as "silent synapses" — structurally present but
    functionally very weak, requiring competitive Hebbian LTP to mature.

    Biological basis:
    - Hebb (1949): correlated activity drives synapse formation
    - Bhatt et al (2009): LTP induces new dendritic spine formation
    - Kwon & Bhatt (2022, Nat Neurosci): new spines form near potentiated
      spines, connecting to previously unrepresented axons

    At our scale (~4000 neurons), we sample co-active pairs directly rather
    than modeling filopodia spatial search — the computational result (co-active
    neurons get connected) is identical.
    """
    if rng is None:
        rng = np.random.default_rng()
    if growth_rate <= 0 or net.ach < 0.01:
        return 0

    from .graph import Region
    region_list = list(Region)
    sensory_idx = region_list.index(Region.SENSORY)
    motor_idx = region_list.index(Region.MOTOR)
    l1_idx = region_list.index(Region.LOCAL_DETECT)

    # Use EMA rates for stable activity signal, fall back to instantaneous
    activity = activity_ema if activity_ema is not None else net.r

    active_mask = activity > 0.02
    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)

    if n_active < 2:
        return 0

    # Co-active pairs (main) + small exploratory fraction (ACh-gated)
    explore_frac = 0.2 * net.ach
    n_explore = int(n_candidates * explore_frac)
    n_coactive = n_candidates - n_explore

    n_cand_active = min(n_coactive, n_active * n_active)
    src_active = active_idx[rng.integers(0, n_active, size=n_cand_active)]
    dst_active = active_idx[rng.integers(0, n_active, size=n_cand_active)]

    src_explore = rng.integers(0, net.n_nodes, size=n_explore)
    dst_explore = rng.integers(0, net.n_nodes, size=n_explore)

    src_all = np.concatenate([src_active, src_explore])
    dst_all = np.concatenate([dst_active, dst_explore])

    # Biological constraints:
    #  - no self-loops
    #  - no edges TO sensory
    #  - no edges FROM motor
    #  - sensory can only project to L1 (thalamocortical targeting —
    #    axon guidance molecules restrict thalamic afferents to layer 4
    #    of primary sensory cortex, not higher areas)
    src_is_sensory = net.regions[src_all] == sensory_idx
    dst_is_l1 = net.regions[dst_all] == l1_idx
    dst_is_motor = net.regions[dst_all] == motor_idx
    valid = (
        (src_all != dst_all)
        & (net.regions[dst_all] != sensory_idx)
        & (net.regions[src_all] != motor_idx)
        & (~src_is_sensory | dst_is_l1)
        & (allow_motor_target | ~dst_is_motor)
    )
    src_all = src_all[valid]
    dst_all = dst_all[valid]

    if len(src_all) == 0:
        return 0

    # Deduplicate against existing edges (vectorized)
    n_nodes = net.n_nodes
    n_existing = net._edge_count
    edge_exists = np.zeros(n_nodes * n_nodes, dtype=np.bool_)
    existing_hash = (
        net._edge_src[:n_existing].astype(np.int64) * n_nodes
        + net._edge_dst[:n_existing].astype(np.int64)
    )
    edge_exists[existing_hash] = True
    cand_hash = src_all.astype(np.int64) * n_nodes + dst_all.astype(np.int64)
    novel = ~edge_exists[cand_hash]
    src_all = src_all[novel]
    dst_all = dst_all[novel]

    if len(src_all) == 0:
        return 0

    # Deduplicate within batch
    cand_hash = src_all.astype(np.int64) * n_nodes + dst_all.astype(np.int64)
    _, unique_idx = np.unique(cand_hash, return_index=True)
    src_all = src_all[unique_idx]
    dst_all = dst_all[unique_idx]

    # Acceptance: co-activation driven + neurotrophin-gated baseline.
    # During infancy (high growth_rate), baseline is substantial — models
    # exuberant, activity-independent growth driven by molecular cues.
    # During adulthood (low growth_rate), baseline shrinks to near zero.
    coact = activity[src_all] * activity[dst_all]
    baseline_p = 0.005 * growth_rate * net.ach
    prob = growth_rate * net.ach * coact + baseline_p

    # Layer-distance decay: connections between distant cortical layers
    # are physically harder to form (longer axons, white matter tracts).
    # Same layer: 1.0x, adjacent: 0.3x, skip: 0.05x
    _LAYER_DISTANCE_SCALE = {0: 1.0, 1: 0.3, 2: 0.05}
    src_layers = np.array([layer_index(int(r)) for r in net.regions[src_all]])
    dst_layers = np.array([layer_index(int(r)) for r in net.regions[dst_all]])
    both_cortical = (src_layers >= 0) & (dst_layers >= 0)
    layer_dist = np.abs(src_layers - dst_layers)
    distance_scale = np.ones(len(prob))
    for dist, scale in _LAYER_DISTANCE_SCALE.items():
        distance_scale[both_cortical & (layer_dist == dist)] = scale
    distance_scale[both_cortical & (layer_dist > 2)] = 0.01
    prob *= distance_scale

    # Layer-aware demand boost: higher cortical layers get preferential
    # growth during development (trophic factors from active L1 drive
    # L2/L3 dendrite elaboration — Bhatt et al 2009).
    if layer_demand_boost:
        for layer_idx, boost in layer_demand_boost.items():
            if boost != 1.0:
                prob[dst_layers == layer_idx] *= boost

    np.clip(prob, 0.0, 1.0, out=prob)
    winners = rng.random(len(prob)) < prob

    new_src = src_all[winners]
    new_dst = dst_all[winners]

    if len(new_src) == 0:
        return 0

    # Silent synapses: weight scaled by co-activation strength
    new_coact = coact[winners]
    base_weight = 0.001 + 0.004 * (new_coact / (new_coact.max() + 1e-8))
    is_inhib = net.node_types[new_src] == _NTYPE_I
    weights = base_weight.copy()
    weights[is_inhib] *= -1.0

    net.add_edges_batch(new_src, new_dst, weights)
    return len(new_src)


# ── Lateral decorrelation ────────────────────────────────────────

def lateral_decorrelation(
    net: DNG,
    layer_neurons: np.ndarray,
    eta: float = 0.005,
    sim_threshold: float = 0.5,
) -> int:
    """
    Anti-Hebbian lateral decorrelation for a cortical layer.

    Biological basis: PV+ basket cell interneurons strengthen inhibitory
    connections between co-active excitatory neurons, forcing them to
    develop different feature selectivities. This is the primary cortical
    mechanism preventing representational collapse (mode collapse) in
    topographic maps.

    For neurons that won WTA (r > 0.01), compute cosine similarity of
    incoming excitatory weight vectors. Apply repulsive updates to
    similar pairs, pushing their weight vectors apart on the hypersphere.

    References:
      - Földiák (1990): anti-Hebbian lateral connections for decorrelation
      - Rubner & Schulten (1990): decorrelation via inhibitory feedback
      - King et al (2013): PV+ interneurons shape feature selectivity in V1
    """
    ne = net._edge_count
    if ne == 0:
        return 0

    active_mask = net.r[layer_neurons] > 0.01
    active = layer_neurons[active_mask]
    n_active = len(active)
    if n_active < 2:
        return 0

    edge_src = net._edge_src[:ne]
    edge_dst = net._edge_dst[:ne]
    edge_w = net._edge_w[:ne]

    active_set = np.zeros(net.n_nodes, dtype=bool)
    active_set[active] = True
    active_map = np.full(net.n_nodes, -1, dtype=np.int64)
    active_map[active] = np.arange(n_active, dtype=np.int64)

    # Excitatory edges incoming to active neurons
    emask = active_set[edge_dst] & (edge_w > 0)
    e_indices = np.where(emask)[0]
    if len(e_indices) < n_active:
        return 0

    _DBG = getattr(lateral_decorrelation, '_debug', False)
    if _DBG:
        print(f"      [decorr] n_active={n_active}, e_indices={len(e_indices)}", flush=True)

    e_rows = active_map[edge_dst[e_indices]]
    e_cols = edge_src[e_indices]
    e_vals = edge_w[e_indices].copy()

    # Map source nodes to dense column indices
    unique_src = np.unique(e_cols)
    src_map = np.full(net.n_nodes, -1, dtype=np.int64)
    src_map[unique_src] = np.arange(len(unique_src), dtype=np.int64)
    n_src = len(unique_src)

    # Dense weight matrix (n_active x n_unique_src)
    W = np.zeros((n_active, n_src), dtype=np.float64)
    W[e_rows, src_map[e_cols]] = e_vals

    # Cosine similarity
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    W_hat = W / norms

    sim = W_hat @ W_hat.T
    np.fill_diagonal(sim, 0.0)

    # Repel only pairs above threshold
    repel = np.where(sim > sim_threshold, sim, 0.0)
    n_repelled = int(np.count_nonzero(repel))
    if _DBG:
        sim_upper = sim[np.triu_indices(n_active, k=1)]
        print(f"      [decorr] sim: mean={sim_upper.mean():.3f} max={sim_upper.max():.3f} "
              f">thresh={int((sim_upper > sim_threshold).sum())} n_repelled={n_repelled}", flush=True)
    if n_repelled == 0:
        return 0

    # dW_i = -eta * sum_j(sim[i,j] * W_hat_j)
    dW = -eta * (repel @ W_hat)
    dW_mag = float(np.abs(dW).mean())
    if dW_mag < 1e-10:
        return 0

    # Write back to edges
    edge_w[e_indices] += dW[e_rows, src_map[e_cols]]
    np.clip(edge_w[e_indices], 1e-7, None, out=edge_w[e_indices])

    net._csr_dirty = True
    return n_repelled
