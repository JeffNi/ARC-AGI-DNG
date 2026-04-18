"""
Template generator: genome -> DNG.

Autonomous mushroom body architecture:
  SENSORY (one-hot color: 10 per cell)
    -> L3 / Kenyon Cells (sparse random expansion, ~10 inputs each)
      -> POSITION (9 neurons, WTA)  — which cell to target
      -> DONE (1 neuron)            — submit signal
      -> MEMORY (10 groups of 10, one per color — MBON compartments)
  MEMORY group k -> ACTION(k)  [fixed strong wiring]
  L3 -> MEMORY               [PLASTIC, depression-based learning site]

Learning: DA-modulated synaptic depression at KC→MBON (L3→MEMORY).
Wrong color commit → depress L3→MEMORY for wrong color's group.
Correct color emerges by elimination (undepressed groups remain strong).

Instinct circuits (hardwired, NOT plastic):
  - Spatial instinct: SENSORY(cell_i) -> POSITION(cell_i)
  - Color echo instinct: position-gated at runtime (tie-breaker for naive brain)
"""

from __future__ import annotations

import numpy as np

from .encoding import NUM_COLORS
from .perception.encoder import sensory_size, FEATURES_PER_CELL
from .genome import Genome
from .graph import DNG, NodeType, Region, DEFAULT_LEAK, _NTYPE_E, _NTYPE_I


N_POSITION = 9   # one per grid cell (3x3)
N_ACTION = 11     # 10 colors + 1 no-op
N_DONE = 1
N_COMMIT = 1      # basal ganglia go/no-go gate for commits
N_MOTOR_NEW = N_POSITION + N_ACTION + N_DONE + N_COMMIT


def create_dng(
    genome: Genome,
    grid_h: int,
    grid_w: int,
    rng: np.random.Generator | None = None,
) -> DNG:
    if rng is None:
        rng = np.random.default_rng()

    n_cells = grid_h * grid_w
    n_sensory = sensory_size(grid_h, grid_w)
    n_int = genome.n_internal
    n_mem = genome.n_memory
    n_gate = genome.n_gate
    n_gaze = genome.n_gaze

    n_l1 = int(n_int * genome.frac_layer1)
    n_l2 = int(n_int * genome.frac_layer2)
    n_l3 = n_int - n_l1 - n_l2

    n_position = n_cells  # one POSITION neuron per grid cell
    n_action = NUM_COLORS + 1  # 10 colors + 1 no-op
    n_done = 1
    n_commit = 1  # basal ganglia go/no-go gate
    n_motor_total = n_position + n_action + n_done + n_commit

    n_total = n_sensory + n_int + n_mem + n_gate + n_gaze + n_motor_total

    sensory_start = 0
    l1_start = n_sensory
    l2_start = n_sensory + n_l1
    l3_start = n_sensory + n_l1 + n_l2
    memory_start = n_sensory + n_int
    gate_start = n_sensory + n_int + n_mem
    gaze_start = n_sensory + n_int + n_mem + n_gate
    position_start = n_sensory + n_int + n_mem + n_gate + n_gaze
    action_start = position_start + n_position
    done_start = action_start + n_action
    commit_start = done_start + n_done

    sensory = np.arange(sensory_start, sensory_start + n_sensory)
    layer1 = np.arange(l1_start, l1_start + n_l1)
    layer2 = np.arange(l2_start, l2_start + n_l2)
    layer3 = np.arange(l3_start, l3_start + n_l3)
    memory = np.arange(memory_start, memory_start + n_mem)
    gate = np.arange(gate_start, gate_start + n_gate)
    gaze = np.arange(gaze_start, gaze_start + n_gaze)
    position = np.arange(position_start, position_start + n_position)
    action = np.arange(action_start, action_start + n_action)
    done = np.arange(done_start, done_start + n_done)
    commit = np.arange(commit_start, commit_start + n_commit)
    motor_all = np.concatenate([position, action, done, commit])

    # Node types: all excitatory
    node_types = np.full(n_total, _NTYPE_E, dtype=int)

    # Regions
    _reg = list(Region)
    regions = np.zeros(n_total, dtype=int)
    regions[sensory] = _reg.index(Region.SENSORY)
    if len(layer1) > 0:
        regions[layer1] = _reg.index(Region.LOCAL_DETECT)
    if len(layer2) > 0:
        regions[layer2] = _reg.index(Region.MID_LEVEL)
    regions[layer3] = _reg.index(Region.ABSTRACT)
    if len(memory) > 0:
        regions[memory] = _reg.index(Region.MEMORY)
    if len(gate) > 0:
        regions[gate] = _reg.index(Region.GATE)
    if len(gaze) > 0:
        regions[gaze] = _reg.index(Region.GAZE)
    regions[position] = _reg.index(Region.POSITION)
    regions[action] = _reg.index(Region.ACTION)
    regions[done] = _reg.index(Region.DONE)
    regions[commit] = _reg.index(Region.COMMIT)

    # Leak rates (from genome)
    _ntype_list = list(NodeType)
    leak_rates = np.array([DEFAULT_LEAK[_ntype_list[t]] for t in node_types])
    leak_rates[layer3] = genome.leak_l3
    if len(memory) > 0:
        leak_rates[memory] = genome.leak_memory
    leak_rates[motor_all] = genome.leak_motor

    max_rate = np.full(n_total, genome.max_rate_E)
    # Sensory neurons are input transducers — output proportional to signal.
    max_rate[sensory] = 20.0
    # L3/KCs need rate headroom so weight variance creates distinguishable
    # activation levels for the APL WTA.  Capping at 1.0 flattens all KCs
    # to the same rate and the WTA zeros everyone.
    max_rate[layer3] = 10.0
    adapt_rate = np.full(n_total, genome.adapt_rate_E)
    adapt_rate[sensory] = 0.0  # no adaptation for input transducers

    # Column IDs: sensory cells get column assignments for diagnostics
    column_ids = np.full(n_total, -1, dtype=np.int32)
    for cell in range(n_cells):
        s0 = sensory_start + cell * FEATURES_PER_CELL
        column_ids[s0:s0 + FEATURES_PER_CELL] = cell

    net = DNG(
        n_nodes=n_total,
        node_types=node_types,
        regions=regions,
        excitability=np.ones(n_total),
        leak_rates=leak_rates,
        max_rate=max_rate,
        adapt_rate=adapt_rate,
        input_nodes=sensory,
        output_nodes=motor_all,
        memory_nodes=memory,
        gate_nodes=gate,
        gaze_nodes=gaze,
        column_ids=column_ids,
        n_columns=0,
        max_h=grid_h,
        max_w=grid_w,
        f_rate=genome.f_rate,
        f_decay=genome.f_decay,
        f_max=genome.f_max,
    )

    if len(memory) > 0:
        net.threshold[memory] = genome.threshold_memory

    if len(commit) > 0:
        net.threshold[commit] = genome.threshold_commit
        net.excitability[commit] = genome.excitability_commit
        net.leak_rates[commit] = genome.leak_commit

    ws = genome.weight_scale

    # ═══════════════════════════════════════════════════════════════════
    # SENSORY -> L3/KC: cell-local sparse projection (mushroom body wiring)
    #
    # Biology: each KC's dendritic claw samples from a LOCAL neighborhood
    # in the calyx.  We partition KCs into cell-assigned groups. Each KC
    # draws most inputs from its assigned cell, with a few from elsewhere.
    # This creates the activation variance the APL WTA needs.
    # ═══════════════════════════════════════════════════════════════════
    _wire_kc_local(
        net, sensory, layer3, n_cells,
        local_fan=genome.kc_local_fan,
        global_fan=genome.kc_global_fan,
        weight_scale=ws * genome.w_scale_sensory_to_l3,
        rng=rng,
    )

    # Store KC cell assignments so engine can apply position-specific gain
    kc_cell_ids = np.full(n_l3, -1, dtype=np.int32)
    n_kc_per_cell = n_l3 // n_cells
    _remainder = n_l3 % n_cells
    _ki = 0
    for _c in range(n_cells):
        _nk = n_kc_per_cell + (1 if _c < _remainder else 0)
        kc_cell_ids[_ki : _ki + _nk] = _c
        _ki += _nk
    net._kc_cell_ids = kc_cell_ids

    # ═══════════════════════════════════════════════════════════════════
    # L3/KC -> POSITION: learned pathway (KC -> MBON analog for spatial)
    # ═══════════════════════════════════════════════════════════════════
    kc_pos_fan = max(1, min(int(0.15 * n_l3), n_l3, genome.max_fan_in))
    _fan_in_edges(net, layer3, position, kc_pos_fan, ws * genome.w_scale_l3_to_position, rng)

    # (L3/KC -> ACTION removed: KCs don't project to motor in biology.
    #  Color selection is handled by the position-gated echo instinct.
    #  Learning happens at KC->MBON (L3->MEMORY), not KC->motor.)

    # ═══════════════════════════════════════════════════════════════════
    # L3/KC -> DONE: learned pathway
    # ═══════════════════════════════════════════════════════════════════
    kc_done_fan = max(1, min(int(0.10 * n_l3), n_l3, genome.max_fan_in))
    _fan_in_edges(net, layer3, done, kc_done_fan, ws * genome.w_scale_l3_to_done, rng)

    # ═══════════════════════════════════════════════════════════════════
    # MEMORY group assignments: 10 groups of 10, one per color.
    # Group k drives ACTION(k). This is the MBON compartment structure.
    # ═══════════════════════════════════════════════════════════════════
    n_groups = min(NUM_COLORS, n_mem)
    group_size = n_mem // n_groups if n_groups > 0 else 0
    memory_group_ids = np.full(n_mem, -1, dtype=np.int32)
    for g in range(n_groups):
        g_start = g * group_size
        g_end = g_start + group_size
        memory_group_ids[g_start:g_end] = g
    # Leftover neurons (if n_mem not divisible) go to last group
    if n_mem > n_groups * group_size:
        memory_group_ids[n_groups * group_size:] = n_groups - 1

    # Store on the DNG so engine can reference group assignments
    net._memory_group_ids = memory_group_ids
    net._memory_start = memory_start
    net._n_memory_groups = n_groups
    net._memory_group_size = group_size

    # ═══════════════════════════════════════════════════════════════════
    # L3 -> MEMORY: PLASTIC, depression-based learning site (KC → MBON)
    # Strong initial weights — depression weakens wrong-color groups.
    # ~15 L3 inputs per MEMORY neuron.
    # ═══════════════════════════════════════════════════════════════════
    if n_mem > 0:
        mem_fan_in = min(50, n_l3)
        l3_mem_start = net._edge_count
        _fan_in_edges(net, layer3, memory, mem_fan_in, ws * genome.w_scale_l3_to_memory, rng,
                      uniform=True)
        l3_mem_end = net._edge_count
        # Store initial weights for spontaneous recovery during sleep
        net._l3_mem_initial_w = net._edge_w[l3_mem_start:l3_mem_end].copy()
        net._l3_mem_edge_slice = (l3_mem_start, l3_mem_end)
        # NOT consolidated — these are the primary learning site

    # ═══════════════════════════════════════════════════════════════════
    # MEMORY -> POSITION: learned plastic
    # ═══════════════════════════════════════════════════════════════════
    if n_mem > 0:
        mem_pos_fan = max(1, min(int(0.20 * n_mem), n_mem))
        _fan_in_edges(net, memory, position, mem_pos_fan, ws * genome.w_scale_memory_to_position, rng)

    # MEMORY→ACTION graph edges REMOVED: the analytical MBON readout in
    # _motor_wta computes the same signal without graph-dynamics lag.
    # Graph edges were double-counting and adding noise.

    # ═══════════════════════════════════════════════════════════════════
    # MEMORY -> DONE: learned plastic
    # ═══════════════════════════════════════════════════════════════════
    if n_mem > 0:
        mem_done_fan = max(1, min(int(0.15 * n_mem), n_mem))
        _fan_in_edges(net, memory, done, mem_done_fan, ws * genome.w_scale_memory_to_done, rng)

    # POSITION→ACTION graph edges REMOVED: random initial weights injected
    # noise that competed with the echo. Action selection is now purely
    # echo + MBON readout + WTA noise (all computed in _motor_wta).

    # ═══════════════════════════════════════════════════════════════════
    # COMMIT neuron: basal ganglia go/no-go gate
    # Receives context (L3, MEMORY) + motor plan (POSITION, ACTION) to
    # learn WHEN to commit. Fires when evidence is sufficient.
    # ═══════════════════════════════════════════════════════════════════
    # L3 -> COMMIT (sensory context)
    kc_commit_fan = max(1, min(int(0.10 * n_l3), n_l3, genome.max_fan_in))
    _fan_in_edges(net, layer3, commit, kc_commit_fan, ws * genome.w_scale_l3_to_commit, rng)
    if n_mem > 0:
        mem_commit_fan = max(1, min(int(0.15 * n_mem), n_mem))
        _fan_in_edges(net, memory, commit, mem_commit_fan, ws * genome.w_scale_memory_to_commit, rng)
    pos_commit_w = _init_weights(rng, ws * genome.w_scale_memory_to_commit, n_position)
    net.add_edges_batch(position, np.full(n_position, commit[0], dtype=np.int32),
                        pos_commit_w)
    act_commit_w = _init_weights(rng, ws * genome.w_scale_memory_to_commit, n_action)
    net.add_edges_batch(action, np.full(n_action, commit[0], dtype=np.int32),
                        act_commit_w)

    # ═══════════════════════════════════════════════════════════════════
    # INSTINCT: Spatial instinct — SENSORY(cell_i) -> POSITION(cell_i)
    # Retinotopic map: all features of cell i drive POSITION neuron i.
    # Hardwired, NOT plastic. Consolidated.
    # ═══════════════════════════════════════════════════════════════════
    spatial_src = []
    spatial_dst = []
    for cell in range(n_cells):
        for feat in range(FEATURES_PER_CELL):
            s_idx = sensory_start + cell * FEATURES_PER_CELL + feat
            p_idx = position_start + cell
            spatial_src.append(s_idx)
            spatial_dst.append(p_idx)
    spatial_src = np.array(spatial_src, dtype=np.int32)
    spatial_dst = np.array(spatial_dst, dtype=np.int32)
    spatial_w = np.full(len(spatial_src), genome.spatial_instinct_weight)
    edge_start = net._edge_count
    net.add_edges_batch(spatial_src, spatial_dst, spatial_w)
    edge_end = net._edge_count
    net._edge_consolidation[edge_start:edge_end] = 20.0

    # Color echo instinct is now position-gated at runtime in engine._motor_wta().
    # No static SENSORY->ACTION echo edges needed.

    # ═══════════════════════════════════════════════════════════════════
    # SENSORY -> GAZE: orienting reflex (stimulus-driven saccades)
    # ═══════════════════════════════════════════════════════════════════
    if n_gaze > 0:
        gaze_fan_in = min(15, n_sensory)
        _fan_in_edges(net, sensory, gaze, gaze_fan_in, ws * genome.w_scale_sensory_to_gaze, rng)

    # Store motor pool info on the DNG for engine use
    net._position_start = position_start
    net._action_start = action_start
    net._done_start = done_start
    net._commit_start = commit_start
    net._n_position = n_position
    net._n_action = n_action
    net._n_done = n_done
    net._n_commit = n_commit

    return net


def _init_weights(
    rng: np.random.Generator, scale: float, n: int,
) -> np.ndarray:
    w = rng.normal(loc=scale, scale=scale * 0.35, size=n)
    np.maximum(w, scale * 0.01, out=w)
    return w


def _fan_in(density: float, source_size: int, cap: int = 9999) -> int:
    return max(1, min(int(density * source_size), source_size, cap))


def _fan_in_edges(
    net: DNG,
    src_nodes: np.ndarray,
    dst_nodes: np.ndarray,
    fan_in: int,
    weight_scale: float,
    rng: np.random.Generator,
    uniform: bool = False,
) -> None:
    """Each dst node gets `fan_in` random connections from src nodes.

    If uniform=True, all weights are set to exactly weight_scale (no noise).
    Used for L3->MEMORY where random variance creates systematic group biases
    that interfere with the MBON readout.
    """
    n_src = len(src_nodes)
    n_dst = len(dst_nodes)
    if n_src == 0 or n_dst == 0 or fan_in <= 0:
        return

    fan_in = min(fan_in, n_src)
    if fan_in >= n_src:
        fan_in = n_src - 1
    if fan_in <= 0:
        return
    max_chunk = max(1, 5_000_000 // n_src)
    all_src_parts, all_dst_parts = [], []

    for start in range(0, n_dst, max_chunk):
        end = min(start + max_chunk, n_dst)
        chunk_dst = dst_nodes[start:end]
        chunk_n = len(chunk_dst)
        rand_matrix = rng.random((chunk_n, n_src))
        chosen_idx = np.argpartition(rand_matrix, fan_in, axis=1)[:, :fan_in]
        all_src_parts.append(src_nodes[chosen_idx.ravel()])
        all_dst_parts.append(np.repeat(chunk_dst, fan_in))

    all_src = np.concatenate(all_src_parts)
    all_dst = np.concatenate(all_dst_parts)
    valid = all_src != all_dst
    sel_src = all_src[valid]
    sel_dst = all_dst[valid]

    if uniform:
        magnitudes = np.full(len(sel_src), weight_scale)
    else:
        magnitudes = _init_weights(rng, weight_scale, len(sel_src))
    signs = np.where(net._mask_I[sel_src], -1.0, 1.0)
    sel_w = signs * magnitudes
    net.add_edges_batch(sel_src, sel_dst, sel_w)


def _wire_kc_local(
    net: DNG,
    sensory: np.ndarray,
    layer3: np.ndarray,
    n_cells: int,
    local_fan: int,
    global_fan: int,
    weight_scale: float,
    rng: np.random.Generator,
) -> None:
    """Cell-local KC wiring: each KC is assigned to one grid cell and draws
    most inputs from that cell's sensory features, with a few from elsewhere.

    Mimics insect MB calyx topography where each KC's dendritic claw
    samples from a restricted neighborhood of projection neuron boutons.
    """
    n_sensory = len(sensory)
    n_l3 = len(layer3)
    if n_sensory == 0 or n_l3 == 0 or n_cells == 0:
        return

    features_per_cell = n_sensory // n_cells
    local_fan = min(local_fan, features_per_cell)
    total_fan = local_fan + global_fan

    n_kc_per_cell = n_l3 // n_cells
    remainder = n_l3 % n_cells

    all_src, all_dst = [], []
    kc_idx = 0

    for cell in range(n_cells):
        n_kc = n_kc_per_cell + (1 if cell < remainder else 0)
        if n_kc == 0:
            continue

        cell_kcs = layer3[kc_idx : kc_idx + n_kc]
        kc_idx += n_kc

        cell_start = cell * features_per_cell
        cell_end = cell_start + features_per_cell
        local_pool = sensory[cell_start:cell_end]

        other_sensory = np.concatenate([
            sensory[:cell_start], sensory[cell_end:]
        ]) if (cell_start > 0 or cell_end < n_sensory) else np.array([], dtype=sensory.dtype)

        actual_global = min(global_fan, len(other_sensory))

        for kc in cell_kcs:
            local_chosen = rng.choice(local_pool, size=local_fan, replace=False)

            if actual_global > 0:
                global_chosen = rng.choice(other_sensory, size=actual_global, replace=False)
                src = np.concatenate([local_chosen, global_chosen])
            else:
                src = local_chosen

            all_src.append(src)
            all_dst.append(np.full(len(src), kc, dtype=np.int32))

    if not all_src:
        return

    all_src = np.concatenate(all_src)
    all_dst = np.concatenate(all_dst)
    magnitudes = _init_weights(rng, weight_scale, len(all_src))
    signs = np.where(net._mask_I[all_src], -1.0, 1.0)
    net.add_edges_batch(all_src, all_dst, signs * magnitudes)
