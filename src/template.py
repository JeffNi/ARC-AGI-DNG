"""
Template generator: genome -> DNG.

Hierarchical cortical layout:
  SENSORY (perception features: 30 per cell + 28 global)
    -> LAYER 1 / LOCAL_DETECT (sparse RF, local feature combination)
      -> LAYER 2 / MID_LEVEL (wider RF, regional integration)
        -> LAYER 3 / ABSTRACT (non-topographic, reasoning)
          -> MOTOR (10 colors per cell, one-hot)
          <-> MEMORY (self-sustaining, episodic)

Information flows bottom-up through the hierarchy. Feedback connections
(top-down) exist but are weak at birth. Lateral connections within each
layer enable local competition. Only Layer 3 connects to Motor (learned
output path). The instinct copy pathway bypasses the hierarchy entirely.

Instinct circuits:
  - Copy pathway: sensory color nodes -> motor color nodes (identity mapping)
  - Memory self-connections: bistable attractor dynamics
  - Memory <-> Layer 3: write and recall paths
"""

from __future__ import annotations

import numpy as np

from .encoding import NUM_COLORS
from .perception.encoder import sensory_size, FEATURES_PER_CELL, GLOBAL_FEATURES
from .genome import Genome
from .graph import DNG, NodeType, Region, DEFAULT_LEAK, _NTYPE_E, _NTYPE_I


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
    n_motor = n_cells * NUM_COLORS
    n_int = genome.n_internal
    n_mem = genome.n_memory

    # Split internal into 3 cortical layers
    n_l1 = int(n_int * genome.frac_layer1)
    n_l2 = int(n_int * genome.frac_layer2)
    n_l3 = n_int - n_l1 - n_l2

    n_total = n_sensory + n_int + n_mem + n_motor

    sensory_start = 0
    l1_start = n_sensory
    l2_start = n_sensory + n_l1
    l3_start = n_sensory + n_l1 + n_l2
    memory_start = n_sensory + n_int
    motor_start = n_sensory + n_int + n_mem

    sensory = np.arange(sensory_start, sensory_start + n_sensory)
    layer1 = np.arange(l1_start, l1_start + n_l1)
    layer2 = np.arange(l2_start, l2_start + n_l2)
    layer3 = np.arange(l3_start, l3_start + n_l3)
    all_internal = np.arange(l1_start, l1_start + n_int)
    memory = np.arange(memory_start, memory_start + n_mem)
    motor = np.arange(motor_start, motor_start + n_motor)

    # Node types: E/I split for all internal layers
    node_types = np.full(n_total, _NTYPE_E, dtype=int)
    n_inhib = int(n_int * genome.frac_inhibitory)
    n_exc = n_int - n_inhib
    internal_types = [_NTYPE_E] * n_exc + [_NTYPE_I] * n_inhib
    rng.shuffle(internal_types)
    node_types[l1_start:l1_start + n_int] = internal_types

    # Regions — assign each layer its own region
    _reg = list(Region)
    regions = np.zeros(n_total, dtype=int)
    regions[sensory] = _reg.index(Region.SENSORY)
    regions[layer1] = _reg.index(Region.LOCAL_DETECT)
    regions[layer2] = _reg.index(Region.MID_LEVEL)
    regions[layer3] = _reg.index(Region.ABSTRACT)
    regions[memory] = _reg.index(Region.MEMORY)
    regions[motor] = _reg.index(Region.MOTOR)

    # Leak rates per node type
    _ntype_list = list(NodeType)
    leak_rates = np.array([DEFAULT_LEAK[_ntype_list[t]] for t in node_types])
    leak_rates[memory] = 0.02

    # Per-node parameters
    max_rate = np.full(n_total, genome.max_rate_E)
    adapt_rate = np.full(n_total, genome.adapt_rate_E)
    i_mask = node_types == _NTYPE_I
    max_rate[i_mask] = genome.max_rate_I
    adapt_rate[i_mask] = genome.adapt_rate_I

    # Column assignments: each layer gets SEPARATE columns so WTA
    # competition is within-layer (V1 neurons compete with V1, not V4).
    # L1 columns: 0..n_cells-1, L2 columns: n_cells..2*n_cells-1.
    # L3: column_id = -1 (non-topographic, competes in global pool).
    column_ids = np.full(n_total, -1, dtype=np.int32)
    column_ids[l1_start:l1_start + n_l1] = np.arange(n_l1) % n_cells
    column_ids[l2_start:l2_start + n_l2] = n_cells + (np.arange(n_l2) % n_cells)
    # Sensory neurons
    for cell in range(n_cells):
        s0 = sensory_start + cell * FEATURES_PER_CELL
        column_ids[s0:s0 + FEATURES_PER_CELL] = cell
    # Motor neurons
    for cell in range(n_cells):
        m0 = motor_start + cell * NUM_COLORS
        column_ids[m0:m0 + NUM_COLORS] = cell

    net = DNG(
        n_nodes=n_total,
        node_types=node_types,
        regions=regions,
        excitability=np.ones(n_total),
        leak_rates=leak_rates,
        max_rate=max_rate,
        adapt_rate=adapt_rate,
        input_nodes=sensory,
        output_nodes=motor,
        memory_nodes=memory,
        column_ids=column_ids,
        n_columns=0,
        max_h=grid_h,
        max_w=grid_w,
        f_rate=genome.f_rate,
        f_decay=genome.f_decay,
        f_max=genome.f_max,
    )

    ws = genome.weight_scale
    cap = genome.max_fan_in

    # ═══════════════════════════════════════════════════════════════════
    # FEEDFORWARD PATHWAY (strong at birth)
    # ═══════════════════════════════════════════════════════════════════

    # SENSORY -> LAYER 1: sparse receptive fields (dendritic sampling)
    rf_prob = 0.05
    _local_rf_edges(
        net, sensory_start, layer1, grid_h, grid_w, n_cells,
        rf_radius=2, long_range_frac=0.15, ws=ws / rf_prob**0.5, rng=rng,
        direction='sensory_to_internal',
        n_sensory=n_sensory,
        connection_prob=rf_prob,
    )

    # LAYER 1 -> LAYER 2: wider effective RF, each L2 neuron samples
    # from L1 neurons across multiple columns (~radius 3).
    # Corticocortical feedforward synapses are individually stronger
    # than thalamocortical ones because they carry post-WTA sparse
    # signals that need amplification at each stage.
    ff_ws = ws * 3.0
    _inter_layer_rf_edges(
        net, layer1, layer2, grid_h, grid_w, n_cells,
        rf_radius=3, connection_prob=0.15,
        ws=ff_ws, rng=rng,
    )

    # LAYER 2 -> LAYER 3: broad sampling (L3 is non-topographic,
    # each L3 neuron receives from a random subset of L2).
    fan_l2_l3 = _fan_in(0.20, n_l2, cap)
    _fan_in_edges(net, layer2, layer3, fan_l2_l3, ff_ws, rng)

    # LAYER 3 -> MOTOR: learned output pathway
    _local_rf_edges(
        net, motor_start, layer3, grid_h, grid_w, n_cells,
        rf_radius=2, long_range_frac=0.35, ws=ws * 0.01 / rf_prob**0.5, rng=rng,
        direction='internal_to_motor',
        n_sensory=n_sensory,
        connection_prob=rf_prob,
    )

    # ═══════════════════════════════════════════════════════════════════
    # FEEDBACK PATHWAY (weak at birth, ~10% of feedforward)
    # ═══════════════════════════════════════════════════════════════════
    feedback_ws = ws * 0.1

    # LAYER 2 -> LAYER 1
    fan_l2_l1 = _fan_in(0.05, n_l2, cap)
    _fan_in_edges(net, layer2, layer1, fan_l2_l1, feedback_ws, rng)

    # LAYER 3 -> LAYER 2
    fan_l3_l2 = _fan_in(0.05, n_l3, cap)
    _fan_in_edges(net, layer3, layer2, fan_l3_l2, feedback_ws, rng)

    # ═══════════════════════════════════════════════════════════════════
    # LATERAL CONNECTIONS (within each layer, weak at birth)
    # ═══════════════════════════════════════════════════════════════════
    lateral_ws = ws * 0.1

    fan_l1_l1 = _fan_in(genome.density_internal_to_internal, n_l1, cap)
    _fan_in_edges(net, layer1, layer1, fan_l1_l1, lateral_ws, rng)

    fan_l2_l2 = _fan_in(genome.density_internal_to_internal, n_l2, cap)
    _fan_in_edges(net, layer2, layer2, fan_l2_l2, lateral_ws, rng)

    fan_l3_l3 = _fan_in(genome.density_internal_to_internal, n_l3, cap)
    _fan_in_edges(net, layer3, layer3, fan_l3_l3, lateral_ws, rng)

    # ═══════════════════════════════════════════════════════════════════
    # OTHER PATHWAYS
    # ═══════════════════════════════════════════════════════════════════

    # MOTOR -> LAYER 3: feedback from motor (proprioceptive-like)
    fan_m2l3 = _fan_in(genome.density_motor_to_internal, n_motor, cap)
    _fan_in_edges(net, motor, layer3, fan_m2l3, lateral_ws, rng)

    # LAYER 1 -> SENSORY: top-down feedback (weak at birth)
    fan_l1_s = _fan_in(genome.density_internal_to_sensory, n_l1, cap)
    _fan_in_edges(net, layer1, sensory, fan_l1_s, lateral_ws, rng)

    # SENSORY -> MOTOR: weak direct path (NOT the copy pathway)
    fan_s2m = _fan_in(genome.density_sensory_to_motor, n_sensory, cap)
    _fan_in_edges(net, sensory, motor, fan_s2m, ws, rng)

    # MEMORY circuit — wired to Layer 3 (abstract/reasoning)
    # Memory self-connections are strong (bistable attractors).
    # Memory↔L3 connections are moderate — L3 should be driven
    # primarily by L2 feedforward, with memory providing context.
    mem_ws = ws * 5
    mem_fan_in = min(max(10, n_mem // 2), cap)
    _fan_in_edges(net, memory, memory, mem_fan_in, mem_ws, rng)
    _fan_in_edges(net, layer3, memory, min(max(10, n_l3 // 3), cap), ws * 2, rng)
    _fan_in_edges(net, memory, layer3, min(max(10, n_mem // 2), cap), ws * 2, rng)
    _fan_in_edges(net, memory, motor, min(max(10, n_mem // 2), cap), ws * 2, rng)
    _fan_in_edges(net, sensory, memory, min(max(5, n_sensory // 15), cap), ws, rng)

    # INSTINCT: Copy pathway — sensory COLOR nodes -> motor nodes (1:1)
    copy_src = []
    copy_dst = []
    for cell in range(n_cells):
        for k in range(NUM_COLORS):
            s_idx = sensory_start + cell * FEATURES_PER_CELL + k
            m_idx = motor_start + cell * NUM_COLORS + k
            copy_src.append(s_idx)
            copy_dst.append(m_idx)
    copy_src = np.array(copy_src, dtype=np.int32)
    copy_dst = np.array(copy_dst, dtype=np.int32)
    copy_w = np.full(len(copy_src), 2.0)
    edge_start = net._edge_count
    net.add_edges_batch(copy_src, copy_dst, copy_w)
    edge_end = net._edge_count
    net._edge_consolidation[edge_start:edge_end] = 20.0

    # Spatial neighbor connections for motor
    _spatial_neighbors(net, motor_start, grid_h, grid_w,
                       genome.density_motor_neighbors, ws, rng)

    return net


def _inter_layer_rf_edges(
    net: DNG,
    src_layer: np.ndarray,
    dst_layer: np.ndarray,
    grid_h: int,
    grid_w: int,
    n_cells: int,
    rf_radius: int,
    connection_prob: float,
    ws: float,
    rng: np.random.Generator,
) -> None:
    """Connect two topographic layers: each dst neuron samples from src
    neurons in a spatial neighborhood around its assigned column."""
    n_src = len(src_layer)
    n_dst = len(dst_layer)
    cell_row = np.arange(n_cells) // grid_w
    cell_col = np.arange(n_cells) % grid_w

    src_cells = np.arange(n_src) % n_cells
    dst_cells = np.arange(n_dst) % n_cells

    # Group src neurons by cell for fast lookup
    src_by_cell = [[] for _ in range(n_cells)]
    for i, c in enumerate(src_cells):
        src_by_cell[c].append(int(src_layer[i]))

    src_list, dst_list = [], []

    for idx in range(n_dst):
        dst_node = int(dst_layer[idx])
        center = dst_cells[idx]
        cr, cc = cell_row[center], cell_col[center]

        # Gather src neurons from all cells within RF radius
        candidate_src = []
        for dr in range(-rf_radius, rf_radius + 1):
            for dc in range(-rf_radius, rf_radius + 1):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    cell = nr * grid_w + nc
                    candidate_src.extend(src_by_cell[cell])

        if not candidate_src:
            continue

        candidates = np.array(candidate_src, dtype=np.int64)
        keep = rng.random(len(candidates)) < connection_prob
        if keep.sum() < 2:
            keep[:min(2, len(keep))] = True
        selected = candidates[keep]

        for s in selected:
            src_list.append(int(s))
            dst_list.append(dst_node)

    if not src_list:
        return

    all_src = np.array(src_list, dtype=np.int32)
    all_dst = np.array(dst_list, dtype=np.int32)
    valid = all_src != all_dst
    sel_src = all_src[valid]
    sel_dst = all_dst[valid]

    magnitudes = _init_weights(rng, ws, len(sel_src))
    signs = np.where(net._mask_I[sel_src], -1.0, 1.0)
    sel_w = signs * magnitudes
    net.add_edges_batch(sel_src, sel_dst, sel_w)


def _local_rf_edges(
    net: DNG,
    io_start: int,
    internal: np.ndarray,
    grid_h: int,
    grid_w: int,
    n_cells: int,
    rf_radius: int,
    long_range_frac: float,
    ws: float,
    rng: np.random.Generator,
    direction: str,
    n_sensory: int = 0,
    connection_prob: float = 0.15,
) -> None:
    """Spatially structured connections between I/O layer and an internal layer.

    Each neuron samples a RANDOM SUBSET of the available inputs in its
    receptive field (controlled by connection_prob). This mimics dendritic
    tree sampling — in real cortex, each neuron contacts only ~5-15% of
    available thalamic axons, creating the initial diversity that competitive
    Hebbian learning amplifies into stimulus selectivity.
    """
    n_int = len(internal)

    if direction == 'sensory_to_internal':
        features_per_cell = FEATURES_PER_CELL
        n_io = n_sensory
    else:
        features_per_cell = NUM_COLORS
        n_io = n_cells * NUM_COLORS

    cell_assignments = np.arange(n_int) % n_cells
    cell_row = np.arange(n_cells) // grid_w
    cell_col = np.arange(n_cells) % grid_w

    if direction == 'sensory_to_internal':
        io_start_val = io_start
    else:
        io_start_val = io_start

    src_list, dst_list = [], []

    for idx, int_node in enumerate(internal):
        center_cell = cell_assignments[idx]
        cr, cc = cell_row[center_cell], cell_col[center_cell]

        local_cells = []
        for dr in range(-rf_radius, rf_radius + 1):
            for dc in range(-rf_radius, rf_radius + 1):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    local_cells.append(nr * grid_w + nc)

        local_io_nodes = []
        for cell in local_cells:
            for k in range(features_per_cell):
                local_io_nodes.append(io_start_val + cell * features_per_cell + k)

        local_io = np.array(local_io_nodes, dtype=np.int64)

        keep_mask = rng.random(len(local_io)) < connection_prob
        if keep_mask.sum() < 3:
            keep_mask[:3] = True
        selected_local = local_io[keep_mask]

        n_long = max(1, int(len(local_io) * long_range_frac * connection_prob))
        all_io = np.arange(io_start_val, io_start_val + n_io)
        n_long = min(n_long, len(all_io))
        long_range = rng.choice(all_io, size=n_long, replace=False)

        all_connected = np.unique(np.concatenate([
            selected_local,
            long_range.astype(np.int64),
        ]))

        if direction == 'sensory_to_internal':
            for io_node in all_connected:
                src_list.append(int(io_node))
                dst_list.append(int(int_node))
        else:
            for io_node in all_connected:
                src_list.append(int(int_node))
                dst_list.append(int(io_node))

    if not src_list:
        return

    all_src = np.array(src_list, dtype=np.int32)
    all_dst = np.array(dst_list, dtype=np.int32)
    valid = all_src != all_dst
    sel_src = all_src[valid]
    sel_dst = all_dst[valid]

    magnitudes = _init_weights(rng, ws, len(sel_src))
    signs = np.where(net._mask_I[sel_src], -1.0, 1.0)
    sel_w = signs * magnitudes
    net.add_edges_batch(sel_src, sel_dst, sel_w)


def _init_weights(
    rng: np.random.Generator, scale: float, n: int,
) -> np.ndarray:
    """Noisy initial weights — small Gaussian perturbation around scale.

    Competitive learning requires initial asymmetry to break symmetry
    (Rumelhart & Zipser 1985, Krotov & Hopfield 2019). Without it,
    identical neurons make identical responses and K-H updates average
    out to zero differentiation.
    """
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
) -> None:
    """Each dst node gets `fan_in` random connections from src nodes."""
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

    magnitudes = _init_weights(rng, weight_scale, len(sel_src))
    signs = np.where(net._mask_I[sel_src], -1.0, 1.0)
    sel_w = signs * magnitudes
    net.add_edges_batch(sel_src, sel_dst, sel_w)


def _spatial_neighbors(
    net: DNG,
    region_offset: int,
    grid_h: int,
    grid_w: int,
    density: float,
    weight_scale: float,
    rng: np.random.Generator,
) -> None:
    """Connect color nodes of 4-connected spatial neighbors."""
    if density <= 0:
        return

    src_list, dst_list = [], []
    for r in range(grid_h):
        for c in range(grid_w):
            cell = r * grid_w + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    neighbor = nr * grid_w + nc
                    for k in range(NUM_COLORS):
                        src_list.append(region_offset + cell * NUM_COLORS + k)
                        dst_list.append(region_offset + neighbor * NUM_COLORS + k)

    if not src_list:
        return

    all_src = np.array(src_list, dtype=np.int32)
    all_dst = np.array(dst_list, dtype=np.int32)
    mask = rng.random(len(all_src)) < density
    sel_src = all_src[mask]
    sel_dst = all_dst[mask]
    magnitudes = _init_weights(rng, weight_scale, len(sel_src))
    signs = np.where(net._mask_I[sel_src], -1.0, 1.0)
    sel_w = signs * magnitudes
    net.add_edges_batch(sel_src, sel_dst, sel_w)
