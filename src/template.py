"""
Template generator: genome -> DNG.

Phase 1 layout (Evolutionary Minimalism):
  SENSORY (perception features: 28 per cell + 28 global)
    -> INTERNAL (E/I neurons, main processing)
    <-> MEMORY (self-sustaining, episodic)
    -> MOTOR (10 colors per cell, one-hot)

Instinct circuits:
  - Copy pathway: sensory color nodes -> motor color nodes (identity mapping)
  - Memory self-connections: bistable attractor dynamics
  - Memory <-> internal: write and recall paths
"""

from __future__ import annotations

import numpy as np

from .encoding import NUM_COLORS
from .perception.encoder import sensory_size, FEATURES_PER_CELL
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
    n_total = n_sensory + n_int + n_mem + n_motor

    sensory_start = 0
    internal_start = n_sensory
    memory_start = n_sensory + n_int
    motor_start = n_sensory + n_int + n_mem

    sensory = np.arange(sensory_start, sensory_start + n_sensory)
    internal = np.arange(internal_start, internal_start + n_int)
    memory = np.arange(memory_start, memory_start + n_mem)
    motor = np.arange(motor_start, motor_start + n_motor)

    # Node types: E/I split for internal only
    node_types = np.full(n_total, _NTYPE_E, dtype=int)
    n_inhib = int(n_int * genome.frac_inhibitory)
    n_exc = n_int - n_inhib
    internal_types = [_NTYPE_E] * n_exc + [_NTYPE_I] * n_inhib
    rng.shuffle(internal_types)
    node_types[internal_start:internal_start + n_int] = internal_types

    # Regions
    _reg = list(Region)
    regions = np.zeros(n_total, dtype=int)
    regions[sensory] = _reg.index(Region.SENSORY)
    regions[internal] = _reg.index(Region.INTERNAL)
    regions[memory] = _reg.index(Region.MEMORY)
    regions[motor] = _reg.index(Region.MOTOR)

    # Leak rates per node type
    _ntype_list = list(NodeType)
    leak_rates = np.array([DEFAULT_LEAK[_ntype_list[t]] for t in node_types])
    leak_rates[memory] = 0.02  # very slow leak for persistent activity

    # Per-node parameters: different for E vs I
    max_rate = np.full(n_total, genome.max_rate_E)
    adapt_rate = np.full(n_total, genome.adapt_rate_E)
    i_mask = node_types == _NTYPE_I
    max_rate[i_mask] = genome.max_rate_I
    adapt_rate[i_mask] = genome.adapt_rate_I

    column_ids = np.full(n_total, -1, dtype=np.int32)

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

    # SENSORY -> INTERNAL: local receptive fields
    _local_rf_edges(
        net, sensory_start, internal, grid_h, grid_w, n_cells,
        rf_radius=2, long_range_frac=0.15, ws=ws, rng=rng,
        direction='sensory_to_internal',
        n_sensory=n_sensory,
    )

    # INTERNAL -> MOTOR: start very weak (must learn proper mappings)
    _local_rf_edges(
        net, motor_start, internal, grid_h, grid_w, n_cells,
        rf_radius=2, long_range_frac=0.35, ws=ws * 0.01, rng=rng,
        direction='internal_to_motor',
        n_sensory=n_sensory,
    )

    # INTERNAL <-> INTERNAL: lateral connections
    fan_i2i = _fan_in(genome.density_internal_to_internal, n_int, cap)
    _fan_in_edges(net, internal, internal, fan_i2i, ws, rng)

    # MOTOR -> INTERNAL: feedback
    fan_m2i = _fan_in(genome.density_motor_to_internal, n_motor, cap)
    _fan_in_edges(net, motor, internal, fan_m2i, ws, rng)

    # INTERNAL -> SENSORY: top-down feedback
    fan_i2s = _fan_in(genome.density_internal_to_sensory, n_int, cap)
    _fan_in_edges(net, internal, sensory, fan_i2s, ws, rng)

    # SENSORY -> MOTOR: weak direct path (NOT the copy pathway)
    fan_s2m = _fan_in(genome.density_sensory_to_motor, n_sensory, cap)
    _fan_in_edges(net, sensory, motor, fan_s2m, ws, rng)

    # MEMORY circuit
    mem_ws = ws * 5
    mem_fan_in = min(max(10, n_mem // 2), cap)
    _fan_in_edges(net, memory, memory, mem_fan_in, mem_ws, rng)
    _fan_in_edges(net, internal, memory, min(max(10, n_int // 5), cap), mem_ws, rng)
    _fan_in_edges(net, memory, internal, min(max(10, n_mem // 2), cap), mem_ws, rng)
    _fan_in_edges(net, memory, motor, min(max(10, n_mem // 2), cap), mem_ws, rng)
    _fan_in_edges(net, sensory, memory, min(max(5, n_sensory // 15), cap), mem_ws, rng)

    # INSTINCT: Copy pathway — sensory COLOR nodes -> motor nodes (1:1)
    # Only the first 10 features per cell are color one-hots
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
    # Instinct consolidation: innate pathway, scaling preserves dominance
    net._edge_consolidation[edge_start:edge_end] = 20.0

    # Spatial neighbor connections for motor
    _spatial_neighbors(net, motor_start, grid_h, grid_w,
                       genome.density_motor_neighbors, ws, rng)

    return net


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
) -> None:
    """
    Spatially structured connections between I/O layer and internal layer.

    For sensory_to_internal: connects perception feature nodes (all 28 per cell)
    within the receptive field to internal neurons.

    For internal_to_motor: connects internal to motor color nodes (10 per cell).
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
                local_io_nodes.append(io_start + cell * features_per_cell + k)

        n_long = max(1, int(len(local_io_nodes) * long_range_frac))
        all_io = np.arange(io_start, io_start + n_io)
        n_long = min(n_long, len(all_io))
        long_range = rng.choice(all_io, size=n_long, replace=False)

        all_connected = np.unique(np.concatenate([
            np.array(local_io_nodes, dtype=np.int64),
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

    magnitudes = np.abs(rng.normal(0, ws, size=len(sel_src)))
    signs = np.where(net._mask_I[sel_src], -1.0, 1.0)
    sel_w = signs * magnitudes
    net.add_edges_batch(sel_src, sel_dst, sel_w)


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

    magnitudes = np.abs(rng.normal(0, weight_scale, size=len(sel_src)))
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
    magnitudes = np.abs(rng.normal(0, weight_scale, size=len(sel_src)))
    signs = np.where(net._mask_I[sel_src], -1.0, 1.0)
    sel_w = signs * magnitudes
    net.add_edges_batch(sel_src, sel_dst, sel_w)
