"""
Template generator: genome -> DNG.

Layout:
  Sensory (max_h * max_w * 10)
    -> Internal (n_internal)  <->  Concept (n_concept)
    -> Memory (n_memory, self-sustaining, long-term)
    -> Motor (max_h * max_w * 10)

Instinct circuits:
  - Copy pathway: sensory[cell,color] -> motor[cell,color] (identity mapping)
  - Memory self-connections: bistable, once activated stays on
  - Memory <-> internal: write and recall paths
"""

from __future__ import annotations

import numpy as np

from .encoding import NUM_COLORS
from .genome import Genome
from .graph import DNG, NodeType, Region, DEFAULT_LEAK


def create_dng(
    genome: Genome,
    grid_h: int,
    grid_w: int,
    rng: np.random.Generator | None = None,
) -> DNG:
    if rng is None:
        rng = np.random.default_rng()

    n_cells = grid_h * grid_w
    n_io = n_cells * NUM_COLORS
    n_int = genome.n_internal
    n_concept = genome.n_concept
    n_mem = genome.n_memory
    n_total = n_io + n_int + n_concept + n_mem + n_io

    sensory_start = 0
    internal_start = n_io
    concept_start = n_io + n_int
    memory_start = n_io + n_int + n_concept
    motor_start = n_io + n_int + n_concept + n_mem

    sensory = np.arange(sensory_start, sensory_start + n_io)
    internal = np.arange(internal_start, internal_start + n_int)
    concept = np.arange(concept_start, concept_start + n_concept)
    memory = np.arange(memory_start, memory_start + n_mem)
    motor = np.arange(motor_start, motor_start + n_io)

    # ── Node types ──────────────────────────────────────────────────
    node_types = np.full(n_total, list(NodeType).index(NodeType.E), dtype=int)

    n_inhib = int(n_int * genome.frac_inhibitory)
    n_modul = int(n_int * genome.frac_modulatory)
    n_mem_internal = int(n_int * genome.frac_memory)
    n_exc = n_int - n_inhib - n_modul - n_mem_internal

    internal_types = (
        [list(NodeType).index(NodeType.E)] * n_exc +
        [list(NodeType).index(NodeType.I)] * n_inhib +
        [list(NodeType).index(NodeType.M)] * n_modul +
        [list(NodeType).index(NodeType.Mem)] * n_mem_internal
    )
    rng.shuffle(internal_types)
    node_types[internal_start:internal_start + n_int] = internal_types

    n_concept_I = max(1, int(n_concept * 0.2))
    concept_types = ([list(NodeType).index(NodeType.E)] * (n_concept - n_concept_I) +
                     [list(NodeType).index(NodeType.I)] * n_concept_I)
    rng.shuffle(concept_types)
    node_types[concept_start:concept_start + n_concept] = concept_types

    # Memory pool nodes are all excitatory (self-sustaining)
    node_types[memory_start:memory_start + n_mem] = list(NodeType).index(NodeType.E)

    # ── Regions ─────────────────────────────────────────────────────
    regions = np.zeros(n_total, dtype=int)
    regions[sensory] = list(Region).index(Region.SENSORY)
    regions[internal] = list(Region).index(Region.ABSTRACT)
    regions[concept] = list(Region).index(Region.ABSTRACT)
    regions[memory] = list(Region).index(Region.MEMORY)
    regions[motor] = list(Region).index(Region.MOTOR)

    # ── Leak rates ──────────────────────────────────────────────────
    _ntype_list = list(NodeType)
    leak_rates = np.array([DEFAULT_LEAK[_ntype_list[t]] for t in node_types])
    # Memory pool: very slow leak so activity persists
    leak_rates[memory] = 0.02

    column_ids = np.full(n_total, -1, dtype=np.int32)

    net = DNG(
        n_nodes=n_total,
        node_types=node_types,
        regions=regions,
        excitability=np.ones(n_total),
        leak_rates=leak_rates,
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

    # ── Local receptive field connections (sensory <-> internal) ────
    # Each internal neuron is assigned a grid position and connects to
    # a local patch of sensory neurons (like V1 receptive fields).
    _local_rf_edges(
        net, sensory_start, internal, grid_h, grid_w, n_cells,
        rf_radius=2, long_range_frac=0.15, ws=ws, rng=rng,
        direction='sensory_to_internal',
    )

    # Internal -> Motor: local + long-range connections so motor neurons
    # can receive input from internal neurons at distant grid positions.
    # Essential for spatial transformations (flip, rotate, transpose).
    _local_rf_edges(
        net, motor_start, internal, grid_h, grid_w, n_cells,
        rf_radius=2, long_range_frac=0.35, ws=ws, rng=rng,
        direction='internal_to_motor',
    )

    # Internal <-> Internal: lateral connections (can stay random)
    fan_i2i = _fan_in(genome.density_internal_to_internal, n_int, cap)
    _fan_in_edges(net, internal, internal, fan_i2i, ws, rng)

    # Motor -> Internal (feedback): random but sparse
    fan_m2i = _fan_in(genome.density_motor_to_internal, n_io, cap)
    _fan_in_edges(net, motor, internal, fan_m2i, ws, rng)

    # Internal -> Sensory (top-down feedback): sparse
    fan_i2s = _fan_in(genome.density_internal_to_sensory, n_int, cap)
    _fan_in_edges(net, internal, sensory, fan_i2s, ws, rng)

    # Sensory -> Motor (weak direct path, NOT the copy pathway)
    fan_s2m = _fan_in(genome.density_sensory_to_motor, n_io, cap)
    _fan_in_edges(net, sensory, motor, fan_s2m, ws, rng)

    # Concept pool: pools from multiple internal groups (larger receptive fields)
    fan_c2concept = _fan_in(genome.density_column_to_concept, n_int, cap)
    fan_concept2c = _fan_in(genome.density_concept_to_column, n_concept, cap)
    _fan_in_edges(net, internal, concept, fan_c2concept, ws, rng)
    _fan_in_edges(net, concept, internal, fan_concept2c, ws, rng)
    _fan_in_edges(net, concept, concept, min(max(2, n_concept // 5), cap), ws, rng)
    _fan_in_edges(net, concept, motor, min(max(10, n_concept // 3), cap), ws, rng)
    _fan_in_edges(net, sensory, concept, min(max(10, n_io // 8), cap), ws, rng)

    # ── Hippocampal memory circuit ───────────────────────────────────
    # Dense autoassociative recurrence (CA3-like pattern completion).
    # Each memory node connects to ~half the others so partial cues
    # can reactivate full stored patterns via attractor dynamics.
    mem_ws = ws * 5
    mem_fan_in = min(max(10, n_mem // 2), cap)
    _fan_in_edges(net, memory, memory, mem_fan_in, mem_ws, rng)

    # Internal -> Memory (write path)
    _fan_in_edges(net, internal, memory, min(max(10, n_int // 5), cap), mem_ws, rng)
    # Memory -> Internal (recall path)
    _fan_in_edges(net, memory, internal, min(max(10, n_mem // 2), cap), mem_ws, rng)
    # Concept -> Memory (abstract patterns get stored)
    _fan_in_edges(net, concept, memory, min(max(5, n_concept // 3), cap), mem_ws, rng)
    # Memory -> Concept (recalled memories inform abstraction)
    _fan_in_edges(net, memory, concept, min(max(5, n_mem // 3), cap), mem_ws, rng)
    # Memory -> Motor (direct recall to output -- strong, so memory can drive answers)
    _fan_in_edges(net, memory, motor, min(max(10, n_mem // 2), cap), mem_ws, rng)
    # Sensory -> Memory (direct perception to memory)
    _fan_in_edges(net, sensory, memory, min(max(5, n_io // 15), cap), mem_ws, rng)

    # ── Instinct: Copy pathway ──────────────────────────────────────
    # Direct 1:1 sensory[cell,color] -> motor[cell,color] connections.
    # Weak initial copy -- same magnitude as other connections so CHL
    # can strengthen or weaken it based on whether copying is useful.
    copy_src = sensory.copy()
    copy_dst = motor.copy()
    copy_w = np.full(len(copy_src), ws)
    idx = net._edge_count
    n_new = len(copy_src)
    net._ensure_capacity(idx + n_new)
    net._edge_src[idx:idx + n_new] = copy_src
    net._edge_dst[idx:idx + n_new] = copy_dst
    net._edge_w[idx:idx + n_new] = copy_w
    net._edge_count += n_new
    net._csr_dirty = True

    # Spatial neighbor connections
    _spatial_neighbors(net, sensory_start, grid_h, grid_w,
                       genome.density_sensory_neighbors, ws, rng)
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
) -> None:
    """
    Create spatially structured connections between I/O layer and internal layer.

    Each internal neuron is assigned a grid position. It connects to all
    color channels of cells within `rf_radius` of that position, plus a
    fraction of random long-range connections for global context.

    direction: 'sensory_to_internal' or 'internal_to_motor'
    """
    n_int = len(internal)
    n_io = n_cells * NUM_COLORS

    # Assign each internal neuron to a grid cell (cycling through positions)
    cell_assignments = np.arange(n_int) % n_cells

    # Precompute grid coordinates for each cell
    cell_row = np.arange(n_cells) // grid_w
    cell_col = np.arange(n_cells) % grid_w

    src_list, dst_list = [], []

    for idx, int_node in enumerate(internal):
        center_cell = cell_assignments[idx]
        cr, cc = cell_row[center_cell], cell_col[center_cell]

        # Local patch: all cells within rf_radius (Chebyshev distance)
        local_cells = []
        for dr in range(-rf_radius, rf_radius + 1):
            for dc in range(-rf_radius, rf_radius + 1):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    local_cells.append(nr * grid_w + nc)

        # All color channels of local cells
        local_io_nodes = []
        for cell in local_cells:
            for k in range(NUM_COLORS):
                local_io_nodes.append(io_start + cell * NUM_COLORS + k)

        # Long-range: random sample from ALL io nodes
        n_long = max(1, int(len(local_io_nodes) * long_range_frac))
        all_io = np.arange(io_start, io_start + n_io)
        long_range = rng.choice(all_io, size=n_long, replace=False)

        all_connected = np.unique(np.concatenate([
            np.array(local_io_nodes, dtype=np.int64),
            long_range.astype(np.int64),
        ]))

        if direction == 'sensory_to_internal':
            for io_node in all_connected:
                src_list.append(int(io_node))
                dst_list.append(int(int_node))
        else:  # internal_to_motor
            for io_node in all_connected:
                src_list.append(int(int_node))
                dst_list.append(int(io_node))

    if not src_list:
        return

    all_src = np.array(src_list, dtype=np.int32)
    all_dst = np.array(dst_list, dtype=np.int32)

    # Remove self-loops
    valid = all_src != all_dst
    sel_src = all_src[valid]
    sel_dst = all_dst[valid]

    magnitudes = np.abs(rng.normal(0, ws, size=len(sel_src)))
    signs = np.where(net._mask_I[sel_src], -1.0, 1.0)
    sel_w = signs * magnitudes

    n_new = len(sel_src)
    idx_start = net._edge_count
    net._ensure_capacity(idx_start + n_new)
    net._edge_src[idx_start:idx_start + n_new] = sel_src
    net._edge_dst[idx_start:idx_start + n_new] = sel_dst
    net._edge_w[idx_start:idx_start + n_new] = sel_w
    net._edge_count += n_new
    net._csr_dirty = True


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
    """Each dst node gets `fan_in` random connections from src nodes (vectorized)."""
    n_src = len(src_nodes)
    n_dst = len(dst_nodes)
    if n_src == 0 or n_dst == 0 or fan_in <= 0:
        return

    fan_in = min(fan_in, n_src)

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

    idx = net._edge_count
    n_new = len(sel_src)
    net._ensure_capacity(idx + n_new)
    net._edge_src[idx:idx + n_new] = sel_src
    net._edge_dst[idx:idx + n_new] = sel_dst
    net._edge_w[idx:idx + n_new] = sel_w
    net._edge_count += n_new
    net._csr_dirty = True


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

    idx = net._edge_count
    n_new = len(sel_src)
    net._ensure_capacity(idx + n_new)
    net._edge_src[idx:idx + n_new] = sel_src
    net._edge_dst[idx:idx + n_new] = sel_dst
    net._edge_w[idx:idx + n_new] = sel_w
    net._edge_count += n_new
    net._csr_dirty = True
