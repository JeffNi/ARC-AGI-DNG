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

    # ── Two-layer internal hierarchy ──────────────────────────────────
    # Layer 1 (local features, 60%): small RFs from sensory, like V1/V2.
    # Lateral connections within layer for edge/boundary detection.
    # Layer 2 (patterns, 40%): pools from Layer 1 with larger RFs.
    # Connected to motor for output computation.

    n_L1 = int(n_int * 0.6)
    n_L2 = n_int - n_L1
    layer1 = internal[:n_L1]
    layer2 = internal[n_L1:]

    # Sensory -> Layer 1: small local RFs (radius 2, like V1)
    _local_rf_edges(
        net, sensory_start, layer1, grid_h, grid_w, n_cells,
        rf_radius=2, long_range_frac=0.10, ws=ws, rng=rng,
        direction='sensory_to_internal',
    )

    # Layer 1 -> Layer 2: wider pooling (radius 3 ≈ 7x7 patch on 10x10 grid)
    _inter_layer_rf_edges(
        net, layer1, layer2, grid_h, grid_w, n_cells,
        rf_radius=3, long_range_frac=0.15, ws=ws, rng=rng,
    )

    # Layer 2 -> Motor: local RFs with moderate long-range
    _local_rf_edges(
        net, motor_start, layer2, grid_h, grid_w, n_cells,
        rf_radius=3, long_range_frac=0.15, ws=ws, rng=rng,
        direction='internal_to_motor',
    )

    # ── Seed connections (sparse bootstraps for synaptogenesis) ──────
    # Genetic scaffold: sparse but STRONG enough to actually activate
    # target neurons. A real brain's initial wiring doesn't just exist
    # structurally -- it conducts enough signal to keep neurons alive.
    # Synaptogenesis then grows dense wiring where neurons are co-active.
    #
    # Seed weight is 10x the base weight_scale so that even a few
    # connections produce enough input to cross the firing threshold.

    SEED = 10
    seed_ws = ws * 10  # strong enough to actually activate targets

    # Layer 1 lateral (nearest-neighbor interactions)
    _fan_in_edges(net, layer1, layer1, SEED, seed_ws, rng)

    # Layer 2 lateral
    _fan_in_edges(net, layer2, layer2, SEED, seed_ws, rng)

    # Feedback paths — top-down signal flow is critical for CHL.
    # During clamped phase, correct output on motor neurons must
    # propagate backwards to shape internal correlations.
    FEEDBACK_FAN = 20
    _fan_in_edges(net, motor, layer2, min(FEEDBACK_FAN, n_io), seed_ws, rng)
    _fan_in_edges(net, layer2, layer1, min(FEEDBACK_FAN, n_L2), seed_ws, rng)
    _fan_in_edges(net, layer1, sensory, min(SEED, n_L1), ws, rng)

    # Sensory -> Motor (non-copy, bypasses internal layers)
    _fan_in_edges(net, sensory, motor, min(5, n_io), ws, rng)

    # Concept pool: association cortex — sparse at birth, matures through
    # experience. Needs enough input to participate in activity, but the
    # real wiring comes from synaptogenesis during infancy/childhood.
    CONCEPT_FAN = 25
    _fan_in_edges(net, layer2, concept, CONCEPT_FAN, seed_ws, rng)
    _fan_in_edges(net, layer1, concept, min(15, n_L1), seed_ws, rng)
    _fan_in_edges(net, concept, layer2, min(CONCEPT_FAN, n_concept), seed_ws, rng)
    _fan_in_edges(net, concept, layer1, min(10, n_concept), seed_ws, rng)
    _fan_in_edges(net, concept, concept, min(15, n_concept), seed_ws, rng)
    _fan_in_edges(net, concept, motor, min(15, n_concept), seed_ws, rng)
    _fan_in_edges(net, sensory, concept, min(10, n_io), seed_ws, rng)

    # ── Hippocampal memory circuit ───────────────────────────────────
    # Hippocampus has moderate initial wiring, specialized for rapid
    # one-shot encoding. Needs strong bidirectional paths to both
    # internal layers and concept pool for episodic storage/recall.
    MEM_FAN = 20
    mem_seed_ws = seed_ws * 2
    _fan_in_edges(net, memory, memory, MEM_FAN, mem_seed_ws, rng)
    _fan_in_edges(net, internal, memory, MEM_FAN, mem_seed_ws, rng)
    _fan_in_edges(net, memory, internal, min(MEM_FAN, n_mem), mem_seed_ws, rng)
    _fan_in_edges(net, concept, memory, min(15, n_concept), mem_seed_ws, rng)
    _fan_in_edges(net, memory, concept, min(15, n_mem), mem_seed_ws, rng)
    _fan_in_edges(net, memory, motor, min(15, n_mem), seed_ws, rng)
    _fan_in_edges(net, sensory, memory, min(10, n_io), mem_seed_ws, rng)

    # ── Instinct: Copy pathway (reflex arc) ─────────────────────────
    # Direct 1:1 sensory[cell,color] -> motor[cell,color].
    # Strong enough to be the default behavior (like a newborn reflex).
    # Learning later modifies this to compute transformations.
    COPY_W = 0.3
    copy_src = sensory.copy()
    copy_dst = motor.copy()
    copy_w = np.full(len(copy_src), COPY_W)
    idx = net._edge_count
    n_new = len(copy_src)
    net._ensure_capacity(idx + n_new)
    net._edge_src[idx:idx + n_new] = copy_src
    net._edge_dst[idx:idx + n_new] = copy_dst
    net._edge_w[idx:idx + n_new] = copy_w
    net._edge_count += n_new
    net._csr_dirty = True

    # ── Motor self-recurrence (working memory) ───────────────────────
    # Each motor neuron connects to itself. This creates "stickiness":
    # once a pattern is established on motor neurons, it persists even
    # after the driving signal is removed. Like prefrontal recurrent
    # loops that maintain working memory.
    SELF_W = 0.9
    self_src = motor.copy()
    self_dst = motor.copy()
    self_w = np.full(len(self_src), SELF_W)
    idx2 = net._edge_count
    n_self = len(self_src)
    net._ensure_capacity(idx2 + n_self)
    net._edge_src[idx2:idx2 + n_self] = self_src
    net._edge_dst[idx2:idx2 + n_self] = self_dst
    net._edge_w[idx2:idx2 + n_self] = self_w
    net._edge_count += n_self
    net._csr_dirty = True

    # Motor neurons: slow leak so patterns persist (working memory)
    leak_rates[motor] = 0.1

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


def _inter_layer_rf_edges(
    net: DNG,
    src_layer: np.ndarray,
    dst_layer: np.ndarray,
    grid_h: int,
    grid_w: int,
    n_cells: int,
    rf_radius: int,
    long_range_frac: float,
    ws: float,
    rng: np.random.Generator,
) -> None:
    """
    Connect two internal layers with spatially structured RFs (vectorized).

    Each dst_layer neuron is assigned to a grid cell (cycling). It receives
    connections from all src_layer neurons assigned to cells within rf_radius,
    plus a fraction of random long-range connections.
    """
    n_src = len(src_layer)
    n_dst = len(dst_layer)
    if n_src == 0 or n_dst == 0:
        return

    src_cell = np.arange(n_src) % n_cells
    dst_cell = np.arange(n_dst) % n_cells

    cell_row = np.arange(n_cells) // grid_w
    cell_col = np.arange(n_cells) % grid_w

    src_rows = cell_row[src_cell]
    src_cols = cell_col[src_cell]
    dst_rows = cell_row[dst_cell]
    dst_cols = cell_col[dst_cell]

    # Vectorized spatial proximity: for each dst, find all src within radius
    # Use broadcasting: (n_dst, 1) vs (1, n_src) -> (n_dst, n_src)
    row_dist = np.abs(dst_rows[:, None] - src_rows[None, :])
    col_dist = np.abs(dst_cols[:, None] - src_cols[None, :])
    local_mask = (row_dist <= rf_radius) & (col_dist <= rf_radius)

    # Add long-range: for each dst, sample random src indices
    n_local_per_dst = local_mask.sum(axis=1)
    avg_local = max(1, int(n_local_per_dst.mean()))
    n_long = max(1, int(avg_local * long_range_frac))
    long_range_idx = rng.integers(0, n_src, size=(n_dst, n_long))
    long_mask = np.zeros((n_dst, n_src), dtype=bool)
    np.put_along_axis(long_mask, long_range_idx, True, axis=1)

    combined = local_mask | long_mask
    dst_idx, src_idx = np.where(combined)

    all_src = src_layer[src_idx].astype(np.int32)
    all_dst = dst_layer[dst_idx].astype(np.int32)

    valid = all_src != all_dst
    sel_src, sel_dst = all_src[valid], all_dst[valid]

    if len(sel_src) == 0:
        return

    magnitudes = np.abs(rng.normal(0, ws, size=len(sel_src)))
    signs = np.where(net._mask_I[sel_src], -1.0, 1.0)
    sel_w = signs * magnitudes

    n_new = len(sel_src)
    idx = net._edge_count
    net._ensure_capacity(idx + n_new)
    net._edge_src[idx:idx + n_new] = sel_src
    net._edge_dst[idx:idx + n_new] = sel_dst
    net._edge_w[idx:idx + n_new] = sel_w
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
