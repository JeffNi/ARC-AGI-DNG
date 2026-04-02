"""
Activation dynamics modulated by neuromodulators.

Uses Numba JIT-compiled kernels for the hot path.
First call triggers ~1s compilation, then runs as native machine code.
"""

from __future__ import annotations

import numpy as np

from .graph import DNG, Region
from .encoding import NUM_COLORS
from .numba_kernels import run_steps, run_steps_record

_REGION_ABSTRACT = list(Region).index(Region.ABSTRACT)
_REGION_MEMORY = list(Region).index(Region.MEMORY)

_rng = np.random.Generator(np.random.PCG64())

_net_cache: dict = {}


def _get_cached(net: DNG):
    """Cache pool indices, motor info, and empty signal for a network."""
    key = id(net.regions)
    if key not in _net_cache:
        pool_mask = (net.regions == _REGION_ABSTRACT)
        pool_idx = np.where(pool_mask)[0].astype(np.int64)
        mem_mask = (net.regions == _REGION_MEMORY)
        mem_pool_idx = np.where(mem_mask)[0].astype(np.int64)
        empty_signal = np.zeros(net.n_nodes, dtype=np.float64)
        motor_start = int(net.output_nodes[0]) if len(net.output_nodes) > 0 else 0
        n_motor_cells = len(net.output_nodes) // NUM_COLORS
        _net_cache[key] = (pool_idx, mem_pool_idx, empty_signal, motor_start, n_motor_cells)
    return _net_cache[key]


def think(
    net: DNG,
    signal: np.ndarray | None = None,
    steps: int = 50,
    noise_std: float = 0.05,
) -> None:
    """Run N steps of continuous dynamics (Numba-compiled)."""
    if steps <= 0:
        return

    pool_idx, mem_pool_idx, empty_signal, motor_start, n_motor_cells = _get_cached(net)

    W = net.get_weight_matrix()
    W_data = np.ascontiguousarray(W.data)
    W_indices = np.ascontiguousarray(W.indices.astype(np.int64))
    W_indptr = np.ascontiguousarray(W.indptr.astype(np.int64))

    noise_eff = noise_std * max(0.2, 1.0 - 0.5 * net.ne)
    noise_matrix = _rng.standard_normal((steps, net.n_nodes))

    sig = signal if signal is not None else empty_signal
    has_signal = signal is not None

    k_frac = net.wta_k_frac * max(0.5, 1.0 - 0.3 * net.ne)
    wta_k = max(1, int(len(pool_idx) * k_frac))
    mem_k = max(1, int(len(mem_pool_idx) * 0.05))

    f_rate_eff = net.f_rate * (1.0 + net.ach)

    run_steps(
        net.V, net.r, net.prev_r, net.f,
        net.threshold, net.leak_rates, net.excitability, net.adaptation,
        W_data, W_indices, W_indptr,
        sig, noise_eff,
        net.max_rate, f_rate_eff, net.f_decay, net.f_max,
        net.adapt_rate, net.adapt_decay,
        pool_idx, wta_k,
        mem_pool_idx, mem_k,
        has_signal, net.n_nodes, steps,
        noise_matrix,
        motor_start, n_motor_cells, NUM_COLORS,
    )


def step(
    net: DNG,
    signal: np.ndarray | None = None,
    noise_std: float = 0.05,
) -> None:
    """Single timestep (delegates to think with steps=1)."""
    think(net, signal=signal, steps=1, noise_std=noise_std)


def record_think(
    net: DNG,
    signal: np.ndarray | None = None,
    steps: int = 50,
    noise_std: float = 0.05,
) -> np.ndarray:
    """
    Run N steps and accumulate edge correlations (all compiled).
    Returns correlation array of length edge_capacity.
    """
    n_edges = net._edge_count
    corr = np.zeros(net._edge_capacity, dtype=np.float64)
    if n_edges == 0 or steps <= 0:
        return corr

    pool_idx, mem_pool_idx, empty_signal, motor_start, n_motor_cells = _get_cached(net)

    W = net.get_weight_matrix()
    W_data = np.ascontiguousarray(W.data)
    W_indices = np.ascontiguousarray(W.indices.astype(np.int64))
    W_indptr = np.ascontiguousarray(W.indptr.astype(np.int64))

    noise_eff = noise_std * max(0.2, 1.0 - 0.5 * net.ne)
    noise_matrix = _rng.standard_normal((steps, net.n_nodes))

    sig = signal if signal is not None else empty_signal
    has_signal = signal is not None

    k_frac = net.wta_k_frac * max(0.5, 1.0 - 0.3 * net.ne)
    wta_k = max(1, int(len(pool_idx) * k_frac))
    mem_k = max(1, int(len(mem_pool_idx) * 0.05))

    f_rate_eff = net.f_rate * (1.0 + net.ach)

    run_steps_record(
        net.V, net.r, net.prev_r, net.f,
        net.threshold, net.leak_rates, net.excitability, net.adaptation,
        W_data, W_indices, W_indptr,
        sig, noise_eff,
        net.max_rate, f_rate_eff, net.f_decay, net.f_max,
        net.adapt_rate, net.adapt_decay,
        pool_idx, wta_k,
        mem_pool_idx, mem_k,
        has_signal, net.n_nodes, steps,
        noise_matrix,
        net._edge_src[:n_edges].astype(np.int64),
        net._edge_dst[:n_edges].astype(np.int64),
        corr[:n_edges],
        n_edges,
        motor_start, n_motor_cells, NUM_COLORS,
    )

    return corr
