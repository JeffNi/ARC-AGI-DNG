"""
GPU-accelerated kernels using CuPy.

Provides drop-in replacements for the Numba CPU kernels in numba_kernels.py.
Auto-detection: import `use_gpu` to check availability, call `gpu_run_steps`
and `gpu_run_steps_record` as replacements for CPU versions.

Data stays on CPU (DNG uses NumPy arrays). Each call transfers to GPU,
computes, transfers back. For ~2500+ nodes the GPU sparse matmul dominates
and this is faster than Numba CPU despite the transfer overhead.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    _HAS_GPU = True
    # Warm up: force CuPy to initialize CUDA context now (first call is slow)
    cp.array([0.0])
except Exception:
    _HAS_GPU = False

GPU_NODE_THRESHOLD = 2000


def use_gpu(n_nodes: int) -> bool:
    return _HAS_GPU and n_nodes >= GPU_NODE_THRESHOLD


def _wta_pool_gpu(r: cp.ndarray, pool_indices: cp.ndarray, k: int):
    """WTA on GPU: zero all but top-k in pool."""
    n = len(pool_indices)
    if n == 0 or k >= n:
        return
    pool_vals = r[pool_indices]
    # Partition: indices of top-k values
    threshold_idx = n - k
    partitioned = cp.argpartition(pool_vals, threshold_idx)
    losers = pool_indices[partitioned[:threshold_idx]]
    r[losers] = 0.0


def _wta_motor_gpu(r: cp.ndarray, motor_start: int, n_cells: int, n_colors: int):
    """Per-cell WTA for motor neurons on GPU."""
    if n_cells == 0:
        return
    motor_block = r[motor_start:motor_start + n_cells * n_colors].reshape(n_cells, n_colors)
    winners = cp.argmax(motor_block, axis=1)
    mask = cp.ones_like(motor_block, dtype=cp.bool_)
    mask[cp.arange(n_cells), winners] = False
    motor_block[mask] = 0.0


def gpu_run_steps(
    V, r, prev_r, f, threshold, leak, excitability, adaptation,
    W_data, W_indices, W_indptr,
    signal, noise_scale,
    max_rate, f_rate_eff, f_decay, f_max,
    adapt_rate, adapt_decay,
    pool_indices, wta_k,
    mem_pool_indices, mem_wta_k,
    has_signal, n_nodes, n_steps,
    noise_matrix,
    motor_start, n_motor_cells, n_colors,
):
    """GPU version of run_steps: transfer, compute, transfer back."""
    # Transfer state to GPU
    d_V = cp.asarray(V)
    d_r = cp.asarray(r)
    d_f = cp.asarray(f)
    d_threshold = cp.asarray(threshold)
    d_leak = cp.asarray(leak)
    d_excitability = cp.asarray(excitability)
    d_adaptation = cp.asarray(adaptation)
    d_signal = cp.asarray(signal) if has_signal else None
    d_noise = cp.asarray(noise_matrix)

    # Transfer sparse matrix to GPU
    W_gpu = cpsp.csr_matrix(
        (cp.asarray(W_data), cp.asarray(W_indices), cp.asarray(W_indptr)),
        shape=(n_nodes, n_nodes),
    )

    d_pool = cp.asarray(pool_indices)
    d_mem_pool = cp.asarray(mem_pool_indices)

    one_minus_leak = 1.0 - d_leak
    one_minus_fdecay = 1.0 - f_decay
    adapt_decay_val = 1.0 - adapt_decay

    for s in range(n_steps):
        # Facilitated rates
        buf = d_r * (1.0 + d_f)

        # Sparse matmul (cuSPARSE)
        syn = W_gpu @ buf

        # State update (all vectorized on GPU)
        d_V = one_minus_leak * d_V + d_leak * d_excitability * syn
        if d_signal is not None:
            d_V += d_signal
        d_V += d_noise[s] * noise_scale
        d_V -= d_adaptation

        d_r = cp.clip(d_V - d_threshold, 0.0, max_rate)
        d_f = cp.clip(d_f * one_minus_fdecay + f_rate_eff * d_r, 0.0, f_max)
        d_adaptation = cp.clip(d_adaptation * adapt_decay_val + adapt_rate * d_r, 0.0, 5.0)

        # WTA
        _wta_pool_gpu(d_r, d_pool, wta_k)
        _wta_pool_gpu(d_r, d_mem_pool, mem_wta_k)
        _wta_motor_gpu(d_r, motor_start, n_motor_cells, n_colors)

    # Transfer back
    V[:] = cp.asnumpy(d_V)
    r[:] = cp.asnumpy(d_r)
    prev_r[:] = r  # last step's r is current r
    f[:] = cp.asnumpy(d_f)
    adaptation[:] = cp.asnumpy(d_adaptation)


def gpu_run_steps_record(
    V, r, prev_r, f, threshold, leak, excitability, adaptation,
    W_data, W_indices, W_indptr,
    signal, noise_scale,
    max_rate, f_rate_eff, f_decay, f_max,
    adapt_rate, adapt_decay,
    pool_indices, wta_k,
    mem_pool_indices, mem_wta_k,
    has_signal, n_nodes, n_steps,
    noise_matrix,
    edge_src, edge_dst, corr, n_edges,
    motor_start, n_motor_cells, n_colors,
):
    """GPU version of run_steps_record: includes correlation accumulation."""
    # Transfer state to GPU
    d_V = cp.asarray(V)
    d_r = cp.asarray(r)
    d_f = cp.asarray(f)
    d_threshold = cp.asarray(threshold)
    d_leak = cp.asarray(leak)
    d_excitability = cp.asarray(excitability)
    d_adaptation = cp.asarray(adaptation)
    d_signal = cp.asarray(signal) if has_signal else None
    d_noise = cp.asarray(noise_matrix)

    W_gpu = cpsp.csr_matrix(
        (cp.asarray(W_data), cp.asarray(W_indices), cp.asarray(W_indptr)),
        shape=(n_nodes, n_nodes),
    )

    d_pool = cp.asarray(pool_indices)
    d_mem_pool = cp.asarray(mem_pool_indices)
    d_edge_src = cp.asarray(edge_src)
    d_edge_dst = cp.asarray(edge_dst)
    d_corr = cp.zeros(n_edges, dtype=cp.float64)

    one_minus_leak = 1.0 - d_leak
    one_minus_fdecay = 1.0 - f_decay
    adapt_decay_val = 1.0 - adapt_decay
    inv_steps = 1.0 / n_steps

    for s in range(n_steps):
        buf = d_r * (1.0 + d_f)
        syn = W_gpu @ buf

        d_V = one_minus_leak * d_V + d_leak * d_excitability * syn
        if d_signal is not None:
            d_V += d_signal
        d_V += d_noise[s] * noise_scale
        d_V -= d_adaptation

        d_r = cp.clip(d_V - d_threshold, 0.0, max_rate)
        d_f = cp.clip(d_f * one_minus_fdecay + f_rate_eff * d_r, 0.0, f_max)
        d_adaptation = cp.clip(d_adaptation * adapt_decay_val + adapt_rate * d_r, 0.0, 5.0)

        _wta_pool_gpu(d_r, d_pool, wta_k)
        _wta_pool_gpu(d_r, d_mem_pool, mem_wta_k)
        _wta_motor_gpu(d_r, motor_start, n_motor_cells, n_colors)

        # Accumulate correlations
        d_corr += d_r[d_edge_src] * d_r[d_edge_dst]

    d_corr *= inv_steps

    # Transfer back
    V[:] = cp.asnumpy(d_V)
    r[:] = cp.asnumpy(d_r)
    prev_r[:] = r
    f[:] = cp.asnumpy(d_f)
    adaptation[:] = cp.asnumpy(d_adaptation)
    corr[:n_edges] = cp.asnumpy(d_corr)
