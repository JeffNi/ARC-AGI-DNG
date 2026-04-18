"""
GPU-accelerated kernels using CuPy.

Provides drop-in replacements for the Numba CPU kernels in numba_kernels.py.
Auto-detection: import `use_gpu` to check availability.

Hybrid approach: GPU cuSPARSE for the expensive sparse matmul, CPU Numba
for WTA (tiny arrays, ~200 columns of ~15 neurons each — GPU kernel launch
overhead dominates). Data is uploaded once per step(), with CPU<->GPU transfers
only for the ~5600-element rate vector per sub-step.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    _HAS_GPU = True
    cp.array([0.0])
except Exception:
    _HAS_GPU = False

GPU_NODE_THRESHOLD = 2000


def use_gpu(n_nodes: int) -> bool:
    return _HAS_GPU and n_nodes >= GPU_NODE_THRESHOLD


def _free_gpu_pool():
    """Release unused GPU memory back to the driver."""
    if _HAS_GPU:
        cp.get_default_memory_pool().free_all_blocks()


def gpu_run_steps_v2(
    V, r, prev_r, f, threshold, leak, excitability, adaptation,
    edge_src, edge_dst, edge_w, n_edges, inh_scale,
    signal, noise_scale,
    DA, eta, w_max, plasticity_interval,
    max_rate, f_rate_eff, f_decay, f_max,
    adapt_rate, adapt_decay,
    col_pool, col_sizes, col_offsets, n_cols, wta_k_frac,
    mem_pool_indices, mem_wta_k,
    l3_pool_indices, l3_wta_k,
    has_signal, n_nodes, n_steps,
    noise_matrix,
    motor_start, n_motor_cells, n_colors,
    edge_plastic,
    edge_elig, elig_decay,
    refractory_mask,
    eta_local,
    r_pre_wta,
    node_col_id,
    col_mean_r,
    ema_rate_dendritic,
    csr_indptr, csr_indices, csr_data,
):
    """GPU dynamics+plasticity matching the CPU run_steps_plastic behavior.

    Uses cuSPARSE for the sparse matmul with active-input normalization.
    WTA and column means run on CPU (Numba) where they're faster.
    """
    from .numba_kernels import wta_columnar, wta_pool, wta_pool_soft, wta_motor_cells as _wta_motor

    _free_gpu_pool()

    # Upload weight matrix (CSR for cuSPARSE)
    d_csr_data = cp.asarray(csr_data)
    d_indices = cp.asarray(csr_indices)
    d_indptr = cp.asarray(csr_indptr)
    W_gpu = cpsp.csr_matrix(
        (d_csr_data, d_indices, d_indptr), shape=(n_nodes, n_nodes),
    )

    # Upload node arrays
    d_V = cp.asarray(V)
    d_r = cp.asarray(r)
    d_f = cp.asarray(f)
    d_threshold = cp.asarray(threshold)
    d_leak = cp.asarray(leak)
    d_exc = cp.asarray(excitability)
    d_adaptation = cp.asarray(adaptation)
    d_max_rate = cp.asarray(max_rate)
    d_adapt_rate = cp.asarray(adapt_rate)
    d_refrac = cp.asarray(refractory_mask)
    d_signal = cp.asarray(signal) if has_signal else None
    d_noise = cp.asarray(noise_matrix)
    d_prev_r = cp.asarray(prev_r)

    # Upload edge arrays for plasticity
    d_edge_src = cp.asarray(edge_src)
    d_edge_dst = cp.asarray(edge_dst)
    d_edge_w = cp.asarray(edge_w)
    d_edge_plastic = cp.asarray(edge_plastic)
    d_edge_elig = cp.asarray(edge_elig)
    d_node_col_id = cp.asarray(node_col_id)
    d_r_pre_wta = cp.asarray(r_pre_wta)
    d_col_mean_r = cp.asarray(col_mean_r)
    d_ema_dend = cp.asarray(ema_rate_dendritic)

    # Pre-compute constants
    one_minus_leak = 1.0 - d_leak
    one_minus_fdecay = 1.0 - f_decay
    adapt_decay_val = 1.0 - adapt_decay
    _EPS = 1e-6
    _ACTIVE_THRESH = 0.001

    # Pre-allocate temp buffers
    d_buf = cp.empty(n_nodes, dtype=cp.float64)
    d_syn = cp.empty(n_nodes, dtype=cp.float64)

    for s in range(n_steps):
        # Active-input normalized spmv
        cp.multiply(d_r, 1.0 + d_f, out=d_buf)
        d_buf_masked = cp.where(d_buf > _ACTIVE_THRESH, d_buf, 0.0)
        cp.copyto(d_syn, W_gpu @ d_buf_masked)

        # Count active inputs per destination from CSR structure
        active_data = (d_buf[d_indices] > _ACTIVE_THRESH).astype(cp.float64)
        W_count = cpsp.csr_matrix(
            (active_data, d_indices, d_indptr), shape=(n_nodes, n_nodes),
        )
        n_active = cp.asarray(W_count.sum(axis=1)).ravel()
        d_syn /= cp.sqrt(n_active + 1.0)

        # Node dynamics
        d_prev_r[:] = d_r
        d_V[:] = one_minus_leak * d_V + d_leak * d_exc * d_syn
        if d_signal is not None:
            d_V += d_signal
        d_V += d_noise[s] * noise_scale
        d_V -= d_adaptation
        cp.clip(d_V, -1.0, 5.0, out=d_V)

        d_r[:] = cp.clip(d_V - d_threshold, 0.0, None)
        cp.minimum(d_r, d_max_rate, out=d_r)
        refrac_mask = d_refrac & (d_prev_r > 0.8 * d_max_rate)
        d_r[refrac_mask] *= 0.1

        cp.clip(d_f * one_minus_fdecay + f_rate_eff * d_r, 0.0, f_max, out=d_f)
        cp.clip(
            d_adaptation * adapt_decay_val + d_adapt_rate * d_r, 0.0, 5.0,
            out=d_adaptation,
        )

        # pre-WTA rates on GPU
        d_r_pre_wta[:] = d_r

        # WTA on CPU (Numba — much faster for 200 tiny columns)
        r_cpu = cp.asnumpy(d_r)
        r_pre_cpu = cp.asnumpy(d_r_pre_wta)
        wta_columnar(r_cpu, col_pool, col_sizes, col_offsets, n_cols, wta_k_frac)
        wta_pool(r_cpu, mem_pool_indices, mem_wta_k)
        wta_pool_soft(r_cpu, l3_pool_indices, l3_wta_k)
        _wta_motor(r_cpu, motor_start, n_motor_cells, n_colors)
        d_r = cp.asarray(r_cpu)

        # Column mean for plasticity (CPU)
        for col in range(n_cols):
            o = col_offsets[col]
            sz = col_sizes[col]
            if sz > 0:
                col_mean_r[col] = np.mean(r_pre_cpu[col_pool[o:o + sz]])
        d_col_mean_r = cp.asarray(col_mean_r)

        # Eligibility trace update (GPU — vectorized over all edges)
        d_edge_elig *= elig_decay
        pre_r = d_r[d_edge_src]
        post_r = d_r[d_edge_dst]
        active_edges = (pre_r > 0.001) | (post_r > 0.001)
        d_edge_elig += cp.where(active_edges, pre_r * post_r, 0.0)

        # Competitive Hebbian plasticity
        if plasticity_interval > 0 and s > 0 and s % plasticity_interval == 0:
            post_wta = d_r[d_edge_dst]
            pre_r = d_r[d_edge_src]
            r_pre_src = d_r_pre_wta[d_edge_dst]
            dst_col = d_node_col_id[d_edge_dst]
            h_star = cp.where(
                dst_col >= 0,
                d_col_mean_r[cp.clip(dst_col, 0, max(n_cols - 1, 0))],
                0.0,
            )

            delta = r_pre_src - h_star
            g_h = cp.where(delta > 0.0, delta * delta, 0.0)

            w_abs = cp.abs(d_edge_w)
            dw = eta_local * g_h * (pre_r - w_abs)
            dw += eta * DA * g_h * pre_r
            dw = cp.where(d_edge_w < 0.0, -dw, dw)
            active = ((pre_r > 0.001) | (post_wta > 0.001)) & d_edge_plastic
            dw = cp.where(active, dw, 0.0)
            d_edge_w += dw

            pos = d_edge_w > 0.0
            neg = d_edge_w < 0.0
            d_edge_w = cp.where(pos, cp.clip(d_edge_w, _EPS, w_max), d_edge_w)
            d_edge_w = cp.where(neg, cp.clip(d_edge_w, -w_max, -_EPS), d_edge_w)

    # Transfer results back to CPU
    V[:] = cp.asnumpy(d_V)
    r[:] = cp.asnumpy(d_r)
    prev_r[:] = cp.asnumpy(d_prev_r)
    f[:] = cp.asnumpy(d_f)
    adaptation[:] = cp.asnumpy(d_adaptation)
    edge_w[:] = cp.asnumpy(d_edge_w)
    edge_elig[:] = cp.asnumpy(d_edge_elig)
    r_pre_wta[:] = cp.asnumpy(d_r_pre_wta)
    col_mean_r[:] = col_mean_r  # already updated in-place above
    ema_rate_dendritic[:] = cp.asnumpy(d_ema_dend)

    _free_gpu_pool()
