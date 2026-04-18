"""
SciPy/NumPy vectorized kernels — drop-in replacements for numba_kernels.py.

Uses scipy sparse CSR matvec (OpenBLAS/MKL-backed) instead of hand-rolled
numba edge loops. ~3-6x faster than the numba COO path for our matrix sizes
(~5K nodes, ~1M edges) because:
  - CSR row-contiguous storage → better cache behaviour
  - BLAS sparse matvec uses SIMD/AVX
  - NumPy vectorized ops avoid Python-level element loops

Trade-off: Python loop over time steps adds ~10μs overhead per step,
negligible for n_nodes > 1000.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .numba_kernels import wta_columnar, wta_pool, wta_pool_soft, wta_motor_cells


def run_steps_plastic_scipy(
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
):
    """Vectorized dynamics with competitive Hebbian plasticity.

    Same semantics as numba_kernels.run_steps_plastic, but uses scipy CSR
    matvec for synaptic input and numpy vectorized ops for state/plasticity.
    WTA kernels remain numba-compiled (small arrays, already fast).
    """
    _EPS = 1e-6
    one_minus_leak = 1.0 - leak
    one_minus_fdecay = 1.0 - f_decay
    adapt_retain = 1.0 - adapt_decay
    elig_retain = elig_decay

    # Build CSR and topology helper once (refreshed after plasticity)
    W, W_struct = _build_csr(edge_w, edge_dst, edge_src, n_edges, n_nodes, inh_scale)

    # Precompute refractory threshold per node (0.8 * max_rate)
    refract_thresh = 0.8 * max_rate

    # Pre-index plastic edges for vectorized plasticity
    plastic_mask = edge_plastic.astype(bool)

    for s in range(n_steps):
        buf = r * (1.0 + f)

        # --- Synaptic input: CSR matvec (the main speedup) ---
        buf_active = np.where(buf > 0.001, buf, 0.0)
        syn = W.dot(buf_active)
        # Active-input normalization: approximate with structural degree
        # since recounting per step is expensive. Uses the binary topology
        # matrix to count which presynaptic inputs are actually active.
        n_active = W_struct.dot((buf > 0.001).astype(np.float64))
        denom = np.sqrt(n_active + 1.0)
        syn /= denom

        # --- State update (vectorized numpy) ---
        prev_r[:] = r

        V[:] = one_minus_leak * V + leak * excitability * syn
        if has_signal:
            V += signal
        V += noise_matrix[s] * noise_scale
        V -= adaptation
        np.clip(V, -1.0, 5.0, out=V)

        r[:] = V - threshold
        np.clip(r, 0.0, None, out=r)
        np.minimum(r, max_rate, out=r)

        # Refractory suppression
        refract = refractory_mask & (prev_r > refract_thresh)
        r[refract] *= 0.1

        r_pre_wta[:] = r

        # --- Facilitation ---
        f *= one_minus_fdecay
        f += f_rate_eff * r
        np.clip(f, 0.0, f_max, out=f)

        # --- Adaptation ---
        adaptation *= adapt_retain
        adaptation += adapt_rate * r
        np.clip(adaptation, 0.0, 5.0, out=adaptation)

        # --- Per-column plasticity threshold (before WTA) ---
        for c in range(n_cols):
            off = col_offsets[c]
            sz = col_sizes[c]
            if sz == 0:
                col_mean_r[c] = 0.0
                continue
            vals = r_pre_wta[col_pool[off:off + sz]]
            mean_v = vals.mean()
            col_mean_r[c] = mean_v + 0.5 * (vals.max() - mean_v)

        # --- WTA (numba-compiled, fast for small pools) ---
        wta_columnar(r, col_pool, col_sizes, col_offsets, n_cols, wta_k_frac)
        wta_pool(r, mem_pool_indices, mem_wta_k)
        wta_pool_soft(r, l3_pool_indices, l3_wta_k)
        wta_motor_cells(r, motor_start, n_motor_cells, n_colors)

        # --- Eligibility traces (vectorized) ---
        edge_elig *= elig_retain
        pre_r_e = r[edge_src]
        post_r_e = r[edge_dst]
        co = pre_r_e * post_r_e
        elig_update = plastic_mask & (co > 0.001)
        edge_elig[elig_update] += co[elig_update]

        # --- Competitive Hebbian plasticity (vectorized K-H rule) ---
        if plasticity_interval > 0 and s > 0 and s % plasticity_interval == 0:
            _kh_plasticity_vectorized(
                edge_src, edge_dst, edge_w, n_edges,
                plastic_mask, r_pre_wta, node_col_id, col_mean_r,
                ema_rate_dendritic, eta_local, eta, DA, w_max, _EPS,
            )
            # Rebuild CSR with updated weights
            W, W_struct = _build_csr(
                edge_w, edge_dst, edge_src, n_edges, n_nodes, inh_scale,
                W_struct,
            )


def _build_csr(edge_w, edge_dst, edge_src, n_edges, n_nodes, inh_scale,
               existing_struct=None):
    """Build weighted CSR and (optionally reuse) binary topology CSR."""
    w = edge_w.copy()
    if inh_scale != 1.0:
        neg = w < 0
        w[neg] *= inh_scale
    W = csr_matrix((w, (edge_dst, edge_src)), shape=(n_nodes, n_nodes))

    if existing_struct is not None:
        W_struct = existing_struct
    else:
        ones = np.ones(n_edges, dtype=np.float64)
        W_struct = csr_matrix((ones, (edge_dst, edge_src)), shape=(n_nodes, n_nodes))
    return W, W_struct


def _kh_plasticity_vectorized(
    edge_src, edge_dst, edge_w, n_edges,
    plastic_mask, r_pre_wta, node_col_id, col_mean_r,
    ema_rate_dendritic, eta_local, eta, DA, w_max, eps,
):
    """Vectorized Krotov-Hopfield competitive plasticity update."""
    pre_r = r_pre_wta[edge_src]
    post_r = r_pre_wta[edge_dst]

    active = ((pre_r > 0.001) | (post_r > 0.001)) & plastic_mask
    if not np.any(active):
        return

    # Column thresholds per edge
    dst_col = node_col_id[edge_dst]
    n_cols = len(col_mean_r)
    safe_col = np.clip(dst_col, 0, max(n_cols - 1, 0))
    h_star = np.where(dst_col >= 0, col_mean_r[safe_col], 0.0)

    # Conscience: frequent winners get a higher bar
    dst_ema = ema_rate_dendritic[edge_dst]
    conscience = dst_ema - h_star
    h_star = np.where(conscience > 0.0, h_star + conscience, h_star)

    delta = post_r - h_star
    g_h = np.where(delta > 0.0, delta * delta, 0.0)

    update = active & (g_h > 1e-8)
    if not np.any(update):
        return

    # Only compute dw for edges that will be updated
    idx = np.where(update)[0]
    w_old = edge_w[idx]
    w_abs = np.abs(w_old)
    g = g_h[idx]
    pr = pre_r[idx]

    dw = eta_local * g * (pr - w_abs)
    dw += eta * DA * g * pr
    dw = np.where(w_old < 0.0, -dw, dw)

    w_new = w_old + dw

    # Clamp: positive stays positive, negative stays negative
    pos = w_old > 0.0
    neg = w_old < 0.0
    w_new = np.where(pos, np.clip(w_new, eps, w_max), w_new)
    w_new = np.where(neg, np.clip(w_new, -w_max, -eps), w_new)

    edge_w[idx] = w_new
