"""
Numba JIT-compiled kernels for the DNG hot paths.

Phase 1 additions:
  - Refractory suppression (all kernels)
  - Per-node max_rate and adapt_rate arrays
  - Eligibility trace accumulation (run_steps_plastic)
"""

from __future__ import annotations

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def csr_matvec(data, indices, indptr, x, out):
    """CSR sparse matrix-vector multiply: out = A @ x"""
    n_rows = len(indptr) - 1
    for i in range(n_rows):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += data[j] * x[indices[j]]
        out[i] = s


@numba.jit(nopython=True, cache=True)
def compute_facilitated(r, f, out, n):
    """out = r * (1 + f)"""
    for i in range(n):
        out[i] = r[i] * (1.0 + f[i])


@numba.jit(nopython=True, cache=True)
def update_state(
    V, r, prev_r, f, threshold, leak, excitability, adaptation,
    synaptic_input, signal, noise_buf,
    max_rate, f_rate_eff, f_decay, f_max,
    adapt_rate, adapt_decay, has_signal, n,
):
    """All element-wise state updates in a single compiled pass."""
    for i in range(n):
        prev_r[i] = r[i]

        vi = (1.0 - leak[i]) * V[i] + leak[i] * excitability[i] * synaptic_input[i]
        if has_signal:
            vi += signal[i]
        vi += noise_buf[i]
        vi -= adaptation[i]
        V[i] = vi

        ri = vi - threshold[i]
        if ri < 0.0:
            ri = 0.0
        elif ri > max_rate[i]:
            ri = max_rate[i]

        # Refractory suppression: if was strongly active last step, suppress
        if prev_r[i] > 0.8 * max_rate[i]:
            ri *= 0.1

        r[i] = ri

        fi = f[i] * (1.0 - f_decay) + f_rate_eff * ri
        if fi < 0.0:
            fi = 0.0
        elif fi > f_max:
            fi = f_max
        f[i] = fi

        ai = adaptation[i] * (1.0 - adapt_decay) + adapt_rate[i] * ri
        if ai < 0.0:
            ai = 0.0
        elif ai > 5.0:
            ai = 5.0
        adaptation[i] = ai


@numba.jit(nopython=True, cache=True)
def wta_pool(r, pool_indices, k):
    """Zero out all but top-k rates in the pool."""
    n = len(pool_indices)
    if n == 0 or k >= n:
        return

    order = np.empty(n, dtype=numba.int64)
    for i in range(n):
        order[i] = i
    for i in range(1, n):
        key_idx = order[i]
        key_val = r[pool_indices[key_idx]]
        j = i - 1
        while j >= 0 and r[pool_indices[order[j]]] < key_val:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key_idx

    for i in range(k, n):
        r[pool_indices[order[i]]] = 0.0


@numba.jit(nopython=True, cache=True)
def wta_motor_cells(r, motor_start, n_cells, n_colors):
    """Per-cell WTA for motor neurons: only strongest color survives."""
    for cell in range(n_cells):
        base = motor_start + cell * n_colors
        best_val = -1.0
        best_idx = 0
        for c in range(n_colors):
            val = r[base + c]
            if val > best_val:
                best_val = val
                best_idx = c
        for c in range(n_colors):
            if c != best_idx:
                r[base + c] = 0.0


@numba.jit(nopython=True, cache=True)
def accumulate_corr(r, edge_src, edge_dst, corr, n_edges):
    """Accumulate pre*post correlations for all edges."""
    for i in range(n_edges):
        corr[i] += r[edge_src[i]] * r[edge_dst[i]]


@numba.jit(nopython=True, cache=True)
def run_steps(
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
    refractory_mask,
):
    """Run multiple steps entirely in compiled code."""
    buf = np.empty(n_nodes)

    for s in range(n_steps):
        for i in range(n_nodes):
            buf[i] = r[i] * (1.0 + f[i])

        syn = np.empty(n_nodes)
        n_rows = len(W_indptr) - 1
        for i in range(n_rows):
            acc = 0.0
            for j in range(W_indptr[i], W_indptr[i + 1]):
                acc += W_data[j] * buf[W_indices[j]]
            syn[i] = acc

        for i in range(n_nodes):
            prev_r[i] = r[i]

            vi = (1.0 - leak[i]) * V[i] + leak[i] * excitability[i] * syn[i]
            if has_signal:
                vi += signal[i]
            vi += noise_matrix[s, i] * noise_scale
            vi -= adaptation[i]
            V[i] = vi

            ri = vi - threshold[i]
            if ri < 0.0:
                ri = 0.0
            elif ri > max_rate[i]:
                ri = max_rate[i]

            if refractory_mask[i] and prev_r[i] > 0.8 * max_rate[i]:
                ri *= 0.1

            r[i] = ri

            fi = f[i] * (1.0 - f_decay) + f_rate_eff * ri
            if fi < 0.0:
                fi = 0.0
            elif fi > f_max:
                fi = f_max
            f[i] = fi

            ai = adaptation[i] * (1.0 - adapt_decay) + adapt_rate[i] * ri
            if ai < 0.0:
                ai = 0.0
            elif ai > 5.0:
                ai = 5.0
            adaptation[i] = ai

        wta_pool(r, pool_indices, wta_k)
        wta_pool(r, mem_pool_indices, mem_wta_k)
        wta_motor_cells(r, motor_start, n_motor_cells, n_colors)


@numba.jit(nopython=True, cache=True)
def run_steps_plastic(
    V, r, prev_r, f, threshold, leak, excitability, adaptation,
    edge_src, edge_dst, edge_w, n_edges, inh_scale,
    signal, noise_scale,
    DA, eta, w_max, bcm_theta, plasticity_interval,
    max_rate, f_rate_eff, f_decay, f_max,
    adapt_rate, adapt_decay,
    pool_indices, wta_k,
    mem_pool_indices, mem_wta_k,
    has_signal, n_nodes, n_steps,
    noise_matrix,
    motor_start, n_motor_cells, n_colors,
    edge_plastic,
    edge_elig, elig_decay,
    refractory_mask,
):
    """
    Dynamics with continuous local plasticity + eligibility trace accumulation.

    Every step: eligibility traces accumulate from pre*post co-activity.
    Every plasticity_interval steps: Hebbian+DA update on plastic edges.
    """
    _EPS = 1e-6
    buf = np.empty(n_nodes)
    syn = np.empty(n_nodes)

    for s in range(n_steps):
        for i in range(n_nodes):
            buf[i] = r[i] * (1.0 + f[i])

        for i in range(n_nodes):
            syn[i] = 0.0
        for e in range(n_edges):
            w_eff = edge_w[e]
            if w_eff < 0.0:
                w_eff *= inh_scale
            syn[edge_dst[e]] += w_eff * buf[edge_src[e]]

        for i in range(n_nodes):
            prev_r[i] = r[i]

            vi = (1.0 - leak[i]) * V[i] + leak[i] * excitability[i] * syn[i]
            if has_signal:
                vi += signal[i]
            vi += noise_matrix[s, i] * noise_scale
            vi -= adaptation[i]
            V[i] = vi

            ri = vi - threshold[i]
            if ri < 0.0:
                ri = 0.0
            elif ri > max_rate[i]:
                ri = max_rate[i]

            if refractory_mask[i] and prev_r[i] > 0.8 * max_rate[i]:
                ri *= 0.1

            r[i] = ri

            fi = f[i] * (1.0 - f_decay) + f_rate_eff * ri
            if fi < 0.0:
                fi = 0.0
            elif fi > f_max:
                fi = f_max
            f[i] = fi

            ai = adaptation[i] * (1.0 - adapt_decay) + adapt_rate[i] * ri
            if ai < 0.0:
                ai = 0.0
            elif ai > 5.0:
                ai = 5.0
            adaptation[i] = ai

        wta_pool(r, pool_indices, wta_k)
        wta_pool(r, mem_pool_indices, mem_wta_k)
        wta_motor_cells(r, motor_start, n_motor_cells, n_colors)

        # Eligibility trace accumulation — only for plastic edges with strong co-activity
        for e in range(n_edges):
            edge_elig[e] *= elig_decay
            if not edge_plastic[e]:
                continue
            pre_r = r[edge_src[e]]
            post_r = r[edge_dst[e]]
            if pre_r > 0.15 and post_r > 0.15:
                edge_elig[e] += pre_r * post_r

        # Continuous Hebbian+DA plasticity
        if plasticity_interval > 0 and s > 0 and s % plasticity_interval == 0:
            for e in range(n_edges):
                if not edge_plastic[e]:
                    continue
                pre_r = r[edge_src[e]]
                post_r = r[edge_dst[e]]
                if pre_r < 0.01 or post_r < 0.01:
                    continue
                theta = bcm_theta[edge_dst[e]]
                dw = eta * DA * pre_r * (post_r - theta)
                w_old = edge_w[e]
                if w_old < 0.0:
                    dw = -dw
                w_new = w_old + dw
                if w_old > 0.0:
                    if w_new < _EPS:
                        w_new = _EPS
                    elif w_new > w_max:
                        w_new = w_max
                else:
                    if w_new > -_EPS:
                        w_new = -_EPS
                    elif w_new < -w_max:
                        w_new = -w_max
                edge_w[e] = w_new


@numba.jit(nopython=True, cache=True)
def run_steps_record(
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
    refractory_mask,
):
    """Run steps + accumulate correlations, all compiled."""
    buf = np.empty(n_nodes)
    syn = np.empty(n_nodes)
    inv_steps = 1.0 / n_steps

    for s in range(n_steps):
        for i in range(n_nodes):
            buf[i] = r[i] * (1.0 + f[i])

        n_rows = len(W_indptr) - 1
        for i in range(n_rows):
            acc = 0.0
            for j in range(W_indptr[i], W_indptr[i + 1]):
                acc += W_data[j] * buf[W_indices[j]]
            syn[i] = acc

        for i in range(n_nodes):
            prev_r[i] = r[i]
            vi = (1.0 - leak[i]) * V[i] + leak[i] * excitability[i] * syn[i]
            if has_signal:
                vi += signal[i]
            vi += noise_matrix[s, i] * noise_scale
            vi -= adaptation[i]
            V[i] = vi

            ri = vi - threshold[i]
            if ri < 0.0:
                ri = 0.0
            elif ri > max_rate[i]:
                ri = max_rate[i]

            if refractory_mask[i] and prev_r[i] > 0.8 * max_rate[i]:
                ri *= 0.1

            r[i] = ri

            fi = f[i] * (1.0 - f_decay) + f_rate_eff * ri
            if fi < 0.0:
                fi = 0.0
            elif fi > f_max:
                fi = f_max
            f[i] = fi

            ai = adaptation[i] * (1.0 - adapt_decay) + adapt_rate[i] * ri
            if ai < 0.0:
                ai = 0.0
            elif ai > 5.0:
                ai = 5.0
            adaptation[i] = ai

        wta_pool(r, pool_indices, wta_k)
        wta_pool(r, mem_pool_indices, mem_wta_k)
        wta_motor_cells(r, motor_start, n_motor_cells, n_colors)

        for i in range(n_edges):
            corr[i] += r[edge_src[i]] * r[edge_dst[i]]

    for i in range(n_edges):
        corr[i] *= inv_steps
