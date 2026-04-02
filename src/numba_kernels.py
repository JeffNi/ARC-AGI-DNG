"""
Numba JIT-compiled kernels for the DNG hot paths.

These replace numpy dispatch overhead with compiled machine code.
First call incurs ~1s compile time, subsequent calls are native speed.
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
    """
    All element-wise state updates in a single compiled pass.
    Replaces ~10 separate numpy calls with one tight loop.
    """
    for i in range(n):
        prev_r[i] = r[i]

        # V update: (1-leak)*V + leak*excitability*synaptic_input + signal + noise - adaptation
        vi = (1.0 - leak[i]) * V[i] + leak[i] * excitability[i] * synaptic_input[i]
        if has_signal:
            vi += signal[i]
        vi += noise_buf[i]
        vi -= adaptation[i]
        V[i] = vi

        # Firing rate: clip(V - threshold, 0, max_rate)
        ri = vi - threshold[i]
        if ri < 0.0:
            ri = 0.0
        elif ri > max_rate:
            ri = max_rate
        r[i] = ri

        # Facilitation: f = f * (1 - decay) + rate_eff * r
        fi = f[i] * (1.0 - f_decay) + f_rate_eff * ri
        if fi < 0.0:
            fi = 0.0
        elif fi > f_max:
            fi = f_max
        f[i] = fi

        # Adaptation: a = a * (1 - decay) + rate * r
        ai = adaptation[i] * (1.0 - adapt_decay) + adapt_rate * ri
        if ai < 0.0:
            ai = 0.0
        elif ai > 5.0:
            ai = 5.0
        adaptation[i] = ai


@numba.jit(nopython=True, cache=True)
def wta_pool(r, pool_indices, k):
    """Zero out all but top-k rates in the pool. Handles ties correctly."""
    n = len(pool_indices)
    if n == 0 or k >= n:
        return

    # Sort indices by rate (descending) to handle ties deterministically
    order = np.empty(n, dtype=numba.int64)
    for i in range(n):
        order[i] = i
    # Simple insertion sort (n is typically <1000, fast enough)
    for i in range(1, n):
        key_idx = order[i]
        key_val = r[pool_indices[key_idx]]
        j = i - 1
        while j >= 0 and r[pool_indices[order[j]]] < key_val:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key_idx

    # Keep top-k, zero the rest
    for i in range(k, n):
        r[pool_indices[order[i]]] = 0.0


@numba.jit(nopython=True, cache=True)
def wta_motor_cells(r, motor_start, n_cells, n_colors):
    """
    Per-cell WTA for motor neurons: within each cell's color group,
    only the strongest color survives (k=1). This creates competitive
    color selection -- the output cell "decides" on one color.
    """
    for cell in range(n_cells):
        base = motor_start + cell * n_colors
        best_val = -1.0
        best_idx = 0
        for c in range(n_colors):
            val = r[base + c]
            if val > best_val:
                best_val = val
                best_idx = c
        # Zero out all but the winner
        for c in range(n_colors):
            if c != best_idx:
                r[base + c] = 0.0


@numba.jit(nopython=True, cache=True)
def accumulate_corr(r, edge_src, edge_dst, corr, n_edges):
    """Accumulate pre*post correlations for all edges (no temp arrays)."""
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
):
    """
    Run multiple steps entirely in compiled code.
    Eliminates Python loop overhead completely.
    """
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
            elif ri > max_rate:
                ri = max_rate
            r[i] = ri

            fi = f[i] * (1.0 - f_decay) + f_rate_eff * ri
            if fi < 0.0:
                fi = 0.0
            elif fi > f_max:
                fi = f_max
            f[i] = fi

            ai = adaptation[i] * (1.0 - adapt_decay) + adapt_rate * ri
            if ai < 0.0:
                ai = 0.0
            elif ai > 5.0:
                ai = 5.0
            adaptation[i] = ai

        wta_pool(r, pool_indices, wta_k)
        wta_pool(r, mem_pool_indices, mem_wta_k)
        wta_motor_cells(r, motor_start, n_motor_cells, n_colors)


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
):
    """
    Run steps + accumulate correlations, all compiled.
    Replaces record_phase + step loop entirely.
    """
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
            elif ri > max_rate:
                ri = max_rate
            r[i] = ri

            fi = f[i] * (1.0 - f_decay) + f_rate_eff * ri
            if fi < 0.0:
                fi = 0.0
            elif fi > f_max:
                fi = f_max
            f[i] = fi

            ai = adaptation[i] * (1.0 - adapt_decay) + adapt_rate * ri
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
