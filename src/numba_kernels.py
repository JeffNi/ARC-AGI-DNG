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
    """Graded lateral inhibition: top-k fire at full rate, others suppressed.

    Losers retain 0.1% of their rate — biologically non-zero but far below
    the column mean, so competitive Hebbian produces anti-Hebbian LTD.
    """
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
        r[pool_indices[order[i]]] *= 0.001


@numba.jit(nopython=True, cache=True)
def wta_columnar(r, pool_indices, col_sizes, col_offsets, n_cols, k_frac):
    """Per-column WTA: each cortical column competes independently.

    Biologically, lateral inhibition is local — neurons compete with
    their column neighbors, not with neurons across the cortex. This
    forces different columns to specialize for different features.
    """
    for col in range(n_cols):
        offset = col_offsets[col]
        size = col_sizes[col]
        if size == 0:
            continue
        k = max(1, int(size * k_frac))
        if k >= size:
            continue

        # Insertion sort by descending r (small pools, ~15 neurons)
        order = np.empty(size, dtype=numba.int64)
        for i in range(size):
            order[i] = i
        for i in range(1, size):
            key_idx = order[i]
            key_val = r[pool_indices[offset + key_idx]]
            j = i - 1
            while j >= 0 and r[pool_indices[offset + order[j]]] < key_val:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key_idx

        for i in range(k, size):
            r[pool_indices[offset + order[i]]] *= 0.001


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
    col_pool, col_sizes, col_offsets, n_cols, wta_k_frac,
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
            n_act = 0.0
            for j in range(W_indptr[i], W_indptr[i + 1]):
                pre_val = buf[W_indices[j]]
                if pre_val > 0.001:
                    acc += W_data[j] * pre_val
                    n_act += 1.0
            syn[i] = acc / (n_act + 1.0) ** 0.5

        for i in range(n_nodes):
            prev_r[i] = r[i]

            vi = (1.0 - leak[i]) * V[i] + leak[i] * excitability[i] * syn[i]
            if has_signal:
                vi += signal[i]
            vi += noise_matrix[s, i] * noise_scale
            vi -= adaptation[i]
            if vi > 5.0:
                vi = 5.0
            elif vi < -1.0:
                vi = -1.0
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

        wta_columnar(r, col_pool, col_sizes, col_offsets, n_cols, wta_k_frac)
        wta_pool(r, mem_pool_indices, mem_wta_k)
        wta_motor_cells(r, motor_start, n_motor_cells, n_colors)


@numba.jit(nopython=True, parallel=True, cache=True)
def _spmv_csr_parallel(indptr, indices, data, buf, syn, n_nodes):
    """Parallel sparse matmul with active-input normalization.

    Groups by destination node so each thread owns its accumulator —
    no write conflicts, scales linearly with cores.
    """
    for dst in numba.prange(n_nodes):
        acc = 0.0
        n_act = 0.0
        for j in range(indptr[dst], indptr[dst + 1]):
            pre_val = buf[indices[j]]
            if pre_val > 0.001:
                acc += data[j] * pre_val
                n_act += 1.0
        syn[dst] = acc / (n_act + 1.0) ** 0.5


@numba.jit(nopython=True, parallel=True, cache=True)
def _update_r_trace(r_trace, r, trace_decay, n_nodes):
    """Update slow activity trace: EMA of firing rates."""
    alpha = 1.0 - trace_decay
    for i in numba.prange(n_nodes):
        r_trace[i] = r_trace[i] * trace_decay + r[i] * alpha


@numba.jit(nopython=True, parallel=True, cache=True)
def _eligibility_update_parallel(
    edge_elig, edge_src, edge_dst, r, edge_plastic, n_edges, elig_decay,
):
    """Parallel eligibility trace update — ALL edges including motor.

    Motor edges need eligibility traces for DA-based reward learning
    (basal ganglia analog). Only K-H competitive plasticity excludes
    motor edges (via edge_plastic in _plasticity_update_parallel).
    """
    for e in numba.prange(n_edges):
        edge_elig[e] *= elig_decay
        co = r[edge_src[e]] * r[edge_dst[e]]
        if co > 0.001:
            edge_elig[e] += co


@numba.jit(nopython=True, parallel=True, cache=True)
def _plasticity_update_parallel(
    edge_w, edge_src, edge_dst, r_pre_wta, edge_plastic, n_edges,
    node_col_id, col_mean_r, ema_rate_dendritic,
    eta_local, eta, DA, w_max,
    r_trace, edge_is_l1_to_l2, use_trace_rule, trace_contrast_eta,
):
    """Parallel Krotov-Hopfield competitive Hebbian — each edge is independent.

    For L1->L2 edges when use_trace_rule is True, the post-synaptic term
    uses r_trace (slow EMA) instead of instantaneous r_pre_wta. This binds
    temporally adjacent representations across saccades (DiCarlo & Cox 2007).

    Includes a heterosynaptic LTD term for L1->L2: when an L2 neuron was
    recently active (high trace) but is NOT active now, weaken the connection
    from any active L1 input. This improves discrimination across patterns.
    """
    _EPS = 1e-6
    for e in numba.prange(n_edges):
        if not edge_plastic[e]:
            continue
        pre_r = r_pre_wta[edge_src[e]]
        post_pre_wta = r_pre_wta[edge_dst[e]]
        if pre_r < 0.001 and post_pre_wta < 0.001:
            continue

        is_l1_l2_trace = use_trace_rule and edge_is_l1_to_l2[e]

        # For L1->L2 edges: use the slow trace as the post-synaptic signal.
        if is_l1_l2_trace:
            post_for_plasticity = r_trace[edge_dst[e]]
        else:
            post_for_plasticity = post_pre_wta

        col_id = node_col_id[edge_dst[e]]
        if col_id < 0:
            h_star = 0.0
        else:
            h_star = col_mean_r[col_id]

        dst_ema = ema_rate_dendritic[edge_dst[e]]
        conscience = dst_ema - h_star
        if conscience > 0.0:
            h_star += conscience

        delta = post_for_plasticity - h_star
        if delta > 0.0:
            g_h = delta * delta
        else:
            g_h = 0.0

        if g_h < 1e-8 and not is_l1_l2_trace:
            continue

        w_old = edge_w[e]
        w_abs = w_old if w_old > 0.0 else -w_old

        dw = eta_local * g_h * (pre_r - w_abs)
        dw += eta * DA * g_h * pre_r

        # Heterosynaptic LTD for L1->L2: L2 was recently active (trace)
        # but isn't now, while L1 IS active — weaken the connection.
        if is_l1_l2_trace:
            trace_val = r_trace[edge_dst[e]]
            if trace_val > 0.01 and post_pre_wta < 0.01 and pre_r > 0.01:
                dw -= trace_contrast_eta * eta_local * trace_val * pre_r

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
def _col_thresholds(r_pre_wta, col_pool, col_sizes, col_offsets, n_cols, col_mean_r):
    """Per-column plasticity threshold: midpoint of mean and max pre-WTA rate."""
    for c in range(n_cols):
        offset = col_offsets[c]
        size = col_sizes[c]
        if size == 0:
            col_mean_r[c] = 0.0
            continue
        acc = 0.0
        mx = 0.0
        for k in range(size):
            rv = r_pre_wta[col_pool[offset + k]]
            acc += rv
            if rv > mx:
                mx = rv
        mean_r = acc / size
        col_mean_r[c] = mean_r + 0.5 * (mx - mean_r)


@numba.jit(nopython=True, cache=True)
def _node_dynamics(
    V, r, prev_r, f, threshold, leak, excitability, adaptation,
    syn, signal, noise_row, noise_scale, has_signal,
    max_rate, f_rate_eff, f_decay, f_max,
    adapt_rate, adapt_decay, refractory_mask, r_pre_wta, n_nodes,
):
    """Membrane, rate, facilitation, and adaptation update for all nodes."""
    for i in range(n_nodes):
        prev_r[i] = r[i]

        vi = (1.0 - leak[i]) * V[i] + leak[i] * excitability[i] * syn[i]
        if has_signal:
            vi += signal[i]
        vi += noise_row[i] * noise_scale
        vi -= adaptation[i]
        if vi > 5.0:
            vi = 5.0
        elif vi < -1.0:
            vi = -1.0
        V[i] = vi

        ri = vi - threshold[i]
        if ri < 0.0:
            ri = 0.0
        elif ri > max_rate[i]:
            ri = max_rate[i]

        if refractory_mask[i] and prev_r[i] > 0.8 * max_rate[i]:
            ri *= 0.1

        r[i] = ri
        r_pre_wta[i] = ri

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


def run_steps_plastic(
    V, r, prev_r, f, threshold, leak, excitability, adaptation,
    edge_src, edge_dst, edge_w, n_edges, inh_scale,
    signal, noise_scale,
    DA, eta, w_max, plasticity_interval,
    max_rate, f_rate_eff, f_decay, f_max,
    adapt_rate, adapt_decay,
    col_pool, col_sizes, col_offsets, n_cols, wta_k_frac,
    mem_pool_indices, mem_wta_k,
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
    csr_indptr=None, csr_indices=None, csr_data=None,
    r_trace=None, trace_decay=0.95,
    edge_is_l1_to_l2=None, use_trace_rule=False,
    trace_contrast_eta=0.3,
):
    """
    Dynamics with competitive Hebbian plasticity + eligibility traces.

    Dispatches to parallel sub-kernels when CSR arrays are provided,
    otherwise falls back to the sequential COO path.

    The CSR matmul uses inh_scale-baked weights (built once per call)
    so plasticity-driven weight tweaks within the 25-step window are
    negligible — weights are refreshed on the next step() call.
    """
    syn = np.empty(n_nodes)
    buf = np.empty(n_nodes)

    use_parallel = csr_indptr is not None

    for s in range(n_steps):
        for i in range(n_nodes):
            buf[i] = r[i] * (1.0 + f[i])

        if use_parallel:
            _spmv_csr_parallel(csr_indptr, csr_indices, csr_data, buf, syn, n_nodes)
        else:
            n_active = np.empty(n_nodes)
            for i in range(n_nodes):
                syn[i] = 0.0
                n_active[i] = 0.0
            for e in range(n_edges):
                pre_val = buf[edge_src[e]]
                if pre_val > 0.001:
                    w_eff = edge_w[e]
                    if w_eff < 0.0:
                        w_eff *= inh_scale
                    syn[edge_dst[e]] += w_eff * pre_val
                    n_active[edge_dst[e]] += 1.0
            for i in range(n_nodes):
                syn[i] /= (n_active[i] + 1.0) ** 0.5

        _node_dynamics(
            V, r, prev_r, f, threshold, leak, excitability, adaptation,
            syn, signal, noise_matrix[s], noise_scale, has_signal,
            max_rate, f_rate_eff, f_decay, f_max,
            adapt_rate, adapt_decay, refractory_mask, r_pre_wta, n_nodes,
        )

        _col_thresholds(r_pre_wta, col_pool, col_sizes, col_offsets, n_cols, col_mean_r)
        wta_columnar(r, col_pool, col_sizes, col_offsets, n_cols, wta_k_frac)
        wta_pool(r, mem_pool_indices, mem_wta_k)
        wta_motor_cells(r, motor_start, n_motor_cells, n_colors)

        if r_trace is not None:
            _update_r_trace(r_trace, r, trace_decay, n_nodes)

        _eligibility_update_parallel(
            edge_elig, edge_src, edge_dst, r, edge_plastic, n_edges, elig_decay,
        )

        if plasticity_interval > 0 and s > 0 and s % plasticity_interval == 0:
            _trace_rule = use_trace_rule and r_trace is not None and edge_is_l1_to_l2 is not None
            _r_tr = r_trace if r_trace is not None else r_pre_wta
            _l1l2 = edge_is_l1_to_l2 if edge_is_l1_to_l2 is not None else edge_plastic
            _plasticity_update_parallel(
                edge_w, edge_src, edge_dst, r_pre_wta, edge_plastic, n_edges,
                node_col_id, col_mean_r, ema_rate_dendritic,
                eta_local, eta, DA, w_max,
                _r_tr, _l1l2, _trace_rule, trace_contrast_eta,
            )


@numba.jit(nopython=True, cache=True)
def run_steps_record(
    V, r, prev_r, f, threshold, leak, excitability, adaptation,
    W_data, W_indices, W_indptr,
    signal, noise_scale,
    max_rate, f_rate_eff, f_decay, f_max,
    adapt_rate, adapt_decay,
    col_pool, col_sizes, col_offsets, n_cols, wta_k_frac,
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
            n_act = 0.0
            for j in range(W_indptr[i], W_indptr[i + 1]):
                pre_val = buf[W_indices[j]]
                if pre_val > 0.001:
                    acc += W_data[j] * pre_val
                    n_act += 1.0
            syn[i] = acc / (n_act + 1.0) ** 0.5

        for i in range(n_nodes):
            prev_r[i] = r[i]
            vi = (1.0 - leak[i]) * V[i] + leak[i] * excitability[i] * syn[i]
            if has_signal:
                vi += signal[i]
            vi += noise_matrix[s, i] * noise_scale
            vi -= adaptation[i]
            if vi > 5.0:
                vi = 5.0
            elif vi < -1.0:
                vi = -1.0
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

        wta_columnar(r, col_pool, col_sizes, col_offsets, n_cols, wta_k_frac)
        wta_pool(r, mem_pool_indices, mem_wta_k)
        wta_motor_cells(r, motor_start, n_motor_cells, n_colors)

        for i in range(n_edges):
            corr[i] += r[edge_src[i]] * r[edge_dst[i]]

    for i in range(n_edges):
        corr[i] *= inv_steps
