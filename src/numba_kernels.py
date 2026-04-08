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


@numba.jit(nopython=True, cache=True)
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
):
    """
    Dynamics with competitive Hebbian plasticity + eligibility traces.

    Every step: eligibility traces accumulate from pre*post co-activity.
    Every plasticity_interval steps: Krotov-Hopfield competitive update.
    Winners (pre-WTA rate above conscience-adjusted threshold) get
    contrastive LTP. The per-neuron EMA rate acts as a "conscience"
    (DeSieno 1988): frequent winners get a higher threshold, forcing
    the network to rotate winners across stimuli.
    """
    _EPS = 1e-6
    buf = np.empty(n_nodes)
    syn = np.empty(n_nodes)
    n_active = np.empty(n_nodes)

    for s in range(n_steps):
        for i in range(n_nodes):
            buf[i] = r[i] * (1.0 + f[i])

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
            denom = (n_active[i] + 1.0) ** 0.5
            syn[i] /= denom

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

        # Compute per-column plasticity threshold from PRE-WTA rates,
        # before WTA zeroes out losers.
        # Threshold = midpoint between column mean and max pre-WTA rate.
        # This ensures only the top few neurons per column get LTP,
        # creating sharper competition than using the mean alone.
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

        wta_columnar(r, col_pool, col_sizes, col_offsets, n_cols, wta_k_frac)
        wta_pool(r, mem_pool_indices, mem_wta_k)
        wta_motor_cells(r, motor_start, n_motor_cells, n_colors)

        # Eligibility uses post-WTA rates (actual spike-driven signals)
        for e in range(n_edges):
            edge_elig[e] *= elig_decay
            if not edge_plastic[e]:
                continue
            pre_r = r[edge_src[e]]
            post_r = r[edge_dst[e]]
            co = pre_r * post_r
            if co > 0.001:
                edge_elig[e] += co

        # Competitive Hebbian plasticity — proper Krotov-Hopfield (2019).
        #
        # The K-H contrastive rule moves each winning neuron's weight
        # vector TOWARD the current input pattern:
        #
        #   dW = eta * g(h) * (v_pre - |w|)
        #
        # This is self-normalizing: weights converge to the centroid
        # of winning input patterns (like k-means). No separate decay
        # term needed — the (v - w) contrastive form prevents runaway.
        #
        # g(h) = ReLU(delta)^2 where delta = r_pre_wta - threshold.
        # Only winners (delta > 0) learn. Losers' weights are preserved,
        # letting them remain candidates for other stimuli.
        if plasticity_interval > 0 and s > 0 and s % plasticity_interval == 0:
            for e in range(n_edges):
                if not edge_plastic[e]:
                    continue
                pre_r = r_pre_wta[edge_src[e]]
                post_pre_wta = r_pre_wta[edge_dst[e]]
                if pre_r < 0.001 and post_pre_wta < 0.001:
                    continue

                col_id = node_col_id[edge_dst[e]]
                if col_id < 0:
                    h_star = 0.0
                else:
                    h_star = col_mean_r[col_id]

                # Conscience: raise threshold for neurons that fire
                # frequently (EMA > column mean). Frequent winners must
                # exceed a higher bar, giving underused neurons a chance.
                dst_ema = ema_rate_dendritic[edge_dst[e]]
                conscience = dst_ema - h_star
                if conscience > 0.0:
                    h_star += conscience

                delta = post_pre_wta - h_star
                if delta > 0.0:
                    g_h = delta * delta
                else:
                    g_h = 0.0

                if g_h < 1e-8:
                    continue

                w_old = edge_w[e]
                w_abs = w_old if w_old > 0.0 else -w_old

                # Contrastive update: pull weight toward input
                dw = eta_local * g_h * (pre_r - w_abs)
                # DA-gated term for reward-modulated learning
                dw += eta * DA * g_h * pre_r

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
