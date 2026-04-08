"""Quick diagnostic: hierarchical cortical layers at grid=10.
Disposable script — delete after diagnosis.
"""
import numpy as np
import time
from src.genome import Genome
from src.brain import Brain
from src.encoding import grid_to_signal
from src.stimuli import InfancyStimuli
from src.graph import Region, internal_mask as _int_mask


def measure(brain, h, w, label):
    from run import _probe_layer_sim
    from src.stimuli import (
        rectangle, l_shape, random_scatter,
        random_noise, cross_shape, color_blocks,
    )
    _reg = list(Region)
    l1_idx = np.where(brain.net.regions == _reg.index(Region.LOCAL_DETECT))[0]
    l2_idx = np.where(brain.net.regions == _reg.index(Region.MID_LEVEL))[0]
    l3_idx = np.where(brain.net.regions == _reg.index(Region.ABSTRACT))[0]
    ne = brain.net._edge_count

    # Alive = wins WTA for at least one probe stimulus
    rng = np.random.default_rng(42)
    gens = [rectangle, l_shape, random_scatter, random_noise, cross_shape, color_blocks]
    ever_won = np.zeros(brain.net.n_nodes, dtype=bool)
    saved_V = brain.net.V.copy()
    saved_r = brain.net.r.copy()
    saved_f = brain.net.f.copy()
    saved_adapt = brain.net.adaptation.copy()
    for gen in gens:
        grid = gen(h, w, rng)
        signal = grid_to_signal(grid, max_h=h, max_w=w)
        brain.net.V[:] = 0.0
        brain.net.r[:] = 0.0
        brain.net.f[:] = 0.0
        brain.net.adaptation[:] = 0.0
        brain.clear_signal()
        brain.inject_signal(signal)
        brain.step(n_steps=20)
        ever_won |= brain.net.r > 0.01
    brain.net.V[:] = saved_V
    brain.net.r[:] = saved_r
    brain.net.f[:] = saved_f
    brain.net.adaptation[:] = saved_adapt
    brain.clear_signal()

    # Per-layer masks for sim probe
    l1_mask = np.zeros(brain.net.n_nodes, dtype=bool)
    l1_mask[l1_idx] = True
    l2_mask = np.zeros(brain.net.n_nodes, dtype=bool)
    l2_mask[l2_idx] = True
    l3_mask = np.zeros(brain.net.n_nodes, dtype=bool)
    l3_mask[l3_idx] = True

    import sys
    print(f"    probing L1...", end="", flush=True)
    m_l1 = _probe_layer_sim(brain, h, w, l1_mask)
    print(f" L2...", end="", flush=True)
    m_l2 = _probe_layer_sim(brain, h, w, l2_mask)
    print(f" L3...", end="", flush=True)
    m_l3 = _probe_layer_sim(brain, h, w, l3_mask)
    print(" done", flush=True)

    ema = brain.homeostasis.ema_rate
    w_abs = np.abs(brain.net._edge_w[:ne])
    w90 = float(np.percentile(w_abs, 90))

    print(f"  [{label:>16s}]  w90={w90:.4f}")
    print(f"    L1: alive={int(ever_won[l1_idx].sum()):>4}/{len(l1_idx)}  "
          f"sel={float(ema[l1_idx].std()):.4f}  "
          f"simV={m_l1['sim_v']:.3f}  simR={m_l1['sim_r']:.3f}  simCol={m_l1['sim_col']:.3f}")
    print(f"    L2: alive={int(ever_won[l2_idx].sum()):>4}/{len(l2_idx)}  "
          f"sel={float(ema[l2_idx].std()):.4f}  "
          f"simV={m_l2['sim_v']:.3f}  simR={m_l2['sim_r']:.3f}  simCol={m_l2['sim_col']:.3f}")
    print(f"    L3: alive={int(ever_won[l3_idx].sum()):>4}/{len(l3_idx)}  "
          f"sel={float(ema[l3_idx].std()):.4f}  "
          f"simV={m_l3['sim_v']:.3f}  simR={m_l3['sim_r']:.3f}  simCol={m_l3['sim_col']:.3f}")


def run_trial(grid_size, n_days=7, lateral_density=None, n_int=None):
    genome = Genome()
    if lateral_density is not None:
        genome.density_internal_to_internal = lateral_density
    if n_int is not None:
        genome.n_internal = n_int
    h, w = grid_size, grid_size

    n_l1 = int(genome.n_internal * genome.frac_layer1)
    n_l2 = int(genome.n_internal * genome.frac_layer2)
    n_l3 = genome.n_internal - n_l1 - n_l2
    print(f"\n{'='*60}")
    print(f"  GRID={grid_size}x{grid_size}, n_int={genome.n_internal}, {n_days} days")
    print(f"  Layers: L1={n_l1}, L2={n_l2}, L3={n_l3}")
    print(f"{'='*60}")

    import sys
    print("  Creating brain...", flush=True)
    brain = Brain.birth(genome, grid_h=h, grid_w=w, seed=42)

    ne = brain.net._edge_count
    print(f"  nodes={brain.net.n_nodes}, edges={ne:,}", flush=True)

    measure(brain, h, w, "BIRTH")

    wta_start, wta_end, wta_ramp_days = 0.3, 0.1, 15
    stimuli = InfancyStimuli(h, w, rng=brain.rng)
    for day in range(1, n_days + 1):
        t = min(day / wta_ramp_days, 1.0)
        wta_frac = wta_start + t * (wta_end - wta_start)
        brain.stage_manager._to_setpoints.wta_active_frac = wta_frac
        t0 = time.time()
        for _ in range(30):
            grid = stimuli.generate()
            signal = grid_to_signal(grid, max_h=h, max_w=w)
            brain.store_signal(signal)
            brain.inject_signal(signal)
            brain.step(n_steps=50)
            brain.clear_signal()
            brain.step(n_steps=20)
            brain.try_sleep()
        dt = time.time() - t0
        measure(brain, h, w, f"DAY {day} wta={wta_frac:.2f} ({dt:.0f}s)")


if __name__ == "__main__":
    run_trial(10, n_days=10, n_int=1500)
