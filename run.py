"""
run.py — Brain lifecycle entry point.

Usage:
  python run.py                  # birth or resume, run indefinitely
  python run.py --days 10        # run for 10 days
  python run.py --seed 42        # birth with specific seed
  python run.py --grid 5         # 5x5 grid
  python run.py --tasks-per-day 100

Ctrl+C gracefully saves a checkpoint and exits.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from src.genome import Genome
from src.brain import Brain
from src.teacher import Teacher
from src.monitor import Monitor
from src.evaluator import Evaluator
from src.stimuli import InfancyStimuli, load_arc_inputs
from src.encoding import grid_to_signal


_STOP = False


def _probe_repr_similarity(brain: Brain, h: int, w: int, n_probes: int = 6) -> float:
    """Quick probe: do different stimuli produce different internal patterns?

    Uses centered cosine similarity (Pearson correlation) on membrane potential V.
    Centering removes the common-mode DC bias that would otherwise dominate,
    letting us measure whether the *pattern* of activation differs across stimuli.

    Returns mean pairwise Pearson r (1.0 = identical patterns, 0.0 = uncorrelated).
    Lower is better — means the network differentiates inputs.
    """
    from src.stimuli import (
        rectangle, l_shape, random_scatter,
        random_noise, cross_shape, color_blocks,
    )
    from src.graph import internal_mask as _int_mask

    int_mask = _int_mask(brain.net.regions)
    rng = np.random.default_rng(42)
    generators = [rectangle, l_shape, random_scatter,
                  random_noise, cross_shape, color_blocks]

    # Save full network state — probe must not corrupt training dynamics
    ne = brain.net._edge_count
    saved_V = brain.net.V.copy()
    saved_r = brain.net.r.copy()
    saved_adapt = brain.net.adaptation.copy()
    saved_f = brain.net.f.copy()
    saved_w = brain.net._edge_w[:ne].copy()
    saved_elig = brain.net._edge_eligibility[:ne].copy()
    saved_exc = brain.net.excitability.copy()

    patterns = []
    for gen in generators[:n_probes]:
        grid = gen(h, w, rng)
        signal = grid_to_signal(grid, max_h=h, max_w=w)
        brain.net.V[:] = 0.0
        brain.net.r[:] = 0.0
        brain.net.adaptation[:] = 0.0
        brain.clear_signal()
        brain.inject_signal(signal)
        brain.step(n_steps=20)
        patterns.append(brain.net.V[int_mask].copy())

    # Restore full network state (including weights modified by plasticity)
    brain.net._edge_count = ne
    brain.net.V[:] = saved_V
    brain.net.r[:] = saved_r
    brain.net.adaptation[:] = saved_adapt
    brain.net.f[:] = saved_f
    brain.net._edge_w[:ne] = saved_w
    brain.net._edge_eligibility[:ne] = saved_elig
    brain.net.excitability[:] = saved_exc
    brain.clear_signal()

    sims = []
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            a = patterns[i] - patterns[i].mean()
            b = patterns[j] - patterns[j].mean()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-8 and nb > 1e-8:
                sims.append(float(np.dot(a, b) / (na * nb)))
    return float(np.mean(sims)) if sims else 1.0


def _probe_layer_sim(brain: Brain, h: int, w: int, mask: np.ndarray,
                     n_probes: int = 6) -> dict:
    """Probe representational similarity for a layer.

    Returns dict with:
      sim_v:  mean pairwise Pearson r on V (membrane potential)
      sim_r:  mean pairwise Pearson r on r (post-WTA firing rate)
      sim_col: mean within-column sim on r (are column-mates differentiating?)
    """
    from src.stimuli import (
        rectangle, l_shape, random_scatter,
        random_noise, cross_shape, color_blocks,
    )
    rng = np.random.default_rng(42)
    generators = [rectangle, l_shape, random_scatter,
                  random_noise, cross_shape, color_blocks]

    ne = brain.net._edge_count
    saved_V = brain.net.V.copy()
    saved_r = brain.net.r.copy()
    saved_adapt = brain.net.adaptation.copy()
    saved_f = brain.net.f.copy()
    saved_w = brain.net._edge_w[:ne].copy()
    saved_elig = brain.net._edge_eligibility[:ne].copy()
    saved_exc = brain.net.excitability.copy()

    pats_v, pats_r = [], []
    for gen in generators[:n_probes]:
        grid = gen(h, w, rng)
        signal = grid_to_signal(grid, max_h=h, max_w=w)
        brain.net.V[:] = 0.0
        brain.net.r[:] = 0.0
        brain.net.adaptation[:] = 0.0
        brain.clear_signal()
        brain.inject_signal(signal)
        brain.step(n_steps=20)
        pats_v.append(brain.net.V[mask].copy())
        pats_r.append(brain.net.r[mask].copy())

    brain.net._edge_count = ne
    brain.net.V[:] = saved_V
    brain.net.r[:] = saved_r
    brain.net.adaptation[:] = saved_adapt
    brain.net.f[:] = saved_f
    brain.net._edge_w[:ne] = saved_w
    brain.net._edge_eligibility[:ne] = saved_elig
    brain.net.excitability[:] = saved_exc
    brain.clear_signal()

    def _mean_pairwise(patterns):
        sims = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                a = patterns[i] - patterns[i].mean()
                b = patterns[j] - patterns[j].mean()
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-8 and nb > 1e-8:
                    sims.append(float(np.dot(a, b) / (na * nb)))
        return float(np.mean(sims)) if sims else 1.0

    sim_v = _mean_pairwise(pats_v)
    sim_r = _mean_pairwise(pats_r)

    # Within-column similarity: for each column, compare its neurons'
    # response vectors across stimuli. Low = good differentiation.
    layer_indices = np.where(mask)[0]
    col_ids = brain.net.column_ids[layer_indices]
    unique_cols = np.unique(col_ids[col_ids >= 0])

    col_sims = []
    for col in unique_cols:
        col_neurons = np.where(col_ids == col)[0]
        if len(col_neurons) < 2:
            continue
        # For each pair of neurons in this column, compute correlation
        # of their response profiles across stimuli
        responses = np.array([
            [pats_r[s][n] for s in range(len(pats_r))]
            for n in col_neurons
        ])  # (n_neurons_in_col, n_stimuli)
        for a_i in range(len(col_neurons)):
            for b_i in range(a_i + 1, len(col_neurons)):
                ra = responses[a_i] - responses[a_i].mean()
                rb = responses[b_i] - responses[b_i].mean()
                na, nb = np.linalg.norm(ra), np.linalg.norm(rb)
                if na > 1e-8 and nb > 1e-8:
                    col_sims.append(float(np.dot(ra, rb) / (na * nb)))

    sim_col = float(np.mean(col_sims)) if col_sims else 0.0

    return {"sim_v": sim_v, "sim_r": sim_r, "sim_col": sim_col}


def _birth_health_check(brain: Brain, monitor: Monitor) -> bool:
    """Verify the newborn brain is functional before starting development.

    A newborn (post-template) should have signal propagation and
    representational diversity already built in from prenatal wiring.
    """
    from src.stimuli import solid_fill, checkerboard
    from src.graph import internal_mask as _int_mask

    h, w = brain.net.max_h, brain.net.max_w
    rng = np.random.default_rng(123)
    int_idx = np.where(_int_mask(brain.net.regions))[0]

    ne = brain.net._edge_count
    saved_V = brain.net.V.copy()
    saved_r = brain.net.r.copy()
    saved_a = brain.net.adaptation.copy()
    saved_f = brain.net.f.copy()
    saved_w = brain.net._edge_w[:ne].copy()
    saved_elig = brain.net._edge_eligibility[:ne].copy()
    saved_exc = brain.net.excitability.copy()

    alive_counts = []
    for gen in [solid_fill, checkerboard]:
        grid = gen(h, w, rng)
        signal = grid_to_signal(grid, max_h=h, max_w=w)
        brain.net.V[:] = 0; brain.net.r[:] = 0; brain.net.adaptation[:] = 0
        brain.clear_signal()
        brain.inject_signal(signal)
        brain.step(n_steps=30)
        alive_counts.append(int((brain._r_pre_wta[int_idx] > 0.01).sum()))

    brain.net._edge_count = ne
    brain.net.V[:] = saved_V
    brain.net.r[:] = saved_r
    brain.net.adaptation[:] = saved_a
    brain.net.f[:] = saved_f
    brain.net._edge_w[:ne] = saved_w
    brain.net._edge_eligibility[:ne] = saved_elig
    brain.net.excitability[:] = saved_exc
    brain.clear_signal()

    n_int = len(int_idx)
    min_alive = min(alive_counts)
    pct = min_alive / n_int * 100

    sim = _probe_repr_similarity(brain, h, w)

    ok = min_alive >= 0.9 * n_int and sim < 0.95
    status = "OK" if ok else "WARNING"
    monitor.status(
        f"Birth health [{status}]: alive={min_alive}/{n_int} ({pct:.0f}%), "
        f"sim={sim:.3f}"
    )
    if not ok:
        monitor.status(
            "  Birth network is non-functional. Check weight_scale and "
            "template connectivity."
        )
    return ok


def diagnose_infancy(brain: Brain, monitor: Monitor, n_probes: int = 8):
    """
    Probe the post-infancy brain to see if it built meaningful structure.

    Measures:
    1. Representational differentiation — do different stimuli produce
       different internal activity patterns?
    2. Edge distribution — are connections biased toward functional pathways?
    3. Signal propagation — does sensory input reach internal/motor layers?
    """
    from src.stimuli import (
        rectangle, l_shape, random_scatter,
        random_noise, cross_shape, color_blocks, diagonal_line, border_frame,
    )
    from src.graph import Region, internal_mask as _int_mask

    h, w = brain.net.max_h, brain.net.max_w
    rng = np.random.default_rng(99)
    region_list = list(Region)

    generators = [
        rectangle, l_shape, random_scatter,
        random_noise, cross_shape, color_blocks, diagonal_line, border_frame,
    ]

    # --- 1. Representational differentiation ---
    internal_mask = _int_mask(brain.net.regions)
    motor_mask = brain.net.regions == region_list.index(Region.MOTOR)
    sensory_mask = brain.net.regions == region_list.index(Region.SENSORY)

    patterns = []
    pattern_names = []
    propagation_stats = []

    for gen in generators[:n_probes]:
        grid = gen(h, w, rng)
        signal = grid_to_signal(grid, max_h=h, max_w=w)

        brain.clear_signal()
        brain.inject_signal(signal)
        brain.step(n_steps=30)

        r = brain.net.r.copy()
        patterns.append(r[internal_mask].copy())
        pattern_names.append(gen.__name__)

        propagation_stats.append({
            "name": gen.__name__,
            "sensory_mean": float(r[sensory_mask].mean()),
            "internal_mean": float(r[internal_mask].mean()),
            "motor_mean": float(r[motor_mask].mean()),
            "internal_active": int((r[internal_mask] > 0.01).sum()),
            "motor_active": int((r[motor_mask] > 0.01).sum()),
        })

    brain.clear_signal()

    # Pairwise cosine similarity between internal representations
    n_pat = len(patterns)
    similarities = []
    for i in range(n_pat):
        for j in range(i + 1, n_pat):
            a, b = patterns[i], patterns[j]
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a > 1e-8 and norm_b > 1e-8:
                sim = float(np.dot(a, b) / (norm_a * norm_b))
            else:
                sim = 0.0
            similarities.append(sim)

    mean_sim = float(np.mean(similarities)) if similarities else 0.0
    min_sim = float(np.min(similarities)) if similarities else 0.0
    max_sim = float(np.max(similarities)) if similarities else 0.0

    # --- 2. Edge distribution by region pathway ---
    n = brain.net._edge_count
    src_reg = brain.net.regions[brain.net._edge_src[:n]]
    dst_reg = brain.net.regions[brain.net._edge_dst[:n]]

    pathway_counts = {}
    for sr in Region:
        for dr in Region:
            si = region_list.index(sr)
            di = region_list.index(dr)
            count = int(((src_reg == si) & (dst_reg == di)).sum())
            if count > 0:
                key = f"{sr.value}->{dr.value}"
                pathway_counts[key] = count

    # --- Print report ---
    monitor.status("=" * 60)
    monitor.status("INFANCY DIAGNOSTIC — Is the brain building something useful?")
    monitor.status("=" * 60)

    monitor.status("")
    monitor.status("1. SIGNAL PROPAGATION (does input reach deeper layers?)")
    for s in propagation_stats:
        monitor.status(
            f"   {s['name']:20s}  sensory={s['sensory_mean']:.4f}  "
            f"internal={s['internal_mean']:.4f} ({s['internal_active']} active)  "
            f"motor={s['motor_mean']:.4f} ({s['motor_active']} active)"
        )

    monitor.status("")
    monitor.status("2. REPRESENTATIONAL DIFFERENTIATION (do different stimuli look different inside?)")
    monitor.status(f"   Pairwise cosine similarity of internal activity vectors:")
    monitor.status(f"   mean={mean_sim:.4f}  min={min_sim:.4f}  max={max_sim:.4f}")
    if mean_sim < 0.85:
        monitor.status(f"   -> GOOD: representations are differentiated (mean < 0.85)")
    elif mean_sim < 0.95:
        monitor.status(f"   -> MARGINAL: some differentiation but patterns are similar")
    else:
        monitor.status(f"   -> BAD: all stimuli produce nearly identical internal activity")

    monitor.status("")
    monitor.status("3. EDGE DISTRIBUTION (where are the connections?)")
    total = sum(pathway_counts.values())
    for pathway, count in sorted(pathway_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total if total > 0 else 0
        monitor.status(f"   {pathway:25s}  {count:>10,}  ({pct:5.1f}%)")

    monitor.status("=" * 60)


def _handle_sigint(sig, frame):
    global _STOP
    if _STOP:
        print("\nForced exit.")
        sys.exit(1)
    _STOP = True
    print("\nGraceful shutdown requested... finishing current task and saving.")


def run_infancy(
    brain: Brain,
    monitor: Monitor,
    stimuli: InfancyStimuli,
    birth_edges: int,
    max_days: int = 100,
    stimuli_per_day: int = 60,
    observe_steps: int = 25,
    rest_steps: int = 10,
    growth_target: float = 2.5,
    spontaneous_threshold: float = 0.03,
) -> int:
    """
    Run the infancy phase: pure sensory exposure, no tasks.

    The brain sees random visual stimuli, builds synapses through
    Hebbian co-activation and sleep-driven synaptogenesis, and
    develops internal representations. No reward, no CHL, no motor
    readout. Just looking at the world.

    Returns the number of infancy days completed.

    Transition to childhood when:
      - Primary: synapse count >= growth_target * birth_edges
      - Fallback: max_days reached
    """
    h = brain.net.max_h
    w = brain.net.max_w

    monitor.status(
        f"=== INFANCY BEGINS === "
        f"birth edges={birth_edges:,}, "
        f"target={int(birth_edges * growth_target):,} "
        f"({growth_target}x), max {max_days} days"
    )

    # E/I balance maturation: WTA sharpens over infancy as PV+
    # interneurons mature (Hensch 2005). Broad early WTA lets neurons
    # establish baseline activity; sharp late WTA forces specialization.
    wta_start, wta_end, wta_ramp_days = 0.3, 0.1, 15

    day = 0
    for day in range(1, max_days + 1):
        if _STOP:
            break

        # Developmental WTA ramp: 0.3 → 0.1 over wta_ramp_days
        t = min(day / wta_ramp_days, 1.0)
        wta_frac = wta_start + t * (wta_end - wta_start)
        brain.stage_manager._to_setpoints.wta_active_frac = wta_frac

        day_sleeps = 0
        day_pruned = 0
        day_start = time.time()
        age_start = brain.age
        edges_start = brain.net._edge_count

        for stim_i in range(stimuli_per_day):
            if _STOP:
                break

            grid = stimuli.generate()
            signal = grid_to_signal(grid, max_h=h, max_w=w)
            brain.store_signal(signal)
            brain.inject_signal(signal)
            brain.step(n_steps=observe_steps)

            brain.clear_signal()
            brain.step(n_steps=rest_steps)

            sleep_stats = brain.try_sleep()
            if sleep_stats:
                monitor.sleep_event(sleep_stats, brain.age)
                day_sleeps += 1
                day_pruned += sleep_stats.get("pruned", 0)

        # End-of-day: measure spontaneous activity with no input
        brain.clear_signal()
        brain.step(n_steps=10)

        # Representational similarity probe
        repr_sim = _probe_repr_similarity(brain, h, w)

        day_elapsed = time.time() - day_start
        day_steps = brain.age - age_start
        day_grown = brain.net._edge_count - edges_start + day_pruned
        snap = monitor.infancy_snapshot(
            brain, day=day, birth_edges=birth_edges,
            day_sleeps=day_sleeps, day_pruned=day_pruned,
            day_grown=day_grown, day_secs=day_elapsed,
            day_steps=day_steps, repr_sim=repr_sim,
        )

        # Milestone save
        if day % 5 == 0:
            brain.save_milestone(f"infancy_day_{day}")

        # Safety: edge count should never drop during infancy
        current_edges = brain.net._edge_count
        if current_edges < birth_edges * 0.95:
            monitor.status(
                f"BUG: infancy edge count dropped to {current_edges:,} "
                f"({current_edges / max(1, birth_edges):.2f}x birth). "
                f"Halting infancy — pruning during infancy is a bug."
            )
            break

        # Transition check
        growth_ratio = current_edges / max(1, birth_edges)
        if growth_ratio >= growth_target:
            monitor.status(
                f"Infancy transition: synapse target reached "
                f"({current_edges:,} = {growth_ratio:.2f}x birth)"
            )
            break

    monitor.status(
        f"=== INFANCY COMPLETE === "
        f"day {day}, edges={brain.net._edge_count:,} "
        f"({brain.net._edge_count / max(1, birth_edges):.2f}x birth), "
        f"age={brain.age}"
    )
    brain.save_milestone("infancy_complete")

    return day


def main():
    parser = argparse.ArgumentParser(description="Brain lifecycle runner")
    parser.add_argument("--days", type=int, default=0, help="Childhood days to run (0=indefinite)")
    parser.add_argument("--tasks-per-day", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grid", type=int, default=10, help="Grid dimension (square)")
    parser.add_argument("--checkpoint-dir", type=str, default="runs/life")
    parser.add_argument("--task-dir", type=str, default="micro_tasks")
    parser.add_argument("--fresh", action="store_true", help="Force new brain (ignore checkpoints)")
    parser.add_argument("--eval-every", type=int, default=5, help="Run eval snapshot every N days (0=never)")
    parser.add_argument("--snapshot-every", type=int, default=1, help="Log brain health snapshot every N days")
    parser.add_argument("--infancy-days", type=int, default=20, help="Max infancy days (0=skip)")
    parser.add_argument("--infancy-stimuli", type=int, default=60, help="Stimuli per infancy day")
    parser.add_argument("--growth-target", type=float, default=2.5, help="Synapse growth target (multiple of birth)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)
    print("Starting brain lifecycle...", flush=True)

    genome = Genome(
        n_internal=1500,
        n_memory=200,
        max_h=args.grid,
        max_w=args.grid,
    )
    monitor = Monitor(log_dir=args.checkpoint_dir)

    # Birth or resume
    ckpt_dir = args.checkpoint_dir
    if not args.fresh:
        try:
            brain = Brain.resume(genome, checkpoint_dir=ckpt_dir)
            monitor.status(f"Resumed brain at age {brain.age}")
        except FileNotFoundError:
            brain = Brain.birth(
                genome,
                grid_h=args.grid,
                grid_w=args.grid,
                seed=args.seed,
                checkpoint_dir=ckpt_dir,
            )
            monitor.status(f"Born new brain: {brain.net.n_nodes} nodes, "
                          f"{brain.net._edge_count:,} edges")
    else:
        brain = Brain.birth(
            genome,
            grid_h=args.grid,
            grid_w=args.grid,
            seed=args.seed,
            checkpoint_dir=ckpt_dir,
        )
        monitor.status(f"Born new brain: {brain.net.n_nodes} nodes, "
                      f"{brain.net._edge_count:,} edges")

    # Initial brain health snapshot
    monitor.brain_snapshot(brain, label="birth")
    _birth_health_check(brain, monitor)
    birth_edges = brain.net._edge_count

    # ── INFANCY PHASE ─────────────────────────────────────────────
    if args.infancy_days > 0 and brain.stage_manager.current_stage == "infancy":
        arc_grids = load_arc_inputs(args.task_dir, args.grid, args.grid)
        monitor.status(f"Loaded {len(arc_grids)} ARC input grids for infancy exposure")

        stimuli = InfancyStimuli(
            max_h=args.grid,
            max_w=args.grid,
            rng=brain.rng,
            arc_input_grids=arc_grids,
            arc_mix_ratio=0.5,
        )

        run_infancy(
            brain=brain,
            monitor=monitor,
            stimuli=stimuli,
            birth_edges=birth_edges,
            max_days=args.infancy_days,
            stimuli_per_day=args.infancy_stimuli,
            growth_target=args.growth_target,
        )

        # Diagnose what infancy built
        diagnose_infancy(brain, monitor)

        # Transition to childhood
        brain.stage_manager.transition_to("childhood")
        monitor.status("Stage transition: infancy -> childhood")
        monitor.brain_snapshot(brain, label="childhood_start")

    # ── CHILDHOOD PHASE ───────────────────────────────────────────
    teacher = Teacher(
        brain=brain,
        monitor=monitor,
        task_dir=args.task_dir,
    )

    day = 0
    try:
        while not _STOP:
            day += 1
            if args.days > 0 and day > args.days:
                break

            result = teacher.run_day(max_tasks=args.tasks_per_day)

            # Periodic brain health snapshot
            if args.snapshot_every > 0 and day % args.snapshot_every == 0:
                monitor.brain_snapshot(brain, label=f"childhood_day_{day}")

            # Milestone save + evaluation snapshot
            if day % 5 == 0:
                brain.save_milestone(f"childhood_day_{day}")

            if args.eval_every > 0 and day % args.eval_every == 0:
                monitor.status(f"Running eval snapshot at day {day}...")
                evaluator = Evaluator(
                    brain=brain,
                    task_dir=args.task_dir,
                )
                report = evaluator.run_full_eval(console=True)
                monitor.eval_report(report, log_dir=ckpt_dir)

        monitor.status("Lifecycle complete" if not _STOP else "Interrupted")

    finally:
        path = brain.save(tag="final")
        monitor.status(f"Final checkpoint saved: {path}")


if __name__ == "__main__":
    main()
