"""Unified life loop — one training script for all developmental stages.

Activity schedule is driven by `babble_ratio` and `task_mix_ratio` from
the StageManager's interpolated setpoints. Stage transitions are triggered
by a user-provided schedule (day -> stage name).

Three waking activities:
  1. Motor babble + mimicry (babble_ratio)
  2. Curriculum tasks (task_mix_ratio)
  3. Sensory stimuli (remainder)

Day cycle: fixed 2100-step budget regardless of stage.
  - Waking: 1400 steps (activities + free time)
  - Sleep:   700 steps (rest + consolidation)
"""
import sys
import numpy as np
import time
from src.genome import Genome
from src.brain import Brain
from src.display_buffer import DisplayBuffer
from src.encoding import grid_to_signal
from src.gaze_log import GazeLogger, print_gaze_summary
from src.stimuli import InfancyStimuli, load_arc_inputs
from src.graph import Region
from src.teacher import Teacher
from src.monitor import Monitor

DAY_STEPS = 2100
WAKE_STEPS = 1400
SLEEP_STEPS = DAY_STEPS - WAKE_STEPS
OBSERVE_STEPS = 25
REST_STEPS = 10
TEMPORAL_REST = 5  # shorter rest between temporal contiguity views
STEPS_PER_STIMULUS = OBSERVE_STEPS + REST_STEPS

BABBLE_STEPS = 35
MIMICRY_STEPS = 60

CHECKPOINT_DAYS = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60}
FUNC_TEST_DAYS = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0


def l1_functional_test(brain, h, w):
    """Quick L1 functional probe: consistency, discrimination, column switching."""
    from src.stimuli import (
        rectangle, l_shape, random_scatter, random_noise,
        cross_shape, color_blocks, solid_fill, checkerboard,
    )
    _reg = list(Region)
    l1_idx = np.where(brain.net.regions == _reg.index(Region.LOCAL_DETECT))[0]

    ne = brain.net._edge_count
    saved = {k: getattr(brain.net, k).copy() for k in ['V', 'r', 'adaptation', 'f']}
    saved['w'] = brain.net._edge_w[:ne].copy()
    saved['elig'] = brain.net._edge_eligibility[:ne].copy()
    saved['exc'] = brain.net.excitability.copy()

    def get_fp(grid):
        signal = grid_to_signal(grid, max_h=h, max_w=w)
        brain.net.V[:] = 0; brain.net.r[:] = 0; brain.net.adaptation[:] = 0
        brain.clear_signal()
        brain.inject_signal(signal)
        brain.step(n_steps=20)
        return brain.net.r[l1_idx].copy()

    def restore():
        brain.net.V[:] = saved['V']; brain.net.r[:] = saved['r']
        brain.net.adaptation[:] = saved['adaptation']; brain.net.f[:] = saved['f']
        brain.net._edge_w[:ne] = saved['w']
        brain.net._edge_eligibility[:ne] = saved['elig']
        brain.net.excitability[:] = saved['exc']; brain.clear_signal()

    rng2 = np.random.default_rng(99)
    grids = {n: g(h, w, rng2) for n, g in [
        ('rect', rectangle), ('L', l_shape), ('scatter', random_scatter),
        ('noise', random_noise), ('cross', cross_shape), ('color', color_blocks),
        ('solid', solid_fill), ('checker', checkerboard),
    ]}

    consistencies = []
    for grid in grids.values():
        fp1 = get_fp(grid); restore()
        fp2 = get_fp(grid); restore()
        consistencies.append(cosine(fp1, fp2))

    fps = {}
    for name, grid in grids.items():
        fps[name] = get_fp(grid); restore()
    names = list(fps.keys())
    off_diag = [cosine(fps[names[i]], fps[names[j]])
                for i in range(len(names)) for j in range(i + 1, len(names))]

    col_ids = brain._node_col_id[l1_idx]
    unique_cols = np.unique(col_ids[col_ids >= 0])
    n_sw = 0; n_tot = 0
    for col in unique_cols[:100]:
        cn = np.where(col_ids == col)[0]
        if len(cn) < 2:
            continue
        n_tot += 1
        w_set = set(int(np.argmax(fps[n][cn])) for n in names)
        if len(w_set) > 1:
            n_sw += 1

    cons = np.mean(consistencies)
    disc = np.mean(off_diag)
    print(f"    L1 FUNC: consistency={cons:.3f}  cross_sim={disc:.3f}  "
          f"gap={cons - disc:.3f}  col_switch={n_sw}/{n_tot}", flush=True)


def l2_functional_test(brain, h, w):
    """L2 functional probe: consistency, discrimination, and added value over L1."""
    from src.stimuli import (
        rectangle, l_shape, random_scatter, random_noise,
        cross_shape, color_blocks, solid_fill, checkerboard,
    )
    _reg = list(Region)
    l1_idx = np.where(brain.net.regions == _reg.index(Region.LOCAL_DETECT))[0]
    l2_idx = np.where(brain.net.regions == _reg.index(Region.MID_LEVEL))[0]

    ne = brain.net._edge_count
    saved = {k: getattr(brain.net, k).copy() for k in ['V', 'r', 'adaptation', 'f']}
    saved['w'] = brain.net._edge_w[:ne].copy()
    saved['elig'] = brain.net._edge_eligibility[:ne].copy()
    saved['exc'] = brain.net.excitability.copy()

    def get_fps(grid):
        signal = grid_to_signal(grid, max_h=h, max_w=w)
        brain.net.V[:] = 0; brain.net.r[:] = 0; brain.net.adaptation[:] = 0
        brain.clear_signal()
        brain.inject_signal(signal)
        brain.step(n_steps=20)
        return brain.net.r[l1_idx].copy(), brain.net.r[l2_idx].copy()

    def restore():
        brain.net.V[:] = saved['V']; brain.net.r[:] = saved['r']
        brain.net.adaptation[:] = saved['adaptation']; brain.net.f[:] = saved['f']
        brain.net._edge_w[:ne] = saved['w']
        brain.net._edge_eligibility[:ne] = saved['elig']
        brain.net.excitability[:] = saved['exc']; brain.clear_signal()

    rng2 = np.random.default_rng(99)
    grids = {n: g(h, w, rng2) for n, g in [
        ('rect', rectangle), ('L', l_shape), ('scatter', random_scatter),
        ('noise', random_noise), ('cross', cross_shape), ('color', color_blocks),
        ('solid', solid_fill), ('checker', checkerboard),
    ]}

    # Consistency: same stimulus -> same L2 response
    consistencies = []
    for grid in grids.values():
        _, fp1 = get_fps(grid); restore()
        _, fp2 = get_fps(grid); restore()
        consistencies.append(cosine(fp1, fp2))

    # Discrimination: different stimuli -> different L2 responses
    l1_fps, l2_fps = {}, {}
    for name, grid in grids.items():
        l1_fps[name], l2_fps[name] = get_fps(grid); restore()
    names = list(l2_fps.keys())
    l2_off_diag = [cosine(l2_fps[names[i]], l2_fps[names[j]])
                   for i in range(len(names)) for j in range(i + 1, len(names))]
    l1_off_diag = [cosine(l1_fps[names[i]], l1_fps[names[j]])
                   for i in range(len(names)) for j in range(i + 1, len(names))]

    # Added value: for L1's most-confused pairs, does L2 do better?
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    l1_sims = [cosine(l1_fps[names[i]], l1_fps[names[j]]) for i, j in pairs]
    l2_sims = [cosine(l2_fps[names[i]], l2_fps[names[j]]) for i, j in pairs]
    top5 = sorted(range(len(l1_sims)), key=lambda k: -l1_sims[k])[:5]
    l1_top5_avg = np.mean([l1_sims[k] for k in top5])
    l2_top5_avg = np.mean([l2_sims[k] for k in top5])

    # Sparsity: what fraction of L2 neurons are active per stimulus
    active_fracs = []
    for fp in l2_fps.values():
        active_fracs.append(float((fp > 0.01).mean()))

    cons = np.mean(consistencies)
    disc = np.mean(l2_off_diag)
    sparsity = np.mean(active_fracs)
    print(f"    L2 FUNC: consistency={cons:.3f}  cross_sim={disc:.3f}  "
          f"gap={cons - disc:.3f}  sparsity={sparsity:.3f}", flush=True)
    print(f"    L2 vs L1 on confusable pairs: L1={l1_top5_avg:.3f}  "
          f"L2={l2_top5_avg:.3f}  improvement={l1_top5_avg - l2_top5_avg:.3f}",
          flush=True)


def l2_l3_diagnostic(brain, h, w):
    """Layer-specific diagnostics for L2 and L3.

    L2: transformation tolerance, cross-category separation, driven activity.
    L3: driven activity plus active set diversity.
    """
    from src.stimuli import (
        rectangle, l_shape, cross_shape, color_blocks,
        shifted_variant, recolored_variant,
    )
    _reg = list(Region)
    l1_idx = np.where(brain.net.regions == _reg.index(Region.LOCAL_DETECT))[0]
    l2_idx = np.where(brain.net.regions == _reg.index(Region.MID_LEVEL))[0]
    l3_idx = np.where(brain.net.regions == _reg.index(Region.ABSTRACT))[0]

    ne = brain.net._edge_count
    saved = {k: getattr(brain.net, k).copy() for k in ['V', 'r', 'adaptation', 'f']}
    saved['w'] = brain.net._edge_w[:ne].copy()
    saved['elig'] = brain.net._edge_eligibility[:ne].copy()
    saved['exc'] = brain.net.excitability.copy()

    def get_r(grid):
        signal = grid_to_signal(grid, max_h=h, max_w=w)
        brain.net.V[:] = 0; brain.net.r[:] = 0; brain.net.adaptation[:] = 0
        brain.clear_signal()
        brain.inject_signal(signal)
        brain.step(n_steps=20, learn=False)
        return (brain.net.r[l1_idx].copy(),
                brain.net.r[l2_idx].copy(),
                brain.net.r[l3_idx].copy())

    def restore():
        brain.net.V[:] = saved['V']; brain.net.r[:] = saved['r']
        brain.net.adaptation[:] = saved['adaptation']; brain.net.f[:] = saved['f']
        brain.net._edge_w[:ne] = saved['w']
        brain.net._edge_eligibility[:ne] = saved['elig']
        brain.net.excitability[:] = saved['exc']; brain.clear_signal()

    rng2 = np.random.default_rng(77)
    base_gens = [rectangle, l_shape, cross_shape, color_blocks]
    bases = [g(h, w, rng2) for g in base_gens]

    # --- Transformation tolerance ---
    # For each base, generate shifted + recolored variants.
    # Measure cosine similarity between base and variant at L1 vs L2.
    # If L2 > L1, L2 is abstracting over the transformation.
    l1_trans_sims, l2_trans_sims = [], []
    for base_grid in bases:
        r1_l1, r1_l2, _ = get_r(base_grid); restore()
        for tfm in [shifted_variant, recolored_variant]:
            variant = tfm(base_grid, rng2)
            r2_l1, r2_l2, _ = get_r(variant); restore()
            l1_trans_sims.append(cosine(r1_l1, r2_l1))
            l2_trans_sims.append(cosine(r1_l2, r2_l2))

    trans_tol_l1 = float(np.mean(l1_trans_sims))
    trans_tol_l2 = float(np.mean(l2_trans_sims))
    trans_tol = trans_tol_l2 - trans_tol_l1

    # --- Cross-category separation ---
    # Similarity between structurally different stimuli at L2.
    # Lower = better separation.
    base_l1s, base_l2s, base_l3s = [], [], []
    l2_active_sets: list[set[int]] = []
    l3_active_sets: list[set[int]] = []
    for base_grid in bases:
        r_l1, r_l2, r_l3 = get_r(base_grid); restore()
        base_l1s.append(r_l1)
        base_l2s.append(r_l2)
        base_l3s.append(r_l3)
        l2_active_sets.append(set(int(i) for i in np.where(r_l2 > 0.01)[0]))
        l3_active_sets.append(set(int(i) for i in np.where(r_l3 > 0.01)[0]))

    cross_sims = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            cross_sims.append(cosine(base_l2s[i], base_l2s[j]))
    cat_sep = float(np.mean(cross_sims)) if cross_sims else 0.0

    # --- Driven activity ---
    # Fraction of neurons that fire (r > 0.01) for at least one stimulus
    # vs baseline (blank grid).
    blank = np.zeros((h, w), dtype=np.int32)
    _, blank_l2, blank_l3 = get_r(blank); restore()
    blank_l2_active = set(int(i) for i in np.where(blank_l2 > 0.01)[0])
    blank_l3_active = set(int(i) for i in np.where(blank_l3 > 0.01)[0])

    all_l2_active = set().union(*l2_active_sets) if l2_active_sets else set()
    all_l3_active = set().union(*l3_active_sets) if l3_active_sets else set()
    # Driven = neurons active for stimuli but NOT for blank
    l2_driven = len(all_l2_active - blank_l2_active)
    l3_driven = len(all_l3_active - blank_l3_active)
    l2_driven_frac = l2_driven / len(l2_idx) if len(l2_idx) > 0 else 0.0
    l3_driven_frac = l3_driven / len(l3_idx) if len(l3_idx) > 0 else 0.0

    # --- L2 active set diversity (same as L1's uniq/jaccard) ---
    l2_uniq = len(all_l2_active)
    l2_n_active = float(np.mean([len(s) for s in l2_active_sets])) if l2_active_sets else 0.0
    l2_jaccards = []
    for i in range(len(l2_active_sets)):
        for j in range(i + 1, len(l2_active_sets)):
            a, b = l2_active_sets[i], l2_active_sets[j]
            inter = len(a & b)
            union = len(a | b)
            l2_jaccards.append(inter / union if union > 0 else 1.0)
    l2_jacc = float(np.mean(l2_jaccards)) if l2_jaccards else 1.0

    # --- L3 active set diversity and pattern separation ---
    l3_uniq = len(all_l3_active)
    l3_n_active = float(np.mean([len(s) for s in l3_active_sets])) if l3_active_sets else 0.0
    l3_sparseness = l3_n_active / len(l3_idx) if len(l3_idx) > 0 else 0.0

    l3_jaccards = []
    for i in range(len(l3_active_sets)):
        for j in range(i + 1, len(l3_active_sets)):
            a, b = l3_active_sets[i], l3_active_sets[j]
            inter = len(a & b)
            union = len(a | b)
            l3_jaccards.append(inter / union if union > 0 else 1.0)
    l3_jacc = float(np.mean(l3_jaccards)) if l3_jaccards else 1.0

    # L3 cross-category separation (cosine between different stimuli)
    l3_cross_sims = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            l3_cross_sims.append(cosine(base_l3s[i], base_l3s[j]))
    l3_cat_sep = float(np.mean(l3_cross_sims)) if l3_cross_sims else 0.0

    restore()

    return {
        "l2_trans_tol": trans_tol,
        "l2_trans_l1": trans_tol_l1,
        "l2_trans_l2": trans_tol_l2,
        "l2_cat_sep": cat_sep,
        "l2_driven_frac": l2_driven_frac,
        "l2_uniq": l2_uniq,
        "l2_jacc": l2_jacc,
        "l2_n_active": l2_n_active,
        "l3_driven_frac": l3_driven_frac,
        "l3_uniq": l3_uniq,
        "l3_n_active": l3_n_active,
        "l3_sparseness": l3_sparseness,
        "l3_jacc": l3_jacc,
        "l3_cat_sep": l3_cat_sep,
    }


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

    # L1 probe (existing, trusted metrics)
    l1_mask = np.zeros(brain.net.n_nodes, dtype=bool); l1_mask[l1_idx] = True
    print(f"    probing L1...", end="", flush=True)
    m_l1 = _probe_layer_sim(brain, h, w, l1_mask)

    # L2/L3 probe (layer-appropriate metrics)
    print(f" L2/L3...", end="", flush=True)
    m_hl = l2_l3_diagnostic(brain, h, w)
    print(" done", flush=True)

    ema = brain.homeostasis.ema_rate
    w_abs = np.abs(brain.net._edge_w[:ne])
    w90 = float(np.percentile(w_abs, 90))

    print(f"  [{label:>16s}]  edges={ne:,}  w90={w90:.4f}")
    print(f"    L1: alive={m_l1['uniq']}/{len(l1_idx)}  "
          f"sel={float(ema[l1_idx].std()):.4f}  "
          f"uniq={m_l1['uniq']}  jacc={m_l1['jaccard']:.3f}  "
          f"nAct={m_l1['n_active']:.0f}")
    print(f"    L2: driven={m_hl['l2_driven_frac']:.3f}  "
          f"transTol={m_hl['l2_trans_tol']:+.3f}  "
          f"catSep={m_hl['l2_cat_sep']:.3f}  "
          f"uniq={m_hl['l2_uniq']}  jacc={m_hl['l2_jacc']:.3f}  "
          f"nAct={m_hl['l2_n_active']:.0f}")
    print(f"    L3: driven={m_hl['l3_driven_frac']:.3f}  "
          f"sparse={m_hl['l3_sparseness']:.3f}  "
          f"catSep={m_hl['l3_cat_sep']:.3f}  "
          f"uniq={m_hl['l3_uniq']}  jacc={m_hl['l3_jacc']:.3f}  "
          f"nAct={m_hl['l3_n_active']:.0f}")


# ── UNIFIED LIFE LOOP ─────────────────────────────────────────

def run_life(
    n_days: int = 60,
    checkpoint_path: str | None = None,
    resume_day: int | None = None,
    transition_schedule: dict[int, str] | None = None,
    grid_size: int = 10,
    n_int: int = 2000,
    force_stage: str | None = None,
):
    """
    Single training loop that spans all developmental stages.

    Activity mix (stimuli vs tasks) follows task_mix_ratio from setpoints,
    which the StageManager interpolates during transitions. Stage transitions
    are triggered by the user-provided schedule.
    """
    if transition_schedule is None:
        transition_schedule = {20: "late_infancy", 40: "childhood"}

    genome = Genome()
    genome.n_internal = n_int
    h, w = grid_size, grid_size

    n_l1 = int(genome.n_internal * genome.frac_layer1)
    n_l2 = int(genome.n_internal * genome.frac_layer2)
    n_l3 = genome.n_internal - n_l1 - n_l2

    ckpt_dir = "life/unified"

    # --- Init brain ---
    if checkpoint_path is None:
        brain = Brain.birth(genome, grid_h=h, grid_w=w, seed=42,
                            checkpoint_dir=ckpt_dir)
        start_day = 1
        print(f"\n{'='*60}")
        print(f"  LIFE: BIRTH  grid={grid_size}x{grid_size}, n_int={n_int}")
    else:
        brain = Brain.load_from(genome, checkpoint_path, checkpoint_dir=ckpt_dir)
        start_day = resume_day if resume_day is not None else 1
        print(f"\n{'='*60}")
        print(f"  LIFE: RESUME from {checkpoint_path} at day {start_day}")

    if force_stage is not None:
        old_stage = brain.stage_manager.current_stage
        brain.stage_manager.current_stage = force_stage
        from src.homeostasis.stages import STAGES
        brain.stage_manager._to_setpoints = STAGES[force_stage]
        brain.stage_manager._from_setpoints = STAGES[force_stage]
        brain.stage_manager._transition_progress = 1.0
        print(f"  FORCED stage: {old_stage} -> {force_stage}")

    print(f"  Layers: L1={n_l1}, L2={n_l2}, L3={n_l3}")
    print(f"  Day cycle: {DAY_STEPS} total = {WAKE_STEPS} wake + {SLEEP_STEPS} sleep")
    print(f"  Schedule: {transition_schedule}")
    print(f"  nodes={brain.net.n_nodes}, edges={brain.net._edge_count:,}, "
          f"birth_edges={brain.birth_edges:,}")
    print(f"  stage={brain.stage_manager.current_stage}, "
          f"age={brain.age}, peak_growth={genome.peak_growth_target}x")
    print(f"{'='*60}", flush=True)

    measure(brain, h, w, "START")
    l1_functional_test(brain, h, w)
    l2_functional_test(brain, h, w)
    brain.save_milestone("start")

    # --- Infancy stimuli (always available) ---
    arc_grids = load_arc_inputs("micro_tasks", h, w)
    print(f"  Loaded {len(arc_grids)} ARC input grids for mixing", flush=True)
    stimuli = InfancyStimuli(h, w, rng=brain.rng,
                             arc_input_grids=arc_grids, arc_mix_ratio=0.2)

    # --- Display buffer + gaze logger ---
    display_buffer = DisplayBuffer(h, w)
    gaze_logger = GazeLogger(log_path=f"{ckpt_dir}/gaze_log.jsonl")

    # --- Teacher ---
    monitor = Monitor(log_dir=ckpt_dir, console=True)
    teacher = Teacher(
        brain=brain,
        monitor=monitor,
        display_buffer=display_buffer,
        gaze_logger=gaze_logger,
        task_dir="micro_tasks",
        observe_steps=25,
        attempt_steps=40,
    )

    # WTA ramp during pure infancy (before any transition starts)
    wta_start, wta_end, wta_ramp_days = 0.3, 0.15, 15
    infancy_wta_active = True

    for day in range(start_day, start_day + n_days):
        t0 = time.time()

        # --- Check for scheduled stage transition ---
        if day in transition_schedule:
            target_stage = transition_schedule[day]
            if brain.stage_manager.current_stage != target_stage:
                brain.stage_manager.transition_to(target_stage)
                infancy_wta_active = False
                print(f"\n  >>> TRANSITION to {target_stage} at day {day} "
                      f"(tau={brain.stage_manager.transition_tau})", flush=True)

        # --- Manual WTA ramp during pure infancy ---
        if infancy_wta_active and brain.stage_manager.current_stage == "infancy":
            t_ramp = min(day / wta_ramp_days, 1.0)
            wta_frac = wta_start + t_ramp * (wta_end - wta_start)
            brain.stage_manager._to_setpoints.wta_active_frac = wta_frac

        sp = brain.stage_manager.current_setpoints()
        task_ratio = sp.task_mix_ratio
        babble_ratio = sp.babble_ratio
        wake_used = 0
        tasks_attempted = 0
        tasks_solved = 0
        n_babbles = 0
        n_mimicry = 0

        # --- Waking: three-way activity selection ---
        teacher.state.tasks_today = 0
        teacher.state.solves_today = 0
        teacher.state.day += 1

        # Refresh display buffer with infancy stimuli each day
        # (provides consistent structured patterns within a day)
        display_buffer.load_infancy_stimuli(brain.rng)
        stim_refresh_counter = 0
        STIM_REFRESH_INTERVAL = 200  # refresh after ~200 steps

        while wake_used + STEPS_PER_STIMULUS <= WAKE_STEPS:
            roll = brain.rng.random()

            if roll < babble_ratio:
                # Motor activity: pure babble or stimulus mimicry
                if brain.rng.random() < 0.3:
                    # Motor babble: gaze-routed visual feedback
                    brain.motor_babble(
                        display_buffer=display_buffer,
                        gaze_logger=gaze_logger,
                    )
                    wake_used += BABBLE_STEPS
                    n_babbles += 1
                else:
                    # Stimulus mimicry: guide gaze to a random stimulus
                    # slot, then mimicry with gaze-routed feedback
                    stim_slot = int(brain.rng.integers(
                        0, display_buffer.answer_slot))
                    brain.guide_gaze(stim_slot, strength=0.5)
                    slot = brain.apply_gaze(display_buffer)
                    if gaze_logger:
                        stype = display_buffer.slot_types[slot]
                        gaze_logger.record(brain.age, slot, stype)

                    signal = grid_to_signal(
                        display_buffer.get_slot(slot),
                        max_h=h, max_w=w,
                    )
                    brain.store_signal(signal)
                    brain.stimulus_with_feedback(
                        signal,
                        display_buffer=display_buffer,
                        gaze_logger=gaze_logger,
                    )
                    wake_used += MIMICRY_STEPS
                    n_mimicry += 1
            elif roll < babble_ratio + task_ratio:
                pick = teacher.pick_task()
                if pick is not None:
                    task_type, task = pick
                    age_before = brain.age
                    correct = teacher.run_task(task_type, task)
                    task_steps = brain.age - age_before
                    wake_used += task_steps
                    tasks_attempted += 1
                    if correct:
                        tasks_solved += 1
                    # Restore infancy stimuli after task
                    display_buffer.load_infancy_stimuli(brain.rng)
                else:
                    # No task available — temporal contiguity observation
                    from src.stimuli import random_variant
                    base_grid = stimuli.generate()
                    views = [base_grid,
                             random_variant(base_grid, brain.rng),
                             random_variant(base_grid, brain.rng)]
                    for vi, v_grid in enumerate(views):
                        signal = grid_to_signal(v_grid, max_h=h, max_w=w)
                        brain.inject_signal(signal)
                        brain.store_signal(signal)
                        brain.step(n_steps=OBSERVE_STEPS)
                        brain.clear_signal()
                        brain.step(n_steps=TEMPORAL_REST)
                        if vi == 0:
                            brain.decorrelate_layers()
                    brain.decorrelate_layers()
                    wake_used += (OBSERVE_STEPS + TEMPORAL_REST) * 3
            else:
                # Passive sensory observation with temporal contiguity:
                # show a base pattern then 2 variants in sequence so the
                # trace rule can bind them as "same object, different view."
                from src.stimuli import random_variant
                base_grid = stimuli.generate()
                views = [base_grid,
                         random_variant(base_grid, brain.rng),
                         random_variant(base_grid, brain.rng)]
                for vi, v_grid in enumerate(views):
                    signal = grid_to_signal(v_grid, max_h=h, max_w=w)
                    brain.inject_signal(signal)
                    brain.store_signal(signal)
                    brain.step(n_steps=OBSERVE_STEPS)
                    brain.clear_signal()
                    brain.step(n_steps=TEMPORAL_REST)
                    if vi == 0:
                        brain.decorrelate_layers()
                brain.decorrelate_layers()
                wake_used += (OBSERVE_STEPS + TEMPORAL_REST) * 3

            # Periodically refresh display buffer stimuli
            stim_refresh_counter += STEPS_PER_STIMULUS
            if stim_refresh_counter >= STIM_REFRESH_INTERVAL:
                display_buffer.load_infancy_stimuli(brain.rng)
                stim_refresh_counter = 0

            brain.decorrelate_layers()

        # --- Free waking time ---
        free_steps = WAKE_STEPS - wake_used
        if free_steps > 0:
            brain.clear_signal()
            brain.step(n_steps=free_steps)

        # --- Sleep ---
        brain.clear_signal()
        brain.step(n_steps=SLEEP_STEPS)
        brain.try_sleep()
        brain.fatigue.reset()

        dt = time.time() - t0

        # --- Diagnostics ---
        ne = brain.net._edge_count
        growth_ratio = ne / max(1, brain.birth_edges)
        density_f = max(0.0, 1.0 - growth_ratio / genome.peak_growth_target)
        ema_d = brain.homeostasis.ema_rate_dendritic
        demand_f = float((ema_d < sp.target_rate).mean())

        stage_tag = brain.stage_manager.current_stage[:4]
        trans = brain.stage_manager._transition_progress

        # Per-layer edge counts
        from src.graph import layer_index as _li
        _src_l = np.array([_li(int(r)) for r in brain.net.regions[brain.net._edge_src[:ne]]])
        _dst_l = np.array([_li(int(r)) for r in brain.net.regions[brain.net._edge_dst[:ne]]])
        _src_l[_src_l < 0] = 0
        _dst_l[_dst_l < 0] = 0
        _edge_layer = np.maximum(_src_l, _dst_l)
        n_l1_edges = int((_edge_layer == 0).sum())
        n_l2_edges = int((_edge_layer == 1).sum())
        n_l3_edges = int((_edge_layer == 2).sum())

        print(f"\n  --- Day {day} [{stage_tag}] ({dt:.0f}s) ---", flush=True)
        print(f"  babble={n_babbles}  mimicry={n_mimicry}  "
              f"tasks={tasks_solved}/{tasks_attempted}  free={free_steps}", flush=True)
        print(f"  growth={growth_ratio:.2f}x  density_f={density_f:.3f}  "
              f"demand_f={demand_f:.3f}  trans={trans:.3f}  "
              f"wta={sp.wta_active_frac:.3f}  "
              f"babble_r={babble_ratio:.2f}  task_r={task_ratio:.2f}", flush=True)
        print(f"  edges: total={ne:,}  L1={n_l1_edges:,}  "
              f"L2={n_l2_edges:,}  L3={n_l3_edges:,}", flush=True)

        measure(brain, h, w, f"DAY {day} ({dt:.0f}s)")

        # Gaze summary (every 5 days or on checkpoint days)
        if gaze_logger and (day % 5 == 0 or day in CHECKPOINT_DAYS):
            events = gaze_logger.get_events()
            if events:
                print(f"  Gaze behavior (day {day}):", flush=True)
                print_gaze_summary(events, display_buffer.n_slots,
                                   display_buffer.answer_slot)
            gaze_logger.flush()
            gaze_logger.clear()

        if day in FUNC_TEST_DAYS:
            l1_functional_test(brain, h, w)
            l2_functional_test(brain, h, w)
        if day in CHECKPOINT_DAYS:
            path = brain.save_milestone(f"day{day}")
            print(f"  Saved checkpoint: {path}", flush=True)

    # Final gaze summary
    if gaze_logger:
        events = gaze_logger.get_events()
        if events:
            print(f"\n  Final gaze summary:", flush=True)
            print_gaze_summary(events, display_buffer.n_slots,
                               display_buffer.answer_slot)
        gaze_logger.flush()

    print(f"\n  DONE. Final edges: {brain.net._edge_count:,}  "
          f"growth={brain.net._edge_count / max(1, brain.birth_edges):.2f}x", flush=True)
    if tasks_attempted > 0:
        print(f"  Final curriculum tier: {teacher.state.current_tier}", flush=True)
        for t_name, tracker in teacher.state.type_trackers.items():
            if tracker.attempts > 0:
                print(f"    {t_name}: {tracker.solves}/{tracker.attempts} "
                      f"({100*tracker.solve_rate:.0f}%) "
                      f"{'MASTERED' if tracker.mastered else ''}", flush=True)


if __name__ == "__main__":
    # Parse: python diag_grid10.py [n_days] [checkpoint_path] [day:stage ...]
    n_days = 60
    ckpt = None
    schedule = {}

    args = sys.argv[1:]
    positional = []
    forced_stage = None
    for arg in args:
        if arg.startswith("--force-stage="):
            forced_stage = arg.split("=", 1)[1]
        elif ':' in arg:
            day_str, stage = arg.split(':', 1)
            schedule[int(day_str)] = stage
        else:
            positional.append(arg)

    resume_day = None
    if len(positional) >= 1:
        n_days = int(positional[0])
    if len(positional) >= 2:
        ckpt = positional[1]
    if len(positional) >= 3:
        resume_day = int(positional[2])

    if not schedule:
        schedule = {20: "late_infancy", 40: "childhood"}

    run_life(
        n_days=n_days,
        checkpoint_path=ckpt,
        resume_day=resume_day,
        transition_schedule=schedule,
        force_stage=forced_stage,
    )
