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
from src.encoding import grid_to_signal
from src.stimuli import InfancyStimuli, load_arc_inputs
from src.graph import Region
from src.teacher import Teacher
from src.monitor import Monitor

DAY_STEPS = 2100
WAKE_STEPS = 1400
SLEEP_STEPS = DAY_STEPS - WAKE_STEPS
OBSERVE_STEPS = 25
REST_STEPS = 10
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
        brain.net.V[:] = 0.0; brain.net.r[:] = 0.0
        brain.net.f[:] = 0.0; brain.net.adaptation[:] = 0.0
        brain.clear_signal()
        brain.inject_signal(signal)
        brain.step(n_steps=20)
        ever_won |= brain.net.r > 0.01
    brain.net.V[:] = saved_V; brain.net.r[:] = saved_r
    brain.net.f[:] = saved_f; brain.net.adaptation[:] = saved_adapt
    brain.clear_signal()

    l1_mask = np.zeros(brain.net.n_nodes, dtype=bool); l1_mask[l1_idx] = True
    l2_mask = np.zeros(brain.net.n_nodes, dtype=bool); l2_mask[l2_idx] = True
    l3_mask = np.zeros(brain.net.n_nodes, dtype=bool); l3_mask[l3_idx] = True

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

    print(f"  [{label:>16s}]  edges={ne:,}  w90={w90:.4f}")
    print(f"    L1: alive={int(ever_won[l1_idx].sum()):>4}/{len(l1_idx)}  "
          f"sel={float(ema[l1_idx].std()):.4f}  "
          f"simV={m_l1['sim_v']:.3f}  simR={m_l1['sim_r']:.3f}  simCol={m_l1['sim_col']:.3f}")
    print(f"    L2: alive={int(ever_won[l2_idx].sum()):>4}/{len(l2_idx)}  "
          f"sel={float(ema[l2_idx].std()):.4f}  "
          f"simV={m_l2['sim_v']:.3f}  simR={m_l2['sim_r']:.3f}  simCol={m_l2['sim_col']:.3f}")
    print(f"    L3: alive={int(ever_won[l3_idx].sum()):>4}/{len(l3_idx)}  "
          f"sel={float(ema[l3_idx].std()):.4f}  "
          f"simV={m_l3['sim_v']:.3f}  simR={m_l3['sim_r']:.3f}  simCol={m_l3['sim_col']:.3f}")


# ── UNIFIED LIFE LOOP ─────────────────────────────────────────

def run_life(
    n_days: int = 60,
    checkpoint_path: str | None = None,
    transition_schedule: dict[int, str] | None = None,
    grid_size: int = 10,
    n_int: int = 1500,
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
        start_day = 1
        print(f"\n{'='*60}")
        print(f"  LIFE: RESUME from {checkpoint_path}")

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

    # --- Teacher (lazy init, only used when tasks are selected) ---
    monitor = Monitor(log_dir=ckpt_dir, console=True)
    teacher = Teacher(
        brain=brain,
        monitor=monitor,
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

        while wake_used + STEPS_PER_STIMULUS <= WAKE_STEPS:
            roll = brain.rng.random()

            if roll < babble_ratio:
                # Motor activity: pure babble or stimulus mimicry
                if brain.rng.random() < 0.3:
                    brain.motor_babble()
                    wake_used += BABBLE_STEPS
                    n_babbles += 1
                else:
                    grid = stimuli.generate()
                    signal = grid_to_signal(grid, max_h=h, max_w=w)
                    brain.store_signal(signal)
                    brain.stimulus_with_feedback(signal)
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
                else:
                    grid = stimuli.generate()
                    signal = grid_to_signal(grid, max_h=h, max_w=w)
                    brain.store_signal(signal)
                    brain.inject_signal(signal)
                    brain.step(n_steps=OBSERVE_STEPS)
                    brain.clear_signal()
                    brain.step(n_steps=REST_STEPS)
                    wake_used += STEPS_PER_STIMULUS
            else:
                # Sensory stimulus (no motor feedback)
                grid = stimuli.generate()
                signal = grid_to_signal(grid, max_h=h, max_w=w)
                brain.store_signal(signal)
                brain.inject_signal(signal)
                brain.step(n_steps=OBSERVE_STEPS)
                brain.clear_signal()
                brain.step(n_steps=REST_STEPS)
                wake_used += STEPS_PER_STIMULUS

            # Anti-Hebbian decorrelation after every activity.
            # Safe even after babble/mimicry because learn=False during
            # feedback phases prevents L1 from learning noisy features.
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

        if day in FUNC_TEST_DAYS:
            l1_functional_test(brain, h, w)
            l2_functional_test(brain, h, w)
        if day in CHECKPOINT_DAYS:
            path = brain.save_milestone(f"day{day}")
            print(f"  Saved checkpoint: {path}", flush=True)

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
    for arg in args:
        if ':' in arg:
            day_str, stage = arg.split(':', 1)
            schedule[int(day_str)] = stage
        else:
            positional.append(arg)

    if len(positional) >= 1:
        n_days = int(positional[0])
    if len(positional) >= 2:
        ckpt = positional[1]

    if not schedule:
        schedule = {20: "late_infancy", 40: "childhood"}

    run_life(
        n_days=n_days,
        checkpoint_path=ckpt,
        transition_schedule=schedule,
    )
