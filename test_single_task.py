"""
Single-task sanity tests: can the fixed architecture learn transformations?

Tests progressively harder tasks:
1. Color swap (1<->2) on 3x3
2. Horizontal flip on 3x3
3. Both tasks simultaneously (multi-task)
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np

from src.genome import Genome
from src.template import create_dng
from src.pipeline import LifecycleConfig, study_task_round, study_day, solve_task
from src.episodic_memory import EpisodicMemory
from src.graph import Region

MAX_H = MAX_W = 5

def make_config():
    return LifecycleConfig(
        observe_steps=40, think_steps=60, consolidation_steps=20,
        eta=0.05, w_max=2.5,
        free_phase_steps=40, clamped_phase_steps=40,
        attempts_per_round=6, n_rounds=4, stuck_patience=2,
        noise_std=0.02, focus_strength=0.5,
        rest_steps=20, rest_noise_std=0.03,
        sleep_downscale=0.995, sleep_tag_threshold=0.001,
        prune_weak_threshold=0.0005, prune_cycles_required=20,
        growth_rate=0.5, growth_candidates=50000,
    )

def make_net(seed=42):
    genome = Genome(
        max_h=MAX_H, max_w=MAX_W,
        n_internal=400, n_concept=80, n_memory=120,
        max_fan_in=100, weight_scale=0.02, frac_inhibitory=0.25,
        f_rate=0.05, f_decay=0.02, f_max=3.0,
    )
    rng = np.random.default_rng(seed)
    net = create_dng(genome, MAX_H, MAX_W, rng)
    net.wta_k_frac = 0.25
    net.threshold[:] = 0.15
    motor_mask = net.regions == list(Region).index(Region.MOTOR)
    net.threshold[motor_mask] = 0.05
    net.adapt_rate = 0.05
    return net, rng

def evaluate(net, train_pairs, test_in, test_out, episodic=None):
    eval_cfg = LifecycleConfig(
        observe_steps=40, think_steps=80, noise_std=0.02,
        focus_strength=0.5, t_max=200,
    )
    result = solve_task(net, train_pairs, test_in, eval_cfg,
                        output_h=test_out.shape[0], output_w=test_out.shape[1],
                        episodic=episodic)
    fg = test_out != 0
    fg_acc = float(np.mean(result.grid[fg] == test_out[fg])) if fg.any() else 0.0
    return fg_acc, result.grid

# ── Task 1: Color swap ────────────────────────────────────────────
swap_train = [
    (np.array([[1,2,0],[0,1,2],[2,0,1]]), np.array([[2,1,0],[0,2,1],[1,0,2]])),
    (np.array([[1,1,2],[2,2,1],[1,2,2]]), np.array([[2,2,1],[1,1,2],[2,1,1]])),
    (np.array([[0,1,0],[2,0,2],[0,1,0]]), np.array([[0,2,0],[1,0,1],[0,2,0]])),
]
swap_test_in = np.array([[2,1,2],[1,2,1],[2,1,2]])
swap_test_out = np.array([[1,2,1],[2,1,2],[1,2,1]])

# ── Task 2: Horizontal flip ──────────────────────────────────────
flip_train = [
    (np.array([[1,0,3],[0,2,0],[4,0,5]]), np.array([[3,0,1],[0,2,0],[5,0,4]])),
    (np.array([[6,7,0],[0,0,0],[0,8,9]]), np.array([[0,7,6],[0,0,0],[9,8,0]])),
    (np.array([[1,2,3],[4,5,6],[7,8,9]]), np.array([[3,2,1],[6,5,4],[9,8,7]])),
]
flip_test_in = np.array([[0,0,1],[0,2,0],[3,0,0]])
flip_test_out = np.array([[1,0,0],[0,2,0],[0,0,3]])

def run_test(name, train_pairs, test_in, test_out, n_days=50, seed=42):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    net, rng = make_net(seed)
    config = make_config()
    print(f"Network: {net.n_nodes} nodes, {net.edge_count()} edges")

    from src.plasticity import sleep_selective, prune_sustained, homeostatic_excitability_update
    ema_r = None

    for day in range(1, n_days + 1):
        net.ach = 1.0; net.da = 0.0; net.ne = 0.0

        result = study_task_round(
            net, train_pairs, test_in, test_out, config,
            prev_best_reward=0.0, is_first_visit=True,
        )

        sleep_selective(net, config.sleep_downscale, config.sleep_tag_threshold)
        prune_sustained(net, config.prune_weak_threshold, config.prune_cycles_required)
        ema_r = homeostatic_excitability_update(net, ema_r=ema_r)

        if day <= 5 or day % 10 == 0 or result.reward >= 1.0:
            fg_acc, grid = evaluate(net, train_pairs, test_in, test_out)
            print(f"  Day {day:3d}: train={result.reward:.0%}({result.n_attempts}) "
                  f"eval_fg={fg_acc:.0%}  out={grid.tolist()}", flush=True)
            if fg_acc >= 1.0:
                print(f"  -> SOLVED on day {day}!")
                return day
    print(f"  -> NOT SOLVED after {n_days} days")
    return None

# ── Run tests ─────────────────────────────────────────────────────
t0 = time.time()

d1 = run_test("Color swap (1<->2)", swap_train, swap_test_in, swap_test_out, n_days=50)
d2 = run_test("Horizontal flip", flip_train, flip_test_in, flip_test_out, n_days=50)

# ── Multi-task test (with episodic memory) ───────────────────────
print(f"\n{'='*60}")
print(f"TEST: Multi-task (swap + flip, episodic memory buffer)")
print(f"{'='*60}")
net, rng = make_net(99)
config = make_config()
config.n_rounds = 2
config.attempts_per_round = 4
config.eta = 0.03             # strong CHL
config.replay_eta = 0.03      # match waking strength
config.replay_passes = 10
config.sleep_downscale = 1.0
config.prune_weak_threshold = 0.0
config.prune_cycles_required = 999
net.wta_k_frac = 0.05         # very sparse: ~24 abstract neurons active
print(f"Network: {net.n_nodes} nodes, {net.edge_count()} edges")

ema_r = None
replay_buffer = {}

day_tasks = [
    (swap_train, swap_test_in, swap_test_out),
    (flip_train, flip_test_in, flip_test_out),
]

for day in range(1, 31):
    net.ach = 1.0; net.da = 0.0; net.ne = 0.0

    day_result, ema_r = study_day(net, day_tasks, config, ema_r, rng,
                                  episodic=None, replay_buffer=replay_buffer)

    if day <= 10 or day % 5 == 0:
        acc_swap, g_swap = evaluate(net, swap_train, swap_test_in, swap_test_out)
        acc_flip, g_flip = evaluate(net, flip_train, flip_test_in, flip_test_out)
        rewards = [f"{a.reward:.0%}" for a in day_result.attempts]
        cons = net._edge_consolidation[:net._edge_count]
        n_cons = int((cons > 0.1).sum())
        detail = ""
        if day <= 10:
            detail = f"  s={g_swap.tolist()} f={g_flip.tolist()}"
        print(f"  Day {day:3d}: swap={acc_swap:.0%} flip={acc_flip:.0%} "
              f"train={rewards} cons={n_cons}{detail}", flush=True)

print(f"\nTotal time: {time.time()-t0:.0f}s")
