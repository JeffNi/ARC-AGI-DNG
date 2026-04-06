"""
Infancy phase: 300 days of unsupervised concept formation.

A baby doesn't solve puzzles. It LOOKS at the world while its brain
grows explosively. By the time it starts stacking blocks, it already
has a rich vocabulary of visual features that self-organized through
exposure + Hebbian learning + pruning.

All developmental parameters are smooth, continuous functions of
maturity (0 → 1 over 300 days). Nothing changes abruptly.

Checkpointing: saves full training state every CHECKPOINT_EVERY days.
Resume: pass --resume to pick up from the latest checkpoint.

Usage:
  python train_infancy.py                 # start fresh
  python train_infancy.py --resume        # resume from latest checkpoint
  python train_infancy.py --days 500      # override total days
"""
import sys, time, os, argparse, json
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.childhood import extract_tasks
from src.curriculum import sort_by_difficulty
from src.pipeline import (
    LifecycleConfig, nursery_exposure, solve_task, NurseryResult,
    study_task_round, sleep, rest, _soft_reset,
)
from src.patterns import generate_all
from src.encoding import NUM_COLORS, grid_to_signal
from src.dynamics import think
from src.graph import DNG, Region
from src.episodic_memory import EpisodicMemory
from src.plasticity import synaptogenesis

# ── CLI ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run infancy training")
parser.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint")
parser.add_argument("--days", type=int, default=300,
                    help="Total infancy days (default: 300)")
parser.add_argument("--grids-per-day", type=int, default=100,
                    help="Grids presented per day (default: 100)")
parser.add_argument("--checkpoint-every", type=int, default=10,
                    help="Save checkpoint every N days (default: 10)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
parser.add_argument("--tag", type=str, default=None,
                    help="Checkpoint subdirectory tag (e.g. 'v2')")
args = parser.parse_args()

# ── Constants ─────────────────────────────────────────────────────

MAX_H = MAX_W = 10
N_INFANCY_DAYS = args.days
GRIDS_PER_DAY = args.grids_per_day
CHECKPOINT_EVERY = args.checkpoint_every
_ckpt_name = f"infancy_{args.tag}" if args.tag else "infancy_checkpoints"
CHECKPOINT_DIR = os.path.join("models", _ckpt_name)
N_TASK_TEST = 12

# ── Scaled-up genome ─────────────────────────────────────────────

GENOME = Genome(
    max_h=MAX_H, max_w=MAX_W,
    n_internal=1500, n_concept=200, n_memory=300,
    max_fan_in=200, weight_scale=0.01,
)

CONFIG = LifecycleConfig(
    observe_steps=30,
    think_steps=40,
    free_phase_steps=40,
    clamped_phase_steps=40,
    eta=0.05,
    w_max=2.5,
    noise_std=0.02,
    focus_strength=0.5,
    attempts_per_round=3,
    n_rounds=1,
    rest_steps=15,
    rest_noise_std=0.03,
    sleep_downscale=0.998,
    sleep_tag_threshold=0.001,
    prune_weak_threshold=0.003,
    prune_cycles_required=20,
    replay_eta=0.005,
    replay_passes=2,
    replay_steps=20,
    memory_hint_strength=1.5,
    spontaneous_strength=0.2,
    nursery_binding_eta=0.08,
    nursery_growth_rate=0.8,
    nursery_growth_candidates=500000,
)

# ── Checkpointing ────────────────────────────────────────────────

def save_checkpoint(net, day, maturity, ema_r, rng_state, path):
    """Save complete training state so we can resume exactly.

    Crucially saves maturity and developmental hyperparams so that on
    resume the network picks up with a smooth continuation -- no sudden
    parameter jumps even if --days changes between runs.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    net_path = path + ".net.npz"
    net.save(net_path)
    state = {
        "day": day,
        "maturity": maturity,
        "total_days": N_INFANCY_DAYS,
        "grids_per_day": GRIDS_PER_DAY,
        # Snapshot the developmental hyperparams that were active
        "dev": {
            "base_wta_k_frac": net.wta_k_frac,
            "nursery_binding_eta": CONFIG.nursery_binding_eta,
            "nursery_growth_rate": CONFIG.nursery_growth_rate,
            "nursery_growth_candidates": CONFIG.nursery_growth_candidates,
            "sleep_downscale": CONFIG.sleep_downscale,
            "prune_weak_threshold": CONFIG.prune_weak_threshold,
            "prune_cycles_required": CONFIG.prune_cycles_required,
        },
    }
    state_path = path + ".state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    thresh_path = path + ".threshold.npy"
    np.save(thresh_path, net.threshold)
    # base_threshold is saved via the net's threshold at checkpoint time
    # (nursery_exposure sets net.threshold = base * scale, so we derive
    # base from current threshold / scale). Saved separately for clarity.
    ema_path = path + ".ema.npy"
    if ema_r is not None:
        np.save(ema_path, ema_r)
    rng_path = path + ".rng.npz"
    np.savez(rng_path, **{k: v for k, v in rng_state.items()})
    print(f"  [checkpoint] saved day {day} (maturity={maturity:.4f}) -> {path}")


def load_checkpoint(path):
    """Load complete training state including developmental params."""
    net_path = path + ".net.npz"
    net = DNG.load(net_path)
    state_path = path + ".state.json"
    with open(state_path, "r") as f:
        state = json.load(f)
    # Restore saved base thresholds if available
    thresh_path = path + ".threshold.npy"
    if os.path.exists(thresh_path):
        net.threshold[:] = np.load(thresh_path)
    ema_path = path + ".ema.npy"
    ema_r = np.load(ema_path) if os.path.exists(ema_path) else None
    rng_path = path + ".rng.npz"
    rng_state = {}
    if os.path.exists(rng_path):
        data = np.load(rng_path, allow_pickle=True)
        rng_state = {k: data[k] for k in data.files}
    return net, state, ema_r, rng_state


def find_latest_checkpoint():
    """Find the most recent checkpoint by day number."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    best_day = -1
    best_path = None
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.endswith(".state.json"):
            base = fname[:-len(".state.json")]
            try:
                with open(os.path.join(CHECKPOINT_DIR, fname), "r") as f:
                    state = json.load(f)
                day = state["day"]
                if day > best_day:
                    best_day = day
                    best_path = os.path.join(CHECKPOINT_DIR, base)
            except (json.JSONDecodeError, KeyError):
                continue
    return best_path

# ── Representational quality metrics ──────────────────────────────

def measure_representations(net, test_grids, config, rng, label=""):
    """Present a fixed set of grids and measure internal neuron responses."""
    abstract_idx = list(Region).index(Region.ABSTRACT)
    internal_nodes = np.where(net.regions == abstract_idx)[0]
    n_internal = len(internal_nodes)
    mh, mw = net.max_h, net.max_w
    n_total = net.n_nodes

    from src.pipeline import _focus_mask

    activations = []
    for grid in test_grids:
        grid = np.asarray(grid)
        gh, gw = grid.shape
        net.V[:] *= 0.1
        net.r[:] *= 0.1
        net.prev_r[:] *= 0.1
        focus = _focus_mask(net, gh, gw, config.focus_strength)
        sig = grid_to_signal(grid, 0, n_total, max_h=mh, max_w=mw) + focus
        think(net, signal=sig, steps=30, noise_std=0.01)
        activations.append(net.r[internal_nodes].copy())

    A = np.array(activations)

    active_frac = np.mean(A > 0.05, axis=1)
    mean_sparsity = 1.0 - float(np.mean(active_frac))

    neuron_std = np.std(A, axis=0)
    neuron_mean = np.mean(A, axis=0)
    alive = neuron_mean > 0.01
    n_alive = int(alive.sum())
    mean_selectivity = float(np.mean(neuron_std[alive])) if n_alive > 0 else 0.0

    n_grids = len(test_grids)
    if n_grids >= 2:
        sample_idx = rng.choice(n_grids, size=min(30, n_grids), replace=False)
        sample_A = A[sample_idx]
        norms = np.linalg.norm(sample_A, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = sample_A / norms
        cos_sim = normed @ normed.T
        np.fill_diagonal(cos_sim, 0)
        mean_diversity = 1.0 - float(np.mean(np.abs(cos_sim)))
    else:
        mean_diversity = 0.0

    consistency_scores = []
    for i in range(min(10, n_grids)):
        grid = np.asarray(test_grids[i])
        gh, gw = grid.shape
        net.V[:] *= 0.1
        net.r[:] *= 0.1
        net.prev_r[:] *= 0.1
        focus = _focus_mask(net, gh, gw, config.focus_strength)
        sig = grid_to_signal(grid, 0, n_total, max_h=mh, max_w=mw) + focus
        think(net, signal=sig, steps=30, noise_std=0.01)
        act2 = net.r[internal_nodes].copy()
        norm1 = np.linalg.norm(activations[i])
        norm2 = np.linalg.norm(act2)
        if norm1 > 1e-8 and norm2 > 1e-8:
            corr = float(np.dot(activations[i], act2) / (norm1 * norm2))
            consistency_scores.append(corr)
    mean_consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.0

    return {
        'n_alive': n_alive,
        'n_internal': n_internal,
        'sparsity': mean_sparsity,
        'selectivity': mean_selectivity,
        'diversity': mean_diversity,
        'consistency': mean_consistency,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(args.seed)
    start_day = 1
    ema_r = None

    # Maturity schedule: by default, ramp 0→1 over all days.
    # On resume, we pick up from the saved maturity and ramp to 1.0
    # over the remaining days -- so even if --days changes, there's
    # no sudden jump in developmental parameters.
    resume_maturity = 0.0  # maturity at (start_day - 1)

    if args.resume:
        ckpt_path = find_latest_checkpoint()
        if ckpt_path is not None:
            print(f"Resuming from checkpoint: {ckpt_path}")
            net, state, ema_r, rng_state = load_checkpoint(ckpt_path)
            start_day = state["day"] + 1
            resume_maturity = state.get("maturity", (start_day - 2) / max(1, N_INFANCY_DAYS - 1))
            if "dev" in state:
                dev = state["dev"]
                print(f"  Saved dev params: wta_k={dev.get('base_wta_k_frac')}, "
                      f"bind_eta={dev.get('nursery_binding_eta')}, "
                      f"growth={dev.get('nursery_growth_rate')}")
            if rng_state:
                try:
                    rng_bit_gen = np.random.PCG64()
                    rng_bit_gen.state = {
                        'bit_generator': 'PCG64',
                        'state': {'state': int(rng_state['state'].flat[0]),
                                  'inc': int(rng_state['inc'].flat[0])},
                        'has_uint32': 0, 'uinteger': 0,
                    }
                    rng = np.random.Generator(rng_bit_gen)
                except Exception:
                    rng = np.random.default_rng(args.seed + start_day)
            print(f"  Resumed at day {start_day}, maturity={resume_maturity:.4f}, "
                  f"edges={net.edge_count():,}")
        else:
            print("No checkpoint found, starting fresh.")
            args.resume = False

    if not args.resume:
        print("Creating fresh network...")
        net = create_dng(GENOME, MAX_H, MAX_W, rng=rng)
        net.adapt_rate = 0.01
        net.wta_k_frac = 0.30  # newborn: weak competition, tightens with maturity
        net.inh_scale = 0.3    # newborn: inhibition immature, grows with maturity
        net.threshold[:] = 0.15
        motor_mask = net.regions == list(Region).index(Region.MOTOR)
        net.threshold[motor_mask] = 0.05

    print(f"Network: {net.n_nodes} nodes, {net.edge_count():,} edges")

    abstract_idx = list(Region).index(Region.ABSTRACT)
    n_internal = int((net.regions == abstract_idx).sum())
    print(f"Internal neurons: {n_internal}")

    # ── Generate exposure pool ─────────────────────────────────────
    print("Generating visual patterns...", flush=True)
    pattern_rng = np.random.default_rng(args.seed + 1000)
    all_patterns = generate_all(MAX_H, MAX_W, rng=pattern_rng, n_per_type=40)
    print(f"  {len(all_patterns)} patterns from generators")

    test_rng = np.random.default_rng(99)
    test_patterns = generate_all(MAX_H, MAX_W, rng=test_rng, n_per_type=5)
    print(f"  {len(test_patterns)} test patterns for metrics")

    train_set, _ = arckit.load_data()
    arc_grids = []
    for task in train_set:
        for inp, _ in task.train:
            inp = np.array(inp)
            if inp.shape[0] <= MAX_H and inp.shape[1] <= MAX_W:
                arc_grids.append(inp)
    print(f"  {len(arc_grids)} real ARC input grids")

    exposure_pool = all_patterns + arc_grids
    rng.shuffle(exposure_pool)
    print(f"  {len(exposure_pool)} total exposure grids")

    # ── Baseline (only if starting fresh) ──────────────────────────
    if start_day == 1:
        print(f"\n{'='*60}")
        print("BASELINE REPRESENTATIONS")
        print(f"{'='*60}")
        baseline = measure_representations(net, test_patterns, CONFIG, rng)
        print(f"  Alive neurons: {baseline['n_alive']}/{baseline['n_internal']}")
        print(f"  Sparsity:      {baseline['sparsity']:.3f}")
        print(f"  Selectivity:   {baseline['selectivity']:.4f}")
        print(f"  Diversity:     {baseline['diversity']:.3f}")
        print(f"  Consistency:   {baseline['consistency']:.3f}")

    # ── Infancy loop ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"INFANCY ({N_INFANCY_DAYS} days, {GRIDS_PER_DAY} grids/day)")
    if start_day > 1:
        print(f"  (resuming from day {start_day})")
    print(f"{'='*60}")

    t_start = time.time()

    remaining_days = max(1, N_INFANCY_DAYS - start_day + 1)
    base_threshold = net.threshold.copy()

    for day in range(start_day, N_INFANCY_DAYS + 1):
        # Maturity: smooth ramp from resume_maturity -> 1.0 over remaining days.
        frac = (day - start_day) / max(1, remaining_days - 1)
        maturity = resume_maturity + (1.0 - resume_maturity) * frac

        day_indices = rng.integers(0, len(exposure_pool), size=GRIDS_PER_DAY)
        day_grids = [exposure_pool[i] for i in day_indices]

        result, ema_r = nursery_exposure(
            net, day_grids, CONFIG, rng=rng, ema_r=ema_r, maturity=maturity,
            base_threshold=base_threshold,
        )

        elapsed = time.time() - t_start

        should_log = (day <= 10 or day % 5 == 0 or
                      day == N_INFANCY_DAYS or day == start_day)
        if should_log:
            print(f"  Day {day:4d}/{N_INFANCY_DAYS} (mat={maturity:.3f}): "
                  f"edges={result.edges_after:,} "
                  f"bound={result.edges_bound:,} "
                  f"+{result.edges_grown:,}/-{result.edges_pruned} "
                  f"({elapsed:.0f}s)", flush=True)
            ss = result.synaptogenesis_stats
            if ss is not None:
                print(f"         syngen: active={ss.n_active} silent={ss.n_silent} "
                      f"mean_r={ss.mean_r_active:.3f} "
                      f"cands={ss.n_candidates} novel={ss.n_novel} "
                      f"reject={ss.reject_existing_frac:.1%} "
                      f"mean_p={ss.mean_prob:.4f} "
                      f"created={ss.n_created}", flush=True)

        # Periodic metrics
        should_measure = (day % 15 == 0 or day == N_INFANCY_DAYS or
                          (day <= 30 and day % 5 == 0))
        if should_measure:
            metrics = measure_representations(net, test_patterns, CONFIG, rng)
            print(f"         repr: alive={metrics['n_alive']}/{metrics['n_internal']} "
                  f"sparse={metrics['sparsity']:.3f} "
                  f"select={metrics['selectivity']:.4f} "
                  f"diverse={metrics['diversity']:.3f} "
                  f"consist={metrics['consistency']:.3f}", flush=True)

        # Checkpoint
        if day % CHECKPOINT_EVERY == 0 or day == N_INFANCY_DAYS:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            ckpt_base = os.path.join(CHECKPOINT_DIR, f"day_{day:04d}")
            rng_bg = rng.bit_generator
            rng_state_dict = {}
            try:
                s = rng_bg.state
                rng_state_dict = {
                    'state': np.array([s['state']['state']]),
                    'inc': np.array([s['state']['inc']]),
                }
            except Exception:
                pass
            save_checkpoint(net, day, maturity, ema_r, rng_state_dict, ckpt_base)

    # ── Post-infancy ───────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("POST-INFANCY REPRESENTATIONS")
    print(f"{'='*60}")
    final = measure_representations(net, test_patterns, CONFIG, rng)
    print(f"  Alive neurons: {final['n_alive']}/{final['n_internal']}")
    print(f"  Sparsity:      {final['sparsity']:.3f}")
    print(f"  Selectivity:   {final['selectivity']:.4f}")
    print(f"  Diversity:     {final['diversity']:.3f}")
    print(f"  Consistency:   {final['consistency']:.3f}")

    # ── Quick task-solving test ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TASK-SOLVING TEST")
    print(f"{'='*60}")

    small_tasks = [t for t in train_set if max_dim(t) <= MAX_H]
    tasks = extract_tasks(small_tasks)
    tasks = sort_by_difficulty(tasks)[:N_TASK_TEST]

    episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=200)

    print("\n  A) Direct solve (no task training):")
    n_solved_direct = 0
    for i, (pairs, ti, to) in enumerate(tasks):
        out_h, out_w = to.shape
        r = solve_task(net, pairs, ti, CONFIG, output_h=out_h, output_w=out_w,
                       episodic=episodic)
        perfect = np.array_equal(r.grid, to)
        fg = to != 0
        fg_acc = (float(np.mean(r.grid[fg] == to[fg])) if fg.any()
                  else (1.0 if np.all(r.grid == 0) else 0.0))
        status = "SOLVED" if perfect else f"fg={fg_acc:.2f}"
        if perfect:
            n_solved_direct += 1
        print(f"    [{i:2d}] {status}")
    print(f"  Direct: {n_solved_direct}/{N_TASK_TEST} solved")

    print("\n  B) With brief task training (5 rounds):")
    n_solved_trained = 0
    for i, (pairs, ti, to) in enumerate(tasks):
        _soft_reset(net)
        best_r = 0.0
        for rnd in range(5):
            result = study_task_round(
                net, pairs, ti, to, CONFIG,
                prev_best_reward=best_r,
                is_first_visit=(rnd == 0),
                episodic=episodic,
            )
            best_r = max(best_r, result.reward)
            if result.reward >= 1.0:
                break
        out_h, out_w = to.shape
        r = solve_task(net, pairs, ti, CONFIG, output_h=out_h, output_w=out_w,
                       episodic=episodic)
        perfect = np.array_equal(r.grid, to)
        fg = to != 0
        fg_acc = (float(np.mean(r.grid[fg] == to[fg])) if fg.any()
                  else (1.0 if np.all(r.grid == 0) else 0.0))
        status = "SOLVED" if perfect else f"fg={fg_acc:.2f} (train_best={best_r:.2f})"
        if perfect:
            n_solved_trained += 1
        print(f"    [{i:2d}] {status}")
    print(f"  Trained: {n_solved_trained}/{N_TASK_TEST} solved")

    # ── Final save ─────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    final_path = os.path.join("models", "infant_final.npz")
    net.save(final_path)
    print(f"\nFinal model saved: {final_path}")
    elapsed_total = time.time() - t_start
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)")


def max_dim(task):
    d = 0
    for inp, out in task.train:
        inp, out = np.array(inp), np.array(out)
        d = max(d, inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
    for inp, out in task.test:
        inp, out = np.array(inp), np.array(out)
        d = max(d, inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
    return d


if __name__ == "__main__":
    main()
