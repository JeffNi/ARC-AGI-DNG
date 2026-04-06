"""
Full childhood training + evaluation on ARC-AGI-1.

Trains the DNG through childhood with curriculum ordering,
then evaluates on held-out test tasks in adult mode (no weight changes).

Each run gets its own timestamped folder under models/.
Checkpoints saved every N days. Resume from checkpoint with --resume.
"""
import sys, time, os, argparse
from datetime import datetime
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.childhood import ChildhoodConfig, run_childhood, extract_tasks
from src.curriculum import sort_by_difficulty
from src.pipeline import LifecycleConfig, solve_task
from src.graph import DNG, Region
from src.episodic_memory import EpisodicMemory

# ── Args ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--max-grid', type=int, default=10,
                    help='Max grid dimension (10 for fast, 30 for full)')
parser.add_argument('--days', type=int, default=200,
                    help='Number of childhood days')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint .npz to resume from')
args = parser.parse_args()

MAX_H = MAX_W = args.max_grid

# ── Run folder ────────────────────────────────────────────────────────

run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join("models", run_id)
os.makedirs(run_dir, exist_ok=True)
print(f"Run folder: {run_dir}", flush=True)

# ── Load tasks ───────────────────────────────────────────────────────

train_set, eval_set = arckit.load_data()

def max_grid_dim(task):
    mh, mw = 0, 0
    for inp, out in task.train:
        inp, out = np.array(inp), np.array(out)
        mh = max(mh, inp.shape[0], out.shape[0])
        mw = max(mw, inp.shape[1], out.shape[1])
    for inp, out in task.test:
        inp, out = np.array(inp), np.array(out)
        mh = max(mh, inp.shape[0], out.shape[0])
        mw = max(mw, inp.shape[1], out.shape[1])
    return mh, mw

valid_tasks = [t for t in train_set
               if all(d <= MAX_H for d in max_grid_dim(t))]
print(f"ARC-AGI-1: {len(valid_tasks)}/{len(train_set)} tasks fit in "
      f"{MAX_H}x{MAX_W}", flush=True)

# Split: 80% train, 20% eval
rng = np.random.default_rng(42)
idx = rng.permutation(len(valid_tasks))
n_train = int(len(valid_tasks) * 0.8)
train_arc = [valid_tasks[i] for i in idx[:n_train]]
eval_arc = [valid_tasks[i] for i in idx[n_train:]]

# Extract and sort by curriculum
all_tasks = extract_tasks(train_arc)
all_tasks = sort_by_difficulty(all_tasks)
eval_tasks_raw = eval_arc

print(f"Train: {len(all_tasks)} tasks (sorted by difficulty)", flush=True)
print(f"Eval:  {len(eval_tasks_raw)} tasks (held out)", flush=True)

# ── Create or resume network ────────────────────────────────────────

if args.resume:
    print(f"\nResuming from {args.resume}...", flush=True)
    net = DNG.load(args.resume)
    print(f"  {net.n_nodes} nodes, {net.edge_count()} edges", flush=True)
else:
    genome = Genome(
        max_h=MAX_H, max_w=MAX_W,
        n_internal=400, n_concept=80, n_memory=120,
        max_fan_in=100,
        weight_scale=0.02, frac_inhibitory=0.25,
        f_rate=0.05, f_decay=0.02, f_max=3.0,
    )

    print("\nCreating network...", flush=True)
    t0_build = time.time()
    net = create_dng(genome, MAX_H, MAX_W, rng)
    net.wta_k_frac = 0.05       # sparse WTA: prevents multi-task interference
    net.threshold[:] = 0.15
    motor_mask = net.regions == list(Region).index(Region.MOTOR)
    net.threshold[motor_mask] = 0.05
    net.adapt_rate = 0.05
    print(f"  {net.n_nodes} nodes, {net.edge_count()} edges "
          f"({len(net.memory_nodes)} memory) "
          f"({time.time()-t0_build:.1f}s)", flush=True)

# ── Childhood ───────────────────────────────────────────────────────

childhood_config = ChildhoodConfig(
    n_days=args.days,
    tasks_per_day=8,
    checkpoint_path=os.path.join(run_dir, "checkpoint"),
    verbose=True,
)

lifecycle = LifecycleConfig(
    observe_steps=40,
    think_steps=60,
    consolidation_steps=20,
    eta=0.0,                     # NO CHL during tasks (CLS: only in sleep)
    w_max=2.5,
    free_phase_steps=40,
    clamped_phase_steps=40,
    attempts_per_round=3,
    n_rounds=2,
    stuck_patience=2,
    noise_std=0.02,
    focus_strength=0.5,
    rest_steps=20,
    rest_noise_std=0.03,
    sleep_downscale=1.0,
    sleep_tag_threshold=0.001,
    prune_weak_threshold=0.0,
    prune_cycles_required=999,
    replay_eta=0.0001,           # CLS: very slow neocortical learning
    replay_passes=3,             # 3 passes through ALL episodic memories
    replay_steps=30,
    consolidation_decay=0.99,
    consolidation_strength=0.0,  # disabled during CLS exploration
    consolidation_threshold=0.9,
    memory_hint_strength=1.5,    # strong episodic recall hints
    spontaneous_strength=0.3,
)

print(f"\n{'='*60}", flush=True)
print(f"CHILDHOOD: {childhood_config.n_days} days, "
      f"{childhood_config.tasks_per_day} tasks/day", flush=True)
print(f"{'='*60}", flush=True)

episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=200)

t0 = time.time()
result = run_childhood(net, all_tasks, childhood_config, lifecycle, rng,
                       episodic=episodic)
elapsed = time.time() - t0

print(f"\nChildhood complete in {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
print(f"Final edges: {result.final_edges}", flush=True)
if len(result.day_rewards) >= 10:
    r = result.day_rewards
    print(f"Reward: day1={r[0]:.2f} day10={r[9]:.2f} "
          f"last={r[-1]:.2f}", flush=True)

final_path = os.path.join(run_dir, "final.npz")
net.save(final_path)
print(f"Saved: {final_path}", flush=True)

# ── Evaluation (adult mode, no weight changes) ──────────────────────

print(f"\n{'='*60}", flush=True)
print(f"EVALUATION on {len(eval_tasks_raw)} held-out tasks (adult mode)", flush=True)
print(f"{'='*60}", flush=True)

eval_config = LifecycleConfig(
    observe_steps=40,
    think_steps=80,
    noise_std=0.02,
    focus_strength=0.5,
    t_max=200,
)

correct = 0
total = 0
copy_scores = []
net_scores = []
samples = []
from src.encoding import pad_grid
for task in eval_tasks_raw[:30]:
    train_pairs = [(np.array(i), np.array(o)) for i, o in task.train]
    for test_inp, test_out in task.test:
        test_inp, test_out = np.array(test_inp), np.array(test_out)
        out_h, out_w = test_out.shape

        result = solve_task(net, train_pairs, test_inp, eval_config,
                            output_h=out_h, output_w=out_w,
                            episodic=episodic)
        fg = test_out != 0
        acc = float(np.mean(result.grid[fg] == test_out[fg])) if fg.any() else 1.0

        # Copy baseline: what if we just echoed the input?
        copy_grid = pad_grid(test_inp, out_h, out_w)[:out_h, :out_w]
        copy_acc = float(np.mean(copy_grid[fg] == test_out[fg])) if fg.any() else 1.0

        if acc == 1.0:
            correct += 1
        total += 1
        net_scores.append(acc)
        copy_scores.append(copy_acc)

        better = "+" if acc > copy_acc else ("=" if acc == copy_acc else "-")
        mh, mw = max_grid_dim(task)
        print(f"  {task.id} ({mh}x{mw}): net={acc:.0%} copy={copy_acc:.0%} {better}",
              flush=True)

        if len(samples) < 8:
            samples.append((task.id, test_inp, test_out, result.grid, copy_grid))

mean_net = float(np.mean(net_scores))
mean_copy = float(np.mean(copy_scores))
print(f"\nEval: {correct}/{total} perfect ({correct/max(1,total):.0%})", flush=True)
print(f"Mean fg acc:  network={mean_net:.1%}  copy_baseline={mean_copy:.1%}  "
      f"delta={mean_net-mean_copy:+.1%}", flush=True)

print(f"\n{'='*60}", flush=True)
print("SAMPLE OUTPUTS (first 8 eval tasks)", flush=True)
print(f"{'='*60}", flush=True)
for tid, inp, expected, guess, copy_g in samples:
    print(f"\n--- {tid} ---", flush=True)
    print(f"Input ({inp.shape[0]}x{inp.shape[1]}):", flush=True)
    for row in inp:
        print("  ", row.tolist(), flush=True)
    print(f"Expected:", flush=True)
    for row in expected:
        print("  ", row.tolist(), flush=True)
    print(f"Network output:", flush=True)
    for row in guess:
        print("  ", row.tolist(), flush=True)
    fg_mask = expected != 0
    if fg_mask.any():
        fg_acc = np.mean(guess[fg_mask] == expected[fg_mask])
        cp_acc = np.mean(copy_g[fg_mask] == expected[fg_mask])
        print(f"  (fg: {fg_acc:.0%} net, {cp_acc:.0%} copy baseline)", flush=True)
    else:
        print(f"  (all-black expected)", flush=True)

print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
