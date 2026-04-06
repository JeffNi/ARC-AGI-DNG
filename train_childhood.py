"""
Childhood phase: error-corrective three-factor learning on ARC tasks.

Picks up from an infancy checkpoint and ramps developmental parameters
(inh_scale, wta_k_frac, excitability cap) from their saved checkpoint
values toward adult targets. Learning happens via attempt -> error
injection -> three-factor update. All starting values are read from
the checkpoint.

Usage:
  python train_childhood.py
  python train_childhood.py --checkpoint models/infancy_checkpoints/day_0090
  python train_childhood.py --days 80 --tasks-per-day 12
"""
import sys, os, json, time, argparse
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.graph import DNG
from src.childhood import (
    ChildhoodConfig, extract_tasks, run_childhood,
)
from src.curriculum import sort_by_difficulty
from src.micro_tasks import load_micro_tasks
from src.pipeline import LifecycleConfig
from src.episodic_memory import EpisodicMemory

# ── CLI ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run childhood training")
parser.add_argument("--checkpoint", type=str,
                    default="models/infancy_checkpoints/day_0140",
                    help="Infancy checkpoint path (without extension)")
parser.add_argument("--days", type=int, default=50,
                    help="Total childhood days (default: 50)")
parser.add_argument("--tasks-per-day", type=int, default=3,
                    help="Tasks studied per day (default: 3)")
parser.add_argument("--seed", type=int, default=99,
                    help="Random seed (default: 99)")
parser.add_argument("--output-dir", type=str,
                    default="models/childhood_checkpoints",
                    help="Checkpoint output directory")
args = parser.parse_args()

MAX_H = MAX_W = 10

# ── Load infancy checkpoint ──────────────────────────────────────

net_path = args.checkpoint + ".net.npz"
state_path = args.checkpoint + ".state.json"
threshold_path = args.checkpoint + ".threshold.npy"

print(f"Loading infancy checkpoint: {args.checkpoint}")
net = DNG.load(net_path)

with open(state_path, "r") as f:
    infancy_state = json.load(f)

start_maturity = infancy_state["maturity"]
infancy_dev = infancy_state.get("dev", {})

if os.path.exists(threshold_path):
    net.threshold = np.load(threshold_path)

# Recover infancy-exit plasticity parameters so childhood can pick up
# smoothly instead of jumping to arbitrary values.
infancy_growth_rate = infancy_dev.get("nursery_growth_rate", 0.8)
infancy_growth_candidates = int(infancy_dev.get("nursery_growth_candidates", 500000))
infancy_sleep_downscale_cfg = infancy_dev.get("sleep_downscale", 0.998)
# nursery_exposure applied: child_downscale = 1 - (1-cfg)*0.2
infancy_sleep_downscale = 1.0 - (1.0 - infancy_sleep_downscale_cfg) * 0.2
infancy_prune_weak = infancy_dev.get("prune_weak_threshold", 0.003)
# Infancy used hardcoded removal_rate=0.01 in nursery_exposure
infancy_prune_removal_rate = 0.01

print(f"  Loaded: {net.edge_count():,} edges, {net.n_nodes} nodes")
print(f"  Infancy maturity: {start_maturity:.4f}")
print(f"  inh_scale={net.inh_scale:.3f}  wta_k_frac={net.wta_k_frac:.4f}")
print(f"  Infancy exit plasticity: growth_rate={infancy_growth_rate}, "
      f"candidates={infancy_growth_candidates}, "
      f"prune_rate={infancy_prune_removal_rate}, "
      f"sleep_ds={infancy_sleep_downscale:.4f}")

# ── Load micro-tasks (pre-ARC curriculum) ────────────────────────

print("Loading micro-tasks...")
micro_tasks = load_micro_tasks("micro_tasks", max_h=MAX_H, max_w=MAX_W)
print(f"  {len(micro_tasks)} micro-tasks loaded")

# ── Load ARC tasks ───────────────────────────────────────────────

print("Loading ARC training tasks...")
train_set, _ = arckit.load_data()
arc_tasks = extract_tasks(train_set, max_h=MAX_H, max_w=MAX_W)
arc_tasks = sort_by_difficulty(arc_tasks)
print(f"  {len(arc_tasks)} ARC tasks loaded (filtered to {MAX_H}x{MAX_W} max)")

# Micro-tasks first, then ARC tasks. Mastery curriculum means the
# network must learn the basics before graduating to real challenges.
all_tasks = micro_tasks + arc_tasks
print(f"  {len(all_tasks)} total tasks (micro + ARC)")

# ── Configure childhood ──────────────────────────────────────────

childhood_config = ChildhoodConfig(
    n_days=args.days,
    tasks_per_day=args.tasks_per_day,
    checkpoint_path=os.path.join(args.output_dir, "childhood"),
    verbose=True,
)

lifecycle = LifecycleConfig(
    observe_steps=30,
    think_steps=80,
    eta=0.03,
    w_max=2.5,
    error_prop_steps=15,
    noise_std=0.02,
    focus_strength=0.5,
    attempts_per_round=3,
    n_rounds=2,
    rest_steps=15,
    rest_noise_std=0.03,
    # Structural plasticity: start from infancy exit values.
    # _schedule() interpolates from these toward phase targets.
    growth_rate=infancy_growth_rate,
    growth_candidates=infancy_growth_candidates,
    prune_removal_rate=infancy_prune_removal_rate,
    prune_weak_threshold=infancy_prune_weak,
    sleep_downscale=infancy_sleep_downscale,
    sleep_tag_threshold=0.001,
    prune_cycles_required=20,
    replay_eta=0.005,
    replay_passes=2,
    replay_steps=20,
    memory_hint_strength=1.5,
    spontaneous_strength=0.2,
)

rng = np.random.default_rng(args.seed)
episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=200)

os.makedirs(args.output_dir, exist_ok=True)

# ── Run childhood ────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  CHILDHOOD TRAINING: {args.days} days, {args.tasks_per_day} tasks/day")
print(f"{'='*60}\n")

t0 = time.time()

result = run_childhood(
    net,
    all_tasks,
    config=childhood_config,
    lifecycle=lifecycle,
    rng=rng,
    episodic=episodic,
    start_maturity=start_maturity,
)

elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"  CHILDHOOD COMPLETE")
print(f"  {elapsed:.0f}s elapsed ({elapsed/60:.1f} min)")
print(f"  Final edges: {result.final_edges:,}")
print(f"  Best reward: {max(result.day_rewards):.3f}")
print(f"  Last-5 avg:  {np.mean(result.day_rewards[-5:]):.3f}")
print(f"{'='*60}")

final_path = os.path.join(args.output_dir, "childhood_final.net.npz")
net.save(final_path)
print(f"  Final network saved to {final_path}")
