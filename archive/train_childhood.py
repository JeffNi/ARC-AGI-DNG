"""
Childhood phase: mastery-based curriculum on micro-tasks.

Picks up from an infancy checkpoint.  Learning happens via continuous
local plasticity (Hebbian+DA inside the dynamics loop).

Curriculum advances through task types one at a time:
  - Focus on one type until mastered (all variants solved 2 consecutive days)
  - Review previously mastered types for retention diagnostics
  - Flag types as STUCK if never solved after 10+ days

Logs to timestamped folder.  Curriculum state saved for resume.

Usage:
  python train_childhood.py
  python train_childhood.py --checkpoint models/infancy_checkpoints/day_0140
  python train_childhood.py --resume logs/childhood_20260404_153012
"""
import sys, os, json, time, argparse, datetime, io

sys.path.insert(0, '.')
import numpy as np

from src.graph import DNG
from src.childhood import (
    ChildhoodConfig, run_childhood, load_curriculum_state,
)
from src.micro_tasks import load_micro_tasks_tagged
from src.pipeline import LifecycleConfig
from src.episodic_memory import EpisodicMemory


class TeeWriter:
    """Write to both stdout and a log file."""
    def __init__(self, log_path: str):
        self._stdout = sys.stdout
        self._file = open(log_path, "a", encoding="utf-8")

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


parser = argparse.ArgumentParser(description="Run childhood training (mastery curriculum)")
parser.add_argument("--checkpoint", type=str,
                    default="models/infancy_checkpoints/day_0140",
                    help="Infancy checkpoint path (without extension)")
parser.add_argument("--days", type=int, default=50,
                    help="Total childhood days (default: 50)")
parser.add_argument("--focus-count", type=int, default=5,
                    help="Variants of focus type per day (default: 5)")
parser.add_argument("--review-count", type=int, default=2,
                    help="Review tasks from mastered types (default: 2)")
parser.add_argument("--seed", type=int, default=99,
                    help="Random seed (default: 99)")
parser.add_argument("--output-dir", type=str,
                    default="models/childhood_checkpoints",
                    help="Checkpoint output directory")
parser.add_argument("--resume", type=str, default=None,
                    help="Resume from a log folder (loads curriculum_state.json)")
args = parser.parse_args()

MAX_H = MAX_W = 10

# ── Create timestamped log folder ─────────────────────────────────

if args.resume:
    log_dir = args.resume
else:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"childhood_{ts}")

os.makedirs(log_dir, exist_ok=True)
tee = TeeWriter(os.path.join(log_dir, "training.log"))
sys.stdout = tee

# ── Load infancy checkpoint ──────────────────────────────────────

net_path = args.checkpoint + ".net.npz"

print(f"Loading infancy checkpoint: {args.checkpoint}")
net = DNG.load(net_path)

state_path = args.checkpoint + ".state.json"
if os.path.exists(state_path):
    with open(state_path, "r") as f:
        infancy_state = json.load(f)
    print(f"  Infancy maturity: {infancy_state.get('maturity', '?')}")

threshold_path = args.checkpoint + ".threshold.npy"
if os.path.exists(threshold_path):
    net.threshold = np.load(threshold_path)

print(f"  Loaded: {net.edge_count():,} edges, {net.n_nodes} nodes")
print(f"  inh_scale={net.inh_scale:.3f}  wta_k_frac={net.wta_k_frac:.4f}")

# ── Load micro-tasks (tagged) ────────────────────────────────────

print("Loading micro-tasks (tagged)...")
tagged_tasks = load_micro_tasks_tagged("micro_tasks", max_h=MAX_H, max_w=MAX_W)

total_tasks = sum(len(v) for v in tagged_tasks.values())
print(f"  {total_tasks} tasks across {len(tagged_tasks)} types:")
for tname, tlist in tagged_tasks.items():
    print(f"    {tname}: {len(tlist)} variants (tier {tlist[0].tier})")

if total_tasks == 0:
    print("ERROR: No tasks available. Run scripts/generate_micro_tasks.py first.")
    sys.exit(1)

# ── Resume curriculum state if applicable ─────────────────────────

resume_state = None
if args.resume:
    cs_path = os.path.join(args.resume, "curriculum_state.json")
    if os.path.exists(cs_path):
        with open(cs_path) as f:
            resume_state = json.load(f)
        print(f"  Resuming from day {resume_state.get('day', '?')}, "
              f"focus={resume_state.get('focus_type', '?')}")
    else:
        print(f"  Warning: no curriculum_state.json in {args.resume}, starting fresh")

# ── Configure childhood ──────────────────────────────────────────

childhood_config = ChildhoodConfig(
    n_days=args.days,
    checkpoint_path=os.path.join(args.output_dir, "childhood"),
    verbose=True,
    focus_count=args.focus_count,
    review_count=args.review_count,
    log_dir=log_dir,
)

lifecycle = LifecycleConfig(
    observe_steps=30,
    think_steps=80,
    consolidation_steps=20,
    eta=0.001,
    w_max=2.5,
    bcm_scale=2.0,
    plasticity_interval=5,
    da_baseline_obs=0.3,
    da_baseline_attempt=0.0,
    da_rpe_scale=2.0,
    noise_std=0.02,
    focus_strength=0.5,
    attempts_per_round=3,
    n_rounds=2,
    rest_steps=15,
    rest_noise_std=0.03,
    growth_rate=0.3,
    growth_candidates=50000,
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
print(f"  CHILDHOOD TRAINING (mastery curriculum)")
print(f"  {args.days} days, {args.focus_count} focus + {args.review_count} review/day")
print(f"  eta={lifecycle.eta:.4f}  DA_obs={lifecycle.da_baseline_obs:.2f}  "
      f"DA_attempt={lifecycle.da_baseline_attempt:.2f}")
print(f"  Log folder: {log_dir}")
print(f"{'='*60}\n")

t0 = time.time()

result = run_childhood(
    net,
    tagged_tasks,
    config=childhood_config,
    lifecycle=lifecycle,
    rng=rng,
    episodic=episodic,
    resume_state=resume_state,
)

elapsed = time.time() - t0

# ── Summary ──────────────────────────────────────────────────────

cs = result.curriculum_state
n_mastered = sum(1 for v in cs["types"].values() if v["status"] == "mastered")
n_stuck = sum(1 for v in cs["types"].values() if v["status"] == "stuck")
n_total = len(cs["types"])

print(f"\n{'='*60}")
print(f"  CHILDHOOD COMPLETE")
print(f"  {elapsed:.0f}s elapsed ({elapsed/60:.1f} min)")
print(f"  Final edges: {result.final_edges:,}")
print(f"  Mastered: {n_mastered}/{n_total}  Stuck: {n_stuck}/{n_total}")
print(f"  Best reward: {max(result.day_rewards):.3f}")
print(f"  Last-5 avg:  {np.mean(result.day_rewards[-5:]):.3f}")

print(f"\n  Per-type summary:")
for tname in cs.get("type_order", []):
    ts = cs["types"][tname]
    status_tag = ts["status"].upper()
    solve_rate = (ts["total_solves"] / ts["total_attempts"]
                  if ts["total_attempts"] > 0 else 0)
    print(f"    {tname:30s} [{status_tag:8s}] "
          f"days={ts['days_focused']:2d}  "
          f"solves={ts['total_solves']}/{ts['total_attempts']} "
          f"({solve_rate:.0%})  "
          f"best_day={ts['best_day_solve_rate']:.0%}")

print(f"{'='*60}")

final_path = os.path.join(args.output_dir, "childhood_final.net.npz")
net.save(final_path)
print(f"  Final network saved to {final_path}")
print(f"  Logs: {log_dir}")

# Restore stdout
sys.stdout = tee._stdout
tee.close()
