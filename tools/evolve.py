"""
Evolutionary parameter search — discover learning-critical parameter ranges.

Creates a population of genomes, evaluates them in parallel against a fixed
task set, selects survivors by fitness, and mutates for the next generation.
Results saved to runs/evolve_<timestamp>/.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np

# Force unbuffered prints so output appears in real time
print = functools.partial(print, flush=True)

# ── Project imports (delayed inside worker to avoid pickling issues) ──

GRID_H = 3
GRID_W = 3

# Evolution constants
POP_SIZE = 16
N_GENERATIONS = 10
TOP_K = 4
MUTATION_STRENGTH_INIT = 0.30
MUTATION_STRENGTH_REPRO = 0.15
TRIALS_PER_TYPE = 30
TASK_TYPES = ["identity", "solid_fill", "color_swap", "flip_h", "color_extract"]

# All params that mutate_learning() touches — used for printouts/convergence
EVOLVE_PARAMS = [
    "copy_bias",
    "action_explore_noise", "action_noise_coeff", "mbon_strength", "attention_gain",
    "commit_gain", "commit_noise", "commit_lr", "min_act_rate",
    "suppress_strength",
    "depression_eta", "depression_floor", "recovery_rate",
    "w_scale_sensory_to_l3", "w_scale_l3_to_memory",
    "kc_local_fan", "kc_global_fan",
    "apl_target_sparseness", "apl_gain",
    "leak_l3", "leak_memory", "noise_std",
    "observe_steps",
    "da_correct_commit", "da_wrong_commit", "da_global_wrong", "da_global_correct",
]

# Harder tasks are worth more — evolution shouldn't optimize for the easy wins
TASK_WEIGHTS = {
    "identity": 0.5,
    "solid_fill": 1.0,
    "color_swap": 2.0,
    "flip_h": 3.0,
    "color_extract": 2.0,
}


def evaluate_genome(args: tuple) -> dict:
    """Evaluate a single genome on the fixed task set. Runs in a worker process.

    Returns a dict with fitness, per-type scores, and genome params.
    """
    genome_dict, genome_id, gen_idx, seed = args

    from src.genome import Genome
    from src.brain import Brain
    from src.teacher import Teacher
    from src.monitor import Monitor

    genome = Genome(**genome_dict)
    rng_seed = seed + genome_id * 1000

    run_dir = f"runs/evolve_tmp/gen{gen_idx:02d}_genome{genome_id:02d}"
    os.makedirs(run_dir, exist_ok=True)

    try:
        brain = Brain.birth(genome, grid_h=GRID_H, grid_w=GRID_W,
                            seed=rng_seed, checkpoint_dir=run_dir)
        monitor = Monitor(log_dir=run_dir, console=False)
        teacher = Teacher(brain, monitor, task_dir="micro_tasks")

        train_results: dict[str, list[float]] = {t: [] for t in TASK_TYPES}
        holdout_results: dict[str, list[float]] = {t: [] for t in TASK_TYPES}

        MAX_RETRIES = 10

        for task_type_name in TASK_TYPES:
            tier_key = f"gen_{task_type_name}"
            tasks = teacher._tasks_by_type.get(tier_key, [])
            if not tasks:
                continue

            n_train = max(1, int(len(tasks) * 0.7))
            train_tasks = tasks[:n_train]
            holdout_tasks = tasks[n_train:]

            # Train: retry each instance until solved or MAX_RETRIES,
            # MBON weights persist across instances (generalization).
            trials_used = 0
            task_idx = 0
            while trials_used < TRIALS_PER_TYPE and task_idx < len(train_tasks):
                task = train_tasks[task_idx]
                for retry in range(MAX_RETRIES):
                    if trials_used >= TRIALS_PER_TYPE:
                        break
                    solved = teacher.run_task(tier_key, task)
                    train_results[task_type_name].append(teacher.last_cell_acc)
                    trials_used += 1
                    if solved:
                        break
                task_idx += 1

            # Holdout: single attempt per instance (tests generalization)
            for task in holdout_tasks:
                teacher.run_task(tier_key, task)
                holdout_results[task_type_name].append(teacher.last_cell_acc)

        # ── Fitness computation ──
        per_type_fitness = {}
        weighted_train_acc = 0.0
        weighted_holdout_acc = 0.0
        weighted_trend = 0.0
        total_weight = 0.0
        tasks_with_signal = 0

        for ttype in TASK_TYPES:
            t_scores = train_results[ttype]
            h_scores = holdout_results[ttype]
            if not t_scores:
                continue
            tw = TASK_WEIGHTS.get(ttype, 1.0)
            total_weight += tw

            train_acc = float(np.mean(t_scores))
            holdout_acc = float(np.mean(h_scores)) if h_scores else 0.0

            # Robust learning trend: Spearman rank correlation on cell_acc
            # over training trials. Only rewarded if statistically significant
            # (positive correlation with p < 0.1) to filter out noise.
            trend_bonus = 0.0
            n = len(t_scores)
            if n >= 5:
                from scipy.stats import spearmanr
                rho, p_value = spearmanr(range(n), t_scores)
                if rho > 0 and p_value < 0.10:
                    trend_bonus = rho * 0.5
            elif n >= 3:
                first_half = np.mean(t_scores[:n // 2])
                second_half = np.mean(t_scores[n // 2:])
                delta = second_half - first_half
                if delta > 0.05:
                    trend_bonus = delta

            weighted_train_acc += train_acc * tw
            weighted_holdout_acc += holdout_acc * tw
            weighted_trend += trend_bonus * tw

            if train_acc > 0.15 or holdout_acc > 0.15:
                tasks_with_signal += 1

            per_type_fitness[ttype] = {
                "train_acc": round(train_acc, 4),
                "holdout_acc": round(holdout_acc, 4),
                "trend_bonus": round(trend_bonus, 4),
                "weight": tw,
                "train_scores": [round(s, 3) for s in t_scores],
                "holdout_scores": [round(s, 3) for s in h_scores],
            }

        if total_weight > 0:
            avg_train = weighted_train_acc / total_weight
            avg_holdout = weighted_holdout_acc / total_weight
            avg_trend = weighted_trend / total_weight
        else:
            avg_train = avg_holdout = avg_trend = 0.0

        # Breadth bonus: reward having signal on multiple task types.
        # A specialist acing 1 task loses to a generalist with signal on 3.
        n_types = len(TASK_TYPES)
        breadth_bonus = (tasks_with_signal / n_types) ** 2 if n_types > 0 else 0.0

        # Fitness = accuracy + learning trend + breadth
        fitness = (0.3 * avg_train
                   + 0.4 * avg_holdout
                   + 0.15 * avg_trend
                   + 0.15 * breadth_bonus)

        return {
            "genome_id": genome_id,
            "generation": gen_idx,
            "fitness": round(fitness, 6),
            "avg_train": round(avg_train, 4),
            "avg_holdout": round(avg_holdout, 4),
            "avg_trend": round(avg_trend, 4),
            "breadth": tasks_with_signal,
            "per_type": per_type_fitness,
            "genome": genome_dict,
        }

    except Exception as e:
        return {
            "genome_id": genome_id,
            "generation": gen_idx,
            "fitness": -1.0,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "genome": genome_dict,
        }


def create_initial_population(rng: np.random.Generator,
                              seed_genome_path: str | None = None) -> list[dict]:
    """Build initial population. If seed_genome_path is provided, load that
    genome as the base; otherwise start from Genome defaults."""
    from src.genome import Genome

    if seed_genome_path:
        with open(seed_genome_path) as f:
            data = json.load(f)
        genome_dict = data["genome"] if "genome" in data else data
        # Filter to only known Genome fields
        known = {f.name for f in Genome.__dataclass_fields__.values()}
        genome_dict = {k: v for k, v in genome_dict.items() if k in known}
        base = Genome(**genome_dict)
        print(f"  Seed genome loaded from: {seed_genome_path}")
        print(f"  Seed fitness was: {data.get('fitness', '?')}")
    else:
        base = Genome()

    pop = [asdict(base)]

    for i in range(1, POP_SIZE):
        child = base.mutate_learning(rng, strength=MUTATION_STRENGTH_INIT)
        pop.append(asdict(child))

    return pop


def select_and_reproduce(
    results: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    """Rank by fitness, keep top K, reproduce to fill population."""
    from src.genome import Genome

    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["fitness"], reverse=True)

    survivors = valid[:TOP_K]
    new_pop = []

    for surv in survivors:
        new_pop.append(surv["genome"])

    children_per_survivor = (POP_SIZE - len(new_pop)) // len(survivors)
    extra = (POP_SIZE - len(new_pop)) - children_per_survivor * len(survivors)

    for i, surv in enumerate(survivors):
        parent = Genome(**surv["genome"])
        n_children = children_per_survivor + (1 if i < extra else 0)
        for _ in range(n_children):
            child = parent.mutate_learning(rng, strength=MUTATION_STRENGTH_REPRO)
            new_pop.append(asdict(child))

    return new_pop[:POP_SIZE]


def print_leaderboard(results: list[dict], gen_idx: int):
    """Print a compact leaderboard for this generation."""
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["fitness"], reverse=True)

    print(f"\n{'='*70}")
    print(f"  Generation {gen_idx} Leaderboard")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'ID':<5} {'Fitness':<10} {'Train':<8} {'Hold':<8} {'Trend':<7} {'Br':<3} {'Details'}")
    print(f"  {'-'*75}")

    for rank, r in enumerate(valid[:8], 1):
        types_str = ""
        for ttype in TASK_TYPES:
            info = r.get("per_type", {}).get(ttype, {})
            t_acc = info.get("train_acc", 0)
            h_acc = info.get("holdout_acc", 0)
            types_str += f" {ttype[:4]}={t_acc:.2f}/{h_acc:.2f}"

        marker = " *" if rank <= TOP_K else ""
        breadth = r.get("breadth", 0)
        print(f"  {rank:<5} {r['genome_id']:<5} {r['fitness']:<10.4f} "
              f"{r.get('avg_train', 0):<8.3f} {r.get('avg_holdout', 0):<8.3f} "
              f"{r.get('avg_trend', 0):<7.3f} {breadth:<3}{types_str}{marker}")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n  ({len(errors)} genome(s) crashed)")

    print()


def save_generation(results: list[dict], gen_idx: int, output_dir: Path):
    """Save full generation results to JSON."""
    gen_file = output_dir / f"gen_{gen_idx:02d}.json"
    with open(gen_file, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evolutionary parameter search")
    parser.add_argument("--seed-genome", type=str, default=None,
                        help="Path to a best_genome.json from a previous run")
    args = parser.parse_args()

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    rng = np.random.default_rng(42)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"runs/evolve_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evolutionary Parameter Search")
    print(f"  Population: {POP_SIZE}, Generations: {N_GENERATIONS}")
    print(f"  Top-{TOP_K} survive, {TRIALS_PER_TYPE} trials/type, "
          f"{len(TASK_TYPES)} task types")
    print(f"  Output: {output_dir}")
    print()

    population = create_initial_population(rng, seed_genome_path=args.seed_genome)

    # Save config
    config = {
        "pop_size": POP_SIZE,
        "n_generations": N_GENERATIONS,
        "top_k": TOP_K,
        "mutation_init": MUTATION_STRENGTH_INIT,
        "mutation_repro": MUTATION_STRENGTH_REPRO,
        "trials_per_type": TRIALS_PER_TYPE,
        "task_types": TASK_TYPES,
        "grid": f"{GRID_H}x{GRID_W}",
        "seed": 42,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    best_ever = {"fitness": -1.0}

    n_workers = min(POP_SIZE, max(1, os.cpu_count() // 2))
    print(f"  Using {n_workers} parallel workers")

    for gen_idx in range(N_GENERATIONS):
        gen_start = time.time()
        print(f"\n--- Generation {gen_idx} ---")

        seed_base = int(rng.integers(0, 2**31))
        work_items = [
            (population[i], i, gen_idx, seed_base)
            for i in range(len(population))
        ]

        with Pool(processes=n_workers) as pool:
            results = pool.map(evaluate_genome, work_items)

        gen_time = time.time() - gen_start
        print(f"  Evaluated {len(results)} genomes in {gen_time:.1f}s "
              f"({gen_time/len(results):.1f}s/genome)")

        print_leaderboard(results, gen_idx)
        save_generation(results, gen_idx, output_dir)

        valid = [r for r in results if "error" not in r]
        if valid:
            gen_best = max(valid, key=lambda r: r["fitness"])
            if gen_best["fitness"] > best_ever["fitness"]:
                best_ever = gen_best
                with open(output_dir / "best_genome.json", "w") as f:
                    json.dump(gen_best, f, indent=2, default=str)
                print(f"  New best: fitness={gen_best['fitness']:.4f} (genome {gen_best['genome_id']})")

        if gen_idx < N_GENERATIONS - 1:
            population = select_and_reproduce(results, rng)
            print(f"  Reproduced: {len(population)} genomes for next generation")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  Evolution Complete!")
    print(f"{'='*70}")
    print(f"  Best fitness: {best_ever['fitness']:.4f}")
    print(f"  Breadth: {best_ever.get('breadth', '?')}/{len(TASK_TYPES)} task types with signal")
    print(f"  Avg train: {best_ever.get('avg_train', 0):.4f}  "
          f"Avg holdout: {best_ever.get('avg_holdout', 0):.4f}  "
          f"Avg trend: {best_ever.get('avg_trend', 0):.4f}")

    if "per_type" in best_ever:
        print(f"\n  Per-task breakdown:")
        for ttype in TASK_TYPES:
            info = best_ever["per_type"].get(ttype, {})
            t = info.get("train_acc", 0)
            h = info.get("holdout_acc", 0)
            tr = info.get("trend_bonus", 0)
            print(f"    {ttype:<15} train={t:.3f}  holdout={h:.3f}  trend={tr:.3f}")

    print(f"\n  Best genome params (learning-relevant):")
    if "genome" in best_ever:
        g = best_ever["genome"]
        for key in EVOLVE_PARAMS:
            if key in g:
                print(f"    {key}: {g[key]:.6f}")

    print(f"\n  Results saved to: {output_dir}")

    # Analyze convergence across survivors
    print(f"\n  Parameter convergence across last generation survivors:")
    last_gen_file = output_dir / f"gen_{N_GENERATIONS-1:02d}.json"
    if last_gen_file.exists():
        with open(last_gen_file) as f:
            last_results = json.load(f)
        valid_last = sorted([r for r in last_results if "error" not in r],
                           key=lambda r: r["fitness"], reverse=True)[:TOP_K]
        if len(valid_last) >= 2:
            print(f"  {'Parameter':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            print(f"  {'-'*70}")
            for key in EVOLVE_PARAMS:
                vals = [r["genome"].get(key, 0) for r in valid_last]
                print(f"  {key:<30} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} "
                      f"{np.min(vals):>10.4f} {np.max(vals):>10.4f}")


if __name__ == "__main__":
    main()
