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
from pathlib import Path

import numpy as np

from src.genome import Genome
from src.brain import Brain
from src.teacher import Teacher
from src.monitor import Monitor


_STOP = False


def _handle_sigint(sig, frame):
    global _STOP
    if _STOP:
        print("\nForced exit.")
        sys.exit(1)
    _STOP = True
    print("\nGraceful shutdown requested... finishing current task and saving.")


def main():
    parser = argparse.ArgumentParser(description="Brain lifecycle runner")
    parser.add_argument("--days", type=int, default=0, help="Days to run (0=indefinite)")
    parser.add_argument("--tasks-per-day", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grid", type=int, default=10, help="Grid dimension (square)")
    parser.add_argument("--checkpoint-dir", type=str, default="runs/life")
    parser.add_argument("--task-dir", type=str, default="micro_tasks")
    parser.add_argument("--fresh", action="store_true", help="Force new brain (ignore checkpoints)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    genome = Genome(
        n_internal=500,
        n_memory=100,
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

            if day % 5 == 0:
                brain.save_milestone(f"day_{day}")

        monitor.status("Lifecycle complete" if not _STOP else "Interrupted")

    finally:
        path = brain.save(tag="final")
        monitor.status(f"Final checkpoint saved: {path}")


if __name__ == "__main__":
    main()
