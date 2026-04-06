"""
Visualize tutorial step sequences as a slideshow.

Usage:
  python visualize_tutorial.py tutorials/tile_pattern     # view one tutorial
  python visualize_tutorial.py tutorials                   # view all tutorials
  python visualize_tutorial.py tutorials --animate         # animated version
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.tutorials import load_tutorial, load_tutorials, Tutorial

ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: grey
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: cyan
    "#B10DC9",  # 9: maroon
]

ARC_CMAP = mcolors.ListedColormap(ARC_COLORS)
ARC_NORM = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 10.5, 1), ncolors=10)


def _draw_grid(ax: plt.Axes, grid: np.ndarray, title: str = "") -> None:
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    h, w = grid.shape
    for r in range(h + 1):
        ax.axhline(r - 0.5, color="white", linewidth=0.5)
    for c in range(w + 1):
        ax.axvline(c - 0.5, color="white", linewidth=0.5)


def show_tutorial(tut: Tutorial) -> None:
    """Show a single tutorial as a grid of subplots: input, steps..., output."""
    n_steps = len(tut.steps)
    n_panels = 2 + n_steps  # input + steps + output
    cols = min(n_panels, 6)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.suptitle(f"{tut.name}: {tut.description}", fontsize=12, fontweight="bold")

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    panels = [("Input", tut.input_grid)]
    for i, step_grid in enumerate(tut.steps):
        panels.append((f"Step {i + 1}", step_grid))
    panels.append(("Output", tut.output_grid))

    for idx, (label, grid) in enumerate(panels):
        r, c = divmod(idx, cols)
        _draw_grid(axes[r, c], grid, title=label)

    for idx in range(len(panels), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    plt.tight_layout()


def animate_tutorial(tut: Tutorial) -> None:
    """Animate a tutorial: input → steps → output with a pause between frames."""
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib.animation not available, falling back to static view")
        show_tutorial(tut)
        return

    frames = [("Input", tut.input_grid)]
    for i, step_grid in enumerate(tut.steps):
        frames.append((f"Step {i + 1}/{len(tut.steps)}", step_grid))
    frames.append(("Output (final)", tut.output_grid))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle(f"{tut.name}: {tut.description}", fontsize=12, fontweight="bold")

    def update(frame_idx):
        ax.clear()
        label, grid = frames[frame_idx]
        _draw_grid(ax, grid, title=label)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000, repeat=True)
    plt.show()
    return anim  # prevent GC


def main():
    parser = argparse.ArgumentParser(description="Visualize ARC tutorial steps")
    parser.add_argument("path", help="Path to a tutorial folder or the tutorials root")
    parser.add_argument("--animate", action="store_true",
                        help="Animate tutorials instead of static subplots")
    args = parser.parse_args()

    path = Path(args.path)

    if (path / "steps.json").exists():
        tutorials = [load_tutorial(path / "steps.json")]
        tutorials = [t for t in tutorials if t is not None]
    else:
        tutorials = load_tutorials(path)

    if not tutorials:
        print(f"No tutorials found at {path}")
        sys.exit(1)

    print(f"Found {len(tutorials)} tutorial(s)")
    for tut in tutorials:
        print(f"  - {tut.name}: {len(tut.steps)} steps ({tut.input_grid.shape} -> {tut.output_grid.shape})")

    if args.animate:
        for tut in tutorials:
            animate_tutorial(tut)
    else:
        for tut in tutorials:
            show_tutorial(tut)
        plt.show()


if __name__ == "__main__":
    main()
