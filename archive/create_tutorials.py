"""
Auto-generate initial tutorial step sequences from ARC-AGI-1 tasks.

Generates ~5-10 tutorial JSONs using heuristics:
  - Row-by-row progressive fill (works for any task)
  - Color-replacement detection (highlights changed colors one at a time)
  - Quadrant-based tiling (for outputs larger than inputs)

The user should review and edit these via visualize_tutorial.py.

Usage:
  python create_tutorials.py                   # generate tutorials
  python create_tutorials.py --visualize       # generate + show them
"""

from __future__ import annotations

import json
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, ".")


def _save_tutorial(name: str, description: str, inp, out, steps, tutorials_dir="tutorials"):
    folder = os.path.join(tutorials_dir, name)
    os.makedirs(folder, exist_ok=True)
    data = {
        "name": name,
        "description": description,
        "input": np.asarray(inp).tolist(),
        "output": np.asarray(out).tolist(),
        "steps": [np.asarray(s).tolist() for s in steps],
    }
    path = os.path.join(folder, "steps.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Created: {path} ({len(steps)} steps)")


def _row_by_row_steps(inp: np.ndarray, out: np.ndarray) -> list:
    """Build output row by row, starting from blank (zeros)."""
    h, w = out.shape
    steps = []
    canvas = np.zeros_like(out)
    for r in range(h):
        canvas[r, :] = out[r, :]
        steps.append(canvas.copy())
    return steps


def _color_replace_steps(inp: np.ndarray, out: np.ndarray) -> list:
    """
    For tasks where input and output have the same shape:
    change one color at a time from input to output.
    """
    if inp.shape != out.shape:
        return []
    diff_mask = inp != out
    if not np.any(diff_mask):
        return []

    changed_positions = np.argwhere(diff_mask)
    unique_target_colors = np.unique(out[diff_mask])

    steps = []
    canvas = inp.copy()
    for color in sorted(unique_target_colors):
        mask = diff_mask & (out == color)
        if np.any(mask):
            canvas = canvas.copy()
            canvas[mask] = color
            steps.append(canvas.copy())

    return steps


def _tiling_steps(inp: np.ndarray, out: np.ndarray) -> list:
    """
    For tasks where output is a tiled version of input.
    Build the output by copying the input pattern tile by tile.
    """
    ih, iw = inp.shape
    oh, ow = out.shape
    if oh < ih or ow < iw:
        return []
    if oh % ih != 0 or ow % iw != 0:
        return []

    tiles_h = oh // ih
    tiles_w = ow // iw

    expected = np.tile(inp, (tiles_h, tiles_w))
    if not np.array_equal(expected, out):
        return []

    steps = []
    canvas = np.zeros_like(out)
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            canvas[tr * ih:(tr + 1) * ih, tc * iw:(tc + 1) * iw] = inp
            steps.append(canvas.copy())
    return steps


def _cell_by_cell_steps(inp: np.ndarray, out: np.ndarray, max_steps: int = 8) -> list:
    """Progressively reveal output cells that differ from a blank canvas."""
    h, w = out.shape
    total = h * w
    cells_per_step = max(1, total // max_steps)

    flat_out = out.ravel()
    steps = []
    canvas = np.zeros_like(out)
    flat_canvas = canvas.ravel()

    for start in range(0, total, cells_per_step):
        end = min(start + cells_per_step, total)
        flat_canvas[start:end] = flat_out[start:end]
        steps.append(flat_canvas.reshape(h, w).copy())

    return steps


def generate_synthetic_tutorials(tutorials_dir="tutorials"):
    """Create a few hand-crafted synthetic tutorials for common patterns."""

    # 1. Simple copy (identity)
    grid = [[1, 2], [3, 4]]
    _save_tutorial(
        "identity_copy",
        "Copy the input grid to the output unchanged",
        grid, grid,
        [grid],  # single step: it's already the answer
        tutorials_dir,
    )

    # 2. Color replacement: change all 1s to 5s
    inp = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    out = [[5, 0, 5], [0, 5, 0], [5, 0, 5]]
    step1 = [[5, 0, 1], [0, 1, 0], [1, 0, 1]]  # top-left changed
    step2 = [[5, 0, 5], [0, 1, 0], [1, 0, 1]]  # top-right changed
    step3 = [[5, 0, 5], [0, 5, 0], [1, 0, 1]]  # center changed
    step4 = [[5, 0, 5], [0, 5, 0], [5, 0, 1]]  # bottom-left changed
    _save_tutorial(
        "color_replace_1to5",
        "Replace all occurrences of color 1 (blue) with color 5 (grey)",
        inp, out,
        [step1, step2, step3, step4, out],
        tutorials_dir,
    )

    # 3. 2x2 tiling
    tile = [[1, 2], [3, 4]]
    tiled = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
    s1 = [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    s2 = [[1, 2, 1, 2], [3, 4, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0]]
    s3 = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 0, 0], [3, 4, 0, 0]]
    _save_tutorial(
        "tile_2x2",
        "Tile a 2x2 pattern to fill a 4x4 grid",
        tile, tiled,
        [s1, s2, s3, tiled],
        tutorials_dir,
    )

    # 4. Horizontal flip
    inp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    out = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
    s1 = [[3, 2, 1], [0, 0, 0], [0, 0, 0]]
    s2 = [[3, 2, 1], [6, 5, 4], [0, 0, 0]]
    _save_tutorial(
        "horizontal_flip",
        "Flip the grid horizontally (mirror left-right)",
        inp, out,
        [s1, s2, out],
        tutorials_dir,
    )

    # 5. Fill border
    inp = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    out = [[2, 2, 2, 2], [2, 1, 1, 2], [2, 1, 1, 2], [2, 2, 2, 2]]
    s1 = [[2, 2, 2, 2], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    s2 = [[2, 2, 2, 2], [2, 1, 1, 2], [0, 1, 1, 0], [0, 0, 0, 0]]
    s3 = [[2, 2, 2, 2], [2, 1, 1, 2], [2, 1, 1, 2], [0, 0, 0, 0]]
    _save_tutorial(
        "fill_border",
        "Fill the border (background) cells with color 2 (red)",
        inp, out,
        [s1, s2, s3, out],
        tutorials_dir,
    )


def generate_from_arc_tasks(tutorials_dir="tutorials", max_tutorials=5):
    """
    Load ARC tasks, find simple ones, and generate step tutorials.
    Tries tiling, color replacement, and row-by-row strategies.
    """
    try:
        import arckit
    except ImportError:
        print("arckit not installed, skipping ARC-based tutorials")
        return

    train_set, _ = arckit.load_data()

    generated = 0
    for task in train_set:
        if generated >= max_tutorials:
            break

        inp0 = np.array(task.train[0][0])
        out0 = np.array(task.train[0][1])
        ih, iw = inp0.shape
        oh, ow = out0.shape

        if max(ih, iw, oh, ow) > 10:
            continue

        name = f"arc_{task.id}"

        # Try tiling
        steps = _tiling_steps(inp0, out0)
        if steps and len(steps) >= 2:
            _save_tutorial(name, f"ARC task {task.id} (tiling pattern)",
                           inp0, out0, steps, tutorials_dir)
            generated += 1
            continue

        # Try color replacement
        if inp0.shape == out0.shape:
            steps = _color_replace_steps(inp0, out0)
            if steps and 2 <= len(steps) <= 8:
                _save_tutorial(name, f"ARC task {task.id} (color mapping)",
                               inp0, out0, steps, tutorials_dir)
                generated += 1
                continue

        # Fallback: row-by-row
        if max(oh, ow) <= 6:
            steps = _row_by_row_steps(inp0, out0)
            if 2 <= len(steps) <= 8:
                _save_tutorial(name, f"ARC task {task.id} (row-by-row construction)",
                               inp0, out0, steps, tutorials_dir)
                generated += 1

    print(f"\nGenerated {generated} tutorials from ARC tasks")


def main():
    parser = argparse.ArgumentParser(description="Generate tutorial JSONs")
    parser.add_argument("--visualize", action="store_true",
                        help="Also show the generated tutorials")
    parser.add_argument("--dir", default="tutorials",
                        help="Output directory")
    args = parser.parse_args()

    print("Generating synthetic tutorials...")
    generate_synthetic_tutorials(args.dir)

    print("\nGenerating tutorials from ARC tasks...")
    generate_from_arc_tasks(args.dir)

    if args.visualize:
        from visualize_tutorial import main as viz_main
        sys.argv = ["visualize_tutorial.py", args.dir]
        viz_main()


if __name__ == "__main__":
    main()
