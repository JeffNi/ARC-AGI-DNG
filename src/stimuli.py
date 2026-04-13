"""
Infancy stimulus generator — random visual input for sensory exposure.

Produces grids with varied visual properties: solid fills, stripes,
shapes, scattered pixels, noise. No task pairs, no expected output.
The brain just *sees things* and builds internal representations.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

NUM_COLORS = 10


def solid_fill(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Single color fills the entire grid."""
    color = rng.integers(0, NUM_COLORS)
    return np.full((h, w), color, dtype=np.int32)


def horizontal_stripes(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Alternating horizontal stripes of 2 colors."""
    c1, c2 = rng.choice(NUM_COLORS, size=2, replace=False)
    stripe_w = rng.integers(1, max(2, h // 2) + 1)
    grid = np.zeros((h, w), dtype=np.int32)
    for r in range(h):
        grid[r, :] = c1 if (r // stripe_w) % 2 == 0 else c2
    return grid


def vertical_stripes(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Alternating vertical stripes of 2 colors."""
    c1, c2 = rng.choice(NUM_COLORS, size=2, replace=False)
    stripe_w = rng.integers(1, max(2, w // 2) + 1)
    grid = np.zeros((h, w), dtype=np.int32)
    for c in range(w):
        grid[:, c] = c1 if (c // stripe_w) % 2 == 0 else c2
    return grid


def random_scatter(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Random colored pixels scattered on a background."""
    bg = rng.integers(0, NUM_COLORS)
    grid = np.full((h, w), bg, dtype=np.int32)
    n_pixels = rng.integers(1, max(2, h * w // 3))
    for _ in range(n_pixels):
        r, c = rng.integers(0, h), rng.integers(0, w)
        grid[r, c] = rng.integers(0, NUM_COLORS)
    return grid


def rectangle(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A filled rectangle on a background."""
    bg = rng.integers(0, NUM_COLORS)
    fg = (bg + rng.integers(1, NUM_COLORS)) % NUM_COLORS
    grid = np.full((h, w), bg, dtype=np.int32)
    r1 = rng.integers(0, max(1, h - 1))
    c1 = rng.integers(0, max(1, w - 1))
    r2 = rng.integers(r1 + 1, h + 1)
    c2 = rng.integers(c1 + 1, w + 1)
    grid[r1:r2, c1:c2] = fg
    return grid


def l_shape(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """An L-shaped figure on a background."""
    bg = rng.integers(0, NUM_COLORS)
    fg = (bg + rng.integers(1, NUM_COLORS)) % NUM_COLORS
    grid = np.full((h, w), bg, dtype=np.int32)
    arm_h = rng.integers(2, max(3, h))
    arm_w = rng.integers(2, max(3, w))
    thickness = rng.integers(1, max(2, min(arm_h, arm_w) // 2) + 1)
    r0 = rng.integers(0, max(1, h - arm_h + 1))
    c0 = rng.integers(0, max(1, w - arm_w + 1))
    grid[r0:r0 + arm_h, c0:c0 + thickness] = fg
    grid[r0 + arm_h - thickness:r0 + arm_h, c0:c0 + arm_w] = fg
    return grid


def diagonal_line(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A diagonal line across the grid."""
    bg = rng.integers(0, NUM_COLORS)
    fg = (bg + rng.integers(1, NUM_COLORS)) % NUM_COLORS
    grid = np.full((h, w), bg, dtype=np.int32)
    direction = rng.integers(0, 2)  # 0=top-left to bottom-right, 1=top-right to bottom-left
    for i in range(min(h, w)):
        r = i
        c = i if direction == 0 else (w - 1 - i)
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = fg
    return grid


def checkerboard(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Checkerboard pattern."""
    c1, c2 = rng.choice(NUM_COLORS, size=2, replace=False)
    cell_size = rng.integers(1, max(2, min(h, w) // 2) + 1)
    grid = np.zeros((h, w), dtype=np.int32)
    for r in range(h):
        for c in range(w):
            grid[r, c] = c1 if ((r // cell_size) + (c // cell_size)) % 2 == 0 else c2
    return grid


def border_frame(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A colored border around the grid."""
    bg = rng.integers(0, NUM_COLORS)
    border_color = (bg + rng.integers(1, NUM_COLORS)) % NUM_COLORS
    grid = np.full((h, w), bg, dtype=np.int32)
    thickness = rng.integers(1, max(2, min(h, w) // 3) + 1)
    grid[:thickness, :] = border_color
    grid[-thickness:, :] = border_color
    grid[:, :thickness] = border_color
    grid[:, -thickness:] = border_color
    return grid


def random_noise(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Pure random noise — every cell independent."""
    return rng.integers(0, NUM_COLORS, size=(h, w)).astype(np.int32)


def sparse_dots(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A few isolated dots on a background."""
    bg = rng.integers(0, NUM_COLORS)
    grid = np.full((h, w), bg, dtype=np.int32)
    n_dots = rng.integers(1, max(2, min(h, w)))
    for _ in range(n_dots):
        r, c = rng.integers(0, h), rng.integers(0, w)
        grid[r, c] = (bg + rng.integers(1, NUM_COLORS)) % NUM_COLORS
    return grid


def color_blocks(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Grid divided into 2-4 colored rectangular blocks."""
    grid = np.zeros((h, w), dtype=np.int32)
    split_h = rng.integers(0, 2)  # horizontal or vertical split
    if split_h:
        mid = rng.integers(1, max(2, h))
        grid[:mid, :] = rng.integers(0, NUM_COLORS)
        grid[mid:, :] = rng.integers(0, NUM_COLORS)
    else:
        mid = rng.integers(1, max(2, w))
        grid[:, :mid] = rng.integers(0, NUM_COLORS)
        grid[:, mid:] = rng.integers(0, NUM_COLORS)
    return grid


def cross_shape(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A plus/cross shape centered in the grid."""
    bg = rng.integers(0, NUM_COLORS)
    fg = (bg + rng.integers(1, NUM_COLORS)) % NUM_COLORS
    grid = np.full((h, w), bg, dtype=np.int32)
    cy, cx = h // 2, w // 2
    thickness = rng.integers(1, max(2, min(h, w) // 4) + 1)
    grid[max(0, cy - thickness):min(h, cy + thickness + 1), :] = fg
    grid[:, max(0, cx - thickness):min(w, cx + thickness + 1)] = fg
    return grid


# ── Transformation helpers for diagnostic probes ──────────────

def shifted_variant(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift grid by a random offset, filling vacated cells with the mode color."""
    h, w = grid.shape
    dy = rng.integers(-h // 3, h // 3 + 1)
    dx = rng.integers(-w // 3, w // 3 + 1)
    if dy == 0 and dx == 0:
        dy = 1
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[counts.argmax()])
    out = np.full_like(grid, bg)
    src_r = slice(max(0, -dy), min(h, h - dy))
    src_c = slice(max(0, -dx), min(w, w - dx))
    dst_r = slice(max(0, dy), min(h, h + dy))
    dst_c = slice(max(0, dx), min(w, w + dx))
    out[dst_r, dst_c] = grid[src_r, src_c]
    return out


def recolored_variant(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Remap every color to a different color, preserving spatial structure."""
    present = np.unique(grid)
    perm = rng.permutation(NUM_COLORS)
    while any(perm[c] == c for c in present):
        perm = rng.permutation(NUM_COLORS)
    return perm[grid].astype(np.int32)


def random_variant(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random transformation (shift, recolor, or rotate)."""
    choice = int(rng.integers(0, 3))
    if choice == 0:
        return shifted_variant(grid, rng)
    elif choice == 1:
        return recolored_variant(grid, rng)
    else:
        k = int(rng.integers(1, 4))
        return np.rot90(grid, k).copy()


# All generators with equal probability by default
GENERATORS: list[Callable] = [
    solid_fill,
    horizontal_stripes,
    vertical_stripes,
    random_scatter,
    rectangle,
    l_shape,
    diagonal_line,
    checkerboard,
    border_frame,
    random_noise,
    sparse_dots,
    color_blocks,
    cross_shape,
]


class InfancyStimuli:
    """
    Generates an endless stream of visual stimuli for infancy exposure.
    Optionally mixes in real ARC task inputs (without outputs) for
    exposure to natural visual complexity.
    """

    def __init__(
        self,
        max_h: int = 10,
        max_w: int = 10,
        rng: np.random.Generator | None = None,
        arc_input_grids: list[np.ndarray] | None = None,
        arc_mix_ratio: float = 0.2,
    ):
        self.max_h = max_h
        self.max_w = max_w
        self.rng = rng or np.random.default_rng()
        self.arc_grids = arc_input_grids or []
        self.arc_mix_ratio = arc_mix_ratio if self.arc_grids else 0.0

    def generate(self) -> np.ndarray:
        """Generate one random stimulus grid."""
        if self.arc_grids and self.rng.random() < self.arc_mix_ratio:
            idx = self.rng.integers(0, len(self.arc_grids))
            grid = self.arc_grids[idx]
            from .encoding import pad_grid
            return pad_grid(grid, self.max_h, self.max_w)

        # Random sub-grid size for variety
        h = self.rng.integers(2, self.max_h + 1)
        w = self.rng.integers(2, self.max_w + 1)

        gen = self.rng.choice(GENERATORS)
        grid = gen(h, w, self.rng)

        from .encoding import pad_grid
        return pad_grid(grid, self.max_h, self.max_w)

    def generate_batch(self, n: int) -> list[np.ndarray]:
        """Generate n stimuli."""
        return [self.generate() for _ in range(n)]


def load_arc_inputs(task_dir: str, max_h: int, max_w: int) -> list[np.ndarray]:
    """
    Load just the input grids from micro_tasks for use as infancy stimuli.
    Filters out grids that don't fit.
    """
    from pathlib import Path
    import json

    grids = []
    task_path = Path(task_dir)
    if not task_path.exists():
        return grids

    for tier_dir in task_path.iterdir():
        if not tier_dir.is_dir():
            continue
        for task_file in tier_dir.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                for pair in task.get("train", []):
                    g = np.array(pair["input"], dtype=np.int32)
                    if g.shape[0] <= max_h and g.shape[1] <= max_w:
                        grids.append(g)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return grids
