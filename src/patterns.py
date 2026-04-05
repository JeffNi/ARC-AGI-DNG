"""
Synthetic pattern generators for unsupervised infancy exposure.

These aren't ARC tasks -- they're raw visual stimuli that let the
network develop feature detectors through Hebbian self-organization,
the way a baby's visual cortex develops from seeing the world.

Pattern types (roughly ordered by complexity):
  1. Solid fills (single color awareness)
  2. Horizontal/vertical lines (orientation)
  3. Rectangles and borders (enclosed regions)
  4. Color boundaries (where one color meets another)
  5. Diagonal patterns (non-axis-aligned features)
  6. Checkerboards and grids (periodic structure)
  7. Symmetry (horizontal, vertical, rotational)
  8. Simple shapes (L, T, +, O)
  9. Noise (negative examples -- no structure to find)
"""

from __future__ import annotations
import numpy as np


def generate_all(
    max_h: int,
    max_w: int,
    rng: np.random.Generator | None = None,
    n_per_type: int = 30,
) -> list[np.ndarray]:
    """Generate a diverse set of grid patterns for nursery exposure."""
    if rng is None:
        rng = np.random.default_rng()

    grids = []
    generators = [
        _solid_fills,
        _horizontal_lines,
        _vertical_lines,
        _rectangles,
        _borders,
        _color_boundaries_h,
        _color_boundaries_v,
        _diagonals,
        _checkerboards,
        _h_symmetry,
        _v_symmetry,
        _simple_shapes,
        _stripes,
        _dots,
        _random_noise,
    ]

    for gen in generators:
        for _ in range(n_per_type):
            h = rng.integers(2, max_h + 1)
            w = rng.integers(2, max_w + 1)
            g = gen(h, w, rng)
            grids.append(g)

    rng.shuffle(grids)
    return grids


def _rcolor(rng, exclude=None):
    """Random color 1-9 (non-background)."""
    while True:
        c = int(rng.integers(1, 10))
        if exclude is None or c != exclude:
            return c


def _solid_fills(h, w, rng):
    return np.full((h, w), _rcolor(rng), dtype=int)


def _horizontal_lines(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    row = rng.integers(0, h)
    g[row, :] = _rcolor(rng)
    if rng.random() > 0.5 and h > 2:
        row2 = rng.integers(0, h)
        g[row2, :] = _rcolor(rng)
    return g


def _vertical_lines(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    col = rng.integers(0, w)
    g[:, col] = _rcolor(rng)
    if rng.random() > 0.5 and w > 2:
        col2 = rng.integers(0, w)
        g[:, col2] = _rcolor(rng)
    return g


def _rectangles(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    r1, r2 = sorted(rng.integers(0, h, size=2))
    c1, c2 = sorted(rng.integers(0, w, size=2))
    r2 = max(r2, r1 + 1)
    c2 = max(c2, c1 + 1)
    g[r1:r2+1, c1:c2+1] = _rcolor(rng)
    return g


def _borders(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    c = _rcolor(rng)
    g[0, :] = c
    g[-1, :] = c
    g[:, 0] = c
    g[:, -1] = c
    if h > 2 and w > 2 and rng.random() > 0.5:
        g[1:-1, 1:-1] = _rcolor(rng, exclude=c)
    return g


def _color_boundaries_h(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    split = rng.integers(1, h)
    c1, c2 = _rcolor(rng), _rcolor(rng)
    g[:split, :] = c1
    g[split:, :] = c2
    return g


def _color_boundaries_v(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    split = rng.integers(1, w)
    c1, c2 = _rcolor(rng), _rcolor(rng)
    g[:, :split] = c1
    g[:, split:] = c2
    return g


def _diagonals(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    c = _rcolor(rng)
    for i in range(min(h, w)):
        g[i, i] = c
    if rng.random() > 0.5:
        c2 = _rcolor(rng)
        for i in range(min(h, w)):
            g[i, w - 1 - i] = c2
    return g


def _checkerboards(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    c1, c2 = _rcolor(rng), _rcolor(rng)
    period = rng.integers(1, 3)
    for r in range(h):
        for cc in range(w):
            g[r, cc] = c1 if ((r // period) + (cc // period)) % 2 == 0 else c2
    return g


def _h_symmetry(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    half = (w + 1) // 2
    for r in range(h):
        for cc in range(half):
            g[r, cc] = rng.integers(0, 5)
    for r in range(h):
        for cc in range(half, w):
            g[r, cc] = g[r, w - 1 - cc]
    return g


def _v_symmetry(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    half = (h + 1) // 2
    for r in range(half):
        for cc in range(w):
            g[r, cc] = rng.integers(0, 5)
    for r in range(half, h):
        for cc in range(w):
            g[r, cc] = g[h - 1 - r, cc]
    return g


def _simple_shapes(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    c = _rcolor(rng)
    shape = rng.integers(0, 4)
    ch, cw = h // 2, w // 2

    if shape == 0:  # plus/cross
        g[ch, :] = c
        g[:, cw] = c
    elif shape == 1:  # L shape
        g[ch:, cw] = c
        g[-1, cw:] = c
    elif shape == 2:  # T shape
        g[0, :] = c
        g[:, cw] = c
    else:  # O/ring
        if h >= 3 and w >= 3:
            g[0, :] = c
            g[-1, :] = c
            g[:, 0] = c
            g[:, -1] = c
    return g


def _stripes(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    horizontal = rng.random() > 0.5
    period = rng.integers(1, 4)
    c1, c2 = _rcolor(rng), _rcolor(rng)
    for r in range(h):
        for cc in range(w):
            idx = r if horizontal else cc
            g[r, cc] = c1 if (idx // period) % 2 == 0 else c2
    return g


def _dots(h, w, rng):
    g = np.zeros((h, w), dtype=int)
    n_dots = rng.integers(1, max(2, h * w // 3))
    for _ in range(n_dots):
        r, cc = rng.integers(0, h), rng.integers(0, w)
        g[r, cc] = _rcolor(rng)
    return g


def _random_noise(h, w, rng):
    return rng.integers(0, 10, size=(h, w))
