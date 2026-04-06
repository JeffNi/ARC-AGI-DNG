"""
Generate micro-tasks for pre-ARC childhood training.

~30 distinct operation types across 10 difficulty tiers, ~350 tasks total.
Each task is saved as a JSON file in micro_tasks/tier_NN_name/.

Tasks use varying grid sizes (2x2 to 7x7), randomized colors from the
full ARC palette (0-9), and have 3 train examples + 1 test each.

Usage:
    python scripts/generate_micro_tasks.py
    python scripts/generate_micro_tasks.py --seed 42
    python scripts/generate_micro_tasks.py --output-dir micro_tasks
"""
import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=str, default="micro_tasks")
args = parser.parse_args()

RNG = np.random.default_rng(args.seed)
OUT_DIR = args.output_dir


# ── Helpers ──────────────────────────────────────────────────────────

def rcolor(exclude=None):
    """Random ARC color 1-9 (non-background)."""
    ex = set()
    if exclude is not None:
        ex = set(exclude) if isinstance(exclude, (list, set, tuple)) else {exclude}
    while True:
        c = int(RNG.integers(1, 10))
        if c not in ex:
            return c


def rcolors(n):
    """Return n distinct non-zero colors."""
    pool = RNG.permutation(9) + 1
    return [int(c) for c in pool[:n]]


def rdims(lo=2, hi=7):
    return int(RNG.integers(lo, hi + 1)), int(RNG.integers(lo, hi + 1))


def random_grid(h, w, n_colors=3, density=0.5):
    colors = rcolors(min(n_colors, 9))
    g = np.zeros((h, w), dtype=int)
    mask = RNG.random((h, w)) < density
    g[mask] = RNG.choice(colors, size=int(mask.sum()))
    return g


def random_sparse_grid(h, w, n_dots=None):
    g = np.zeros((h, w), dtype=int)
    if n_dots is None:
        n_dots = int(RNG.integers(2, max(3, h * w // 3)))
    for _ in range(n_dots):
        r, c = int(RNG.integers(0, h)), int(RNG.integers(0, w))
        g[r, c] = rcolor()
    return g


def save_task(tier_dir, idx, tier, task_type, train, test):
    os.makedirs(tier_dir, exist_ok=True)
    obj = {
        "tier": tier,
        "type": task_type,
        "train": [
            {"input": i.tolist(), "output": o.tolist()} for i, o in train
        ],
        "test": [
            {"input": test[0].tolist(), "output": test[1].tolist()}
        ],
    }
    path = os.path.join(tier_dir, f"{idx:03d}_{task_type}.json")
    with open(path, "w") as f:
        json.dump(obj, f)


def make_task(tier, task_type, tier_dir, idx, fn, n_train=3):
    examples = [fn() for _ in range(n_train + 1)]
    train = examples[:n_train]
    test = examples[n_train]
    save_task(tier_dir, idx, tier, task_type, train, test)


# ═══════════════════════════════════════════════════════════════════
# TIER 0: Identity
# ═══════════════════════════════════════════════════════════════════

def gen_tier_0(base_dir, n=20):
    d = os.path.join(base_dir, "tier_00_identity")
    for i in range(n):
        h, w = rdims()
        nc = int(RNG.integers(2, 6))

        def fn(_h=h, _w=w, _nc=nc):
            g = random_grid(_h, _w, n_colors=_nc)
            return g, g.copy()

        make_task(0, "identity", d, i, fn)


# ═══════════════════════════════════════════════════════════════════
# TIER 1: Constant output
# ═══════════════════════════════════════════════════════════════════

def gen_tier_1(base_dir, n_each=10):
    d = os.path.join(base_dir, "tier_01_constant")
    idx = 0

    for i in range(n_each):
        h, w = rdims()
        fill_c = rcolor()

        def fn(_h=h, _w=w, _c=fill_c):
            inp = random_grid(_h, _w)
            out = np.full((_h, _w), _c, dtype=int)
            return inp, out

        make_task(1, "solid_fill", d, idx, fn); idx += 1

    for i in range(n_each):
        h, w = rdims()
        bin_c = rcolor()

        def fn(_h=h, _w=w, _c=bin_c):
            inp = random_grid(_h, _w, density=0.6)
            out = np.where(inp > 0, _c, 0).astype(int)
            return inp, out

        make_task(1, "binarize", d, idx, fn); idx += 1


# ═══════════════════════════════════════════════════════════════════
# TIER 2: Pointwise color operations
# ═══════════════════════════════════════════════════════════════════

def gen_tier_2(base_dir, n_each=10):
    d = os.path.join(base_dir, "tier_02_pointwise")
    idx = 0

    # Color swap: A -> B
    for i in range(n_each):
        h, w = rdims()
        a, b = rcolors(2)

        def fn(_h=h, _w=w, _a=a, _b=b):
            inp = random_grid(_h, _w)
            inp[RNG.random((_h, _w)) < 0.3] = _a
            out = inp.copy()
            out[inp == _a] = _b
            return inp, out

        make_task(2, "color_swap", d, idx, fn); idx += 1

    # Color remove: C -> 0
    for i in range(n_each):
        h, w = rdims()
        target = rcolor()

        def fn(_h=h, _w=w, _t=target):
            inp = random_grid(_h, _w)
            inp[RNG.random((_h, _w)) < 0.3] = _t
            out = inp.copy()
            out[inp == _t] = 0
            return inp, out

        make_task(2, "color_remove", d, idx, fn); idx += 1

    # Single-color extract: keep only C
    for i in range(n_each):
        h, w = rdims()
        keep_c = rcolor()

        def fn(_h=h, _w=w, _c=keep_c):
            inp = random_grid(_h, _w)
            inp[RNG.random((_h, _w)) < 0.3] = _c
            out = np.where(inp == _c, _c, 0).astype(int)
            return inp, out

        make_task(2, "color_extract", d, idx, fn); idx += 1

    # Color invert: non-zero c -> 10-c
    for i in range(n_each):
        h, w = rdims()

        def fn(_h=h, _w=w):
            inp = random_grid(_h, _w, density=0.7)
            out = np.where(inp > 0, 10 - inp, 0).astype(int)
            return inp, out

        make_task(2, "color_invert", d, idx, fn); idx += 1


# ═══════════════════════════════════════════════════════════════════
# TIER 3: Spatial transforms
# ═══════════════════════════════════════════════════════════════════

def gen_tier_3(base_dir, n_each=10):
    d = os.path.join(base_dir, "tier_03_spatial")
    idx = 0

    # Horizontal flip
    for i in range(n_each):
        h, w = rdims()

        def fn(_h=h, _w=w):
            inp = random_grid(_h, _w)
            return inp, np.fliplr(inp)

        make_task(3, "flip_h", d, idx, fn); idx += 1

    # Vertical flip
    for i in range(n_each):
        h, w = rdims()

        def fn(_h=h, _w=w):
            inp = random_grid(_h, _w)
            return inp, np.flipud(inp)

        make_task(3, "flip_v", d, idx, fn); idx += 1

    # Rotate 180
    for i in range(n_each):
        h, w = rdims()

        def fn(_h=h, _w=w):
            inp = random_grid(_h, _w)
            return inp, np.rot90(inp, 2)

        make_task(3, "rotate_180", d, idx, fn); idx += 1

    # Translate (shift, clip at edges)
    for i in range(n_each):
        h, w = rdims(3, 7)
        direction = int(RNG.integers(0, 4))
        shift = int(RNG.integers(1, 3))

        def fn(_h=h, _w=w, _d=direction, _s=shift):
            inp = random_grid(_h, _w, density=0.4)
            out = np.zeros_like(inp)
            if _d == 0:    out[:_h - _s, :] = inp[_s:, :]
            elif _d == 1:  out[_s:, :] = inp[:_h - _s, :]
            elif _d == 2:  out[:, :_w - _s] = inp[:, _s:]
            else:          out[:, _s:] = inp[:, :_w - _s]
            return inp, out

        make_task(3, "translate", d, idx, fn); idx += 1

    # Transpose
    for i in range(n_each):
        h, w = rdims(2, 7)

        def fn(_h=h, _w=w):
            inp = random_grid(_h, _w)
            return inp, inp.T

        make_task(3, "transpose", d, idx, fn); idx += 1


# ═══════════════════════════════════════════════════════════════════
# TIER 4: Crop and extract
# ═══════════════════════════════════════════════════════════════════

def _bbox(g):
    rows = np.any(g > 0, axis=1)
    cols = np.any(g > 0, axis=0)
    if not rows.any():
        return 0, 0, g.shape[0], g.shape[1]
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return rmin, cmin, rmax + 1, cmax + 1


def gen_tier_4(base_dir, n_each=10):
    d = os.path.join(base_dir, "tier_04_crop_extract")
    idx = 0

    # Bounding box crop
    for i in range(n_each):
        h, w = rdims(4, 7)

        def fn(_h=h, _w=w):
            inp = np.zeros((_h, _w), dtype=int)
            oh = int(RNG.integers(2, _h))
            ow = int(RNG.integers(2, _w))
            r0 = int(RNG.integers(0, _h - oh + 1))
            c0 = int(RNG.integers(0, _w - ow + 1))
            inp[r0:r0 + oh, c0:c0 + ow] = random_grid(oh, ow, density=0.8)
            rmin, cmin, rmax, cmax = _bbox(inp)
            if rmin >= rmax or cmin >= cmax:
                inp[_h // 2, _w // 2] = rcolor()
                rmin, cmin, rmax, cmax = _bbox(inp)
            return inp, inp[rmin:rmax, cmin:cmax]

        make_task(4, "bbox_crop", d, idx, fn); idx += 1

    # Remove border
    for i in range(n_each):
        h, w = rdims(4, 7)

        def fn(_h=h, _w=w):
            inp = random_grid(_h, _w, density=0.6)
            bc = rcolor()
            inp[0, :] = bc; inp[-1, :] = bc
            inp[:, 0] = bc; inp[:, -1] = bc
            return inp, inp[1:-1, 1:-1].copy()

        make_task(4, "remove_border", d, idx, fn); idx += 1

    # Compact rows
    for i in range(n_each):
        h, w = rdims(4, 7)

        def fn(_h=h, _w=w):
            inp = np.zeros((_h, _w), dtype=int)
            n_filled = int(RNG.integers(2, _h))
            rows = sorted(RNG.choice(_h, size=n_filled, replace=False))
            for r in rows:
                inp[r] = RNG.integers(1, 10, size=_w)
            out = inp[inp.any(axis=1)]
            if out.size == 0:
                inp[0] = RNG.integers(1, 10, size=_w)
                out = inp[inp.any(axis=1)]
            return inp, out

        make_task(4, "compact_rows", d, idx, fn); idx += 1

    # Compact columns
    for i in range(n_each):
        h, w = rdims(4, 7)

        def fn(_h=h, _w=w):
            inp = np.zeros((_h, _w), dtype=int)
            n_filled = int(RNG.integers(2, _w))
            cols = sorted(RNG.choice(_w, size=n_filled, replace=False))
            for c in cols:
                inp[:, c] = RNG.integers(1, 10, size=_h)
            out = inp[:, inp.any(axis=0)]
            if out.size == 0:
                inp[:, 0] = RNG.integers(1, 10, size=_h)
                out = inp[:, inp.any(axis=0)]
            return inp, out

        make_task(4, "compact_cols", d, idx, fn); idx += 1


# ═══════════════════════════════════════════════════════════════════
# TIER 5: Add and augment
# ═══════════════════════════════════════════════════════════════════

def gen_tier_5(base_dir, n_each=10):
    d = os.path.join(base_dir, "tier_05_add_augment")
    idx = 0

    # Add border
    for i in range(n_each):
        h, w = rdims(2, 5)
        bc = rcolor()

        def fn(_h=h, _w=w, _bc=bc):
            core = random_grid(_h, _w, density=0.7)
            out = np.full((_h + 2, _w + 2), _bc, dtype=int)
            out[1:-1, 1:-1] = core
            return core, out

        make_task(5, "add_border", d, idx, fn); idx += 1

    # Fill background
    for i in range(n_each):
        h, w = rdims()
        fill_c = rcolor()

        def fn(_h=h, _w=w, _c=fill_c):
            inp = random_grid(_h, _w, density=0.4)
            out = inp.copy()
            out[inp == 0] = _c
            return inp, out

        make_task(5, "fill_background", d, idx, fn); idx += 1

    # Add row
    for i in range(n_each):
        h, w = rdims(2, 6)
        rc = rcolor()
        top = bool(RNG.integers(0, 2))

        def fn(_h=h, _w=w, _c=rc, _top=top):
            inp = random_grid(_h, _w, density=0.6)
            row = np.full((1, _w), _c, dtype=int)
            out = np.vstack([row, inp]) if _top else np.vstack([inp, row])
            return inp, out

        make_task(5, "add_row", d, idx, fn); idx += 1

    # Draw line between two same-color pixels
    for i in range(n_each):
        h, w = rdims(4, 7)
        lc = rcolor()
        horiz = bool(RNG.integers(0, 2))

        def fn(_h=h, _w=w, _c=lc, _horiz=horiz):
            inp = np.zeros((_h, _w), dtype=int)
            if _horiz:
                r = int(RNG.integers(0, _h))
                c1, c2 = sorted(int(x) for x in RNG.choice(_w, size=2, replace=False))
                inp[r, c1] = _c; inp[r, c2] = _c
                out = inp.copy()
                out[r, c1:c2 + 1] = _c
            else:
                col = int(RNG.integers(0, _w))
                r1, r2 = sorted(int(x) for x in RNG.choice(_h, size=2, replace=False))
                inp[r1, col] = _c; inp[r2, col] = _c
                out = inp.copy()
                out[r1:r2 + 1, col] = _c
            return inp, out

        make_task(5, "draw_line", d, idx, fn); idx += 1


# ═══════════════════════════════════════════════════════════════════
# TIER 6: Duplicate and scale
# ═══════════════════════════════════════════════════════════════════

def gen_tier_6(base_dir, n_each=10):
    d = os.path.join(base_dir, "tier_06_duplicate_scale")
    idx = 0

    # Tile horizontally
    for i in range(n_each):
        h, w = rdims(2, 4)

        def fn(_h=h, _w=w):
            inp = random_grid(_h, _w, density=0.6)
            return inp, np.hstack([inp, inp])

        make_task(6, "tile_h", d, idx, fn); idx += 1

    # Tile vertically
    for i in range(n_each):
        h, w = rdims(2, 4)

        def fn(_h=h, _w=w):
            inp = random_grid(_h, _w, density=0.6)
            return inp, np.vstack([inp, inp])

        make_task(6, "tile_v", d, idx, fn); idx += 1

    # Scale up 2x
    for i in range(n_each):
        h, w = rdims(2, 4)

        def fn(_h=h, _w=w):
            inp = random_grid(_h, _w, density=0.6)
            return inp, np.repeat(np.repeat(inp, 2, axis=0), 2, axis=1)

        make_task(6, "scale_up_2x", d, idx, fn); idx += 1

    # Duplicate object to marked position
    for i in range(n_each):
        h, w = rdims(5, 7)

        def fn(_h=h, _w=w):
            obj_h = int(RNG.integers(1, max(2, _h // 2)))
            obj_w = int(RNG.integers(1, max(2, _w // 2)))
            obj = random_grid(obj_h, obj_w, density=0.8)
            obj[obj == 0] = rcolor()

            inp = np.zeros((_h, _w), dtype=int)
            r1 = int(RNG.integers(0, max(1, _h - obj_h * 2 - 1)))
            c1 = int(RNG.integers(0, max(1, _w - obj_w)))
            inp[r1:r1 + obj_h, c1:c1 + obj_w] = obj

            marker_c = rcolor(exclude=list(np.unique(obj)))
            r2 = min(r1 + obj_h + 1, _h - obj_h)
            r2 = max(r2, 0)
            inp[r2, c1] = marker_c

            out = inp.copy()
            r2e = min(r2 + obj_h, _h)
            c1e = min(c1 + obj_w, _w)
            out[r2:r2e, c1:c1e] = obj[:r2e - r2, :c1e - c1]
            return inp, out

        make_task(6, "duplicate_obj", d, idx, fn); idx += 1


# ═══════════════════════════════════════════════════════════════════
# TIER 7: Gravity and physics
# ═══════════════════════════════════════════════════════════════════

def _gravity_down(g):
    h, w = g.shape
    out = np.zeros_like(g)
    for c in range(w):
        col = g[:, c]
        nz = col[col > 0]
        if len(nz) > 0:
            out[h - len(nz):, c] = nz
    return out


def _gravity_left(g):
    h, w = g.shape
    out = np.zeros_like(g)
    for r in range(h):
        row = g[r, :]
        nz = row[row > 0]
        if len(nz) > 0:
            out[r, :len(nz)] = nz
    return out


def _gravity_right(g):
    h, w = g.shape
    out = np.zeros_like(g)
    for r in range(h):
        row = g[r, :]
        nz = row[row > 0]
        if len(nz) > 0:
            out[r, w - len(nz):] = nz
    return out


def gen_tier_7(base_dir, n_each=10):
    d = os.path.join(base_dir, "tier_07_gravity")
    idx = 0

    for name, func in [("gravity_down", _gravity_down),
                        ("gravity_left", _gravity_left),
                        ("gravity_right", _gravity_right)]:
        for i in range(n_each):
            h, w = rdims(3, 7)

            def fn(_h=h, _w=w, _f=func):
                inp = random_sparse_grid(_h, _w)
                if not inp.any():
                    inp[int(RNG.integers(0, _h)),
                        int(RNG.integers(0, _w))] = rcolor()
                return inp, _f(inp)

            make_task(7, name, d, idx, fn); idx += 1


# ═══════════════════════════════════════════════════════════════════
# TIER 8: Symmetry
# ═══════════════════════════════════════════════════════════════════

def gen_tier_8(base_dir, n_each=10):
    d = os.path.join(base_dir, "tier_08_symmetry")
    idx = 0

    # Complete horizontal symmetry (left half -> fill right)
    for i in range(n_each):
        h = int(RNG.integers(3, 7))
        half_w = int(RNG.integers(2, 4))
        w = half_w * 2

        def fn(_h=h, _hw=half_w, _w=w):
            left = random_grid(_h, _hw, density=0.6)
            full = np.hstack([left, np.fliplr(left)])
            inp = full.copy()
            inp[:, _hw:] = 0
            return inp, full

        make_task(8, "symmetry_h", d, idx, fn); idx += 1

    # Complete vertical symmetry (top half -> fill bottom)
    for i in range(n_each):
        half_h = int(RNG.integers(2, 4))
        h = half_h * 2
        w = int(RNG.integers(3, 7))

        def fn(_h=h, _hh=half_h, _w=w):
            top = random_grid(_hh, _w, density=0.6)
            full = np.vstack([top, np.flipud(top)])
            inp = full.copy()
            inp[_hh:, :] = 0
            return inp, full

        make_task(8, "symmetry_v", d, idx, fn); idx += 1

    # Complete 4-fold symmetry (top-left quadrant -> fill rest)
    for i in range(n_each):
        qh = int(RNG.integers(2, 4))
        qw = int(RNG.integers(2, 4))

        def fn(_qh=qh, _qw=qw):
            quad = random_grid(_qh, _qw, density=0.6)
            top = np.hstack([quad, np.fliplr(quad)])
            full = np.vstack([top, np.flipud(top)])
            inp = full.copy()
            inp[_qh:, :] = 0
            inp[:, _qw:] = 0
            inp[:_qh, :_qw] = quad
            return inp, full

        make_task(8, "symmetry_4fold", d, idx, fn); idx += 1


# ═══════════════════════════════════════════════════════════════════
# TIER 9: Pattern and region
# ═══════════════════════════════════════════════════════════════════

def gen_tier_9(base_dir, n_each=10):
    d = os.path.join(base_dir, "tier_09_pattern_region")
    idx = 0

    # Fill enclosed region
    for i in range(n_each):
        h, w = rdims(4, 7)
        bc = rcolor()
        fc = rcolor(exclude=bc)

        def fn(_h=h, _w=w, _bc=bc, _fc=fc):
            inp = np.zeros((_h, _w), dtype=int)
            r1 = int(RNG.integers(0, _h // 2))
            r2 = int(RNG.integers(_h // 2 + 1, _h))
            c1 = int(RNG.integers(0, _w // 2))
            c2 = int(RNG.integers(_w // 2 + 1, _w))
            inp[r1, c1:c2 + 1] = _bc
            inp[r2, c1:c2 + 1] = _bc
            inp[r1:r2 + 1, c1] = _bc
            inp[r1:r2 + 1, c2] = _bc
            out = inp.copy()
            if r1 + 1 < r2 and c1 + 1 < c2:
                out[r1 + 1:r2, c1 + 1:c2] = _fc
            return inp, out

        make_task(9, "fill_enclosed", d, idx, fn); idx += 1

    # Denoise (remove isolated single pixels)
    for i in range(n_each):
        h, w = rdims(4, 7)

        def fn(_h=h, _w=w):
            base = random_grid(_h, _w, density=0.3)
            # Ensure some clusters exist
            cluster_c = rcolor()
            cr, cc = int(RNG.integers(0, max(1, _h - 1))), int(RNG.integers(0, max(1, _w - 1)))
            base[cr, cc] = cluster_c
            base[min(cr + 1, _h - 1), cc] = cluster_c

            inp = base.copy()
            n_noise = int(RNG.integers(2, max(3, _h * _w // 4)))
            for _ in range(n_noise):
                r, c = int(RNG.integers(0, _h)), int(RNG.integers(0, _w))
                if inp[r, c] == 0:
                    inp[r, c] = rcolor()

            out = inp.copy()
            for r in range(_h):
                for c in range(_w):
                    if inp[r, c] == 0:
                        continue
                    color = inp[r, c]
                    has_neighbor = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < _h and 0 <= nc < _w and inp[nr, nc] == color:
                            has_neighbor = True
                            break
                    if not has_neighbor:
                        out[r, c] = 0
            return inp, out

        make_task(9, "denoise", d, idx, fn); idx += 1

    # Majority color fill
    for i in range(n_each):
        h, w = rdims()

        def fn(_h=h, _w=w):
            inp = random_grid(_h, _w, density=0.7)
            nz = inp[inp > 0]
            if len(nz) == 0:
                inp[0, 0] = rcolor()
                nz = inp[inp > 0]
            vals, counts = np.unique(nz, return_counts=True)
            majority = int(vals[np.argmax(counts)])
            return inp, np.full((_h, _w), majority, dtype=int)

        make_task(9, "majority_fill", d, idx, fn); idx += 1

    # Extend stripe pattern (fill missing row)
    for i in range(n_each):
        w = int(RNG.integers(3, 7))
        n_rows = int(RNG.integers(4, 7))
        period = int(RNG.integers(2, 4))

        def fn(_w=w, _nr=n_rows, _p=period):
            base_rows = [RNG.integers(1, 10, size=_w).astype(int) for _ in range(_p)]
            full = np.zeros((_nr, _w), dtype=int)
            for r in range(_nr):
                full[r] = base_rows[r % _p]
            gap = int(RNG.integers(1, _nr - 1))
            inp = full.copy()
            inp[gap] = 0
            return inp, full

        make_task(9, "extend_stripe", d, idx, fn); idx += 1


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print(f"Generating micro-tasks into {OUT_DIR}/")

    generators = [
        ("Tier 0: Identity",           gen_tier_0),
        ("Tier 1: Constant output",    gen_tier_1),
        ("Tier 2: Pointwise color",    gen_tier_2),
        ("Tier 3: Spatial transforms",  gen_tier_3),
        ("Tier 4: Crop & extract",     gen_tier_4),
        ("Tier 5: Add & augment",      gen_tier_5),
        ("Tier 6: Duplicate & scale",  gen_tier_6),
        ("Tier 7: Gravity",            gen_tier_7),
        ("Tier 8: Symmetry",           gen_tier_8),
        ("Tier 9: Pattern & region",   gen_tier_9),
    ]

    total = 0
    for name, gen_fn in generators:
        count_before = _count_files(OUT_DIR)
        gen_fn(OUT_DIR)
        count_after = _count_files(OUT_DIR)
        n = count_after - count_before
        total += n
        print(f"  {name}: {n} tasks")

    print(f"\nTotal: {total} tasks generated")


def _count_files(base):
    n = 0
    if not os.path.isdir(base):
        return 0
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if os.path.isdir(p):
            n += len([f for f in os.listdir(p) if f.endswith(".json")])
    return n


if __name__ == "__main__":
    main()
