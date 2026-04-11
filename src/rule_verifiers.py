"""
Rule-based task verification.

Instead of checking pixel-level correctness (which rewards shortcuts like
the copy pathway), verify whether the network applied the correct
*transformation rule*. Each micro_task type has a known rule; the verifier
checks the network's output against that rule applied to the input.

For complex types where rule verification from input+output alone is
ambiguous (denoise, fill_enclosed, extend_stripe, etc.), we fall back to
comparing against the expected output — but the architecture is in place
to replace these with true rule checks later.
"""

from __future__ import annotations

import numpy as np


def verify_rule(
    task_type: str,
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    expected_grid: np.ndarray,
    task: dict | None = None,
) -> bool:
    """Dispatch to the appropriate rule verifier.

    Args:
        task_type: the task's "type" field (e.g. "identity", "flip_h")
        input_grid: the test input (padded)
        output_grid: what the network produced
        expected_grid: the ground-truth output
        task: full task dict (with train pairs) for rules that need
              to infer parameters from demonstrations

    Returns:
        True if the network applied the correct rule.
    """
    fn = _VERIFIERS.get(task_type)
    if fn is not None:
        return fn(input_grid, output_grid, expected_grid, task)
    return _verify_pixel_match(input_grid, output_grid, expected_grid, task)


# ── Deterministic rule verifiers ──────────────────────────────
# These check whether the OUTPUT follows the rule applied to the INPUT,
# independent of the expected output.  The expected output is only used
# as a fallback or for parameter inference.

def _verify_identity(inp, out, expected, task):
    return np.array_equal(out, inp)


def _verify_flip_h(inp, out, expected, task):
    return np.array_equal(out, np.fliplr(inp))


def _verify_flip_v(inp, out, expected, task):
    return np.array_equal(out, np.flipud(inp))


def _verify_rotate_180(inp, out, expected, task):
    return np.array_equal(out, np.rot90(inp, 2))


def _verify_transpose(inp, out, expected, task):
    return np.array_equal(out, inp.T)


def _verify_solid_fill(inp, out, expected, task):
    """All output cells are the same color (matching expected)."""
    if out.size == 0:
        return False
    target = expected.flat[0]
    return np.all(out == target)


def _verify_binarize(inp, out, expected, task):
    """Non-zero input cells map to target color, zero cells stay zero."""
    if expected.size == 0:
        return False
    target = int(expected[inp != 0].flat[0]) if np.any(inp != 0) else 0
    reconstructed = np.where(inp != 0, target, 0)
    return np.array_equal(out, reconstructed)


def _verify_color_swap(inp, out, expected, task):
    """Each color in input maps to a specific other color in output.
    Infer the mapping from expected, then verify output follows it."""
    mapping = {}
    for iv, ev in zip(inp.flat, expected.flat):
        iv, ev = int(iv), int(ev)
        if iv in mapping:
            if mapping[iv] != ev:
                return np.array_equal(out, expected)
        else:
            mapping[iv] = ev
    reconstructed = np.vectorize(lambda x: mapping.get(int(x), int(x)))(inp)
    return np.array_equal(out, reconstructed)


def _verify_color_extract(inp, out, expected, task):
    """Only cells matching the target color are preserved, rest are 0.
    Infer target color from expected output."""
    nonzero = expected[expected != 0]
    if nonzero.size == 0:
        return np.all(out == 0)
    target = int(nonzero.flat[0])
    reconstructed = np.where(inp == target, target, 0)
    return np.array_equal(out, reconstructed)


def _verify_color_remove(inp, out, expected, task):
    """Remove one specific color (set to 0). Infer which from expected."""
    removed = set(inp.flat) - set(expected.flat) - {0}
    if not removed:
        return np.array_equal(out, expected)
    target = removed.pop()
    reconstructed = np.where(inp == int(target), 0, inp)
    return np.array_equal(out, reconstructed)


def _verify_color_invert(inp, out, expected, task):
    """Infer the color mapping from expected and verify."""
    return _verify_color_swap(inp, out, expected, task)


def _verify_fill_background(inp, out, expected, task):
    """Replace all 0s with a fill color. Infer fill from expected."""
    fills = expected[inp == 0]
    if fills.size == 0:
        return np.array_equal(out, inp)
    fill_color = int(fills.flat[0])
    reconstructed = np.where(inp == 0, fill_color, inp)
    return np.array_equal(out, reconstructed)


def _verify_majority_fill(inp, out, expected, task):
    """Fill entire grid with the most common non-zero color in input."""
    if expected.size == 0:
        return False
    target = int(expected.flat[0])
    return np.all(out == target)


def _verify_translate(inp, out, expected, task):
    """The non-zero pattern appears shifted. Verify against expected
    since offset inference from a single pair is ambiguous."""
    return np.array_equal(out, expected)


def _verify_scale_up_2x(inp, out, expected, task):
    """Each input cell becomes a 2x2 block in output."""
    h, w = inp.shape
    if out.shape != (h * 2, w * 2):
        return False
    for r in range(h):
        for c in range(w):
            block = out[r*2:r*2+2, c*2:c*2+2]
            if not np.all(block == inp[r, c]):
                return False
    return True


def _verify_tile_h(inp, out, expected, task):
    """Output is input tiled horizontally (repeated columns)."""
    h, w = inp.shape
    oh, ow = out.shape
    if oh != h or ow % w != 0:
        return np.array_equal(out, expected)
    n_tiles = ow // w
    for t in range(n_tiles):
        if not np.array_equal(out[:, t*w:(t+1)*w], inp):
            return False
    return True


def _verify_tile_v(inp, out, expected, task):
    """Output is input tiled vertically (repeated rows)."""
    h, w = inp.shape
    oh, ow = out.shape
    if ow != w or oh % h != 0:
        return np.array_equal(out, expected)
    n_tiles = oh // h
    for t in range(n_tiles):
        if not np.array_equal(out[t*h:(t+1)*h, :], inp):
            return False
    return True


def _verify_symmetry_h(inp, out, expected, task):
    """Output has horizontal (left-right) symmetry derived from input."""
    return np.array_equal(out, expected)


def _verify_symmetry_v(inp, out, expected, task):
    """Output has vertical (top-bottom) symmetry derived from input."""
    return np.array_equal(out, expected)


def _verify_symmetry_4fold(inp, out, expected, task):
    """Output has 4-fold symmetry derived from input."""
    return np.array_equal(out, expected)


def _verify_gravity(inp, out, expected, task):
    """Gravity in some direction. Verify against expected."""
    return np.array_equal(out, expected)


# ── Fallback for complex / ambiguous types ────────────────────

def _verify_pixel_match(inp, out, expected, task):
    """Fallback: exact pixel match against expected output."""
    return np.array_equal(out, expected)


# ── Dispatcher table ──────────────────────────────────────────

_VERIFIERS = {
    # Tier 0
    "identity": _verify_identity,
    # Tier 1
    "solid_fill": _verify_solid_fill,
    "binarize": _verify_binarize,
    # Tier 2 (pointwise)
    "color_swap": _verify_color_swap,
    "color_extract": _verify_color_extract,
    "color_remove": _verify_color_remove,
    "color_invert": _verify_color_invert,
    # Tier 3 (spatial)
    "flip_h": _verify_flip_h,
    "flip_v": _verify_flip_v,
    "rotate_180": _verify_rotate_180,
    "translate": _verify_translate,
    "transpose": _verify_transpose,
    # Tier 4 (crop/extract) — output dims differ, need expected
    "bbox_crop": _verify_pixel_match,
    "compact_rows": _verify_pixel_match,
    "compact_cols": _verify_pixel_match,
    "remove_border": _verify_pixel_match,
    # Tier 5 (add/augment)
    "add_border": _verify_pixel_match,
    "add_row": _verify_pixel_match,
    "fill_background": _verify_fill_background,
    "draw_line": _verify_pixel_match,
    # Tier 6 (duplicate/scale)
    "tile_h": _verify_tile_h,
    "tile_v": _verify_tile_v,
    "scale_up_2x": _verify_scale_up_2x,
    "duplicate_obj": _verify_pixel_match,
    # Tier 7 (gravity)
    "gravity_down": _verify_gravity,
    "gravity_left": _verify_gravity,
    "gravity_right": _verify_gravity,
    # Tier 8 (symmetry)
    "symmetry_h": _verify_symmetry_h,
    "symmetry_v": _verify_symmetry_v,
    "symmetry_4fold": _verify_symmetry_4fold,
    # Tier 9 (pattern/region)
    "denoise": _verify_pixel_match,
    "fill_enclosed": _verify_pixel_match,
    "majority_fill": _verify_majority_fill,
    "extend_stripe": _verify_pixel_match,
}
