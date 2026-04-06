"""Evaluate infant-appropriate capabilities of the DNG.

Biologically grounded probes based on real infant cognition paradigms:

  1. Signal flow: sensory -> motor conduction (VEP analog)
  2. Discrimination: different inputs -> different representations
  3. Habituation: response decrease to repeated stimuli (Fantz 1964)
  4. Novelty: dishabituation to new stimulus after repetition
  5. Spatial organization: sensitivity to spatial structure vs shuffled pixels
  6. Translation invariance: similar response to shifted patterns
  7. Structural similarity: same structure, different colors

Compares fresh (untrained) vs trained infant checkpoints.
"""
import sys
sys.path.insert(0, '.')
import argparse
import numpy as np

from src.graph import DNG, Region
from src.encoding import grid_to_signal, signal_to_grid, NUM_COLORS
from src.dynamics import think
from src.pipeline import LifecycleConfig, _focus_mask
from src.patterns import generate_all
from src.template import create_dng
from src.genome import Genome

MAX_H = MAX_W = 10


def load_network(path, maturity):
    """Load a checkpoint with its developmental parameters."""
    net = DNG.load(path)
    abstract_idx = list(Region).index(Region.ABSTRACT)
    memory_idx = list(Region).index(Region.MEMORY)
    developing = np.where(
        (net.regions == abstract_idx) | (net.regions == memory_idx)
    )[0]

    net.wta_k_frac = 0.30 * (1.0 - maturity) + 0.05 * maturity
    net.inh_scale = 0.3 + 0.7 * maturity
    max_exc = 1.5 + 3.5 * maturity
    net.excitability[developing] = np.clip(
        net.excitability[developing], 0.1, max_exc
    )
    return net


def create_fresh():
    """Create a fresh untrained network with newborn parameters."""
    genome = Genome(
        max_h=MAX_H, max_w=MAX_W,
        n_internal=1500, n_concept=200, n_memory=300,
        max_fan_in=200, weight_scale=0.01,
    )
    rng = np.random.default_rng(42)
    net = create_dng(genome, MAX_H, MAX_W, rng=rng)
    net.adapt_rate = 0.01
    net.wta_k_frac = 0.30
    net.inh_scale = 0.3
    net.threshold[:] = 0.15
    motor_mask = net.regions == list(Region).index(Region.MOTOR)
    net.threshold[motor_mask] = 0.05
    return net


def _get_abstract_nodes(net):
    return np.where(net.regions == list(Region).index(Region.ABSTRACT))[0]


def present_grid(net, grid, config, full_reset=False):
    """Present a grid and return (output_grid, motor_r, internal_r)."""
    grid = np.asarray(grid)
    gh, gw = grid.shape
    mh, mw = net.max_h, net.max_w
    motor_offset = int(net.output_nodes[0])

    if full_reset:
        net.V[:] = 0.0
        net.r[:] = 0.0
        net.prev_r[:] = 0.0
    else:
        net.V[:] *= 0.1
        net.r[:] *= 0.1
        net.prev_r[:] *= 0.1

    focus = _focus_mask(net, gh, gw, config.focus_strength)
    sig = grid_to_signal(grid, 0, net.n_nodes, max_h=mh, max_w=mw) + focus
    think(net, signal=sig, steps=config.observe_steps, noise_std=config.noise_std)

    abstract_nodes = _get_abstract_nodes(net)
    motor_r = net.r[net.output_nodes].copy()
    internal_r = net.r[abstract_nodes].copy()
    output_grid = signal_to_grid(
        net.V, gh, gw, node_offset=motor_offset, max_h=mh, max_w=mw
    )
    return output_grid, motor_r, internal_r


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Test 1: Signal Flow ──────────────────────────────────────────

def test_signal_flow(net, config, grids):
    """Does sensory input reach motor output? (VEP analog)"""
    print("\n  [1] Signal Flow")
    counts = []
    for grid in grids[:10]:
        _, motor_r, _ = present_grid(net, grid, config, full_reset=True)
        counts.append(int((motor_r > 0.02).sum()))
    mean_active = np.mean(counts)
    print(f"      Motor active: {mean_active:.0f}/{len(net.output_nodes)}")
    return float(mean_active)


# ── Test 2: Discrimination ───────────────────────────────────────

def test_discrimination(net, config, grids):
    """Do different inputs produce different internal patterns?"""
    print("\n  [2] Discrimination")
    reps = []
    for grid in grids[:20]:
        _, _, internal_r = present_grid(net, grid, config, full_reset=True)
        reps.append(internal_r)

    arr = np.array(reps)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = arr / norms
    cos = normed @ normed.T
    np.fill_diagonal(cos, np.nan)
    mean_sim = float(np.nanmean(cos))
    print(f"      Mean internal cosine: {mean_sim:.3f}  (lower = more discriminative)")
    return mean_sim


# ── Test 3: Habituation ──────────────────────────────────────────

def test_habituation(net, config, grid):
    """Does response decrease to repeated stimulus? (Fantz paradigm)"""
    print("\n  [3] Habituation")
    grid = np.asarray(grid)
    responses = []
    # Don't full-reset between repetitions — residual activity IS habituation
    net.V[:] = 0.0; net.r[:] = 0.0; net.prev_r[:] = 0.0
    for rep in range(8):
        _, _, internal_r = present_grid(net, grid, config, full_reset=False)
        responses.append(float(internal_r.sum()))

    first = np.mean(responses[:2])
    last = np.mean(responses[-2:])
    ratio = last / max(first, 1e-8)
    print(f"      First 2 avg: {first:.2f}  Last 2 avg: {last:.2f}  "
          f"Ratio: {ratio:.3f}  (<1 = habituating)")
    return ratio


# ── Test 4: Novelty ──────────────────────────────────────────────

def test_novelty(net, config, familiar_grid, novel_grid):
    """After habituation, does a novel stimulus produce stronger response?"""
    print("\n  [4] Novelty (dishabituation)")
    net.V[:] = 0.0; net.r[:] = 0.0; net.prev_r[:] = 0.0

    for _ in range(6):
        present_grid(net, familiar_grid, config, full_reset=False)

    _, _, familiar_r = present_grid(net, familiar_grid, config, full_reset=False)
    familiar_sum = float(familiar_r.sum())

    _, _, novel_r = present_grid(net, novel_grid, config, full_reset=False)
    novel_sum = float(novel_r.sum())

    ratio = novel_sum / max(familiar_sum, 1e-8)
    print(f"      Familiar: {familiar_sum:.2f}  Novel: {novel_sum:.2f}  "
          f"Ratio: {ratio:.2f}x  (>1 = novelty response)")
    return ratio


# ── Test 5: Spatial Organization ─────────────────────────────────

def test_spatial_organization(net, config, rng):
    """Does the network encode spatial structure, not just pixel histograms?

    Present a grid, then present a shuffled version (same colors, random
    positions). If the network does spatial processing, representations
    should be very different despite identical pixel statistics.
    """
    print("\n  [5] Spatial Organization")
    sims = []
    for _ in range(15):
        h, w = rng.integers(4, 8), rng.integers(4, 8)
        grid = rng.integers(0, 6, size=(h, w))

        shuffled = grid.ravel().copy()
        rng.shuffle(shuffled)
        shuffled = shuffled.reshape(h, w)

        _, _, rep_orig = present_grid(net, grid, config, full_reset=True)
        _, _, rep_shuf = present_grid(net, shuffled, config, full_reset=True)
        sims.append(cosine_sim(rep_orig, rep_shuf))

    mean_sim = float(np.mean(sims))
    # Low similarity = network cares about spatial arrangement
    # High similarity = network only sees color histogram
    print(f"      Original vs shuffled cosine: {mean_sim:.3f}  "
          f"(lower = spatially aware)")
    return mean_sim


# ── Test 6: Translation Invariance ───────────────────────────────

def test_translation_invariance(net, config, rng):
    """Does shifting a pattern produce a similar representation?

    Embed a small pattern at two different positions in an empty grid.
    Compare representations. Also compare against a completely different
    pattern as control. Invariance = shifted > different.
    """
    print("\n  [6] Translation Invariance")
    shifted_sims = []
    control_sims = []

    for _ in range(15):
        pattern = rng.integers(1, 6, size=(3, 3))

        grid_a = np.zeros((8, 8), dtype=int)
        grid_b = np.zeros((8, 8), dtype=int)
        grid_c = np.zeros((8, 8), dtype=int)

        # Position A: top-left region
        r1 = rng.integers(0, 2)
        c1 = rng.integers(0, 2)
        grid_a[r1:r1+3, c1:c1+3] = pattern

        # Position B: shifted by 3+ cells
        r2 = rng.integers(4, 6)
        c2 = rng.integers(4, 6)
        grid_b[r2:r2+3, c2:c2+3] = pattern

        # Control: different pattern at position A
        diff_pattern = rng.integers(1, 6, size=(3, 3))
        grid_c[r1:r1+3, c1:c1+3] = diff_pattern

        _, _, rep_a = present_grid(net, grid_a, config, full_reset=True)
        _, _, rep_b = present_grid(net, grid_b, config, full_reset=True)
        _, _, rep_c = present_grid(net, grid_c, config, full_reset=True)

        shifted_sims.append(cosine_sim(rep_a, rep_b))
        control_sims.append(cosine_sim(rep_a, rep_c))

    mean_shifted = float(np.mean(shifted_sims))
    mean_control = float(np.mean(control_sims))
    gap = mean_shifted - mean_control
    print(f"      Same pattern shifted: {mean_shifted:.3f}")
    print(f"      Different pattern:    {mean_control:.3f}")
    print(f"      Gap: {gap:+.3f}  (>0 = some translation invariance)")
    return gap


# ── Test 7: Structural Similarity ────────────────────────────────

def test_structural_similarity(net, config, rng):
    """Same spatial structure with swapped colors vs completely different grid.

    Tests whether the network encodes structure (edges, regions, shape)
    independently from specific color identity.
    """
    print("\n  [7] Structural Similarity (color swap)")
    recolor_sims = []
    different_sims = []

    for _ in range(15):
        h, w = rng.integers(4, 8), rng.integers(4, 8)
        grid = rng.integers(0, 6, size=(h, w))

        # Color permutation: shuffle which color maps to which
        perm = rng.permutation(10)
        recolored = perm[grid]

        # Completely different grid (same size)
        different = rng.integers(0, 6, size=(h, w))

        _, _, rep_orig = present_grid(net, grid, config, full_reset=True)
        _, _, rep_recol = present_grid(net, recolored, config, full_reset=True)
        _, _, rep_diff = present_grid(net, different, config, full_reset=True)

        recolor_sims.append(cosine_sim(rep_orig, rep_recol))
        different_sims.append(cosine_sim(rep_orig, rep_diff))

    mean_recolor = float(np.mean(recolor_sims))
    mean_diff = float(np.mean(different_sims))
    gap = mean_recolor - mean_diff
    print(f"      Recolored (same structure): {mean_recolor:.3f}")
    print(f"      Different grid:             {mean_diff:.3f}")
    print(f"      Gap: {gap:+.3f}  (>0 = encodes structure beyond color)")
    return gap


# ── Full evaluation ──────────────────────────────────────────────

def evaluate(net, label, config, grids, rng):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {net.n_nodes} nodes, {net.edge_count():,} edges")
    print(f"  WTA={net.wta_k_frac:.3f}  inh_scale={net.inh_scale:.3f}")
    print(f"{'='*60}")

    results = {}
    results['signal_flow'] = test_signal_flow(net, config, grids)
    results['discrimination'] = test_discrimination(net, config, grids)
    results['habituation'] = test_habituation(net, config, grids[0])
    results['novelty'] = test_novelty(net, config, grids[0], grids[15])
    results['spatial_org'] = test_spatial_organization(net, config, rng)
    results['translation_inv'] = test_translation_invariance(net, config, rng)
    results['structural_sim'] = test_structural_similarity(net, config, rng)
    return results


def print_comparison(all_results):
    labels = list(all_results.keys())
    metrics = [
        ('signal_flow',     'Signal flow (motor active)',  True),
        ('discrimination',  'Discrimination (cosine)',     False),
        ('habituation',     'Habituation ratio',           False),
        ('novelty',         'Novelty ratio',               True),
        ('spatial_org',     'Spatial org (orig vs shuf)',   False),
        ('translation_inv', 'Translation invariance gap',  True),
        ('structural_sim',  'Structural similarity gap',   True),
    ]

    header = f"{'Metric':<32s}"
    for label in labels:
        header += f" {label:>12s}"
    header += f" {'Best':>12s}"

    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for key, name, higher_is_better in metrics:
        row = f"  {name:<32s}"
        values = []
        for label in labels:
            v = all_results[label][key]
            values.append(v)
            row += f" {v:>12.3f}"

        if higher_is_better:
            best_idx = int(np.argmax(values))
        else:
            best_idx = int(np.argmin(values))
        row += f" {labels[best_idx]:>12s}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=str,
                        default="models/infancy_checkpoints/day_0090.net.npz")
    parser.add_argument("--v1-maturity", type=float, default=0.298)
    parser.add_argument("--v2", type=str,
                        default="models/infancy_v2/day_0060.net.npz")
    parser.add_argument("--v2-maturity", type=float, default=0.663)
    args = parser.parse_args()

    config = LifecycleConfig(observe_steps=30, noise_std=0.02, focus_strength=0.5)

    rng = np.random.default_rng(99)
    grids = generate_all(MAX_H, MAX_W, rng=rng, n_per_type=10)
    rng.shuffle(grids)
    print(f"Generated {len(grids)} test grids")

    all_results = {}

    # Fresh baseline
    rng_eval = np.random.default_rng(777)
    fresh = create_fresh()
    all_results['Fresh'] = evaluate(fresh, "FRESH (untrained)", config, grids, rng_eval)

    # v1 day 90
    rng_eval = np.random.default_rng(777)
    v1 = load_network(args.v1, args.v1_maturity)
    all_results['v1-d90'] = evaluate(v1, "V1 DAY 90 (old scaffold)", config, grids, rng_eval)

    # v2 day 60
    rng_eval = np.random.default_rng(777)
    v2 = load_network(args.v2, args.v2_maturity)
    all_results['v2-d60'] = evaluate(v2, "V2 DAY 60 (new scaffold)", config, grids, rng_eval)

    print_comparison(all_results)


if __name__ == "__main__":
    main()
