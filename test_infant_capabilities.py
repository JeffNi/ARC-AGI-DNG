"""Test infant-appropriate capabilities of the DNG.

NOT task solving. Tests what an infant brain should be able to do:
  1. Signal flow: does sensory input reach motor output?
  2. Discrimination: do different inputs produce different outputs?
  3. Habituation: does response decrease to repeated stimuli?
  4. Novelty: does a new stimulus after repetition produce stronger response?
  5. Similarity: do similar inputs produce similar representations?
"""
import sys
sys.path.insert(0, '.')
import numpy as np

from src.graph import DNG, Region
from src.encoding import grid_to_signal, signal_to_grid, NUM_COLORS
from src.dynamics import think
from src.pipeline import LifecycleConfig, _focus_mask
from src.patterns import generate_all
from src.template import create_dng
from src.genome import Genome

MAX_H = MAX_W = 10


def load_infant(path, maturity):
    """Load infant with its developmental parameters."""
    net = DNG.load(path)
    abstract_idx = list(Region).index(Region.ABSTRACT)
    memory_idx = list(Region).index(Region.MEMORY)
    abstract = np.where(net.regions == abstract_idx)[0]
    memory = np.where(net.regions == memory_idx)[0]
    developing = np.concatenate([abstract, memory])

    net.wta_k_frac = 0.30 * (1.0 - maturity) + 0.05 * maturity
    net.inh_scale = 0.3 + 0.7 * maturity
    max_exc = 1.5 + 3.5 * maturity
    net.excitability[developing] = np.clip(net.excitability[developing], 0.1, max_exc)
    return net


def create_fresh():
    """Create fresh untrained network."""
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


def present_grid(net, grid, config):
    """Present a grid and return motor output + internal activity."""
    grid = np.asarray(grid)
    gh, gw = grid.shape
    mh, mw = net.max_h, net.max_w
    motor_offset = int(net.output_nodes[0])

    net.V[:] *= 0.1
    net.r[:] *= 0.1
    net.prev_r[:] *= 0.1

    focus = _focus_mask(net, gh, gw, config.focus_strength)
    sig = grid_to_signal(grid, 0, net.n_nodes, max_h=mh, max_w=mw) + focus
    think(net, signal=sig, steps=config.observe_steps, noise_std=config.noise_std)

    abstract_idx = list(Region).index(Region.ABSTRACT)
    abstract_nodes = np.where(net.regions == abstract_idx)[0]

    motor_r = net.r[net.output_nodes].copy()
    internal_r = net.r[abstract_nodes].copy()
    output_grid = signal_to_grid(net.r, gh, gw,
                                  node_offset=motor_offset, max_h=mh, max_w=mw)
    return output_grid, motor_r, internal_r


def test_signal_flow(net, config, grids):
    """Test 1: Does sensory input produce any motor output?"""
    print("\n--- TEST 1: Signal Flow (sensory -> motor) ---")
    motor_activities = []
    for i, grid in enumerate(grids[:10]):
        _, motor_r, _ = present_grid(net, grid, config)
        motor_active = int((motor_r > 0.02).sum())
        motor_mean = float(motor_r.mean())
        motor_max = float(motor_r.max())
        motor_activities.append(motor_active)
        if i < 5:
            print(f"  Grid {i}: motor_active={motor_active}/{len(motor_r)} "
                  f"mean={motor_mean:.4f} max={motor_max:.3f}")

    mean_motor = np.mean(motor_activities)
    print(f"  Average motor active: {mean_motor:.1f}/{len(net.output_nodes)}")
    return mean_motor


def test_discrimination(net, config, grids):
    """Test 2: Do different inputs produce different outputs?"""
    print("\n--- TEST 2: Discrimination (different inputs -> different outputs) ---")
    outputs = []
    internals = []
    for grid in grids[:20]:
        out_grid, motor_r, internal_r = present_grid(net, grid, config)
        outputs.append(out_grid.ravel())
        internals.append(internal_r)

    # How many unique output grids?
    unique_outputs = len(set(tuple(o) for o in outputs))
    print(f"  Unique motor outputs: {unique_outputs}/{len(outputs)}")

    # Internal representation similarity
    int_arr = np.array(internals)
    norms = np.linalg.norm(int_arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = int_arr / norms
    cos_sim = normed @ normed.T
    np.fill_diagonal(cos_sim, np.nan)
    mean_sim = float(np.nanmean(cos_sim))
    print(f"  Mean internal cosine similarity: {mean_sim:.3f}")
    print(f"  (1.0 = all identical, 0.0 = all orthogonal)")

    # Do similar grids produce more similar representations?
    # Compare pairs of same-type vs different-type grids
    return unique_outputs, mean_sim


def test_habituation(net, config, grid):
    """Test 3: Does response decrease when same stimulus is repeated?"""
    print("\n--- TEST 3: Habituation (response to repeated stimulus) ---")
    grid = np.asarray(grid)
    responses = []
    for rep in range(8):
        _, motor_r, internal_r = present_grid(net, grid, config)
        total_activity = float(internal_r.sum())
        motor_activity = float(motor_r.sum())
        responses.append(total_activity)
        print(f"  Rep {rep}: internal_sum={total_activity:.2f} motor_sum={motor_activity:.3f}")

    first_half = np.mean(responses[:4])
    second_half = np.mean(responses[4:])
    ratio = second_half / max(first_half, 1e-8)
    print(f"  First 4 avg: {first_half:.2f}, Last 4 avg: {second_half:.2f}")
    print(f"  Habituation ratio: {ratio:.3f} (<1.0 = habituating)")
    return ratio


def test_novelty(net, config, familiar_grid, novel_grid):
    """Test 4: After habituating, does a novel stimulus produce stronger response?"""
    print("\n--- TEST 4: Novelty Response ---")

    # Habituate to familiar
    for _ in range(6):
        present_grid(net, familiar_grid, config)

    _, _, familiar_r = present_grid(net, familiar_grid, config)
    familiar_sum = float(familiar_r.sum())
    familiar_active = int((familiar_r > 0.02).sum())

    _, _, novel_r = present_grid(net, novel_grid, config)
    novel_sum = float(novel_r.sum())
    novel_active = int((novel_r > 0.02).sum())

    print(f"  Familiar: sum={familiar_sum:.2f} active={familiar_active}")
    print(f"  Novel:    sum={novel_sum:.2f} active={novel_active}")
    ratio = novel_sum / max(familiar_sum, 1e-8)
    print(f"  Novelty ratio: {ratio:.3f} (>1.0 = stronger novel response)")
    return ratio


def test_similarity_structure(net, config, rng):
    """Test 5: Do similar inputs produce similar representations?"""
    print("\n--- TEST 5: Similarity Structure ---")

    # Create pairs of similar grids (same grid with small perturbation)
    base_grids = []
    perturbed_grids = []
    different_grids = []

    for _ in range(10):
        g = rng.integers(0, 5, size=(5, 5))
        base_grids.append(g)
        p = g.copy()
        # Change 2-3 random cells
        for _ in range(rng.integers(2, 4)):
            r, c = rng.integers(0, 5), rng.integers(0, 5)
            p[r, c] = rng.integers(0, 5)
        perturbed_grids.append(p)
        different_grids.append(rng.integers(0, 10, size=(rng.integers(3, 8), rng.integers(3, 8))))

    # Get representations
    base_reps = []
    pert_reps = []
    diff_reps = []
    for b, p, d in zip(base_grids, perturbed_grids, different_grids):
        _, _, br = present_grid(net, b, config)
        base_reps.append(br)
        _, _, pr = present_grid(net, p, config)
        pert_reps.append(pr)
        _, _, dr = present_grid(net, d, config)
        diff_reps.append(dr)

    # Similarity: base vs perturbed should be HIGH
    # Similarity: base vs different should be LOW
    sim_similar = []
    sim_different = []
    for i in range(10):
        bn = np.linalg.norm(base_reps[i])
        pn = np.linalg.norm(pert_reps[i])
        dn = np.linalg.norm(diff_reps[i])
        if bn > 1e-8 and pn > 1e-8:
            sim_similar.append(float(np.dot(base_reps[i], pert_reps[i]) / (bn * pn)))
        if bn > 1e-8 and dn > 1e-8:
            sim_different.append(float(np.dot(base_reps[i], diff_reps[i]) / (bn * dn)))

    mean_sim = np.mean(sim_similar) if sim_similar else 0.0
    mean_diff = np.mean(sim_different) if sim_different else 0.0
    print(f"  Similar pairs (base vs perturbed): cosine={mean_sim:.3f}")
    print(f"  Different pairs (base vs random):  cosine={mean_diff:.3f}")
    print(f"  Gap: {mean_sim - mean_diff:.3f} (>0 = network preserves similarity)")
    return mean_sim, mean_diff


def evaluate(net, label, config, grids, rng):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {net.n_nodes} nodes, {net.edge_count():,} edges")
    print(f"  WTA={net.wta_k_frac:.3f} inh_scale={net.inh_scale:.3f}")
    print(f"{'='*60}")

    motor = test_signal_flow(net, config, grids)
    unique, sim = test_discrimination(net, config, grids)
    hab = test_habituation(net, config, grids[0])
    nov = test_novelty(net, config, grids[0], grids[10])
    sim_s, sim_d = test_similarity_structure(net, config, rng)

    return {
        'motor_active': motor,
        'unique_outputs': unique,
        'mean_similarity': sim,
        'habituation': hab,
        'novelty': nov,
        'sim_similar': sim_s,
        'sim_different': sim_d,
        'sim_gap': sim_s - sim_d,
    }


def main():
    rng = np.random.default_rng(99)
    config = LifecycleConfig(observe_steps=30, noise_std=0.02, focus_strength=0.5)

    grids = generate_all(MAX_H, MAX_W, rng=rng, n_per_type=10)
    rng.shuffle(grids)
    print(f"Generated {len(grids)} test grids")

    # Fresh network (newborn-level parameters)
    fresh = create_fresh()
    fresh_results = evaluate(fresh, "FRESH NEWBORN NETWORK", config, grids, rng)

    # Day 90 infant
    infant = load_infant('models/infancy_checkpoints/day_0090.net.npz', maturity=0.298)
    infant_results = evaluate(infant, "DAY 90 INFANT", config, grids, rng)

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Fresh':>10} {'Infant':>10} {'Better?':>10}")
    print(f"{'-'*60}")
    for key, label, higher_is_better in [
        ('motor_active', 'Motor neurons active', True),
        ('unique_outputs', 'Unique outputs (/20)', True),
        ('mean_similarity', 'Mean internal similarity', False),
        ('habituation', 'Habituation ratio', False),
        ('novelty', 'Novelty ratio', True),
        ('sim_similar', 'Similar pair cosine', True),
        ('sim_different', 'Different pair cosine', False),
        ('sim_gap', 'Similarity gap', True),
    ]:
        f = fresh_results[key]
        i = infant_results[key]
        if higher_is_better:
            winner = "INFANT" if i > f else ("FRESH" if f > i else "TIE")
        else:
            winner = "INFANT" if i < f else ("FRESH" if f < i else "TIE")
        print(f"  {label:<28} {f:>10.3f} {i:>10.3f} {winner:>10}")


if __name__ == "__main__":
    main()
