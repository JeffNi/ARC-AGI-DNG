"""Test childhood learning objectives.

Defines what success looks like for the childhood phase:

  1. IDENTITY: Can the network learn to copy input -> output?
  2. LEARNING CURVE: Does reward improve over study rounds?
  3. RETENTION: After learning task A, does it survive learning task B?
  4. INFANCY PAYOFF: Infant-trained learns faster than fresh network

Run with:
  python test_childhood.py                     # all tests
  python test_childhood.py --test identity     # just identity
  python test_childhood.py --test payoff       # infant vs fresh
"""
import sys
sys.path.insert(0, '.')
import argparse, os
import numpy as np

from src.graph import DNG, Region
from src.encoding import grid_to_signal, signal_to_grid, NUM_COLORS
from src.dynamics import think
from src.pipeline import (
    LifecycleConfig, study_task_round, solve_task, observe_examples,
    _focus_mask, _soft_reset,
)
from src.template import create_dng
from src.genome import Genome
from src.episodic_memory import EpisodicMemory

MAX_H = MAX_W = 10

CHILDHOOD_CONFIG = LifecycleConfig(
    observe_steps=30,
    think_steps=40,
    eta=0.03,
    w_max=2.5,
    noise_std=0.02,
    focus_strength=0.5,
    attempts_per_round=3,
    n_rounds=1,
    rest_steps=15,
    rest_noise_std=0.03,
    sleep_downscale=0.998,
    sleep_tag_threshold=0.001,
    prune_weak_threshold=0.003,
    prune_cycles_required=20,
    replay_eta=0.005,
    replay_passes=2,
    replay_steps=20,
    memory_hint_strength=1.5,
    spontaneous_strength=0.2,
    nursery_binding_eta=0.08,
    nursery_growth_rate=0.8,
    nursery_growth_candidates=500000,
)


def load_infant(path):
    """Load infant checkpoint -- dev params come from the saved network."""
    net = DNG.load(path)
    threshold_path = path.replace(".net.npz", ".threshold.npy")
    if os.path.exists(threshold_path):
        net.threshold = np.load(threshold_path)
    return net


def create_fresh():
    """Create fresh untrained network with newborn parameters."""
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


def make_identity_task(rng, h=5, w=5, n_colors=4):
    """Identity: output = input."""
    grid = rng.integers(0, n_colors, size=(h, w))
    train_pairs = []
    for _ in range(3):
        g = rng.integers(0, n_colors, size=(h, w))
        train_pairs.append((g, g.copy()))
    return train_pairs, grid, grid.copy()


def make_fill_task(rng, h=5, w=5, color=1):
    """Fill: output is all one color regardless of input."""
    target = np.full((h, w), color, dtype=int)
    train_pairs = []
    for _ in range(3):
        g = rng.integers(0, 5, size=(h, w))
        train_pairs.append((g, target.copy()))
    test_in = rng.integers(0, 5, size=(h, w))
    return train_pairs, test_in, target.copy()


def make_recolor_task(rng, h=5, w=5, from_color=1, to_color=3):
    """Recolor: replace one color with another, keep rest."""
    train_pairs = []
    for _ in range(3):
        g = rng.integers(0, 4, size=(h, w))
        out = g.copy()
        out[out == from_color] = to_color
        train_pairs.append((g, out))
    test_in = rng.integers(0, 4, size=(h, w))
    test_out = test_in.copy()
    test_out[test_out == from_color] = to_color
    return train_pairs, test_in, test_out


def eval_guess(net, test_input, test_output, config):
    """Run a solve attempt and return (guess_grid, reward)."""
    out_h, out_w = test_output.shape
    mh, mw = net.max_h, net.max_w
    motor_offset = int(net.output_nodes[0])

    r = solve_task(net, [], test_input, config,
                   output_h=out_h, output_w=out_w)
    fg = test_output != 0
    if fg.any():
        reward = float(np.mean(r.grid[fg] == test_output[fg]))
    else:
        reward = 1.0 if np.array_equal(r.grid, test_output) else 0.0
    perfect = np.array_equal(r.grid, test_output)
    return r.grid, reward, perfect


# ── Test 1: Identity Learning ────────────────────────────────────

def test_identity_learning(net, config, n_rounds=15):
    """Can the network learn to copy input -> output?

    The copy pathway provides a starting point. Hebbian binding
    during observation reinforces the correct pathways.
    """
    print(f"\n  [1] Identity Learning ({n_rounds} rounds)")
    rng = np.random.default_rng(456)
    episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=50)

    train_pairs, test_in, test_out = make_identity_task(rng)

    rewards = []
    for rnd in range(n_rounds):
        result = study_task_round(
            net, train_pairs, test_in, test_out, config,
            prev_best_reward=max(rewards) if rewards else 0.0,
            is_first_visit=(rnd == 0),
            episodic=episodic,
        )
        rewards.append(result.reward)
        if rnd < 5 or rnd % 5 == 4 or result.reward >= 1.0:
            print(f"      Round {rnd+1:2d}: reward={result.reward:.3f}  "
                  f"attempts={result.n_attempts}")
        if result.reward >= 1.0:
            break

    best = max(rewards)
    improving = len(rewards) >= 3 and np.mean(rewards[-3:]) > np.mean(rewards[:3])
    print(f"      Best reward: {best:.3f}  "
          f"Improving: {'YES' if improving else 'NO'}  "
          f"Solved: {'YES' if best >= 1.0 else 'NO'}")
    return best, improving, rewards


# ── Test 2: Learning Curve ───────────────────────────────────────

def test_learning_curve(net, config, n_rounds=20):
    """Does reward improve over rounds on a simple task?"""
    print(f"\n  [2] Learning Curve ({n_rounds} rounds, fill task)")
    rng = np.random.default_rng(789)
    episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=50)

    train_pairs, test_in, test_out = make_fill_task(rng, color=2)

    rewards = []
    for rnd in range(n_rounds):
        result = study_task_round(
            net, train_pairs, test_in, test_out, config,
            prev_best_reward=max(rewards) if rewards else 0.0,
            is_first_visit=(rnd == 0),
            episodic=episodic,
        )
        rewards.append(result.reward)
        if rnd < 3 or rnd % 5 == 4 or result.reward >= 1.0:
            print(f"      Round {rnd+1:2d}: reward={result.reward:.3f}")
        if result.reward >= 1.0:
            break

    first_third = np.mean(rewards[:max(1, len(rewards)//3)])
    last_third = np.mean(rewards[-max(1, len(rewards)//3):])
    improving = last_third > first_third + 0.01
    print(f"      First third avg: {first_third:.3f}  "
          f"Last third avg: {last_third:.3f}  "
          f"Improving: {'YES' if improving else 'NO'}")
    return improving, rewards


# ── Test 3: Retention ────────────────────────────────────────────

def test_retention(net, config, rounds_per_task=10):
    """After learning task A, does it survive learning task B?"""
    print(f"\n  [3] Retention (2 tasks, {rounds_per_task} rounds each)")
    rng = np.random.default_rng(101)
    episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=50)

    task_a = make_identity_task(rng, h=4, w=4)
    task_b = make_fill_task(rng, h=4, w=4, color=3)

    # Learn task A
    print("      Task A (identity):")
    _soft_reset(net)
    for rnd in range(rounds_per_task):
        result = study_task_round(
            net, task_a[0], task_a[1], task_a[2], config,
            prev_best_reward=0.0, is_first_visit=(rnd == 0),
            episodic=episodic,
        )
    reward_a_after_a = result.reward
    print(f"        After learning A: reward={reward_a_after_a:.3f}")

    # Learn task B
    print("      Task B (fill):")
    for rnd in range(rounds_per_task):
        result = study_task_round(
            net, task_b[0], task_b[1], task_b[2], config,
            prev_best_reward=0.0, is_first_visit=(rnd == 0),
            episodic=episodic,
        )
    reward_b_after_b = result.reward
    print(f"        After learning B: reward={reward_b_after_b:.3f}")

    # Re-test task A (without further training)
    _soft_reset(net)
    observe_examples(net, task_a[0], config, episodic=episodic)
    _, reward_a_after_b, perfect_a = eval_guess(net, task_a[1], task_a[2], config)
    print(f"        Task A after learning B: reward={reward_a_after_b:.3f}")

    retained = reward_a_after_b >= reward_a_after_a * 0.5
    print(f"      Retained: {'YES' if retained else 'NO'} "
          f"(A went {reward_a_after_a:.3f} -> {reward_a_after_b:.3f})")
    return retained, reward_a_after_a, reward_a_after_b


# ── Test 4: Infancy Payoff ───────────────────────────────────────

def test_infancy_payoff(infant_path, config, n_rounds=15):
    """Does the infant-trained network learn faster than a fresh one?"""
    print(f"\n  [4] Infancy Payoff (infant vs fresh, {n_rounds} rounds)")

    rng_tasks = np.random.default_rng(222)
    task = make_identity_task(rng_tasks, h=5, w=5)

    results = {}
    for label, net in [
        ("Fresh", create_fresh()),
        ("Infant", load_infant(infant_path)),
    ]:
        print(f"\n      {label} ({net.edge_count():,} edges, "
              f"inh_scale={net.inh_scale:.2f}):")
        episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=50)
        rewards = []
        for rnd in range(n_rounds):
            result = study_task_round(
                net, task[0], task[1], task[2], config,
                prev_best_reward=max(rewards) if rewards else 0.0,
                is_first_visit=(rnd == 0),
                episodic=episodic,
            )
            rewards.append(result.reward)
            if rnd < 3 or rnd % 5 == 4 or result.reward >= 1.0:
                print(f"        Round {rnd+1:2d}: reward={result.reward:.3f}")
            if result.reward >= 1.0:
                break
        results[label] = rewards

    fresh_avg = np.mean(results['Fresh'][-3:])
    infant_avg = np.mean(results['Infant'][-3:])
    infant_faster = infant_avg > fresh_avg + 0.01
    print(f"\n      Fresh last-3 avg:  {fresh_avg:.3f}")
    print(f"      Infant last-3 avg: {infant_avg:.3f}")
    print(f"      Infant advantage: {'YES' if infant_faster else 'NO'}")
    return infant_faster, results


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "identity", "curve",
                                 "retention", "payoff"],
                        help="Which test to run")
    parser.add_argument("--infant", type=str,
                        default="models/infancy_checkpoints/day_0090.net.npz")
    parser.add_argument("--eta", type=float, default=0.03)
    args = parser.parse_args()

    config = CHILDHOOD_CONFIG
    config.eta = args.eta

    run_all = args.test == "all"

    print(f"{'='*60}")
    print(f"  CHILDHOOD LEARNING TESTS")
    print(f"  eta={config.eta}  think_steps={config.think_steps}  "
          f"observe_steps={config.observe_steps}")
    print(f"{'='*60}")

    if run_all or args.test == "identity":
        net = load_infant(args.infant)
        print(f"\n  Network: {net.edge_count():,} edges  "
              f"inh_scale={net.inh_scale:.2f}  wta={net.wta_k_frac:.3f}")
        test_identity_learning(net, config)

    if run_all or args.test == "curve":
        net = load_infant(args.infant)
        test_learning_curve(net, config)

    if run_all or args.test == "retention":
        net = load_infant(args.infant)
        test_retention(net, config)

    if run_all or args.test == "payoff":
        test_infancy_payoff(args.infant, config)


if __name__ == "__main__":
    main()
