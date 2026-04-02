"""
Multi-task diagnostic: test the current network on diverse ARC tasks.

Creates a single network sized for max 30x30 grids, trains on a mix
of tasks with different grid sizes, and evaluates generalization.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.dynamics import think
from src.plasticity import prediction_error_update
from src.encoding import grid_to_signal, signal_to_grid

# ── Load ARC data ──────────────────────────────────────────────────

train_set, eval_set = arckit.load_data()

MAX_H, MAX_W = 30, 30

def max_grid_dim(task):
    mh, mw = 0, 0
    for inp, out in task.train:
        inp, out = np.array(inp), np.array(out)
        mh = max(mh, inp.shape[0], out.shape[0])
        mw = max(mw, inp.shape[1], out.shape[1])
    for inp, out in task.test:
        inp, out = np.array(inp), np.array(out)
        mh = max(mh, inp.shape[0], out.shape[0])
        mw = max(mw, inp.shape[1], out.shape[1])
    return mh, mw

# Pick diverse tasks with varying grid sizes
rng_sel = np.random.default_rng(123)
size_buckets = {}  # (h,w) -> [tasks]
for t in train_set:
    mh, mw = max_grid_dim(t)
    if mh <= MAX_H and mw <= MAX_W:
        key = (mh, mw)
        size_buckets.setdefault(key, []).append(t)

selected = []
target_sizes = [(3, 3), (5, 5), (7, 7), (3, 3), (10, 10),
                (4, 4), (6, 6), (9, 9), (5, 5), (8, 8)]
for th, tw in target_sizes:
    best_key = None
    best_dist = 999
    for key in size_buckets:
        d = abs(key[0] - th) + abs(key[1] - tw)
        if d < best_dist and size_buckets[key]:
            best_dist = d
            best_key = key
    if best_key and size_buckets[best_key]:
        idx = rng_sel.integers(len(size_buckets[best_key]))
        task = size_buckets[best_key].pop(idx)
        selected.append(task)

print(f"Testing {len(selected)} tasks with varying grid sizes:")
for t in selected:
    mh, mw = max_grid_dim(t)
    print(f"  {t.id}: {mh}x{mw}, {len(t.train)} train pairs")

# ── Create one network for all tasks ────────────────────────────────

genome = Genome(
    max_h=MAX_H,
    max_w=MAX_W,
    n_internal=500,
    n_concept=100,
    max_fan_in=150,
    density_sensory_to_internal=0.3,
    density_internal_to_motor=0.3,
    density_internal_to_internal=0.3,
    density_motor_to_internal=0.1,
    density_sensory_to_motor=0.05,
    density_internal_to_sensory=0.05,
    density_column_to_concept=0.3,
    density_concept_to_column=0.3,
    weight_scale=0.15,
    frac_inhibitory=0.25,
)

rng = np.random.default_rng(42)
net = create_dng(genome, MAX_H, MAX_W, rng)
net.wta_k_frac = 0.25
motor_offset = int(net.output_nodes[0])
n = net.n_nodes

print(f"\nNetwork: {n} nodes, {net.edge_count()} edges")
print(f"  max grid: {net.max_h}x{net.max_w}")

# ── Train and evaluate each task ────────────────────────────────────

ETA = 0.05
W_MAX = 2.5
NOISE = 0.02
INPUT_STEPS = 60
N_UPDATES = 8
EPOCHS = 15

results = []

for task in selected:
    mh, mw = max_grid_dim(task)
    train_pairs = [(np.array(i), np.array(o)) for i, o in task.train]
    test_in = np.array(task.test[0][0])
    test_out = np.array(task.test[0][1])
    out_h, out_w = test_out.shape

    # Fresh network per task
    rng_task = np.random.default_rng(42)
    net_task = create_dng(genome, MAX_H, MAX_W, rng_task)
    net_task.wta_k_frac = 0.25
    print(f"  Network: {net_task.n_nodes} nodes, {net_task.edge_count()} edges")

    print(f"\n{'='*70}")
    print(f"Task: {task.id} ({mh}x{mw})")

    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_acc = 0.0
        for inp, out in train_pairs:
            net_task.V[:] *= 0.15
            net_task.r[:] *= 0.15
            net_task.prev_r[:] *= 0.15
            net_task.adaptation[:] *= 0.15

            sig_in = grid_to_signal(inp, node_offset=0, n_total_nodes=n,
                                    max_h=MAX_H, max_w=MAX_W)
            think(net_task, signal=sig_in, steps=INPUT_STEPS, noise_std=NOISE)

            guess = signal_to_grid(net_task.r, out.shape[0], out.shape[1],
                                   node_offset=motor_offset,
                                   max_h=MAX_H, max_w=MAX_W)
            train_acc += float(np.mean(guess == out))

            target_r = grid_to_signal(out, node_offset=motor_offset,
                                      n_total_nodes=n, max_h=MAX_H, max_w=MAX_W)
            combined = sig_in + grid_to_signal(out, motor_offset, n,
                                               max_h=MAX_H, max_w=MAX_W)
            for _ in range(N_UPDATES):
                prediction_error_update(net_task, target_r, eta=ETA, w_max=W_MAX,
                                        error_propagation=0.3)
                think(net_task, signal=combined, steps=8, noise_std=NOISE)
            think(net_task, signal=combined, steps=30, noise_std=NOISE)

        train_acc /= len(train_pairs)

        if epoch in [1, 5, 15]:
            bk = (net_task.V.copy(), net_task.r.copy(),
                  net_task.adaptation.copy(), net_task.prev_r.copy())
            net_task.V[:] = 0
            net_task.r[:] = 0
            net_task.adaptation[:] = 0
            net_task.prev_r[:] = 0

            for inp, out in train_pairs:
                sig = (grid_to_signal(inp, 0, n, max_h=MAX_H, max_w=MAX_W) +
                       grid_to_signal(out, motor_offset, n, max_h=MAX_H, max_w=MAX_W))
                think(net_task, signal=sig, steps=60, noise_std=NOISE * 0.3)

            think(net_task, signal=grid_to_signal(test_in, 0, n, max_h=MAX_H, max_w=MAX_W),
                  steps=120, noise_std=NOISE * 0.3)
            pred = signal_to_grid(net_task.r, out_h, out_w,
                                  node_offset=motor_offset,
                                  max_h=MAX_H, max_w=MAX_W)
            test_acc = float(np.mean(pred == test_out))

            print(f"  Ep {epoch:2d}: train={train_acc:.2f} test={test_acc:.0%}")
            net_task.V[:], net_task.r[:], net_task.adaptation[:], net_task.prev_r[:] = bk

    elapsed = time.time() - t0
    results.append({
        'task_id': task.id,
        'grid': f"{mh}x{mw}",
        'final_train_acc': train_acc,
        'final_test_acc': test_acc,
        'elapsed': elapsed,
    })
    print(f"  Time: {elapsed:.1f}s")

# ── Summary ─────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for r in results:
    print(f"  {r['task_id']} ({r['grid']}): "
          f"train={r['final_train_acc']:.2f} test={r['final_test_acc']:.0%} "
          f"({r['elapsed']:.1f}s)")

train_accs = [r['final_train_acc'] for r in results]
test_accs = [r['final_test_acc'] for r in results]
print(f"\nMean train acc: {np.mean(train_accs):.2f}")
print(f"Mean test acc:  {np.mean(test_accs):.2f}")
print(f"Tasks with test > 0%: {sum(1 for a in test_accs if a > 0)}/{len(test_accs)}")
