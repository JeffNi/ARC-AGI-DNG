"""
Diagnostic: Is the network "alive"? Does it try different things?
Show predictions every epoch + multiple samples from same state.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.dynamics import think, step
from src.plasticity import prediction_error_update
from src.encoding import grid_to_signal, signal_to_grid

train_set, _ = arckit.load_data()
task = {t.id: t for t in train_set}['0d3d703e']
train_pairs = [(np.array(i), np.array(o)) for i, o in task.train]
test_in, test_out = np.array(task.test[0][0]), np.array(task.test[0][1])

print(f"Task: {task.id}")
print(f"  Test input:    {test_in.tolist()}")
print(f"  Test expected: {test_out.tolist()}")

genome = Genome(
    n_internal=450,
    density_sensory_to_internal=0.4,
    density_internal_to_motor=0.4,
    density_internal_to_internal=0.2,
    density_motor_to_internal=0.15,
    density_sensory_to_motor=0.1,
    density_internal_to_sensory=0.05,
    weight_scale=0.15,
    frac_inhibitory=0.3,
)

net = create_dng(genome, 3, 3, np.random.default_rng(42))
net.wta_k_frac = 0.3
motor_offset = int(net.output_nodes[0])
n = net.n_nodes

ETA = 0.05
W_MAX = 2.5
NOISE = 0.02

print(f"Network: {n} nodes, {net.edge_count()} edges\n")

def evaluate(net, label=""):
    """Run test input 5 times with different noise, show variation."""
    preds = []
    for trial in range(5):
        bk = (net.V.copy(), net.r.copy(), net.adaptation.copy(), net.prev_r.copy())
        net.V[:] = 0; net.r[:] = 0; net.adaptation[:] = 0; net.prev_r[:] = 0

        # Show training examples first
        for inp, out in train_pairs:
            sig = grid_to_signal(inp, 0, n) + grid_to_signal(out, motor_offset, n)
            think(net, signal=sig, steps=60, noise_std=NOISE * 0.5)

        # Test input
        think(net, signal=grid_to_signal(test_in, 0, n), steps=120, noise_std=NOISE * 0.5)
        pred = signal_to_grid(net.r, 3, 3, node_offset=motor_offset)
        preds.append(pred.tolist())

        net.V[:], net.r[:], net.adaptation[:], net.prev_r[:] = bk

    # Also check train predictions
    bk = (net.V.copy(), net.r.copy(), net.adaptation.copy(), net.prev_r.copy())
    net.V[:] = 0; net.r[:] = 0; net.adaptation[:] = 0; net.prev_r[:] = 0
    train_preds = []
    for inp, out in train_pairs:
        sig = grid_to_signal(inp, 0, n) + grid_to_signal(out, motor_offset, n)
        think(net, signal=sig, steps=60, noise_std=NOISE * 0.5)
    for inp, out in train_pairs:
        net.V[:] *= 0.2; net.r[:] *= 0.2; net.adaptation[:] *= 0.2
        think(net, signal=grid_to_signal(inp, 0, n), steps=100, noise_std=NOISE * 0.5)
        p = signal_to_grid(net.r, 3, 3, node_offset=motor_offset)
        train_preds.append((inp[0].tolist(), out[0].tolist(), p[0].tolist()))
    net.V[:], net.r[:], net.adaptation[:], net.prev_r[:] = bk

    # Count unique predictions
    unique = len(set(str(p) for p in preds))
    print(f"{label}")
    print(f"  Test predictions ({unique}/5 unique):")
    for i, p in enumerate(preds):
        match = "  " + "".join("*" if p[r][c] == test_out[r][c] else "." for r in range(3) for c in range(3))
        print(f"    trial {i}: {p[0]}{match}")
    print(f"  Train sample (row 0 only):")
    for inp_r, out_r, pred_r in train_preds[:2]:
        print(f"    {inp_r} -> {out_r}  guess: {pred_r}")

# Before any learning
print("="*60)
evaluate(net, "BEFORE LEARNING (epoch 0)")

# Train
print(f"\n{'='*60}")
print("Training with prediction-error learning...")
print("="*60)

for epoch in range(1, 51):
    for inp, out in train_pairs:
        net.V[:] *= 0.15; net.r[:] *= 0.15
        net.prev_r[:] *= 0.15; net.adaptation[:] *= 0.15

        sig_in = grid_to_signal(inp, node_offset=0, n_total_nodes=n)
        think(net, signal=sig_in, steps=60, noise_std=NOISE)

        target_r = grid_to_signal(out, node_offset=motor_offset, n_total_nodes=n)
        combined = sig_in + grid_to_signal(out, motor_offset, n)

        for _ in range(8):
            prediction_error_update(net, target_r, eta=ETA, w_max=W_MAX,
                                    error_propagation=0.3)
            think(net, signal=combined, steps=8, noise_std=NOISE)

        think(net, signal=combined, steps=30, noise_std=NOISE)

    if epoch in [1, 2, 5, 10, 20, 30, 50]:
        evaluate(net, f"EPOCH {epoch}")

print(f"\nDone. Edges: {net.edge_count()}")
