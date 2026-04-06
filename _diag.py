"""Copy pathway health check across infancy checkpoints."""
import sys, os, glob
sys.path.insert(0, '.')
import numpy as np

from src.graph import DNG, Region
from src.encoding import grid_to_signal, signal_to_grid, NUM_COLORS
from src.dynamics import think
from src.pipeline import LifecycleConfig, observe_examples, _focus_mask

_s = list(Region).index(Region.SENSORY)
_m = list(Region).index(Region.MOTOR)

checkpoints = sorted(glob.glob("models/infancy_checkpoints/day_*.net.npz"))
if not checkpoints:
    print("No checkpoints found")
    sys.exit(1)

rng = np.random.default_rng(42)
inp = rng.integers(0, 4, size=(5, 5))
config = LifecycleConfig(observe_steps=30, think_steps=80, noise_std=0.02)

print(f"{'Day':>5s}  {'Edges':>7s}  {'CopyEdg':>7s}  {'MeanW':>7s}  {'MedianW':>8s}  "
      f"{'MaxW':>6s}  {'>0.10':>6s}  {'>0.30':>6s}  {'inh':>5s}  {'wta':>6s}  "
      f"{'MotorReward':>11s}  {'GuessColors':>12s}")
print("-" * 120)

for cp_path in checkpoints:
    day_str = os.path.basename(cp_path).replace("day_", "").replace(".net.npz", "")
    net = DNG.load(cp_path)
    thr_path = cp_path.replace(".net.npz", ".threshold.npy")
    if os.path.exists(thr_path):
        net.threshold = np.load(thr_path)

    ec = net._edge_count
    src_reg = net.regions[net._edge_src[:ec]]
    dst_reg = net.regions[net._edge_dst[:ec]]
    copy_mask = (src_reg == _s) & (dst_reg == _m)
    copy_w = net._edge_w[:ec][copy_mask]

    n_copy = int(copy_mask.sum())
    mean_w = float(copy_w.mean()) if n_copy > 0 else 0.0
    med_w = float(np.median(copy_w)) if n_copy > 0 else 0.0
    max_w = float(copy_w.max()) if n_copy > 0 else 0.0
    above_10 = int((copy_w > 0.10).sum()) if n_copy > 0 else 0
    above_30 = int((copy_w > 0.30).sum()) if n_copy > 0 else 0

    # Motor output test
    mh, mw = net.max_h, net.max_w
    n = net.n_nodes
    motor_off = int(net.output_nodes[0])

    net.adaptation[:] = 0.0
    net.V[net.output_nodes] = 0.0
    net.r[net.output_nodes] = 0.0
    focus = _focus_mask(net, 5, 5, config.focus_strength)
    test_sig = grid_to_signal(inp, 0, n, max_h=mh, max_w=mw) + focus
    think(net, signal=test_sig, steps=80, noise_std=0.02)

    guess = signal_to_grid(net.V, 5, 5, node_offset=motor_off, max_h=mh, max_w=mw)
    fg = inp != 0
    reward = float(np.mean(guess[fg] == inp[fg])) if fg.any() else 1.0
    colors = np.unique(guess)

    print(f"{day_str:>5s}  {ec:>7,d}  {n_copy:>7,d}  {mean_w:>7.4f}  {med_w:>8.5f}  "
          f"{max_w:>6.3f}  {above_10:>6,d}  {above_30:>6,d}  {net.inh_scale:>5.3f}  "
          f"{net.wta_k_frac:>6.4f}  {reward:>11.3f}  {str(colors.tolist()):>12s}")

# Detailed test on latest checkpoint
print(f"\n{'='*60}")
print(f"Detailed motor test on latest checkpoint: {checkpoints[-1]}")
print(f"{'='*60}")
net = DNG.load(checkpoints[-1])
thr_path = checkpoints[-1].replace(".net.npz", ".threshold.npy")
if os.path.exists(thr_path):
    net.threshold = np.load(thr_path)

# Test with observation first (like childhood would)
train_pairs = [(inp, inp.copy())]
observe_examples(net, train_pairs, config, binding_eta=0.06)
net.adaptation[net.output_nodes] = 0.0
net.V[net.output_nodes] = 0.0
net.r[net.output_nodes] = 0.0

mh, mw = net.max_h, net.max_w
n = net.n_nodes
motor_off = int(net.output_nodes[0])
focus = _focus_mask(net, 5, 5, config.focus_strength)
test_sig = grid_to_signal(inp, 0, n, max_h=mh, max_w=mw) + focus
think(net, signal=test_sig, steps=80, noise_std=0.02)

guess = signal_to_grid(net.V, 5, 5, node_offset=motor_off, max_h=mh, max_w=mw)
fg = inp != 0
reward = float(np.mean(guess[fg] == inp[fg])) if fg.any() else 1.0
print(f"  After observe+think: reward={reward:.3f}  colors={np.unique(guess).tolist()}")

# Per-pixel check on a few positions — show both V and r
motor_V = net.V[net.output_nodes]
motor_r = net.r[net.output_nodes]
for row in range(3):
    for col in range(3):
        pixel_idx = row * mh + col
        volts = [motor_V[pixel_idx * NUM_COLORS + c] for c in range(min(5, NUM_COLORS))]
        rates = [motor_r[pixel_idx * NUM_COLORS + c] for c in range(min(5, NUM_COLORS))]
        expected = inp[row, col]
        winner = int(np.argmax(volts))
        print(f"  ({row},{col}): expect={expected} winner={winner}  "
              f"V={[f'{v:.2f}' for v in volts]}  "
              f"r={[f'{r:.3f}' for r in rates]}")
