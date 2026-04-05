"""
Debug the echo test failure: why does the baby echo pass but the
real-task echo produce garbage?

Tests echo under two conditions:
  A) Clean network (like baby test) - should pass
  B) After observe_examples (like real task) - where does it break?
"""
import sys
sys.path.insert(0, '.')
import numpy as np

from src.genome import Genome
from src.template import create_dng
from src.dynamics import think
from src.encoding import NUM_COLORS, grid_to_signal, signal_to_grid
from src.graph import DNG, Region
from src.pipeline import _focus_mask, observe_examples, _soft_reset, LifecycleConfig
from src.episodic_memory import EpisodicMemory

MAX_H = MAX_W = 10
rng = np.random.default_rng(42)

genome = Genome(
    max_h=MAX_H, max_w=MAX_W,
    n_internal=400, n_concept=80, n_memory=120,
    max_fan_in=100, weight_scale=0.02, frac_inhibitory=0.25,
)

print("Creating 10x10 network...", flush=True)
net = create_dng(genome, MAX_H, MAX_W, rng)
net.wta_k_frac = 0.05
print(f"  {net.n_nodes} nodes, {net.edge_count()} edges", flush=True)

mh, mw = net.max_h, net.max_w
n_total = net.n_nodes
motor_offset = int(net.output_nodes[0])
config = LifecycleConfig()

# The task 2 from focus training (gravity shift)
train_pairs = [
    (np.array([[0,2,2],[0,0,2],[0,0,0]]), np.array([[0,0,0],[0,2,2],[0,0,2]])),
    (np.array([[0,0,0],[1,1,1],[0,0,0]]), np.array([[0,0,0],[0,0,0],[1,1,1]])),
    (np.array([[0,1,0],[1,1,0],[0,0,0]]), np.array([[0,0,0],[0,1,0],[1,1,0]])),
    (np.array([[1,1,1],[0,0,0],[0,0,0]]), np.array([[0,0,0],[1,1,1],[0,0,0]])),
]
test_in = np.array([[0,0,0],[0,1,0],[0,0,0]])
test_out = np.array([[0,0,0],[0,0,0],[0,1,0]])


def fmt(g):
    return " / ".join(" ".join(str(v) for v in row) for row in g)

def read_motor(h, w):
    return signal_to_grid(net.r, h, w, node_offset=motor_offset,
                          max_h=mh, max_w=mw)

def full_reset():
    net.V[:] = 0.0
    net.r[:] = 0.0
    net.prev_r[:] = 0.0
    net.adaptation[:] = 0.0
    net.reset_facilitation()

def motor_reset():
    net.V[net.output_nodes] = 0.0
    net.r[net.output_nodes] = 0.0
    net.prev_r[net.output_nodes] = 0.0

def run_echo(label, show_steps=40, recall_steps=30, noise=0.01):
    """Run echo test and report at every checkpoint."""
    out_h, out_w = test_out.shape
    in_h, in_w = test_in.shape
    focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w), 0.5)

    full_sig = (
        grid_to_signal(test_in, 0, n_total, max_h=mh, max_w=mw) +
        grid_to_signal(test_out, motor_offset, n_total, max_h=mh, max_w=mw) +
        focus
    )
    input_sig = (
        grid_to_signal(test_in, 0, n_total, max_h=mh, max_w=mw) + focus
    )

    motor_reset()
    print(f"\n  [{label}] Show phase ({show_steps} steps)...", flush=True)
    think(net, signal=full_sig, steps=show_steps, noise_std=noise)
    after_show = read_motor(out_h, out_w)
    show_match = int(np.sum(after_show == test_out))
    print(f"    After show:    {fmt(after_show)}  ({show_match}/9 match expected)")

    print(f"  [{label}] Recall phase (input only)...")
    for step_count in [5, 10, 20, 30]:
        delta = step_count - (0 if step_count == 5 else [5, 10, 20, 30][[5, 10, 20, 30].index(step_count) - 1])
        think(net, signal=input_sig, steps=delta, noise_std=noise)
        after = read_motor(out_h, out_w)
        match_out = int(np.sum(after == test_out))
        match_in = int(np.sum(after == test_in))
        print(f"    After {step_count:3d} steps: {fmt(after)}  "
              f"(out={match_out}/9 in={match_in}/9)")

    return read_motor(out_h, out_w)


print(f"\nExpected: {fmt(test_out)}")
print(f"Input:    {fmt(test_in)}")

# ── Test A: Clean network ──────────────────────────────────────────
print(f"\n{'='*60}")
print("TEST A: Echo on CLEAN network")
print(f"{'='*60}")
full_reset()
run_echo("CLEAN")

# ── Test B: After observing examples ───────────────────────────────
print(f"\n{'='*60}")
print("TEST B: Echo AFTER observe_examples (like real task)")
print(f"{'='*60}")
full_reset()
ep = EpisodicMemory(max_h=mh, max_w=mw)
print("  Observing 4 training examples...", flush=True)
observe_examples(net, train_pairs, config, ep)
print("  Soft reset...", flush=True)
_soft_reset(net)
run_echo("AFTER_OBS")

# ── Test C: After observe + full network reset ──────────────────────
print(f"\n{'='*60}")
print("TEST C: Echo after observe + FULL reset (is the wiring ruined?)")
print(f"{'='*60}")
full_reset()
observe_examples(net, train_pairs, config, ep)
full_reset()
run_echo("FULL_RESET")

# ── Test D-sim: Simulate focus_session then echo ───────────────────
print(f"\n{'='*60}")
print("TEST D-sim: Echo after simulating focus_session")
print("  (observe + think × 5 attempts, then echo)")
print(f"{'='*60}")
full_reset()
ep2 = EpisodicMemory(max_h=mh, max_w=mw)
focus_sig = (_focus_mask(net, 3, 3, 0.5) +
             grid_to_signal(test_in, 0, n_total, max_h=mh, max_w=mw))

for attempt in range(5):
    _soft_reset(net)
    observe_examples(net, train_pairs, config, ep2)
    think(net, signal=focus_sig, steps=80, noise_std=0.01)
    g = read_motor(3, 3)
    print(f"  Attempt {attempt+1}: {fmt(g)}", flush=True)

print("  Soft reset before echo...", flush=True)
_soft_reset(net)

# Check memory activity
mem_mask = net.regions == list(Region).index(Region.MEMORY)
mem_r = net.r[mem_mask]
print(f"  Memory r after soft_reset: nonzero={np.sum(mem_r > 0.01)} "
      f"max={mem_r.max():.3f} mean={mem_r.mean():.3f}")

run_echo("AFTER_FOCUS_SIM")


# ── Test E: Same but with full reset (proving memory is the culprit)
print(f"\n{'='*60}")
print("TEST E: Same simulation but FULL reset before echo")
print(f"{'='*60}")
full_reset()
ep3 = EpisodicMemory(max_h=mh, max_w=mw)
for attempt in range(5):
    _soft_reset(net)
    observe_examples(net, train_pairs, config, ep3)
    think(net, signal=focus_sig, steps=80, noise_std=0.01)

full_reset()
run_echo("FULL_AFTER_SIM")


# ── Test F: Soft reset but also kill memory neurons ────────────────
print(f"\n{'='*60}")
print("TEST F: Soft reset + zero memory neurons before echo")
print(f"{'='*60}")
full_reset()
ep4 = EpisodicMemory(max_h=mh, max_w=mw)
for attempt in range(5):
    _soft_reset(net)
    observe_examples(net, train_pairs, config, ep4)
    think(net, signal=focus_sig, steps=80, noise_std=0.01)

_soft_reset(net)
mem_nodes = net.memory_nodes
net.V[mem_nodes] = 0.0
net.r[mem_nodes] = 0.0
net.prev_r[mem_nodes] = 0.0
run_echo("SOFT+MEM_KILL")


# ── Test D: Inspect motor V distribution ────────────────────────────
print(f"\n{'='*60}")
print("TEST D: What is motor neuron state after soft reset?")
print(f"{'='*60}")
full_reset()
observe_examples(net, train_pairs, config, ep)
motor_V = net.V[net.output_nodes].copy()
motor_r = net.r[net.output_nodes].copy()
_soft_reset(net)
motor_V_after = net.V[net.output_nodes]
motor_r_after = net.r[net.output_nodes]
print(f"  Motor V before soft reset: min={motor_V.min():.3f} max={motor_V.max():.3f} "
      f"mean={motor_V.mean():.3f} nonzero={np.sum(motor_V > 0.01)}")
print(f"  Motor V after soft reset:  min={motor_V_after.min():.3f} max={motor_V_after.max():.3f} "
      f"mean={motor_V_after.mean():.3f} nonzero={np.sum(motor_V_after > 0.01)}")
print(f"  Motor r after soft reset:  min={motor_r_after.min():.3f} max={motor_r_after.max():.3f} "
      f"nonzero={np.sum(motor_r_after > 0.01)}")

# Internal neuron activity
abstract_mask = net.regions == list(Region).index(Region.ABSTRACT)
internal_r = net.r[abstract_mask]
print(f"\n  Internal r after soft reset: min={internal_r.min():.3f} max={internal_r.max():.3f} "
      f"mean={internal_r.mean():.3f} nonzero={np.sum(internal_r > 0.01)}")
print(f"  Internal facilitation: min={net.f[abstract_mask].min():.3f} "
      f"max={net.f[abstract_mask].max():.3f} mean={net.f[abstract_mask].mean():.3f}")
