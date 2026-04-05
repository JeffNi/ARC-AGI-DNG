"""
Baby milestone tests: can the network do what a newborn can do?

Before reasoning, before learning, the basic infrastructure must work:

  Milestone 1 - HOLD:  Clamp a pattern on motor, remove clamp.
                        Does the pattern persist? (working memory)

  Milestone 2 - COPY:  Present input on sensory neurons only.
                        Does the same pattern appear on motor? (reflex arc)

  Milestone 3 - ECHO:  Show input + different output together, then
                        remove output. Does motor show the OUTPUT
                        (not the input)? (association / short-term memory)

If these fail, nothing else can work.
"""
import sys
sys.path.insert(0, '.')
import numpy as np

from src.genome import Genome
from src.template import create_dng
from src.dynamics import think
from src.encoding import NUM_COLORS, grid_to_signal, signal_to_grid
from src.graph import DNG, Region
from src.pipeline import _focus_mask

MAX_H = MAX_W = 10
rng = np.random.default_rng(42)

genome = Genome(
    max_h=MAX_H, max_w=MAX_W,
    n_internal=400, n_concept=80, n_memory=120,
    max_fan_in=100,
    weight_scale=0.02, frac_inhibitory=0.25,
    f_rate=0.05, f_decay=0.02, f_max=3.0,
)

print("Creating network...", flush=True)
net = create_dng(genome, MAX_H, MAX_W, rng)
net.wta_k_frac = 0.05
net.threshold[:] = 0.15
motor_mask = net.regions == list(Region).index(Region.MOTOR)
net.threshold[motor_mask] = 0.05
net.adapt_rate = 0.01
print(f"  {net.n_nodes} nodes, {net.edge_count()} edges", flush=True)

mh, mw = net.max_h, net.max_w
n_total = net.n_nodes
motor_offset = int(net.output_nodes[0])


def fmt_grid(g):
    return "\n    ".join(" ".join(str(v) for v in row) for row in g)

def reset():
    net.V[:] = 0.0
    net.r[:] = 0.0
    net.prev_r[:] = 0.0
    net.adaptation[:] = 0.0
    net.reset_facilitation()

def read_motor(h, w):
    return signal_to_grid(net.r, h, w, node_offset=motor_offset,
                          max_h=mh, max_w=mw)


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MILESTONE 1: HOLD A PATTERN")
print("  Clamp output on motor -> remove clamp -> does it persist?")
print(f"{'='*60}")

reset()
target = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

motor_signal = grid_to_signal(target, motor_offset, n_total,
                               max_h=mh, max_w=mw)

print(f"\n  Target pattern:")
print(f"    {fmt_grid(target)}")

print(f"\n  Phase 1: Clamping for 30 steps...", flush=True)
think(net, signal=motor_signal, steps=30, noise_std=0.01)
clamped_out = read_motor(3, 3)
print(f"  During clamp: {fmt_grid(clamped_out)}")
clamp_match = np.array_equal(clamped_out, target)
print(f"  Clamp match: {clamp_match}")

print(f"\n  Phase 2: Clamp removed, free running 10 steps...", flush=True)
think(net, signal=None, steps=10, noise_std=0.01)
hold_10 = read_motor(3, 3)
match_10 = int(np.sum(hold_10 == target))
print(f"  After 10 steps: {fmt_grid(hold_10)}")
print(f"  Cells matching: {match_10}/9")

print(f"\n  Phase 3: Free running 30 more steps...", flush=True)
think(net, signal=None, steps=30, noise_std=0.01)
hold_40 = read_motor(3, 3)
match_40 = int(np.sum(hold_40 == target))
print(f"  After 40 steps: {fmt_grid(hold_40)}")
print(f"  Cells matching: {match_40}/9")

think(net, signal=None, steps=50, noise_std=0.01)
hold_90 = read_motor(3, 3)
match_90 = int(np.sum(hold_90 == target))
print(f"  After 90 steps: {fmt_grid(hold_90)}")
print(f"  Cells matching: {match_90}/9")

hold_pass = match_40 >= 7
print(f"\n  MILESTONE 1: {'PASS' if hold_pass else 'FAIL'} "
      f"({match_40}/9 cells held after 40 steps)")


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MILESTONE 2: COPY (REFLEX)")
print("  Present input on sensory -> does motor show the same pattern?")
print(f"{'='*60}")

reset()
test_input = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

input_signal = grid_to_signal(test_input, 0, n_total, max_h=mh, max_w=mw)
focus = _focus_mask(net, 3, 3, 0.5)

print(f"\n  Input pattern:")
print(f"    {fmt_grid(test_input)}")

print(f"\n  Presenting input for 50 steps...", flush=True)
think(net, signal=input_signal + focus, steps=50, noise_std=0.01)
copy_out = read_motor(3, 3)
copy_match = int(np.sum(copy_out == test_input))
print(f"  Motor output: {fmt_grid(copy_out)}")
print(f"  Cells matching input: {copy_match}/9")

copy_pass = copy_match >= 7
print(f"\n  MILESTONE 2: {'PASS' if copy_pass else 'FAIL'} "
      f"({copy_match}/9 cells copied)")


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MILESTONE 3: ECHO (ASSOCIATION)")
print("  Show input+output together -> remove output -> does motor")
print("  show the OUTPUT (not the input)?")
print(f"{'='*60}")

reset()
echo_input = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

echo_output = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]])

print(f"\n  Input (diagonal):")
print(f"    {fmt_grid(echo_input)}")
print(f"  Output (reverse diagonal):")
print(f"    {fmt_grid(echo_output)}")

in_sig = grid_to_signal(echo_input, 0, n_total, max_h=mh, max_w=mw)
out_sig = grid_to_signal(echo_output, motor_offset, n_total, max_h=mh, max_w=mw)
focus = _focus_mask(net, 3, 3, 0.5)

print(f"\n  Phase 1: Showing input+output together for 40 steps...", flush=True)
think(net, signal=in_sig + out_sig + focus, steps=40, noise_std=0.01)
during_show = read_motor(3, 3)
print(f"  During showing: {fmt_grid(during_show)}")

print(f"\n  Phase 2: Output removed, input only for 30 steps...", flush=True)
think(net, signal=in_sig + focus, steps=30, noise_std=0.01)
echo_result = read_motor(3, 3)
echo_match_out = int(np.sum(echo_result == echo_output))
echo_match_in = int(np.sum(echo_result == echo_input))
print(f"  Motor output: {fmt_grid(echo_result)}")
print(f"  Matches OUTPUT: {echo_match_out}/9")
print(f"  Matches INPUT:  {echo_match_in}/9")

# The output should match the SHOWN output, not the input
echo_pass = echo_match_out > echo_match_in and echo_match_out >= 5
print(f"\n  MILESTONE 3: {'PASS' if echo_pass else 'FAIL'} "
      f"(output={echo_match_out}/9 vs input={echo_match_in}/9)")
if not echo_pass and echo_match_in > echo_match_out:
    print(f"  (Network reverted to copying input instead of remembering output)")


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MILESTONE 3b: ECHO (HARDER - all cells differ)")
print("  Input and output share NO cells (except color 0 background).")
print(f"{'='*60}")

reset()
echo_input_b = np.array([[2, 3, 4],
                          [5, 6, 7],
                          [8, 9, 1]])

echo_output_b = np.array([[9, 8, 7],
                           [6, 5, 4],
                           [3, 2, 0]])

print(f"\n  Input:")
print(f"    {fmt_grid(echo_input_b)}")
print(f"  Output (completely different):")
print(f"    {fmt_grid(echo_output_b)}")

in_sig_b = grid_to_signal(echo_input_b, 0, n_total, max_h=mh, max_w=mw)
out_sig_b = grid_to_signal(echo_output_b, motor_offset, n_total, max_h=mh, max_w=mw)
focus_b = _focus_mask(net, 3, 3, 0.5)

print(f"\n  Phase 1: Showing input+output for 40 steps...", flush=True)
think(net, signal=in_sig_b + out_sig_b + focus_b, steps=40, noise_std=0.01)

print(f"\n  Phase 2: Output removed, input only for 30 steps...", flush=True)
think(net, signal=in_sig_b + focus_b, steps=30, noise_std=0.01)
echo_b_result = read_motor(3, 3)
echo_b_match_out = int(np.sum(echo_b_result == echo_output_b))
echo_b_match_in = int(np.sum(echo_b_result == echo_input_b))
print(f"  Motor output: {fmt_grid(echo_b_result)}")
print(f"  Matches OUTPUT: {echo_b_match_out}/9")
print(f"  Matches INPUT:  {echo_b_match_in}/9")

echo_b_pass = echo_b_match_out > echo_b_match_in and echo_b_match_out >= 5
print(f"\n  MILESTONE 3b: {'PASS' if echo_b_pass else 'FAIL'} "
      f"(output={echo_b_match_out}/9 vs input={echo_b_match_in}/9)")


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MILESTONE 4: COPY THEN OVERWRITE")
print("  Copy input -> then show different output -> does output win?")
print(f"{'='*60}")

reset()
copy_first = np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])
overwrite_with = np.array([[2, 2, 2],
                            [2, 2, 2],
                            [2, 2, 2]])

in_sig_c = grid_to_signal(copy_first, 0, n_total, max_h=mh, max_w=mw)
out_sig_c = grid_to_signal(overwrite_with, motor_offset, n_total, max_h=mh, max_w=mw)
focus_c = _focus_mask(net, 3, 3, 0.5)

print(f"\n  Phase 1: Present input (all 1s) for 30 steps (copy establishes)...")
think(net, signal=in_sig_c + focus_c, steps=30, noise_std=0.01)
after_copy = read_motor(3, 3)
print(f"  After copy: {fmt_grid(after_copy)}")

print(f"\n  Phase 2: Reset motor, then clamp output (all 2s) for 30 steps...")
net.V[net.output_nodes] = 0.0
net.r[net.output_nodes] = 0.0
net.prev_r[net.output_nodes] = 0.0
think(net, signal=in_sig_c + out_sig_c + focus_c, steps=30, noise_std=0.01)
after_overwrite = read_motor(3, 3)
print(f"  After overwrite: {fmt_grid(after_overwrite)}")

print(f"\n  Phase 3: Remove output, keep input for 30 steps...")
think(net, signal=in_sig_c + focus_c, steps=30, noise_std=0.01)
after_release = read_motor(3, 3)
overwrite_match = int(np.sum(after_release == overwrite_with))
copy_match_back = int(np.sum(after_release == copy_first))
print(f"  Motor output: {fmt_grid(after_release)}")
print(f"  Matches OVERWRITE (2s): {overwrite_match}/9")
print(f"  Matches COPY (1s):      {copy_match_back}/9")

overwrite_pass = overwrite_match >= 7
print(f"\n  MILESTONE 4: {'PASS' if overwrite_pass else 'FAIL'} "
      f"(overwrite={overwrite_match}/9)")


# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("BABY MILESTONE SUMMARY")
print(f"{'='*60}")
results = [
    ("1. HOLD",           hold_pass),
    ("2. COPY",           copy_pass),
    ("3a. ECHO (basic)",  echo_pass),
    ("3b. ECHO (hard)",   echo_b_pass),
    ("4. OVERWRITE",      overwrite_pass),
]
for name, passed in results:
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status}")

n_pass = sum(p for _, p in results)
print(f"\n  {n_pass}/{len(results)} milestones passed")
if n_pass == len(results):
    print("  Baby infrastructure fully operational!")
elif n_pass >= 3:
    print("  Core infrastructure works. Minor gaps remain.")
else:
    print("  Architecture needs work.")
