"""Diagnostic: trace three-factor learning magnitudes on an identity task."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import DNG, Region
from src.pipeline import (
    LifecycleConfig, think, observe_examples, grid_to_signal,
    signal_to_grid, _focus_mask, _soft_reset, _sharpen_motor,
)
from src.plasticity import error_corrective_update

ckpt = Path("models/infancy_checkpoints/day_0140.net.npz")
net = DNG.load(ckpt)
print(f"Loaded: {net.edge_count()} edges, {net.n_nodes} nodes")

# Simple 2x2 identity task
inp = np.array([[1, 2], [3, 4]])
out = inp.copy()
train_pairs = [(inp, out), (inp, out), (inp, out)]

config = LifecycleConfig(
    observe_steps=40, think_steps=80, error_prop_steps=15,
    eta=0.01, w_max=2.5, noise_std=0.02, attempts_per_round=5,
)

n_total = net.n_nodes
mh, mw = net.max_h, net.max_w
motor_offset = int(net.output_nodes[0])
out_h, out_w = out.shape

region_list = list(Region)
_abstract = region_list.index(Region.ABSTRACT)
_motor = region_list.index(Region.MOTOR)
_sensory = region_list.index(Region.SENSORY)
_memory = region_list.index(Region.MEMORY)

_soft_reset(net)

# Prime
observe_examples(net, train_pairs, config, binding_eta=0.0)

target_motor = grid_to_signal(out, motor_offset, n_total, max_h=mh, max_w=mw)

for attempt in range(1, 6):
    focus = _focus_mask(net, max(inp.shape[0], out_h), max(inp.shape[1], out_w),
                        config.focus_strength)
    test_signal = grid_to_signal(inp, 0, n_total, max_h=mh, max_w=mw) + focus

    think(net, signal=test_signal, steps=config.think_steps, noise_std=config.noise_std)
    r_raw = net.r.copy()

    guess = signal_to_grid(net.V, out_h, out_w, node_offset=motor_offset,
                           max_h=mh, max_w=mw)
    fg = out != 0
    reward = float(np.mean(guess[fg] == out[fg])) if fg.any() else 1.0

    rpe = reward - net.da_baseline
    net.da = max(abs(rpe), 0.1)
    net.da_baseline = 0.9 * net.da_baseline + 0.1 * reward

    # Sharpen motor for learning (like study_task_round does)
    r_guess = _sharpen_motor(r_raw, net.output_nodes, out_h, out_w, mh, mw)

    motor_r = r_guess[net.output_nodes]
    target_r = target_motor[net.output_nodes]
    error_signal = np.zeros(n_total)
    error_signal[net.output_nodes] = target_r - motor_r

    print(f"\n=== Attempt {attempt} ===")
    print(f"  reward={reward:.3f}  DA={net.da:.4f}  baseline={net.da_baseline:.4f}")
    print(f"  guess={guess.flatten()}  target={out.flatten()}")
    print(f"  motor_r(sharp) stats: mean={motor_r.mean():.6f} max={motor_r.max():.6f}")
    print(f"  motor_r(raw)   stats: mean={r_raw[net.output_nodes].mean():.6f} max={r_raw[net.output_nodes].max():.6f}")
    print(f"  target_r stats: mean={target_r[target_r>0].mean():.6f} max={target_r.max():.6f}" if (target_r > 0).any() else "  target_r: all zero")
    print(f"  error_signal: mean_abs={np.abs(error_signal).mean():.6f} "
          f"max={error_signal.max():.6f} min={error_signal.min():.6f}")

    # Error propagation
    think(net, signal=error_signal, steps=config.error_prop_steps,
          noise_std=config.noise_std * 0.25)
    r_corrected = net.r.copy()

    delta_r = r_corrected - r_guess
    print(f"  delta_r (all): mean_abs={np.abs(delta_r).mean():.6f} max={delta_r.max():.6f}")
    print(f"  delta_r (motor): mean_abs={np.abs(delta_r[net.output_nodes]).mean():.6f}")
    print(f"  delta_r (abstract): mean_abs={np.abs(delta_r[net.regions==_abstract]).mean():.6f}")
    print(f"  r_guess (active>0.02): {(r_guess > 0.02).sum()} neurons")
    print(f"  r_guess (sensory): mean={r_guess[net.regions==_sensory].mean():.6f}")
    print(f"  r_guess (abstract): mean={r_guess[net.regions==_abstract].mean():.6f}")
    print(f"  r_guess (motor): mean={r_guess[net.regions==_motor].mean():.6f}")

    # Manually compute what the three-factor update would do
    n = net._edge_count
    src = net._edge_src[:n]
    dst = net._edge_dst[:n]
    pre_active = r_guess[src] > 0.02
    post_shifted = np.abs(delta_r[dst]) > 0.01
    candidate = pre_active & post_shifted
    cand_idx = np.flatnonzero(candidate)
    
    if len(cand_idx) > 0:
        c_src_reg = net.regions[src[cand_idx]]
        c_dst_reg = net.regions[dst[cand_idx]]
        src_ok = (c_src_reg == _abstract) | (c_src_reg == _memory)
        dst_ok = (c_dst_reg == _abstract) | (c_dst_reg == _motor)
        copy_path = (c_src_reg == _sensory) & (c_dst_reg == _motor)
        eligible = (src_ok & dst_ok) | copy_path
        elig_idx = cand_idx[eligible]
        
        if len(elig_idx) > 0:
            pre = r_guess[src[elig_idx]]
            dp = delta_r[dst[elig_idx]]
            da_mod = max(net.da, 0.1)
            dw = config.eta * pre * dp * da_mod
            print(f"  THREE-FACTOR: {len(elig_idx)} eligible edges")
            print(f"    pre:  mean={pre.mean():.6f} max={pre.max():.6f}")
            print(f"    dp:   mean={dp.mean():.6f} mean_abs={np.abs(dp).mean():.6f} max={dp.max():.6f}")
            print(f"    da:   {da_mod:.4f}")
            print(f"    dw:   mean_abs={np.abs(dw).mean():.8f} max={np.abs(dw).max():.8f}")
            print(f"    w_old:mean_abs={np.abs(net._edge_w[elig_idx]).mean():.6f}")
            print(f"    dw/w: mean_abs={(np.abs(dw) / (np.abs(net._edge_w[elig_idx]) + 1e-10)).mean():.6f}")
            
            # Region breakdown
            for rname, ridx in [("abstract->motor", (_abstract, _motor)),
                                ("abstract->abstract", (_abstract, _abstract)),
                                ("sensory->motor", (_sensory, _motor)),
                                ("memory->motor", (_memory, _motor)),
                                ("memory->abstract", (_memory, _abstract))]:
                mask = ((net.regions[src[elig_idx]] == ridx[0]) & 
                        (net.regions[dst[elig_idx]] == ridx[1]))
                if mask.any():
                    print(f"    {rname}: {mask.sum()} edges, mean_abs_dw={np.abs(dw[mask]).mean():.8f}")
        else:
            print(f"  THREE-FACTOR: 0 eligible (of {len(cand_idx)} candidates)")
    else:
        print(f"  THREE-FACTOR: 0 candidates (pre_active={pre_active.sum()}, post_shifted={post_shifted.sum()})")

    # Actually apply the update
    n_changed = error_corrective_update(net, r_guess, r_corrected,
                                        eta=config.eta, w_max=config.w_max)
    print(f"  Applied: {n_changed} edges modified")

print(f"\nFinal edges: {net.edge_count()}")
