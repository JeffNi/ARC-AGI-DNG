"""
Deep focus training: one task at a time, like a human solving puzzles.

Philosophy: A real person either solves an ARC task completely or doesn't.
They don't get "37% of pixels right." They work on one puzzle, think hard,
sleep on it, try again. Only after genuine sustained failure do they move on.

This script mirrors that approach:
  - Focus on ONE task at a time
  - Give it many attempts per session (day)
  - Sleep between sessions for consolidation
  - Track whether it's REASONING, not just pixel accuracy
  - Switch tasks only on: perfect solve, or stuck for N sessions

Reasoning indicators (beyond pixel accuracy):
  - Perfect solve (binary -- the only real metric)
  - Correct color palette (does it know which colors to use?)
  - Foreground density (right amount of non-zero cells?)
  - Output diversity (is it exploring hypotheses or stuck?)
  - Improvement trajectory (getting warmer across sessions?)
"""
import sys, time, os, argparse
from datetime import datetime
sys.path.insert(0, '.')
import numpy as np
import arckit

from src.genome import Genome
from src.template import create_dng
from src.childhood import extract_tasks
from src.curriculum import sort_by_difficulty
from src.pipeline import (
    LifecycleConfig, observe_examples, sleep, rest,
    _focus_mask, _soft_reset,
)
from src.dynamics import think
from src.encoding import NUM_COLORS, grid_to_signal, signal_to_grid
from src.graph import DNG, Region
from src.episodic_memory import EpisodicMemory
from src.plasticity import synaptogenesis

# ── Args ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--max-grid', type=int, default=10,
                    help='Max grid dimension')
parser.add_argument('--max-tasks', type=int, default=30,
                    help='Max tasks to attempt')
parser.add_argument('--max-sessions', type=int, default=5,
                    help='Max sessions (days) per task before forced switch')
parser.add_argument('--attempts', type=int, default=15,
                    help='Thinking attempts per session')
parser.add_argument('--stuck-patience', type=int, default=3,
                    help='Sessions without improvement before switching task')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint .npz to resume from')
args = parser.parse_args()

MAX_H = MAX_W = args.max_grid

# ── Run folder ────────────────────────────────────────────────────────

run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join("models", f"focus_{run_id}")
os.makedirs(run_dir, exist_ok=True)
print(f"Run folder: {run_dir}", flush=True)

# ── Load tasks ───────────────────────────────────────────────────────

train_set, eval_set = arckit.load_data()

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

valid_tasks = [t for t in train_set
               if all(d <= MAX_H for d in max_grid_dim(t))]
print(f"ARC-AGI-1: {len(valid_tasks)}/{len(train_set)} tasks fit in "
      f"{MAX_H}x{MAX_W}", flush=True)

rng = np.random.default_rng(42)
all_tasks = extract_tasks(valid_tasks)
all_tasks = sort_by_difficulty(all_tasks)

n_tasks = min(args.max_tasks, len(all_tasks))
all_tasks = all_tasks[:n_tasks]
print(f"Will attempt {n_tasks} tasks (sorted by difficulty)", flush=True)

# ── Create or resume network ────────────────────────────────────────

if args.resume:
    print(f"\nResuming from {args.resume}...", flush=True)
    net = DNG.load(args.resume)
    print(f"  {net.n_nodes} nodes, {net.edge_count()} edges", flush=True)
else:
    genome = Genome(
        max_h=MAX_H, max_w=MAX_W,
        n_internal=400, n_concept=80, n_memory=120,
        max_fan_in=100,
        weight_scale=0.02, frac_inhibitory=0.25,
        f_rate=0.05, f_decay=0.02, f_max=3.0,
    )

    print("\nCreating network...", flush=True)
    t0_build = time.time()
    net = create_dng(genome, MAX_H, MAX_W, rng)
    net.wta_k_frac = 0.05
    net.threshold[:] = 0.15
    motor_mask = net.regions == list(Region).index(Region.MOTOR)
    net.threshold[motor_mask] = 0.05
    net.adapt_rate = 0.01
    print(f"  {net.n_nodes} nodes, {net.edge_count()} edges "
          f"({len(net.memory_nodes)} memory) "
          f"({time.time()-t0_build:.1f}s)", flush=True)

# ── Config ───────────────────────────────────────────────────────────

config = LifecycleConfig(
    observe_steps=50,
    think_steps=80,
    consolidation_steps=20,
    eta=0.0,
    w_max=2.5,
    noise_std=0.02,
    focus_strength=0.5,
    rest_steps=20,
    rest_noise_std=0.03,
    sleep_downscale=1.0,
    sleep_tag_threshold=0.001,
    prune_weak_threshold=0.0,
    prune_cycles_required=999,
    replay_eta=0.0001,
    replay_passes=3,
    replay_steps=30,
    consolidation_decay=0.99,
    consolidation_strength=0.0,
    consolidation_threshold=0.9,
    memory_hint_strength=1.5,
    spontaneous_strength=0.3,
    growth_rate=0.2,
    growth_candidates=30000,
)

episodic = EpisodicMemory(max_h=MAX_H, max_w=MAX_W, capacity=200)


# ═══════════════════════════════════════════════════════════════════════
# REASONING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_output(guess, expected, input_grid):
    """Structural analysis: does the output show evidence of reasoning?"""
    m = {}

    m['perfect'] = bool(np.array_equal(guess, expected))

    exp_colors = set(np.unique(expected).tolist())
    guess_colors = set(np.unique(guess).tolist())
    m['palette_correct'] = (exp_colors == guess_colors)
    m['palette_precision'] = len(exp_colors & guess_colors) / max(1, len(guess_colors))
    m['palette_recall'] = len(exp_colors & guess_colors) / max(1, len(exp_colors))

    exp_fg = int(np.sum(expected != 0))
    guess_fg = int(np.sum(guess != 0))
    m['fg_expected'] = exp_fg
    m['fg_actual'] = guess_fg
    m['fg_ratio'] = guess_fg / max(1, exp_fg)

    gh, gw = guess.shape
    ih, iw = input_grid.shape
    trim_h, trim_w = min(gh, ih), min(gw, iw)
    trimmed_input = np.zeros_like(guess)
    trimmed_input[:trim_h, :trim_w] = input_grid[:trim_h, :trim_w]
    m['is_copy'] = bool(np.array_equal(guess, trimmed_input))
    m['is_blank'] = bool(np.all(guess == 0))
    m['is_uniform'] = (len(np.unique(guess)) <= 1)

    exp_hist = np.bincount(expected.ravel(), minlength=NUM_COLORS).astype(float)
    guess_hist = np.bincount(guess.ravel(), minlength=NUM_COLORS).astype(float)
    exp_norm = exp_hist / max(1, exp_hist.sum())
    guess_norm = guess_hist / max(1, guess_hist.sum())
    m['color_dist_error'] = float(np.sum(np.abs(exp_norm - guess_norm)))

    fg = expected != 0
    if fg.any():
        m['fg_acc'] = float(np.mean(guess[fg] == expected[fg]))
    else:
        m['fg_acc'] = 1.0 if np.all(guess == 0) else 0.0

    return m


def summarize_reasoning(metrics_list):
    """Summarize reasoning indicators across a session's attempts."""
    if not metrics_list:
        return {}
    n = len(metrics_list)
    return {
        'any_perfect': any(m['perfect'] for m in metrics_list),
        'palette_correct_rate': sum(m['palette_correct'] for m in metrics_list) / n,
        'mean_fg_ratio': np.mean([m['fg_ratio'] for m in metrics_list]),
        'mean_color_error': np.mean([m['color_dist_error'] for m in metrics_list]),
        'copy_rate': sum(m['is_copy'] for m in metrics_list) / n,
        'blank_rate': sum(m['is_blank'] for m in metrics_list) / n,
        'best_fg_acc': max(m['fg_acc'] for m in metrics_list),
    }


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════

def fmt_grid_rows(grid):
    """Format a grid as list of row strings."""
    rows = []
    for row in grid:
        rows.append(" ".join(f"{v}" for v in row))
    return rows


def show_grids(grids_and_labels, indent=4):
    """Show multiple grids side by side."""
    if not grids_and_labels:
        return
    prefix = " " * indent

    all_rows = []
    all_widths = []
    labels = []
    max_nrows = 0

    for grid, label in grids_and_labels:
        rows = fmt_grid_rows(grid)
        w = max((len(r) for r in rows), default=0)
        all_rows.append(rows)
        all_widths.append(w)
        labels.append(label)
        max_nrows = max(max_nrows, len(rows))

    for r_list in all_rows:
        while len(r_list) < max_nrows:
            r_list.append("")

    gap = "  |  "
    label_line = gap.join(f"{l:<{w}}" for l, w in zip(labels, all_widths))
    print(f"{prefix}{label_line}")
    sep_line = gap.join("-" * w for w in all_widths)
    print(f"{prefix}{sep_line}")
    for i in range(max_nrows):
        parts = [f"{all_rows[j][i]:<{all_widths[j]}}" for j in range(len(all_rows))]
        print(f"{prefix}{gap.join(parts)}")
    print()


def show_task_examples(train_pairs):
    """Display a task's training examples."""
    for i, (inp, out) in enumerate(train_pairs):
        print(f"  Example {i+1}: {inp.shape} -> {out.shape}")
        show_grids([(inp, "Input"), (out, "Output")])


# ═══════════════════════════════════════════════════════════════════════
# TEMPORARY HEBBIAN BINDING
# ═══════════════════════════════════════════════════════════════════════

def apply_hebbian_binding(net, eta=0.1):
    """
    Strengthen connections between co-active neurons (one-shot).

    This is "short-term potentiation" -- when the network has just
    processed a combined input+output signal, the pathways that carried
    that signal are temporarily strengthened. This creates routes
    from input representations to output representations.

    Dale's law maintained: excitatory edges get stronger,
    inhibitory edges get more inhibitory.
    """
    n = net._edge_count
    pre = net.r[net._edge_src[:n]]
    post = net.r[net._edge_dst[:n]]

    active = (pre > 0.02) & (post > 0.02)
    n_active = int(active.sum())
    if n_active == 0:
        return 0

    dw = np.zeros(n)
    dw[active] = eta * pre[active] * post[active]
    dw *= np.sign(net._edge_w[:n])

    net._edge_w[:n] += dw
    np.clip(net._edge_w[:n], -2.5, 2.5, out=net._edge_w[:n])

    return n_active


def observe_with_binding(net, train_pairs, cfg, ep_mem, binding_eta):
    """
    Observe training examples AND apply Hebbian binding after each.

    After the network settles with input+output, the active pathways
    are strengthened. This creates temporary associations:
    "when I see input A, the output should look like B."
    """
    mh, mw = net.max_h, net.max_w
    n_total = net.n_nodes
    motor_offset = int(net.output_nodes[0])
    total_bound = 0

    for inp, out in train_pairs:
        inp, out = np.asarray(inp), np.asarray(out)
        in_h, in_w = inp.shape
        out_h, out_w = out.shape

        if ep_mem is not None:
            ep_mem.store(inp, out)

        focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                            cfg.focus_strength)
        sig = (grid_to_signal(inp, 0, n_total, max_h=mh, max_w=mw) +
               grid_to_signal(out, motor_offset, n_total, max_h=mh, max_w=mw) +
               focus)

        think(net, signal=sig, steps=cfg.observe_steps, noise_std=cfg.noise_std)
        total_bound += apply_hebbian_binding(net, eta=binding_eta)

    think(net, signal=None, steps=cfg.observe_steps // 2,
          noise_std=cfg.noise_std)

    return total_bound


def echo_test(net, test_input, test_output, cfg, binding_eta=0.0):
    """
    Show the correct answer, then test if the network can reproduce it.

    1. Full reset of motor neurons (clean slate)
    2. Clamp input + output for 40 steps (teacher shows the answer)
    3. Optionally apply Hebbian binding
    4. Remove output clamp, keep input for 30 steps (can you reproduce it?)
    5. Read motor output and compare

    Returns (guess_grid, metrics_dict)
    """
    test_input = np.asarray(test_input)
    test_output = np.asarray(test_output)
    mh, mw = net.max_h, net.max_w
    n_total = net.n_nodes
    motor_offset = int(net.output_nodes[0])
    in_h, in_w = test_input.shape
    out_h, out_w = test_output.shape

    # Full motor reset so residual self-recurrence doesn't fight the new signal
    net.V[net.output_nodes] = 0.0
    net.r[net.output_nodes] = 0.0
    net.prev_r[net.output_nodes] = 0.0

    focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                        cfg.focus_strength)

    full_signal = (
        grid_to_signal(test_input, 0, n_total, max_h=mh, max_w=mw) +
        grid_to_signal(test_output, motor_offset, n_total, max_h=mh, max_w=mw) +
        focus
    )
    SHOW_STEPS = 40
    think(net, signal=full_signal, steps=SHOW_STEPS,
          noise_std=cfg.noise_std * 0.3)

    if binding_eta > 0:
        apply_hebbian_binding(net, eta=binding_eta)

    input_only = (
        grid_to_signal(test_input, 0, n_total, max_h=mh, max_w=mw) + focus
    )
    RECALL_STEPS = 30
    think(net, signal=input_only, steps=RECALL_STEPS,
          noise_std=cfg.noise_std * 0.3)

    guess = signal_to_grid(net.V, out_h, out_w,
                           node_offset=motor_offset, max_h=mh, max_w=mw)
    m = analyze_output(guess, test_output, test_input)
    return guess, m


# ═══════════════════════════════════════════════════════════════════════
# FOCUS SESSION
# ═══════════════════════════════════════════════════════════════════════

def focus_session(
    net, train_pairs, test_input, test_output,
    cfg, ep_mem, session_num, prev_best,
    n_attempts=15, binding_eta=0.0,
):
    """
    One session (day) of deep focus on a single task.

    If binding_eta > 0: applies temporary Hebbian strengthening during
    observation. Weights are snapshot'd before and restored after, so
    the binding is purely task-local (like short-term potentiation).

    Returns (best_reward, best_guess, metrics_list, all_guesses)
    """
    test_input = np.asarray(test_input)
    test_output = np.asarray(test_output)
    out_h, out_w = test_output.shape
    mh, mw = net.max_h, net.max_w
    n_total = net.n_nodes
    motor_offset = int(net.output_nodes[0])

    # Snapshot weights if using temporary binding
    w_snapshot = None
    if binding_eta > 0:
        w_snapshot = net._edge_w[:net._edge_count].copy()

    if session_num == 1:
        _soft_reset(net)

    if binding_eta > 0:
        n_bound = observe_with_binding(net, train_pairs, cfg, ep_mem, binding_eta)
        print(f"    [binding: {n_bound} edges strengthened]", flush=True)
    else:
        observe_examples(net, train_pairs, cfg, episodic=ep_mem)

    recall_hint = np.zeros(n_total)
    if ep_mem is not None:
        recall_hint = ep_mem.recall_signal(
            test_input, motor_offset, n_total,
            strength=cfg.memory_hint_strength,
        )

    best_reward = prev_best
    best_guess = None
    metrics_list = []
    all_guesses = []

    for attempt in range(1, n_attempts + 1):
        in_h, in_w = test_input.shape
        focus = _focus_mask(net, max(in_h, out_h), max(in_w, out_w),
                            cfg.focus_strength)
        test_signal = (grid_to_signal(test_input, 0, n_total,
                                      max_h=mh, max_w=mw) + focus + recall_hint)

        think(net, signal=test_signal, steps=cfg.think_steps,
              noise_std=cfg.noise_std)

        guess = signal_to_grid(net.V, out_h, out_w,
                               node_offset=motor_offset, max_h=mh, max_w=mw)

        m = analyze_output(guess, test_output, test_input)
        metrics_list.append(m)
        all_guesses.append(guess.copy())

        fg = test_output != 0
        reward = float(np.mean(guess[fg] == test_output[fg])) if fg.any() else 1.0

        improved = reward > best_reward
        if improved:
            best_reward = reward
            best_guess = guess.copy()

        flags = []
        if m['is_copy']:    flags.append("COPY")
        if m['is_blank']:   flags.append("BLANK")
        if m['is_uniform']: flags.append("FLAT")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        arrow = " ^" if improved and attempt > 1 else ""
        star = " ** SOLVED **" if m['perfect'] else ""

        palette_str = "OK" if m['palette_correct'] else "WRONG"
        print(f"    #{attempt:2d}  palette={palette_str:<5s} "
              f"fg={m['fg_actual']:3d}/{m['fg_expected']:<3d} "
              f"color_err={m['color_dist_error']:.2f} "
              f"fg_acc={m['fg_acc']:.0%}"
              f"{flag_str}{arrow}{star}", flush=True)

        if m['perfect']:
            break

        if attempt < n_attempts:
            if binding_eta > 0:
                observe_with_binding(net, train_pairs, cfg, ep_mem, binding_eta)
            else:
                observe_examples(net, train_pairs, cfg, episodic=ep_mem)

    n_unique = len(set(g.tobytes() for g in all_guesses))
    diversity = "diverse" if n_unique > max(1, len(all_guesses) // 3) else "repetitive"
    print(f"    -> {n_unique} unique outputs in {len(all_guesses)} attempts ({diversity})",
          flush=True)

    if n_unique == 1 and not metrics_list[0]['perfect']:
        print(f"    Stuck output:")
        for row in all_guesses[0]:
            print(f"      {' '.join(str(v) for v in row)}")

    # Restore weights if we used temporary binding
    if w_snapshot is not None:
        net._edge_w[:net._edge_count] = w_snapshot

    return best_reward, best_guess, metrics_list, all_guesses


# ═══════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"DEEP FOCUS TRAINING")
print(f"  {n_tasks} tasks, up to {args.max_sessions} sessions/task, "
      f"{args.attempts} attempts/session")
print(f"  Switch on: perfect solve, or stuck for {args.stuck_patience} sessions")
print(f"{'='*70}\n")

t0 = time.time()
ema_r = None

solved_tasks = []
shelved_tasks = []
task_histories = {}

for task_idx in range(n_tasks):
    train_pairs, test_in, test_out = all_tasks[task_idx]

    print(f"\n{'='*70}")
    print(f"TASK {task_idx+1}/{n_tasks}  "
          f"input={test_in.shape[0]}x{test_in.shape[1]} "
          f"output={test_out.shape[0]}x{test_out.shape[1]}  "
          f"examples={len(train_pairs)}")
    print(f"{'='*70}")

    show_task_examples(train_pairs)

    best_reward = 0.0
    stuck_count = 0
    task_history = []

    for session in range(1, args.max_sessions + 1):
        print(f"\n  --- Session {session}/{args.max_sessions} "
              f"(stuck={stuck_count}/{args.stuck_patience}) ---")

        session_best, session_guess, session_metrics, session_guesses = focus_session(
            net, train_pairs, test_in, test_out,
            config, episodic, session,
            prev_best=best_reward,
            n_attempts=args.attempts,
        )

        summary = summarize_reasoning(session_metrics)
        task_history.append({
            'session': session,
            'best_reward': session_best,
            'summary': summary,
            'n_unique_outputs': len(set(g.tobytes() for g in session_guesses)),
        })

        # Exact match is the ONLY valid solve condition
        session_perfect = any(m['perfect'] for m in session_metrics)

        improved_this_session = session_best > best_reward
        if improved_this_session:
            best_reward = session_best
            stuck_count = 0
        else:
            stuck_count += 1

        if session_guess is not None:
            print()
            show_grids([
                (test_in, f"Input"),
                (test_out, f"Expected"),
                (session_guess, f"Best (s{session})"),
            ])

        pal_rate = summary.get('palette_correct_rate', 0)
        fg_ratio = summary.get('mean_fg_ratio', 0)
        color_err = summary.get('mean_color_error', 0)
        copy_rate = summary.get('copy_rate', 0)
        blank_rate = summary.get('blank_rate', 0)
        n_unique = len(set(g.tobytes() for g in session_guesses))

        exact_str = "EXACT MATCH" if session_perfect else f"best_fg_acc={best_reward:.0%}"
        print(f"  Session {session} summary:  {exact_str}  "
              f"palette_ok={pal_rate:.0%}  fg_ratio={fg_ratio:.1f}  "
              f"color_err={color_err:.2f}  unique={n_unique}/{args.attempts}")

        if session_perfect:
            print(f"\n  ** SOLVED in {session} session(s)! **")
            solved_tasks.append(task_idx)
            break

        if stuck_count >= args.stuck_patience:
            # ── DIAGNOSTIC: Your idea -- show the answer, can it reproduce? ──
            print(f"\n  --- DIAGNOSTIC: Showing the answer ---")
            _soft_reset(net)

            # Test 1: Plain echo (no binding)
            echo_g, echo_m = echo_test(net, test_in, test_out, config,
                                       binding_eta=0.0)
            print(f"  Echo (no binding):   perfect={echo_m['perfect']}  "
                  f"fg_acc={echo_m['fg_acc']:.0%}  "
                  f"palette={'OK' if echo_m['palette_correct'] else 'WRONG'}")

            # Test 2: Echo with Hebbian binding
            _soft_reset(net)
            w_snap = net._edge_w[:net._edge_count].copy()
            echo_g2, echo_m2 = echo_test(net, test_in, test_out, config,
                                         binding_eta=0.1)
            net._edge_w[:net._edge_count] = w_snap
            print(f"  Echo (with binding): perfect={echo_m2['perfect']}  "
                  f"fg_acc={echo_m2['fg_acc']:.0%}  "
                  f"palette={'OK' if echo_m2['palette_correct'] else 'WRONG'}")

            # Test 3: Full solve with binding on training examples
            _soft_reset(net)
            w_snap2 = net._edge_w[:net._edge_count].copy()
            print(f"  Solve with binding:", flush=True)
            bind_reward, bind_guess, bind_metrics, _ = focus_session(
                net, train_pairs, test_in, test_out,
                config, episodic, 1, prev_best=0.0,
                n_attempts=5, binding_eta=0.1,
            )
            net._edge_w[:net._edge_count] = w_snap2
            bind_perfect = any(m['perfect'] for m in bind_metrics)
            print(f"  Solve with binding: perfect={bind_perfect}  "
                  f"best_fg_acc={bind_reward:.0%}")

            if bind_guess is not None:
                print()
                show_grids([
                    (test_in, "Input"),
                    (test_out, "Expected"),
                    (echo_g, "Echo(plain)"),
                    (echo_g2, "Echo(bound)"),
                    (bind_guess, "Solve(bound)"),
                ])

            shelved_tasks.append(task_idx)
            break

        # Sleep between sessions
        print(f"  [sleeping...]", end="", flush=True)
        saved_r = net.r.copy()
        net.r[:] = np.maximum(net.r, 0.01)
        n_grown = synaptogenesis(
            net, growth_rate=config.growth_rate,
            n_candidates=config.growth_candidates, rng=rng,
        )
        net.r[:] = saved_r
        rest(net, config)
        n_pruned, ema_r = sleep(net, config, ema_r, rng, episodic=episodic)
        print(f" woke up (+{n_grown}/-{n_pruned} edges, "
              f"now {net.edge_count()}, "
              f"episodic={len(episodic.episodes)})", flush=True)

    else:
        if best_reward < 1.0:
            print(f"\n  Max sessions reached, shelving.")
            shelved_tasks.append(task_idx)

    task_histories[task_idx] = task_history

    # Show improvement trajectory for this task
    if len(task_history) > 1:
        traj = " -> ".join(f"{h['best_reward']:.0%}" for h in task_history)
        print(f"\n  Trajectory: {traj}")
        rewards = [h['best_reward'] for h in task_history]
        if rewards[-1] > rewards[0]:
            print(f"  ^ IMPROVING over sessions (evidence of learning)")
        elif rewards[-1] == rewards[0]:
            print(f"  = No change across sessions")
        else:
            print(f"  v Degraded across sessions")

    # Checkpoint periodically
    if (task_idx + 1) % 5 == 0:
        path = os.path.join(run_dir, f"after_task_{task_idx+1}.npz")
        net.save(path)
        print(f"  [checkpoint: {path}]", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

elapsed = time.time() - t0

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE  ({elapsed:.0f}s / {elapsed/60:.1f}m)")
print(f"{'='*70}")
print(f"  Tasks attempted: {len(solved_tasks) + len(shelved_tasks)}")
print(f"  Solved:  {len(solved_tasks)}")
print(f"  Shelved: {len(shelved_tasks)}")
if solved_tasks:
    print(f"  Solved task indices: {solved_tasks}")

# Reasoning analysis
n_improving = 0
n_static = 0
n_degrading = 0
for idx, history in task_histories.items():
    if len(history) < 2:
        continue
    rewards = [h['best_reward'] for h in history]
    if rewards[-1] > rewards[0]:
        n_improving += 1
    elif rewards[-1] < rewards[0]:
        n_degrading += 1
    else:
        n_static += 1

print(f"\n  Learning signals:")
print(f"    Improving across sessions: {n_improving}")
print(f"    Static (no change):        {n_static}")
print(f"    Degrading:                 {n_degrading}")

# Output behavior analysis
total_copy = 0
total_blank = 0
total_attempts = 0
for idx, history in task_histories.items():
    for h in history:
        s = h.get('summary', {})
        total_copy += s.get('copy_rate', 0) * args.attempts
        total_blank += s.get('blank_rate', 0) * args.attempts
        total_attempts += args.attempts

if total_attempts > 0:
    print(f"\n  Output behavior:")
    print(f"    Copy-of-input rate: {total_copy/total_attempts:.0%}")
    print(f"    Blank output rate:  {total_blank/total_attempts:.0%}")
    print(f"    Producing something: {1 - total_copy/total_attempts - total_blank/total_attempts:.0%}")

# Save final model
final_path = os.path.join(run_dir, "final.npz")
net.save(final_path)
print(f"\n  Saved: {final_path}")

print(f"\n{'='*70}")
print(f"KEY QUESTION: Is it reasoning?")
print(f"{'='*70}")
if len(solved_tasks) > 0:
    print(f"  YES signals: Solved {len(solved_tasks)} task(s) perfectly.")
if n_improving > 0:
    print(f"  YES signals: {n_improving} task(s) showed improvement across sessions.")
if n_static > 0 and len(solved_tasks) == 0:
    print(f"  UNCLEAR: {n_static} task(s) showed no change -- stuck but not degrading.")
if n_degrading > 0:
    print(f"  NO signals: {n_degrading} task(s) got WORSE across sessions.")
if len(solved_tasks) == 0 and n_improving == 0:
    print(f"  VERDICT: No evidence of reasoning yet. Architecture needs work.")
