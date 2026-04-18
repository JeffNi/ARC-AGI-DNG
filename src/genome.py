"""
Genome: the complete genetic specification of an organism.

Every tunable parameter that defines how a brain is built and operates
lives here. Runtime code (Brain, template, Teacher) reads from the
Genome at construction time — it never owns behavioral constants.

Organized into sections so parameters are easy to find by function.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Genome:
    """All parameters needed to instantiate and run a DNG brain."""

    # ══════════════════════════════════════════════════════════════════
    # STRUCTURAL — neuron counts, layer fractions, connectivity
    # These define the physical topology of the network.
    # ══════════════════════════════════════════════════════════════════

    max_h: int = 30
    max_w: int = 30

    n_internal: int = 1500          # total internal processing neurons
    frac_inhibitory: float = 0.0    # KCs are excitatory; inhibition via APL
    frac_modulatory: float = 0.0
    frac_memory: float = 0.0

    frac_layer1: float = 0.0       # no L1 — direct sensory-to-KC
    frac_layer2: float = 0.0       # no L2 — direct sensory-to-KC
    frac_layer3: float = 1.0       # all internal = mushroom body KCs

    n_memory: int = 100             # MBON analog pool
    n_gate: int = 0                 # basal ganglia gate pool
    n_gaze: int = 8                 # oculomotor neurons
    n_concept: int = 100            # shared concept pool

    density_column_to_concept: float = 0.3
    density_concept_to_column: float = 0.3
    density_sensory_to_internal: float = 0.3
    density_internal_to_motor: float = 0.3
    density_internal_to_internal: float = 0.4
    density_motor_to_internal: float = 0.1
    density_sensory_to_motor: float = 0.05
    density_internal_to_sensory: float = 0.05
    density_sensory_neighbors: float = 0.1
    density_motor_neighbors: float = 0.1

    max_fan_in: int = 150

    # KC receptive field: how many sensory inputs each KC draws from its
    # assigned cell (local) vs random other cells (global).  Biological
    # KCs sample from a local calyx neighborhood; the global inputs add
    # sparse cross-cell context for relational patterns.
    kc_local_fan: int = 7       # inputs from assigned cell's features
    kc_global_fan: int = 3      # inputs from random other cells

    # ══════════════════════════════════════════════════════════════════
    # WIRING — initial weight scales per pathway (multiples of weight_scale)
    # These set the birth strength of each connection type.
    # ══════════════════════════════════════════════════════════════════

    weight_scale: float = 0.03              # base scale for all weights

    w_scale_sensory_to_l3: float = 8.0      # sensory → KC feedforward
    w_scale_l3_to_position: float = 2.0     # KC → spatial targeting
    w_scale_l3_to_done: float = 1.5         # KC → completion signal
    w_scale_l3_to_memory: float = 6.0       # KC → MBON (uniform init, plastic)
    w_scale_l3_to_commit: float = 1.5       # KC → go/no-go gate
    w_scale_memory_to_position: float = 1.5 # MBON → spatial output
    w_scale_memory_to_done: float = 1.0     # MBON → completion
    w_scale_memory_to_commit: float = 1.0   # MBON → go/no-go gate
    w_scale_sensory_to_gaze: float = 2.0    # sensory → gaze orienting
    spatial_instinct_weight: float = 0.2    # hardwired retinotopic map

    # ══════════════════════════════════════════════════════════════════
    # NEURON PROPERTIES — per-region leak, threshold, excitability
    # These define how individual neuron populations behave.
    # ══════════════════════════════════════════════════════════════════

    leak_l3: float = 0.05           # KC: slow decay for pattern persistence
    leak_memory: float = 0.005      # MBON: very slow (~200 step persistence)
    leak_motor: float = 0.3         # motor pools: fast for responsive output
    leak_commit: float = 0.15       # commit: slower leak = evidence accumulation

    threshold_memory: float = 0.02  # MBON: low so sparse KC input drives it
    threshold_commit: float = 0.02  # commit: low for sparse input sensitivity
    excitability_commit: float = 3.0  # commit: high gain, biased toward "go"

    max_rate_E: float = 1.0
    max_rate_I: float = 1.5
    adapt_rate_E: float = 0.01
    adapt_rate_I: float = 0.005

    # ══════════════════════════════════════════════════════════════════
    # DYNAMICS — membrane noise, WTA, facilitation
    # Global parameters that affect all neural dynamics.
    # ══════════════════════════════════════════════════════════════════

    noise_std: float = 0.02         # membrane noise standard deviation
    plasticity_interval: int = 5    # steps between competitive plasticity updates
    wta_k: int = 60                 # internal WTA pool size

    f_rate: float = 0.05            # facilitation accumulation rate
    f_decay: float = 0.02           # facilitation decay rate
    f_max: float = 3.0              # max facilitation multiplier

    # ══════════════════════════════════════════════════════════════════
    # MOTOR / ACTION SELECTION — echo, MBON readout, WTA noise
    # These control how the brain picks what color to commit.
    # ══════════════════════════════════════════════════════════════════

    echo_strength: float = 0.15     # (legacy, unused) innate identity echo
    copy_bias: float = 0.05         # instinctive "copy what you see" reflex — must be LOW enough that MBON depression can override it within ~5 retries
    mbon_strength: float = 3.0      # learned MBON readout gain into ACTION
    attention_gain: float = 8.0     # spatial attention: proportional KC boost (higher needed at our small scale)
    action_noise_coeff: float = 0.10  # rate-proportional WTA exploration noise
    action_explore_noise: float = 0.05  # absolute noise floor — bootstraps color selection among observed colors

    # ══════════════════════════════════════════════════════════════════
    # COMMIT GATE — basal ganglia go/no-go decision
    # These control when the brain decides to act.
    # ══════════════════════════════════════════════════════════════════

    commit_gain: float = 3.0        # COMMIT neuron rate → commit probability
    commit_noise: float = 0.03      # spontaneous go-signal noise (tonic DA)
    commit_lr: float = 0.005        # DA learning rate for COMMIT synapses
    min_act_rate: float = 0.05      # minimum action rate to allow a commit

    # ══════════════════════════════════════════════════════════════════
    # EXPLORATION / LOSE-SHIFT — basal ganglia indirect pathway
    # After a wrong commit, suppress that action to force alternatives.
    # ══════════════════════════════════════════════════════════════════

    suppress_steps: int = 30        # how long to suppress punished action
    suppress_strength: float = 0.5  # suppression intensity (0=none, 1=full)

    # ══════════════════════════════════════════════════════════════════
    # TIMING / MOTOR CONTROL — budgets, refractory, gaze
    # How long the brain gets to observe, act, and recover.
    # ══════════════════════════════════════════════════════════════════

    refractory_steps: int = 100     # position inhibition-of-return duration
    auto_submit_steps: int = 250    # inactivity timeout → auto-submit
    min_commits_before_done: int = 3  # DONE gate: min commits first
    done_noise: float = 0.03        # DONE neuron spontaneous noise
    gaze_reflex_steps: int = 3      # post-commit gaze bias toward canvas
    action_budget: int = 500        # max iterations per task attempt
    observe_steps: int = 40         # observation phase duration

    # ══════════════════════════════════════════════════════════════════
    # DA / REWARD SIGNALS — dopamine magnitudes for learning
    # How strongly correct/wrong outcomes modulate plasticity.
    # ══════════════════════════════════════════════════════════════════

    da_correct_commit: float = 0.3  # DA to winning position on correct commit
    da_wrong_commit: float = -0.5   # DA to position on wrong overwrite
    da_global_correct: float = 0.3  # global DA after correct commit
    da_global_wrong: float = -0.3   # global DA after wrong commit
    da_reward_slope: float = 1.1    # end-of-task: DA = offset + slope * accuracy
    da_reward_offset: float = -0.3  # end-of-task: baseline DA for zero accuracy
    da_reward_floor: float = 0.6    # end-of-task: minimum DA when fully correct
    da_novelty_cap: float = 0.4     # max novelty-driven DA spike

    da_baseline_obs: float = 0.2    # DA baseline during observation
    da_baseline_attempt: float = 0.0  # DA baseline during action
    da_baseline_rest: float = 0.05  # DA baseline at rest
    da_decay: float = 0.05          # DA exponential decay rate

    # ══════════════════════════════════════════════════════════════════
    # PLASTICITY / LEARNING — rates, eligibility, depression/potentiation
    # How fast and how much the brain learns from experience.
    # ══════════════════════════════════════════════════════════════════

    chl_eta: float = 0.03           # contrastive Hebbian learning rate
    elig_eta: float = 0.01          # eligibility-modulated update rate
    elig_decay: float = 0.95        # eligibility trace decay per step
    eta_local: float = 0.005        # DA-independent Hebbian (NMDA-like)
    eta_w: float = 0.05             # general weight update rate
    lambda_w: float = 0.001         # weight regularization
    w_max: float = 5.0              # absolute weight cap

    depression_eta: float = 0.5     # L3→MEMORY depression on wrong commits
    depression_floor: float = 0.01  # minimum L3→MEMORY weight
    recovery_rate: float = 0.002    # per-sleep drift toward initial weight

    trace_decay: float = 0.97       # temporal trace EMA (~30-step window)
    trace_contrast_eta: float = 0.3 # heterosynaptic LTD strength

    plasticity_rounds: int = 3
    k_reward: float = 2.0

    # ══════════════════════════════════════════════════════════════════
    # APL — mushroom body global inhibition (sparse coding)
    # Controls how sparse the KC representation is.
    # ══════════════════════════════════════════════════════════════════

    apl_target_sparseness: float = 0.05  # ~75 of 1500 KCs active
    apl_gain: float = 2.0               # APL feedback inhibition strength

    # ══════════════════════════════════════════════════════════════════
    # SLEEP / HOMEOSTASIS — offline consolidation and maintenance
    # What happens during sleep to consolidate and prune.
    # ══════════════════════════════════════════════════════════════════

    sleep_chl_scale: float = 0.3    # CHL eta multiplier during NREM replay
    sleep_shy_downscale: float = 0.97  # SHY synaptic downscaling factor
    sleep_factor: float = 0.95      # general sleep consolidation
    prune_epsilon: float = 0.005    # pruning threshold

    homeostasis_interval: int = 50
    fatigue_rate: float = 0.05
    fatigue_threshold: float = 10.0
    fatigue_sleep_reset: float = 0.1
    peak_growth_target: float = 2.5  # synaptogenesis density ceiling

    # ══════════════════════════════════════════════════════════════════
    # LEGACY — parameters kept for backward compatibility
    # ══════════════════════════════════════════════════════════════════

    think_steps: int = 50
    learn_steps: int = 30
    rest_steps: int = 30
    rest_noise_std: float = 0.05

    # ══════════════════════════════════════════════════════════════════
    # MUTATION
    # ══════════════════════════════════════════════════════════════════

    def mutate(self, rng: np.random.Generator, strength: float = 0.1) -> "Genome":
        """Create a mutated copy — perturbs ALL structural + dynamics params."""
        g = Genome(**{k: v for k, v in self.__dict__.items()})

        def _perturb_int(val, lo, hi):
            delta = max(1, int(abs(val) * strength))
            return int(np.clip(val + rng.integers(-delta, delta + 1), lo, hi))

        def _perturb_float(val, lo, hi):
            return float(np.clip(val + rng.normal(0, abs(val) * strength + 0.001), lo, hi))

        g.n_internal = _perturb_int(g.n_internal, 50, 5000)
        g.n_concept = _perturb_int(g.n_concept, 10, 500)
        g.n_memory = _perturb_int(g.n_memory, 10, 500)
        g.max_fan_in = _perturb_int(g.max_fan_in, 20, 500)
        g.frac_inhibitory = _perturb_float(g.frac_inhibitory, 0.0, 0.5)
        g.frac_modulatory = _perturb_float(g.frac_modulatory, 0.0, 0.3)
        g.frac_memory = _perturb_float(g.frac_memory, 0.0, 0.5)

        g.frac_layer1 = _perturb_float(g.frac_layer1, 0.2, 0.8)
        g.frac_layer2 = _perturb_float(g.frac_layer2, 0.1, 0.5)
        g.frac_layer3 = _perturb_float(g.frac_layer3, 0.05, 0.4)
        layer_sum = g.frac_layer1 + g.frac_layer2 + g.frac_layer3
        g.frac_layer1 /= layer_sum
        g.frac_layer2 /= layer_sum
        g.frac_layer3 /= layer_sum

        for attr in ['density_sensory_to_internal', 'density_internal_to_motor',
                      'density_internal_to_internal', 'density_motor_to_internal',
                      'density_sensory_to_motor', 'density_internal_to_sensory',
                      'density_sensory_neighbors', 'density_motor_neighbors']:
            setattr(g, attr, _perturb_float(getattr(g, attr), 0.0, 1.0))

        g.weight_scale = _perturb_float(g.weight_scale, 0.0001, 1.0)
        g.f_rate = _perturb_float(g.f_rate, 0.001, 0.5)
        g.f_decay = _perturb_float(g.f_decay, 0.001, 0.2)
        g.f_max = _perturb_float(g.f_max, 0.5, 10.0)
        g.eta_w = _perturb_float(g.eta_w, 0.001, 1.0)
        g.w_max = _perturb_float(g.w_max, 1.0, 20.0)
        g.sleep_factor = _perturb_float(g.sleep_factor, 0.5, 0.999)
        g.prune_epsilon = _perturb_float(g.prune_epsilon, 0.0001, 0.1)
        g.think_steps = _perturb_int(g.think_steps, 5, 200)
        g.rest_steps = _perturb_int(g.rest_steps, 5, 200)
        g.rest_noise_std = _perturb_float(g.rest_noise_std, 0.001, 0.5)

        return g

    def mutate_learning(self, rng: np.random.Generator, strength: float = 0.15) -> "Genome":
        """Focused mutation: only the parameters that control learning dynamics.

        Used for evolutionary search over the action-selection / commit /
        exploration / reward parameter space while holding structure fixed.
        """
        g = Genome(**{k: v for k, v in self.__dict__.items()})

        def _pf(val, lo, hi):
            return float(np.clip(val + rng.normal(0, abs(val) * strength + 0.001), lo, hi))

        def _pi(val, lo, hi):
            delta = max(1, int(abs(val) * strength))
            return int(np.clip(val + rng.integers(-delta, delta + 1), lo, hi))

        # Motor / action selection
        g.copy_bias = _pf(g.copy_bias, 0.0, 0.12)
        g.action_explore_noise = _pf(g.action_explore_noise, 0.0, 0.15)
        g.action_noise_coeff = _pf(g.action_noise_coeff, 0.01, 0.5)
        g.mbon_strength = _pf(g.mbon_strength, 0.5, 10.0)
        g.attention_gain = _pf(g.attention_gain, 2.0, 20.0)

        # Commit gate
        g.commit_gain = _pf(g.commit_gain, 0.5, 10.0)
        g.commit_noise = _pf(g.commit_noise, 0.005, 0.15)
        g.commit_lr = _pf(g.commit_lr, 0.0005, 0.05)
        g.min_act_rate = _pf(g.min_act_rate, 0.01, 0.2)

        # Exploration
        g.suppress_strength = _pf(g.suppress_strength, 0.1, 0.9)

        # Learning rates
        g.depression_eta = _pf(g.depression_eta, 0.05, 2.0)
        g.depression_floor = _pf(g.depression_floor, 0.001, 0.1)
        g.recovery_rate = _pf(g.recovery_rate, 0.0005, 0.02)

        # Wiring scales — how strongly signals propagate
        g.w_scale_sensory_to_l3 = _pf(g.w_scale_sensory_to_l3, 2.0, 20.0)
        g.w_scale_l3_to_memory = _pf(g.w_scale_l3_to_memory, 1.0, 15.0)

        # KC receptive field — controls position specificity
        g.kc_local_fan = _pi(g.kc_local_fan, 3, 10)
        g.kc_global_fan = _pi(g.kc_global_fan, 0, 7)

        # KC sparseness — controls position discrimination
        g.apl_target_sparseness = _pf(g.apl_target_sparseness, 0.02, 0.15)
        g.apl_gain = _pf(g.apl_gain, 0.5, 5.0)

        # Neuron dynamics
        g.leak_l3 = _pf(g.leak_l3, 0.01, 0.2)
        g.leak_memory = _pf(g.leak_memory, 0.001, 0.05)
        g.noise_std = _pf(g.noise_std, 0.005, 0.1)

        # Timing
        g.observe_steps = _pi(g.observe_steps, 10, 100)

        # DA magnitudes
        g.da_correct_commit = _pf(g.da_correct_commit, 0.05, 1.0)
        g.da_wrong_commit = _pf(g.da_wrong_commit, -1.5, -0.1)
        g.da_global_wrong = _pf(g.da_global_wrong, -1.0, -0.05)
        g.da_global_correct = _pf(g.da_global_correct, 0.05, 1.0)

        return g
