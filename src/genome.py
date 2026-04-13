"""
Genome: the parameters that define a network's architecture.

See docs/02_Architecture.md and docs/04_ARC_Strategy.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Genome:
    """All parameters needed to instantiate a DNG from a template."""

    # ── Grid dimensions (max supported size) ─────────────────────────
    max_h: int = 30
    max_w: int = 30

    # ── Internal processing region ────────────────────────────────────
    n_internal: int = 2000
    frac_inhibitory: float = 0.2
    frac_modulatory: float = 0.05
    frac_memory: float = 0.1

    # ── Cortical layer fractions (of n_internal) ──────────────────────
    # Bee-inspired: L3 is the mushroom body (expansion layer for pattern
    # separation). L1 kept at 900 (proven stable). L2 = projection neurons.
    frac_layer1: float = 0.45   # Optic lobe — sensory feature extraction
    frac_layer2: float = 0.15   # Projection neurons — intermediate features
    frac_layer3: float = 0.40   # Mushroom body / Kenyon cells — pattern separation

    # ── Long-term memory pool ─────────────────────────────────────────
    n_memory: int = 100

    # ── Basal ganglia gate pool ────────────────────────────────────────
    n_gate: int = 50

    # ── Gaze (oculomotor) neurons ────────────────────────────────────
    n_gaze: int = 8

    # ── Shared concept pool ───────────────────────────────────────────
    n_concept: int = 100
    density_column_to_concept: float = 0.3
    density_concept_to_column: float = 0.3

    # ── Connection densities ──────────────────────────────────────────
    density_sensory_to_internal: float = 0.3
    density_internal_to_motor: float = 0.3
    density_internal_to_internal: float = 0.4
    density_motor_to_internal: float = 0.1
    density_sensory_to_motor: float = 0.05
    density_internal_to_sensory: float = 0.05
    density_sensory_neighbors: float = 0.1
    density_motor_neighbors: float = 0.1

    # ── Fan-in cap ────────────────────────────────────────────────────
    max_fan_in: int = 150

    # ── Weight initialization ─────────────────────────────────────────
    weight_scale: float = 0.03

    # ── Facilitation parameters ───────────────────────────────────────
    f_rate: float = 0.05
    f_decay: float = 0.02
    f_max: float = 3.0

    # ── Growth / lifecycle parameters ─────────────────────────────────
    eta_w: float = 0.05
    lambda_w: float = 0.001
    w_max: float = 5.0
    plasticity_rounds: int = 3
    k_reward: float = 2.0
    sleep_factor: float = 0.95
    prune_epsilon: float = 0.005
    think_steps: int = 50
    learn_steps: int = 30
    rest_steps: int = 30
    rest_noise_std: float = 0.05

    # ── Dynamics / WTA ───────────────────────────────────────────────
    noise_std: float = 0.01
    plasticity_interval: int = 5
    wta_k: int = 60

    # ── Homeostasis ───────────────────────────────────────────────────
    homeostasis_interval: int = 50

    # ── CHL / eligibility / DA parameters (Phase 1) ───────────────────
    chl_eta: float = 0.03
    elig_eta: float = 0.2
    elig_decay: float = 0.85
    eta_local: float = 0.005      # DA-independent Hebbian rate (NMDA-like, always on)

    # ── Temporal trace for L1->L2 invariance learning ────────────────
    trace_decay: float = 0.97     # slow EMA of post-synaptic rates (~30-step window)
    trace_contrast_eta: float = 0.3  # heterosynaptic LTD strength (fraction of positive signal)

    # ── APL (mushroom body global inhibition) ──────────────────────────
    apl_target_sparseness: float = 0.10  # fraction of L3/KC neurons that should be active
    apl_gain: float = 2.0               # strength of APL feedback inhibition

    # ── DA parameters ──────────────────────────────────────────────────
    da_baseline_obs: float = 0.2
    da_baseline_attempt: float = 0.0
    da_baseline_rest: float = 0.05
    da_decay: float = 0.05

    # ── Synaptogenesis regulation ──────────────────────────────────────
    peak_growth_target: float = 2.5          # density ceiling: synaptogenesis -> 0 at this multiple of birth edges

    # ── Fatigue / sleep ────────────────────────────────────────────────
    fatigue_rate: float = 0.05
    fatigue_threshold: float = 10.0
    fatigue_sleep_reset: float = 0.1

    # ── Per-node type defaults ─────────────────────────────────────────
    max_rate_E: float = 1.0
    max_rate_I: float = 1.5
    adapt_rate_E: float = 0.01
    adapt_rate_I: float = 0.005

    def mutate(self, rng: np.random.Generator, strength: float = 0.1) -> "Genome":
        """Create a mutated copy of this genome."""
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

        # Perturb layer fractions then renormalize to sum to 1
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
