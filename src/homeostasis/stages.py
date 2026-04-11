"""
Developmental stages — setpoint configurations and smooth transitions.

Homeostasis is the same code at every stage; only the parameters change.
StageManager handles scheduling transitions and producing the current
blended HomeostasisSetpoints.

Transitions use exponential approach (1 - e^{-t/tau}) rather than linear
ramps, matching the biological time-constant model of developmental
parameter change (Huttenlocher & Dabholkar 1997; Petanjek et al 2011).
"""

from __future__ import annotations

import math

from .setpoints import HomeostasisSetpoints


STAGES: dict[str, HomeostasisSetpoints] = {
    "infancy": HomeostasisSetpoints(
        # During infancy, stability comes from L2 weight normalization,
        # conscience mechanism, SHY downscaling, and health-based pruning.
        # Scaling and intrinsic plasticity are DISABLED — they fight
        # competitive Hebbian differentiation by equalizing firing rates.
        # These are adult maintenance mechanisms, not developmental ones.
        target_rate=0.25,
        scaling_gain=0.0,                # DISABLED during infancy — fights competitive learning
        intrinsic_eta=0.0,               # DISABLED during infancy — resurrects undifferentiated neurons
        ei_target_ratio=0.6,
        ema_tau=0.02,                    # moderate EMA: responsive but not twitchy
        # Developmental — genetic clock
        synaptogenesis_rate=0.5,         # exuberant growth — overproduction then prune
        synaptogenesis_candidates=50_000,   # per-step candidate pairs (continuous growth)
        health_decay_rate=0.01,          # gentle pruning pressure even in infancy
        da_baseline=0.03,                # low resting DA
        da_sensitivity=0.3,              # mild novelty response
        plasticity_rate=0.5,             # critical period: elevated for competitive learning
        leak_rate_target=0.20,           # high leak (unmyelinated, lossy)
        n_demos=3,                       # lots of hand-holding
        spotlight_strength=2.0,          # strong attention guidance
        fatigue_threshold=10.0,          # frequent sleep (like a newborn)
        wta_active_frac=0.3,            # starts broad (PV+ immature), ramped to 0.1 by run_infancy
        noise_std=0.05,                 # spontaneous activity: enough to fire, not enough to drown signal
        task_mix_ratio=0.0,             # pure unsupervised stimuli
        copy_strength=1.0,              # full instinct — copy pathway at birth weight
        # Layer-aware: protect higher layers — they barely exist yet
        health_decay_L2_scale=0.3,      # L2 almost immune to pruning
        health_decay_L3_scale=0.1,      # L3 fully protected (PFC barely wired)
        synaptogenesis_L2_boost=1.0,
        synaptogenesis_L3_boost=1.0,
    ),
    "late_infancy": HomeostasisSetpoints(
        # Motor babbling and mimicry phase.  L1 is stable; the baby can
        # now "see" and starts learning motor control through exploration
        # and imitating the copy pathway.  Synaptogenesis slows — the brain
        # shifts from growing connections to learning how to use them.
        target_rate=0.25,
        scaling_gain=0.0,
        intrinsic_eta=0.0,
        ei_target_ratio=0.6,
        ema_tau=0.02,
        # Developmental
        synaptogenesis_rate=0.25,        # slowing — refine, don't grow
        synaptogenesis_candidates=20_000,
        health_decay_rate=0.02,
        da_baseline=0.03,
        da_sensitivity=0.4,              # slightly higher novelty response for motor exploration
        plasticity_rate=0.5,
        leak_rate_target=0.18,
        n_demos=3,
        spotlight_strength=2.0,
        fatigue_threshold=10.0,
        wta_active_frac=0.15,           # tighter competition than early infancy
        noise_std=0.05,
        task_mix_ratio=0.0,             # no formal tasks yet
        babble_ratio=0.4,               # 40% motor activities — L1 still needs its normal sensory diet
        copy_strength=1.0,              # full strength — it IS the desire to imitate
        # Layer-aware: L2/L3 starting to wire up via babbling co-activity
        health_decay_L2_scale=0.5,
        health_decay_L3_scale=0.2,
        synaptogenesis_L2_boost=1.5,    # motor babbling creates L2 demand
        synaptogenesis_L3_boost=2.0,    # L3 needs a head start
    ),
    "childhood": HomeostasisSetpoints(
        # Identity should already work from mimicry training.
        # Copy pathway begins gradual decay; tasks test learned pathways.
        target_rate=0.15,
        scaling_gain=0.0,                # DISABLED — L2/L3 still differentiating, scaling fights that
        intrinsic_eta=0.0,               # DISABLED — re-enable in adolescence when representations consolidate
        ei_target_ratio=0.8,
        ema_tau=0.02,
        # Developmental
        synaptogenesis_rate=0.15,        # slowing growth (exponential decline from infancy peak)
        synaptogenesis_candidates=20_000,  # L2/L3 peak growth phase (Huttenlocher 1997)
        health_decay_rate=0.05,          # activity-dependent sculpting
        da_baseline=0.05,                # increasing DA tone
        da_sensitivity=0.5,              # stronger novelty response
        plasticity_rate=1.0,             # elevated learning rate
        leak_rate_target=0.12,           # partial myelination
        n_demos=2,                       # less scaffolding
        spotlight_strength=1.5,          # moderate attention guidance
        fatigue_threshold=10.0,          # longer wake periods
        wta_active_frac=0.12,           # slight loosening from infancy 0.1, not 0.2 which undoes L1
        noise_std=0.05,                 # moderate spontaneous activity
        task_mix_ratio=0.7,             # mostly supervised tasks, 30% stimuli continues
        babble_ratio=0.1,               # some continued mimicry alongside tasks
        copy_strength=0.1,              # near-gone — learned pathways must carry motor output now
        # Layer-aware: L2/L3 actively growing, still partially protected
        health_decay_L2_scale=0.7,
        health_decay_L3_scale=0.4,
        synaptogenesis_L2_boost=2.0,    # cross-layer wiring demand
        synaptogenesis_L3_boost=5.0,    # PFC peak growth — needs most help
    ),
    "adolescence": HomeostasisSetpoints(
        # Homeostatic (sleep-only application)
        target_rate=0.10,
        scaling_gain=0.03,
        intrinsic_eta=0.008,
        ei_target_ratio=1.0,
        ema_tau=0.015,
        # Developmental
        synaptogenesis_rate=0.05,        # maintenance only
        synaptogenesis_candidates=200,   # per-step candidate pairs (maintenance)
        health_decay_rate=0.15,          # aggressive pruning pressure
        da_baseline=0.08,                # peak DA (pubertal analog)
        da_sensitivity=0.8,              # peak novelty sensitivity
        plasticity_rate=0.5,             # most windows closing
        leak_rate_target=0.06,           # near-adult myelination
        n_demos=0,                       # no demos — figure it out yourself
        spotlight_strength=1.0,          # no attention guidance
        fatigue_threshold=15.0,          # adult sleep pattern
        wta_active_frac=0.1,            # tight adult E/I balance
        noise_std=0.02,                 # low spontaneous activity (adult cortex)
        task_mix_ratio=1.0,             # fully supervised
        copy_strength=0.0,              # extinct — reflex fully replaced by cortical motor pathways
        # Layer-aware: approaching adult — L3/PFC still last to fully mature
        health_decay_L2_scale=1.0,
        health_decay_L3_scale=0.8,      # PFC matures last (Petanjek 2011)
        synaptogenesis_L2_boost=1.0,
        synaptogenesis_L3_boost=1.0,
    ),
}


class StageManager:
    """
    Manages developmental stage transitions with exponential approach.

    Biological basis: developmental parameters change with exponential
    time constants (Huttenlocher & Dabholkar 1997). Synaptogenesis rate
    declines as 1 - e^{-t/tau}, not linearly. This means rapid initial
    change (the system quickly leaves the old state) with a long tail
    (it gradually and asymptotically approaches the new state).

    `transition_tau` controls the time constant in brain steps. At 3*tau,
    the transition is ~95% complete. We cap at 99.5% and snap to target.
    """

    def __init__(
        self,
        initial_stage: str = "infancy",
        transition_tau: int = 10_000,
    ):
        if initial_stage not in STAGES:
            raise ValueError(f"Unknown stage: {initial_stage}. Choose from {list(STAGES.keys())}")

        self.current_stage = initial_stage
        self.transition_tau = transition_tau

        self._from_setpoints = STAGES[initial_stage]
        self._to_setpoints = STAGES[initial_stage]
        self._transition_progress = 1.0  # fully settled
        self._steps_in_transition = 0

    def transition_to(self, stage: str):
        """Begin a smooth transition to a new stage."""
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}. Choose from {list(STAGES.keys())}")

        self._from_setpoints = self.current_setpoints()
        self._to_setpoints = STAGES[stage]
        self._transition_progress = 0.0
        self._steps_in_transition = 0
        self.current_stage = stage

    def step(self, n: int = 1):
        """Advance the transition by n brain ticks."""
        if self._transition_progress < 1.0:
            self._steps_in_transition += n
            tau = max(1, self.transition_tau)
            self._transition_progress = 1.0 - math.exp(-self._steps_in_transition / tau)
            if self._transition_progress > 0.995:
                self._transition_progress = 1.0

    def current_setpoints(self) -> HomeostasisSetpoints:
        """Get the current blended setpoints."""
        if self._transition_progress >= 1.0:
            return self._to_setpoints
        return self._from_setpoints.interpolate(
            self._to_setpoints, self._transition_progress
        )

    @property
    def is_transitioning(self) -> bool:
        return self._transition_progress < 1.0

    def state_dict(self) -> dict:
        return {
            "current_stage": self.current_stage,
            "transition_progress": self._transition_progress,
            "steps_in_transition": self._steps_in_transition,
            "transition_tau": self.transition_tau,
        }

    def load_state_dict(self, d: dict):
        stage = d.get("current_stage", "infancy")
        if stage in STAGES:
            self.current_stage = stage
            self._to_setpoints = STAGES[stage]
        self._transition_progress = d.get("transition_progress", 1.0)
        self._steps_in_transition = d.get("steps_in_transition", 0)
        self.transition_tau = d.get("transition_tau",
                                    d.get("transition_steps", self.transition_tau))
