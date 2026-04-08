"""
Developmental stages — setpoint configurations and smooth transitions.

Homeostasis is the same code at every stage; only the parameters change.
StageManager handles scheduling transitions and producing the current
blended HomeostasisSetpoints.
"""

from __future__ import annotations

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
        synaptogenesis_rate=0.3,         # acceptance prob for neurotrophin pool
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
    ),
    "childhood": HomeostasisSetpoints(
        # Homeostatic (sleep-only application)
        target_rate=0.15,
        scaling_gain=0.025,
        intrinsic_eta=0.01,
        ei_target_ratio=0.8,
        ema_tau=0.02,
        # Developmental
        synaptogenesis_rate=0.15,        # slowing growth
        synaptogenesis_candidates=2_000,   # per-step candidate pairs
        health_decay_rate=0.05,          # activity-dependent sculpting
        da_baseline=0.05,                # increasing DA tone
        da_sensitivity=0.5,              # stronger novelty response
        plasticity_rate=1.0,             # baseline learning rate
        leak_rate_target=0.12,           # partial myelination
        n_demos=2,                       # less scaffolding
        spotlight_strength=1.5,          # moderate attention guidance
        fatigue_threshold=10.0,          # longer wake periods
        wta_active_frac=0.2,            # E/I balance maturing — lateral inhibition sharpens
        noise_std=0.05,                 # moderate spontaneous activity
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
    ),
}


class StageManager:
    """
    Manages developmental stage transitions with smooth interpolation.

    Transitions never snap — setpoints are linearly interpolated over
    `transition_steps` brain steps.
    """

    def __init__(
        self,
        initial_stage: str = "infancy",
        transition_steps: int = 5000,
    ):
        if initial_stage not in STAGES:
            raise ValueError(f"Unknown stage: {initial_stage}. Choose from {list(STAGES.keys())}")

        self.current_stage = initial_stage
        self.transition_steps = transition_steps

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

    def step(self):
        """Advance the transition by one brain step."""
        if self._transition_progress < 1.0:
            self._steps_in_transition += 1
            self._transition_progress = min(
                1.0, self._steps_in_transition / max(1, self.transition_steps)
            )

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
            "transition_steps": self.transition_steps,
        }

    def load_state_dict(self, d: dict):
        stage = d.get("current_stage", "infancy")
        if stage in STAGES:
            self.current_stage = stage
            self._to_setpoints = STAGES[stage]
        self._transition_progress = d.get("transition_progress", 1.0)
        self._steps_in_transition = d.get("steps_in_transition", 0)
        self.transition_steps = d.get("transition_steps", self.transition_steps)
