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
        target_rate=0.25,
        scaling_gain=0.005,
        bcm_tau=0.02,
        intrinsic_eta=0.008,
        ei_target_ratio=0.6,
        ema_tau=0.03,
    ),
    "childhood": HomeostasisSetpoints(
        target_rate=0.15,
        scaling_gain=0.01,
        bcm_tau=0.01,
        intrinsic_eta=0.005,
        ei_target_ratio=0.8,
        ema_tau=0.02,
    ),
    "adolescence": HomeostasisSetpoints(
        target_rate=0.10,
        scaling_gain=0.02,
        bcm_tau=0.005,
        intrinsic_eta=0.003,
        ei_target_ratio=1.0,
        ema_tau=0.015,
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
