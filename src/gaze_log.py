"""
Gaze logging and analysis for the active vision system.

Records where the network looks at each step, enabling measurement of
whether gaze patterns develop from random (infancy) to structured
(childhood) and whether they resemble human ARC-solving behavior.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


class GazeLogger:
    """Lightweight gaze event recorder.

    Appends fixation events to a JSONL file for offline analysis.
    Tracks the current fixation to compute dwell times.
    """

    def __init__(self, log_path: str | Path | None = None):
        self.log_path = Path(log_path) if log_path else None
        self._current_slot: int = -1
        self._current_type: str = "unknown"
        self._fixation_start: int = 0
        self._events: list[dict] = []
        self._buffer_size = 200

    def record(self, age: int, slot: int, slot_type: str,
               motor_event: bool = False) -> None:
        """Record a gaze observation at the current timestep.

        Args:
            age: Brain age (total steps).
            slot: Which display slot is currently fixated.
            slot_type: Tag for the slot content ("input", "output",
                       "test_input", "answer", "stimulus", "empty").
            motor_event: True if the network just produced motor output
                         (used to measure action-to-canvas mapping).
        """
        if slot != self._current_slot:
            if self._current_slot >= 0:
                self._events.append({
                    "age": self._fixation_start,
                    "slot": self._current_slot,
                    "type": self._current_type,
                    "dwell": age - self._fixation_start,
                })
            self._current_slot = slot
            self._current_type = slot_type
            self._fixation_start = age

        if motor_event:
            self._events.append({
                "age": age,
                "slot": slot,
                "type": "motor_event",
                "motor": True,
            })

        if self.log_path and len(self._events) >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered events to disk."""
        if not self.log_path or not self._events:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            for ev in self._events:
                f.write(json.dumps(ev) + "\n")
        self._events.clear()

    def get_events(self) -> list[dict]:
        """Return all buffered events (for in-memory analysis)."""
        return list(self._events)

    def clear(self) -> None:
        """Clear buffer without writing."""
        self._events.clear()
        self._current_slot = -1
        self._current_type = "unknown"
        self._fixation_start = 0


def analyze_transitions(events: list[dict], n_slots: int) -> np.ndarray:
    """Compute the gaze transition matrix P(next_slot | current_slot).

    Returns an n_slots x n_slots matrix where entry [i, j] is the
    probability of transitioning from slot i to slot j.
    """
    counts = np.zeros((n_slots, n_slots), dtype=np.float64)
    prev_slot = -1
    for ev in events:
        if "motor" in ev:
            continue
        slot = ev.get("slot", -1)
        if prev_slot >= 0 and slot >= 0:
            counts[prev_slot, slot] += 1
        prev_slot = slot

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def analyze_dwell_times(events: list[dict]) -> dict[str, list[int]]:
    """Group dwell times by slot type.

    Returns dict mapping slot_type -> list of dwell durations.
    """
    dwells: dict[str, list[int]] = defaultdict(list)
    for ev in events:
        if "dwell" in ev:
            dwells[ev["type"]].append(ev["dwell"])
    return dict(dwells)


def adjacent_transition_ratio(events: list[dict], n_slots: int) -> float:
    """Fraction of transitions that go to an adjacent slot.

    A ratio above 1/n_slots indicates the network has learned that
    adjacent slots are related (tile association).
    """
    trans = analyze_transitions(events, n_slots)
    adj_sum = 0.0
    total = 0.0
    for i in range(n_slots):
        for j in range(n_slots):
            if i == j:
                continue
            total += trans[i, j]
            if abs(i - j) == 1:
                adj_sum += trans[i, j]
    return adj_sum / max(total, 1e-9)


def action_canvas_probability(events: list[dict],
                              answer_slot: int,
                              window: int = 10) -> float:
    """P(gaze is at answer_slot when motor fires).

    Measures the sensorimotor contingency: does the network look at its
    own output while/right after producing it? Motor events record the
    current gaze slot, so this directly checks co-occurrence.
    """
    motor_events = [ev for ev in events if ev.get("motor")]
    if not motor_events:
        return 0.0

    hits = sum(1 for mev in motor_events if mev["slot"] == answer_slot)
    return hits / len(motor_events)


def print_gaze_summary(events: list[dict], n_slots: int,
                       answer_slot: int) -> None:
    """Print a readable summary of gaze behavior."""
    if not events:
        print("  No gaze events recorded.")
        return

    dwells = analyze_dwell_times(events)
    trans = analyze_transitions(events, n_slots)
    adj_ratio = adjacent_transition_ratio(events, n_slots)
    acp = action_canvas_probability(events, answer_slot)

    n_motor = sum(1 for ev in events if ev.get("motor"))
    print(f"  Total fixation events: {sum(len(v) for v in dwells.values())}  "
          f"motor events: {n_motor}")
    for stype, dlist in sorted(dwells.items()):
        if dlist:
            print(f"    {stype:>12s}: n={len(dlist):4d}  "
                  f"mean_dwell={np.mean(dlist):.1f}  "
                  f"median={np.median(dlist):.0f}")

    print(f"  Adjacent transition ratio: {adj_ratio:.3f} "
          f"(chance={1.0 / max(n_slots - 1, 1):.3f})")
    print(f"  Action->canvas P: {acp:.3f}")

    print("  Transition matrix (top transitions):")
    flat = [(trans[i, j], i, j) for i in range(n_slots)
            for j in range(n_slots) if i != j and trans[i, j] > 0.05]
    flat.sort(reverse=True)
    for prob, i, j in flat[:8]:
        print(f"    slot {i} -> slot {j}: {prob:.3f}")
