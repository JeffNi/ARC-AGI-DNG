"""
Episodic memory: explicit tensor buffer for storing and retrieving experiences.

Replaces spiking memory nodes for storage. The neural network handles
processing; this module handles remembering.

Two retrieval modes:
  1. Deliberate recall: given a query input, find the most similar stored
     input and return the associated output as a motor hint signal.
  2. Spontaneous recall: during thinking, periodically inject a faint
     signal from a relevant stored memory -- like when a related idea
     "pops into your head" while problem-solving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from .encoding import NUM_COLORS, pad_grid


@dataclass
class Episode:
    """A single stored experience: an input-output pair."""
    input_grid: np.ndarray
    output_grid: np.ndarray
    input_flat: np.ndarray   # flattened padded input for fast comparison
    output_flat: np.ndarray  # flattened padded output


def _motor_hint_signal(
    output_grid: np.ndarray,
    motor_offset: int,
    n_total_nodes: int,
    max_h: int,
    max_w: int,
) -> np.ndarray:
    """Encode an output grid as a one-hot motor signal placed at motor_offset."""
    padded = pad_grid(output_grid, max_h, max_w)
    n_cells = max_h * max_w
    sig = np.zeros(n_total_nodes, dtype=np.float64)
    for i in range(n_cells):
        r, c = divmod(i, max_w)
        color = int(padded[r, c])
        sig[motor_offset + i * NUM_COLORS + color] = 1.0
    return sig


class EpisodicMemory:
    """
    Content-addressable memory buffer.

    Stores (input, output) grid pairs. Retrieves by comparing a query
    input to all stored inputs via cell-wise match count.
    """

    def __init__(self, max_h: int, max_w: int, capacity: int = 1000):
        self.max_h = max_h
        self.max_w = max_w
        self.capacity = capacity
        self.episodes: List[Episode] = []

    def store(self, input_grid: np.ndarray, output_grid: np.ndarray) -> None:
        """Store an input-output pair. Deduplicates exact matches."""
        inp = np.asarray(input_grid)
        out = np.asarray(output_grid)
        inp_padded = pad_grid(inp, self.max_h, self.max_w).ravel()
        out_padded = pad_grid(out, self.max_h, self.max_w).ravel()

        # Skip if we already have this exact pair
        for ep in self.episodes:
            if np.array_equal(ep.input_flat, inp_padded) and np.array_equal(ep.output_flat, out_padded):
                return

        self.episodes.append(Episode(
            input_grid=inp.copy(),
            output_grid=out.copy(),
            input_flat=inp_padded,
            output_flat=out_padded,
        ))

        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def store_pairs(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Store multiple input-output pairs."""
        for inp, out in pairs:
            self.store(inp, out)

    def recall(
        self,
        query_input: np.ndarray,
        top_k: int = 1,
    ) -> List[Tuple[Episode, float]]:
        """
        Find the most similar stored inputs and return their episodes.

        Similarity = fraction of non-background cells that match between
        the query and the stored input (ignoring black/0 cells in the query).
        """
        if not self.episodes:
            return []

        query = pad_grid(np.asarray(query_input), self.max_h, self.max_w).ravel()
        fg = query != 0
        n_fg = fg.sum()

        if n_fg == 0:
            return [(self.episodes[0], 0.0)]

        scores = []
        for ep in self.episodes:
            match = (query[fg] == ep.input_flat[fg]).sum()
            scores.append(match / n_fg)

        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.episodes[i], scores[i]) for i in idx]

    def recall_signal(
        self,
        query_input: np.ndarray,
        motor_offset: int,
        n_total_nodes: int,
        strength: float = 0.5,
    ) -> np.ndarray:
        """
        Deliberate recall: combine ALL stored outputs weighted by their
        input similarity to the query. This gives the motor layer a
        blended hint of what answers to similar inputs look like.

        Even low-similarity matches contribute faintly -- like a vague
        feeling that "the answer involves these colors/positions."
        """
        if not self.episodes:
            return np.zeros(n_total_nodes)

        query = pad_grid(np.asarray(query_input), self.max_h, self.max_w).ravel()
        fg = query != 0
        n_fg = max(fg.sum(), 1)

        hint = np.zeros(n_total_nodes)
        total_weight = 0.0

        for ep in self.episodes:
            sim = (query[fg] == ep.input_flat[fg]).sum() / n_fg if fg.any() else 0.0
            if sim < 0.01:
                continue
            w = sim ** 2  # emphasize better matches
            hint += w * _motor_hint_signal(
                ep.output_grid, motor_offset, n_total_nodes,
                self.max_h, self.max_w,
            )
            total_weight += w

        if total_weight > 0:
            hint /= total_weight

        return hint * strength

    def spontaneous_signal(
        self,
        current_activity: np.ndarray,
        input_nodes: np.ndarray,
        motor_offset: int,
        n_total_nodes: int,
        strength: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Spontaneous recall: based on current sensory activity, a relevant
        memory occasionally surfaces -- like an association popping into
        your mind while thinking.

        Reads the current sensory state, finds partially matching memories,
        and injects a faint output signal. The faintness means it doesn't
        override processing, but nudges the network toward relevant patterns.
        """
        if not self.episodes:
            return np.zeros(n_total_nodes)

        if rng is None:
            rng = np.random.default_rng()

        # Decode current sensory activity into a grid
        n_cells = self.max_h * self.max_w
        sensory_start = int(input_nodes[0])
        sensory_r = current_activity[sensory_start:sensory_start + n_cells * NUM_COLORS]
        if len(sensory_r) < n_cells * NUM_COLORS:
            return np.zeros(n_total_nodes)

        sensory_r = sensory_r.reshape(n_cells, NUM_COLORS)
        current_grid = np.argmax(sensory_r, axis=1)

        # Find matching memories
        fg = current_grid != 0
        n_fg = fg.sum()
        if n_fg == 0:
            return np.zeros(n_total_nodes)

        scores = np.array([
            (current_grid[fg] == ep.input_flat[fg]).sum() / n_fg
            for ep in self.episodes
        ])

        # Probabilistic recall: higher similarity = more likely to surface
        probs = scores ** 2
        total = probs.sum()
        if total < 1e-8:
            return np.zeros(n_total_nodes)
        probs /= total

        chosen = rng.choice(len(self.episodes), p=probs)
        ep = self.episodes[chosen]
        sim = scores[chosen]

        hint = _motor_hint_signal(
            ep.output_grid, motor_offset, n_total_nodes,
            self.max_h, self.max_w,
        )

        return hint * strength * sim

    def clear(self) -> None:
        """Wipe all stored memories."""
        self.episodes.clear()

    def __len__(self) -> int:
        return len(self.episodes)

    def __repr__(self) -> str:
        return f"EpisodicMemory({len(self.episodes)} episodes, cap={self.capacity})"
