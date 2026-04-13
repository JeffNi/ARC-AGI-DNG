"""
Core data structure for the Dynamic Neural Graph.

Memory systems:
  Immediate:  V, r (activity)        -- decays in ~5-10 steps
  Short-term: f (facilitation/node)  -- decays in ~100-200 steps, resets between tasks
  Long-term:  memory nodes           -- near-zero decay, self-sustaining, persists across tasks
  Permanent:  weights                -- changed only during childhood

Neuromodulators (global scalars, broadcast to all synapses):
  DA  (dopamine):       reward prediction error → learning magnitude
  ACh (acetylcholine):  learning mode (high=plastic childhood, low=stable adult)
  NE  (norepinephrine): surprise/arousal → sharpens focus, reduces noise

See docs/03_Mathematics.md for full definitions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from scipy import sparse


class NodeType(enum.Enum):
    E = "excitatory"
    I = "inhibitory"
    M = "modulatory"
    Mem = "memory"


class Region(enum.Enum):
    SENSORY = "sensory"
    INTERNAL = "internal"
    MOTOR = "motor"
    MEMORY = "memory"
    LOCAL_DETECT = "local_detect"
    MID_LEVEL = "mid_level"
    ABSTRACT = "abstract"
    GATE = "gate"
    GAZE = "gaze"


# Cortical hierarchy layers (ordered low → high)
CORTICAL_LAYERS = (Region.LOCAL_DETECT, Region.MID_LEVEL, Region.ABSTRACT)
# All regions that count as "internal processing" (for backward compat)
INTERNAL_REGIONS = (Region.INTERNAL,) + CORTICAL_LAYERS

_REG_LIST = list(Region)

_NTYPE_LIST = list(NodeType)
_NTYPE_E = _NTYPE_LIST.index(NodeType.E)
_NTYPE_I = _NTYPE_LIST.index(NodeType.I)
_NTYPE_M = _NTYPE_LIST.index(NodeType.M)
_NTYPE_MEM = _NTYPE_LIST.index(NodeType.Mem)


def internal_mask(regions: np.ndarray) -> np.ndarray:
    """Boolean mask for all internal-processing neurons (any cortical layer)."""
    mask = np.zeros(len(regions), dtype=bool)
    for reg in INTERNAL_REGIONS:
        mask |= regions == _REG_LIST.index(reg)
    return mask


def layer_index(region_val: int) -> int:
    """Map a region integer to cortical layer number (0=L1, 1=L2, 2=L3, -1=not cortical)."""
    reg = _REG_LIST[region_val] if region_val < len(_REG_LIST) else None
    if reg == Region.LOCAL_DETECT:
        return 0
    elif reg == Region.MID_LEVEL:
        return 1
    elif reg == Region.ABSTRACT:
        return 2
    elif reg == Region.INTERNAL:
        return 0
    return -1

DEFAULT_LEAK = {
    NodeType.E: 0.3,
    NodeType.I: 0.5,
    NodeType.M: 0.2,
    NodeType.Mem: 0.02,
}


@dataclass
class DNG:
    """
    Dynamic Neural Graph.

    State per node:
      V[i]          -- membrane potential (real-valued)
      r[i]          -- firing rate = clip(V - threshold, 0, max_rate)
      adaptation[i] -- fatigue current
      f[i]          -- short-term facilitation (presynaptic, per-node)
    """

    n_nodes: int
    node_types: np.ndarray
    regions: np.ndarray

    # Per-node state
    V: np.ndarray = None
    threshold: np.ndarray = None

    # Per-node parameters (arrays, not scalars — different neuron types differ)
    max_rate: np.ndarray = None
    excitability: np.ndarray = None
    leak_rates: np.ndarray = None
    adapt_rate: np.ndarray = None
    adapt_decay: float = 0.1

    # Derived (computed from V each step)
    r: np.ndarray = None
    prev_r: np.ndarray = None
    adaptation: np.ndarray = None

    # Short-term facilitation (presynaptic, per-node)
    f: np.ndarray = None
    f_rate: float = 0.05
    f_decay: float = 0.02
    f_max: float = 3.0

    # Neuromodulators (global scalars)
    da: float = 0.0           # dopamine: reward prediction error
    da_baseline: float = 0.3  # running average of recent rewards
    ach: float = 1.0          # acetylcholine: learning mode (1=childhood, 0=adult)
    ne: float = 0.0           # norepinephrine: surprise/arousal

    # E/I balance (developmental)
    inh_scale: float = 1.0    # multiplier for inhibitory (negative) weights
    _last_inh_scale: float = field(default=-1.0, repr=False)

    # Node groups
    input_nodes: np.ndarray = None
    output_nodes: np.ndarray = None
    memory_nodes: np.ndarray = None
    gate_nodes: np.ndarray = None
    gaze_nodes: np.ndarray = None

    # Edge storage
    _edge_src: np.ndarray = field(default=None, repr=False)
    _edge_dst: np.ndarray = field(default=None, repr=False)
    _edge_w: np.ndarray = field(default=None, repr=False)
    _edge_tag: np.ndarray = field(default=None, repr=False)
    _edge_health: np.ndarray = field(default=None, repr=False)
    _edge_eligibility: np.ndarray = field(default=None, repr=False)
    _edge_count: int = field(default=0, repr=False)
    _edge_capacity: int = field(default=0, repr=False)

    # Cached CSR
    _W_csr: sparse.csr_matrix = field(default=None, repr=False)
    _csr_dirty: bool = field(default=True, repr=False)

    # Grid dimensions
    max_h: int = field(default=0, repr=False)
    max_w: int = field(default=0, repr=False)

    # Column info
    column_ids: np.ndarray = field(default=None, repr=False)
    n_columns: int = field(default=0, repr=False)
    wta_k_frac: float = field(default=0.2, repr=False)

    # Slow activity trace for temporal contiguity learning (trace rule)
    r_trace: np.ndarray = field(default=None, repr=False)

    # Type masks
    _mask_I: np.ndarray = field(default=None, repr=False)
    _mask_E: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        n = self.n_nodes
        if self.V is None:
            self.V = np.zeros(n)
        if self.threshold is None:
            self.threshold = np.full(n, 0.1)
        if self.max_rate is None:
            self.max_rate = np.ones(n)
        if self.adapt_rate is None:
            self.adapt_rate = np.full(n, 0.01)
        if self.excitability is None:
            self.excitability = np.ones(n)
        if self.leak_rates is None:
            self.leak_rates = np.full(n, 0.3)
        if self.r is None:
            self.r = np.zeros(n)
        if self.prev_r is None:
            self.prev_r = np.zeros(n)
        if self.adaptation is None:
            self.adaptation = np.zeros(n)
        if self.f is None:
            self.f = np.zeros(n)
        if self.column_ids is None:
            self.column_ids = np.full(n, -1, dtype=np.int32)
        if self.memory_nodes is None:
            self.memory_nodes = np.array([], dtype=np.int32)
        if self.gate_nodes is None:
            self.gate_nodes = np.array([], dtype=np.int32)
        if self.gaze_nodes is None:
            self.gaze_nodes = np.array([], dtype=np.int32)

        if self.r_trace is None:
            self.r_trace = np.zeros(n)

        if self._edge_src is None:
            cap = max(1000, n * 10)
            self._edge_src = np.empty(cap, dtype=np.int32)
            self._edge_dst = np.empty(cap, dtype=np.int32)
            self._edge_w = np.empty(cap, dtype=np.float64)
            self._edge_tag = np.zeros(cap, dtype=np.float64)
            self._edge_health = np.ones(cap, dtype=np.float64)
            self._edge_consolidation = np.zeros(cap, dtype=np.float64)
            self._edge_eligibility = np.zeros(cap, dtype=np.float64)
            self._edge_count = 0
            self._edge_capacity = cap

        # Ensure eligibility array exists (for networks loaded from old checkpoints)
        if self._edge_eligibility is None:
            self._edge_eligibility = np.zeros(self._edge_capacity, dtype=np.float64)

        self._build_type_masks()

    def _build_type_masks(self):
        self._mask_I = self.node_types == _NTYPE_I
        self._mask_E = ~self._mask_I

    def compute_rates(self):
        self.r = np.clip(self.V - self.threshold, 0.0, None)
        np.minimum(self.r, self.max_rate, out=self.r)

    def reset_facilitation(self):
        """Reset short-term facilitation (between tasks)."""
        self.f[:] = 0.0

    def reset_activity(self):
        """Reset all transient state (between tasks)."""
        self.V[:] = 0.0
        self.r[:] = 0.0
        self.prev_r[:] = 0.0
        self.adaptation[:] = 0.0
        self.f[:] = 0.0

    def _ensure_capacity(self, needed: int):
        if needed > self._edge_capacity:
            new_cap = max(needed, self._edge_capacity * 2)
            self._edge_src = np.resize(self._edge_src, new_cap)
            self._edge_dst = np.resize(self._edge_dst, new_cap)
            self._edge_w = np.resize(self._edge_w, new_cap)
            for attr in ('_edge_tag', '_edge_consolidation', '_edge_eligibility'):
                old = getattr(self, attr)
                new = np.zeros(new_cap, dtype=np.float64)
                new[:len(old)] = old
                setattr(self, attr, new)
            old_health = self._edge_health
            new_health = np.ones(new_cap, dtype=np.float64)
            new_health[:len(old_health)] = old_health
            self._edge_health = new_health
            self._edge_capacity = new_cap

    def add_edge(self, src: int, dst: int, weight: float):
        idx = self._edge_count
        self._ensure_capacity(idx + 1)
        self._edge_src[idx] = src
        self._edge_dst[idx] = dst
        self._edge_w[idx] = weight
        self._edge_tag[idx] = 0.0
        self._edge_health[idx] = 1.0
        self._edge_count += 1
        self._csr_dirty = True

    def add_edges_batch(self, srcs: np.ndarray, dsts: np.ndarray, weights: np.ndarray):
        """Add multiple edges at once (vectorized)."""
        n_new = len(srcs)
        if n_new == 0:
            return
        start = self._edge_count
        self._ensure_capacity(start + n_new)
        self._edge_src[start:start + n_new] = srcs
        self._edge_dst[start:start + n_new] = dsts
        self._edge_w[start:start + n_new] = weights
        self._edge_tag[start:start + n_new] = 0.0
        self._edge_health[start:start + n_new] = 1.0
        self._edge_eligibility[start:start + n_new] = 0.0
        self._edge_count += n_new
        self._csr_dirty = True

    def existing_edge_set(self) -> set:
        """Return set of (src, dst) tuples for all current edges."""
        n = self._edge_count
        return set(zip(self._edge_src[:n].tolist(), self._edge_dst[:n].tolist()))

    def edge_count(self) -> int:
        return self._edge_count

    def get_weight_matrix(self) -> sparse.csr_matrix:
        """CSR weight matrix W where W[dst, src] = weight.

        Applies inh_scale to inhibitory (negative) weights so the stored
        edge weights remain at their base values while the matrix used
        for dynamics reflects the current developmental E/I balance.
        """
        scale_changed = self._last_inh_scale != self.inh_scale
        if self._csr_dirty or self._W_csr is None or scale_changed:
            n = self._edge_count
            w = self._edge_w[:n]
            if self.inh_scale != 1.0:
                w = w.copy()
                neg = w < 0
                w[neg] *= self.inh_scale
            self._W_csr = sparse.csr_matrix(
                (w, (self._edge_dst[:n], self._edge_src[:n])),
                shape=(self.n_nodes, self.n_nodes),
            )
            self._csr_dirty = False
            self._last_inh_scale = self.inh_scale
        return self._W_csr

    def get_csr_permutation(self) -> np.ndarray:
        """Return array mapping edge index -> CSR data position.

        Needed by the GPU plastic kernel to sync CSR data after
        modifying edge weights.  The permutation is the sort order
        that scipy uses when building CSR from COO: lexsort by (row, col)
        = (dst, src).
        """
        n = self._edge_count
        return np.lexsort((
            self._edge_src[:n].astype(np.int64),
            self._edge_dst[:n].astype(np.int64),
        ))

    def compact(self):
        """Remove deleted edges (weight==0 markers)."""
        n = self._edge_count
        mask = self._edge_w[:n] != 0.0
        alive = int(np.sum(mask))
        self._edge_src[:alive] = self._edge_src[:n][mask]
        self._edge_dst[:alive] = self._edge_dst[:n][mask]
        self._edge_w[:alive] = self._edge_w[:n][mask]
        self._edge_tag[:alive] = self._edge_tag[:n][mask]
        self._edge_health[:alive] = self._edge_health[:n][mask]
        self._edge_eligibility[:alive] = self._edge_eligibility[:n][mask]
        self._edge_consolidation[:alive] = self._edge_consolidation[:n][mask]
        self._edge_count = alive
        self._csr_dirty = True

    def save(self, path: Union[str, Path]) -> None:
        n = self._edge_count
        np.savez_compressed(
            Path(path),
            n_nodes=np.array([self.n_nodes]),
            node_types=self.node_types,
            regions=self.regions,
            V=self.V,
            threshold=self.threshold,
            max_rate=self.max_rate,
            excitability=self.excitability,
            leak_rates=self.leak_rates,
            adaptation=self.adaptation,
            adapt_rate=self.adapt_rate,
            adapt_decay=np.array([self.adapt_decay]),
            f=self.f,
            f_rate=np.array([self.f_rate]),
            f_decay=np.array([self.f_decay]),
            f_max=np.array([self.f_max]),
            input_nodes=self.input_nodes,
            output_nodes=self.output_nodes,
            memory_nodes=self.memory_nodes,
            gate_nodes=self.gate_nodes,
            gaze_nodes=self.gaze_nodes,
            edges_src=self._edge_src[:n].copy(),
            edges_dst=self._edge_dst[:n].copy(),
            edges_w=self._edge_w[:n].copy(),
            edges_tag=self._edge_tag[:n].copy(),
            edges_health=self._edge_health[:n].copy(),
            edges_eligibility=self._edge_eligibility[:n].copy(),
            edges_consolidation=self._edge_consolidation[:n].copy(),
            column_ids=self.column_ids,
            n_columns=np.array([self.n_columns]),
            wta_k_frac=np.array([self.wta_k_frac]),
            max_h=np.array([self.max_h]),
            max_w=np.array([self.max_w]),
            da=np.array([self.da]),
            da_baseline=np.array([self.da_baseline]),
            ach=np.array([self.ach]),
            ne=np.array([self.ne]),
            inh_scale=np.array([self.inh_scale]),
            r_trace=self.r_trace,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DNG":
        data = np.load(path)
        src = data["edges_src"]
        dst = data["edges_dst"]
        w = data["edges_w"]
        n_edges = len(src)
        n_nodes = int(data["n_nodes"][0])
        cap = max(n_edges * 2, 1000)

        net = cls(
            n_nodes=n_nodes,
            node_types=data["node_types"],
            regions=data["regions"],
            excitability=data["excitability"],
            leak_rates=data["leak_rates"],
            input_nodes=data["input_nodes"],
            output_nodes=data["output_nodes"],
        )
        net.V = data["V"]
        net.threshold = data["threshold"]

        # Per-node arrays: handle both old (scalar) and new (array) formats
        mr = data["max_rate"]
        if mr.ndim == 0 or len(mr) == 1:
            net.max_rate = np.full(n_nodes, float(mr.flat[0]))
        else:
            net.max_rate = mr.copy()

        net.adaptation = data["adaptation"]

        ar = data["adapt_rate"]
        if ar.ndim == 0 or len(ar) == 1:
            net.adapt_rate = np.full(n_nodes, float(ar.flat[0]))
        else:
            net.adapt_rate = ar.copy()

        net.adapt_decay = float(data["adapt_decay"].flat[0])

        if "f" in data:
            net.f = data["f"]
            net.f_rate = float(data["f_rate"][0])
            net.f_decay = float(data["f_decay"][0])
            net.f_max = float(data["f_max"][0])
        if "memory_nodes" in data:
            net.memory_nodes = data["memory_nodes"]
        if "gate_nodes" in data:
            net.gate_nodes = data["gate_nodes"]
        if "gaze_nodes" in data:
            net.gaze_nodes = data["gaze_nodes"]
        if "column_ids" in data:
            net.column_ids = data["column_ids"]
            net.n_columns = int(data["n_columns"][0])
            net.wta_k_frac = float(data["wta_k_frac"][0])
        if "max_h" in data:
            net.max_h = int(data["max_h"][0])
            net.max_w = int(data["max_w"][0])
        if "da" in data:
            net.da = float(data["da"][0])
            net.da_baseline = float(data["da_baseline"][0])
            net.ach = float(data["ach"][0])
            net.ne = float(data["ne"][0])
        if "inh_scale" in data:
            net.inh_scale = float(data["inh_scale"][0])
        if "r_trace" in data:
            net.r_trace = data["r_trace"].copy()

        net._edge_src = np.empty(cap, dtype=np.int32)
        net._edge_dst = np.empty(cap, dtype=np.int32)
        net._edge_w = np.empty(cap, dtype=np.float64)
        net._edge_tag = np.zeros(cap, dtype=np.float64)
        net._edge_health = np.ones(cap, dtype=np.float64)
        net._edge_eligibility = np.zeros(cap, dtype=np.float64)
        net._edge_consolidation = np.zeros(cap, dtype=np.float64)
        net._edge_src[:n_edges] = src
        net._edge_dst[:n_edges] = dst
        net._edge_w[:n_edges] = w
        if "edges_tag" in data:
            net._edge_tag[:n_edges] = data["edges_tag"]
        if "edges_health" in data:
            net._edge_health[:n_edges] = data["edges_health"]
        elif "edges_weak_count" in data:
            net._edge_health[:n_edges] = 1.0
        if "edges_eligibility" in data:
            net._edge_eligibility[:n_edges] = data["edges_eligibility"]
        if "edges_consolidation" in data:
            net._edge_consolidation[:n_edges] = data["edges_consolidation"]
        net._edge_count = n_edges
        net._edge_capacity = cap
        net._csr_dirty = True
        net.compute_rates()
        return net
