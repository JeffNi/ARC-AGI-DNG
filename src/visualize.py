"""
Visualization utilities for DNG state and ARC grids.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from .graph import DNG, NodeType

ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25',
]
ARC_CMAP = ListedColormap(ARC_COLORS)
ARC_NORM = BoundaryNorm(np.arange(-0.5, 10.5, 1), ARC_CMAP.N)


def plot_grid(ax, grid, title=""):
    grid = np.asarray(grid)
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='#444', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)


def plot_comparison(input_grid, expected_grid, predicted_grid, task_id=""):
    """Show input, expected output, and predicted output side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plot_grid(axes[0], input_grid, "Input")
    plot_grid(axes[1], expected_grid, "Expected")
    plot_grid(axes[2], predicted_grid, "Predicted")

    match = np.array_equal(np.asarray(expected_grid), np.asarray(predicted_grid))
    color = 'green' if match else 'red'
    label = 'CORRECT' if match else 'WRONG'
    fig.suptitle(f"Task {task_id}  [{label}]", fontsize=13,
                 fontweight='bold', color=color)
    fig.tight_layout()
    return fig


def plot_activation_histogram(net: DNG, title="Activation Distribution"):
    """Histogram of current activations, split by node type."""
    fig, ax = plt.subplots(figsize=(6, 3))
    for ntype in NodeType:
        mask = net.node_types == list(NodeType).index(ntype)
        if mask.sum() > 0:
            ax.hist(net.activations[mask], bins=30, alpha=0.6,
                    label=ntype.value, range=(-1, 1))
    ax.set_xlabel("Activation")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_weight_histogram(net: DNG, title="Weight Distribution"):
    """Histogram of all edge weights."""
    if not net.weights:
        return None
    w = np.array(list(net.weights.values()))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(w, bins=50, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}  (n={len(w)}, |mean|={np.mean(np.abs(w)):.4f})")
    fig.tight_layout()
    return fig
