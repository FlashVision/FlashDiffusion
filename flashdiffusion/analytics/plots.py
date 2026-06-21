"""Plotting utilities — training curves and generation grids."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _get_plt():
    """Lazy-import matplotlib to avoid hard dependency at module level."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(
    log: Dict[str, List[float]],
    keys: Optional[Sequence[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
) -> "Figure":
    """Plot one or more scalar metrics from a training log dict.

    Parameters
    ----------
    log : dict[str, list[float]]
        ``{"loss": [...], "lr": [...], ...}``
    keys : sequence of str | None
        Which keys to plot.  *None* plots everything.
    save_path : str | Path | None
        If given, save figure to this path.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()
    keys = keys or list(log.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes.flatten()

    for ax, key in zip(axes, keys):
        values = log[key]
        ax.plot(values, linewidth=1.5)
        ax.set_title(key)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_generation_grid(
    images: list,
    prompts: Optional[List[str]] = None,
    cols: int = 4,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Generated Images",
) -> "Figure":
    """Plot a grid of generated images with optional prompt labels.

    Parameters
    ----------
    images : list[PIL.Image]
        Generated images.
    prompts : list[str] | None
        Optional labels for each image.
    cols : int
        Number of columns.
    save_path : str | Path | None
        Optional save path.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _get_plt()

    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)

    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(np.array(images[i]))
            if prompts and i < len(prompts):
                ax.set_title(prompts[i][:30], fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
