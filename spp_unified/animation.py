"""Animated-GIF rendering for self-propelled-particle trajectories.

Frames come from ``SimulationResult.states`` (a list of ``(positions, headings)``
tuples recorded with ``record_states=True``). Arrows are colored by heading so
that ordered, milling, and turbulent phases are visually distinguishable, and an
optional center-of-mass trail exposes translation and rotation.
"""
from __future__ import annotations

import io
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

from .geometry import relative_to_center


_TWO_SPECIES_COLORS = np.array([[0.11, 0.47, 0.86], [0.87, 0.30, 0.15]])


def _heading_colors(theta: np.ndarray) -> np.ndarray:
    """Map headings in (-pi, pi] to a cyclic hue so direction reads as color."""
    hue = (theta + np.pi) / (2.0 * np.pi)
    return cm.hsv(hue)


def _com_path(states: Sequence[tuple[np.ndarray, np.ndarray]], S: float, periodic: bool) -> np.ndarray:
    """Unwrapped center-of-mass path across frames (circular mean for periodic)."""
    pts = []
    for x, _ in states:
        center, _ = relative_to_center(x, S, periodic)
        pts.append(center)
    return np.asarray(pts)


def render_gif(states: Sequence[tuple[np.ndarray, np.ndarray]], S: float, out_path: str, *,
               species: np.ndarray | None = None, periodic: bool = True,
               title: str = "", subtitle: str = "", metric_series: dict | None = None,
               color_by: str = "heading", trail: bool = True, fps: int = 18,
               dpi: int = 96, arrow_scale: float = 26.0, tail: int = 24,
               max_frames: int = 160) -> str:
    """Render particle frames to an animated GIF and return ``out_path``.

    ``color_by`` is ``"heading"`` (hue = direction), ``"species"`` (two-species
    coloring), or ``"speed_align"`` (local alignment with neighbors, viridis).
    ``metric_series`` optionally maps a label -> per-frame float array drawn as a
    live readout bar so the fitness driving the state is visible as it evolves.
    """
    states = list(states)
    if len(states) > max_frames:
        idx = np.linspace(0, len(states) - 1, max_frames).round().astype(int)
        states = [states[i] for i in idx]
        if metric_series:
            metric_series = {k: np.asarray(v)[np.clip(idx, 0, len(v) - 1)] for k, v in metric_series.items()}
    n_frames = len(states)

    com = _com_path(states, S, periodic) if trail else None

    fig = plt.figure(figsize=(5.0, 5.6), dpi=dpi)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.80])
    bar_ax = fig.add_axes([0.10, 0.855, 0.80, 0.045]) if metric_series else None
    ax.set_xlim(0, S); ax.set_ylim(0, S); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("0.6")

    frames: list[Image.Image] = []
    metric_keys = list(metric_series.keys()) if metric_series else []

    for f, (x, theta) in enumerate(states):
        ax.clear()
        ax.set_xlim(0, S); ax.set_ylim(0, S); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#0c1020")

        if trail and com is not None and f > 0:
            seg = com[max(0, f - tail):f + 1]
            ax.plot(seg[:, 0], seg[:, 1], color="#ffd36b", lw=1.6, alpha=0.85, zorder=1)
            ax.scatter([com[f, 0]], [com[f, 1]], s=18, color="#ffd36b", zorder=2)

        u, v = np.cos(theta), np.sin(theta)
        if color_by == "species" and species is not None:
            colors = _TWO_SPECIES_COLORS[species.astype(int)]
        elif color_by == "speed_align":
            colors = None
        else:
            colors = _heading_colors(theta)

        if color_by == "speed_align":
            ax.quiver(x[:, 0], x[:, 1], u, v, np.cos(theta), cmap="viridis",
                      scale=arrow_scale, width=0.006, pivot="mid", zorder=3)
        else:
            ax.quiver(x[:, 0], x[:, 1], u, v, color=colors,
                      scale=arrow_scale, width=0.006, pivot="mid", zorder=3)

        if title:
            ax.set_title(title, color="white", fontsize=12, pad=6, loc="left")
        ax.text(0.99, 0.015, f"t = {f+1}/{n_frames}", transform=ax.transAxes,
                color="0.75", fontsize=8, ha="right", va="bottom")
        if subtitle:
            ax.text(0.01, 0.985, subtitle, transform=ax.transAxes, color="0.8",
                    fontsize=8.5, ha="left", va="top")

        if bar_ax is not None:
            bar_ax.clear(); bar_ax.axis("off")
            key = metric_keys[0]
            series = metric_series[key]
            fi = min(f, len(series) - 1)
            raw = float(series[fi])
            val = float(np.clip(raw, 0.0, 1.0))
            bar_ax.barh([0], [1.0], color="#232a44", height=0.8)
            bar_ax.barh([0], [val], color="#5ec8ff", height=0.8)
            bar_ax.set_xlim(0, 1); bar_ax.set_ylim(-0.5, 0.5)
            bar_ax.text(0.0, 0.9, f"{key} = {raw:.3f}",
                        transform=bar_ax.transAxes, color="white", fontsize=9,
                        ha="left", va="bottom")

        fig.patch.set_facecolor("#0c1020")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="#0c1020")
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").copy())
        buf.close()

    plt.close(fig)
    duration = int(1000 / max(1, fps))
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True, disposal=2)
    return out_path
