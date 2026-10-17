"""Figure helpers shared by scripts and notebooks.

Style: light print surface, recessive axes and grid, units on every axis, direct labels in
addition to legends. Colors are assigned by role: categorical slots in a fixed order for
unordered series, a single-hue ramp for ordered series (e.g. pump speeds).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from wdtk import pumps as pm
from wdtk.units import flow_from_m3s

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
"""Categorical slots, assigned in this fixed order (never cycled)."""
BLUE_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
"""Ordinal single-hue ramp, light to dark; the lightest step still clears 2:1 on SURFACE."""


def apply_style() -> None:
    """Set matplotlib rcParams for project figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )


def ordinal_colors(n: int) -> list[str]:
    """Return ``n`` colors from the ordinal ramp, spread from light to dark.

    Parameters
    ----------
    n : int
        Number of ordered series (1 to 5).

    Returns
    -------
    list of str
        Hex colors, lightest first.
    """
    if not 1 <= n <= len(BLUE_RAMP):
        raise ValueError(f"n must be between 1 and {len(BLUE_RAMP)}; got {n}.")
    idx = np.round(np.linspace(0, len(BLUE_RAMP) - 1, n)).astype(int)
    return [BLUE_RAMP[i] for i in idx]


def _head_function(curve: pm.PumpCurve) -> Callable[[float], float]:
    def head(q: float) -> float:
        return pm.pump_head_m(curve, q)

    return head


def plot_pump_system(
    ax: Axes,
    curve: pm.PumpCurve,
    system_head: Callable[[float], float],
    speeds: Sequence[float] = (1.0, 0.9, 0.8, 0.7),
    flow_unit: str = "L/s",
) -> list[tuple[float, float]]:
    """Plot a pump curve family (variable speed), a system curve, and operating points.

    Pump curves at each speed ratio come from :func:`wdtk.pumps.scale_curve_speed`
    (affinity laws); operating points from :func:`wdtk.pumps.operating_point`.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    curve : PumpCurve
        Pump curve at full speed (speed ratio 1.0).
    system_head : callable
        Q (m^3/s) -> required system head (m).
    speeds : sequence of float, optional
        Speed ratios N/N_rated to draw, highest first.
    flow_unit : str, optional
        Flow unit for the x-axis (any unit in ``wdtk.units.FLOW_TO_M3S``; default L/s).

    Returns
    -------
    list of tuple of float
        Operating points (flow m^3/s, head m), one per speed.
    """
    q_max = curve.max_flow_m3s
    q = np.linspace(0.0, q_max, 300)
    colors = ordinal_colors(len(speeds))[::-1]  # darkest = full speed
    points: list[tuple[float, float]] = []
    for s, color in zip(speeds, colors, strict=True):
        c_s = pm.scale_curve_speed(curve, s)
        q_s = q[q <= c_s.max_flow_m3s]
        ax.plot(
            flow_from_m3s(q_s, flow_unit),
            pm.pump_head_m(c_s, q_s),
            color=color,
            label=f"Pump at {s:.0%} speed",
        )
        q_op, h_op = pm.operating_point(_head_function(c_s), system_head)
        points.append((q_op, h_op))

    h_sys = np.array([system_head(float(x)) for x in q])
    ax.plot(
        flow_from_m3s(q, flow_unit),
        h_sys,
        color=INK_SECONDARY,
        linestyle="--",
        label="System curve",
    )

    # Operating points: markers on the plot, values in a table in the upper-right corner
    # (labels beside the markers collide with the pump curves, which rise to their left).
    q_col = f"Q ({flow_unit})"
    rows = ["Operating points", f"{'speed':>5}  {q_col:>9}  {'H (m)':>6}"]
    for (q_op, h_op), s in zip(points, speeds, strict=True):
        x = flow_from_m3s(q_op, flow_unit)
        ax.plot(x, h_op, marker="o", ms=8, mfc=INK, mec=SURFACE, mew=2, zorder=5)
        rows.append(f"{s:>5.0%}  {x:>9.1f}  {h_op:>6.1f}")
    ax.text(
        0.98,
        0.97,
        "\n".join(rows),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        family="monospace",
        color=INK_SECONDARY,
        zorder=6,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": SURFACE, "edgecolor": GRID},
    )

    ax.set_xlim(0.0, flow_from_m3s(q_max, flow_unit))
    ax.set_ylim(0.0, curve.a_m * 1.08)
    ax.set_xlabel(f"Flow Q ({flow_unit})")
    ax.set_ylabel("Head H (m)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return points
