"""Figure helpers shared by scripts and notebooks.

Style: light print surface, recessive axes and grid, units on every axis, direct labels in
addition to legends. Colors are assigned by role: categorical slots in a fixed order for
unordered series, a single-hue ramp for ordered series (e.g. pump speeds).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from wdtk import pumps as pm
from wdtk.network.results import SolveResult
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


def plot_convergence(ax: Axes, results: Sequence[SolveResult], title: str = "") -> None:
    """Plot solver convergence histories on a log scale.

    The y-axis is EPANET's convergence measure, sum|dQ| / sum|Q| after each iteration (one
    Newton step for GGA, one sweep over all loops for Hardy Cross), so solvers are compared on
    the same quantity.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    results : sequence of SolveResult
        One result per solver; series are colored by categorical slot in the given order.
    title : str, optional
        Axes title.
    """
    for res, color in zip(results, CATEGORICAL, strict=False):
        it = [r.iteration for r in res.history]
        rel = [max(r.relative_flow_change, 1e-17) for r in res.history]
        ax.semilogy(it, rel, color=color, marker="o", ms=4, label=f"{res.method} ({len(it)})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative flow change, sum|dQ| / sum|Q| (-)")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, title="Solver (iterations)", title_fontsize=8)


SEQUENTIAL = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
"""Single-hue sequential ramp (blue, light to dark) for magnitudes on maps."""


def network_map(
    ax: Axes,
    wn: Any,
    node_values: pd.Series | None = None,
    link_values: pd.Series | None = None,
    *,
    label: str = "",
    reverse: bool = False,
    highlight_links: Sequence[str] = (),
    highlight_label: str = "",
    link_labels: Mapping[str, str] | None = None,
) -> None:
    """Draw a network map colored by node or link values.

    Links without a value are drawn as thin neutral lines; nodes without a value as small
    neutral dots. Tanks (squares) and reservoirs (triangles) are labeled. Coordinates are the
    model's map coordinates (arbitrary units), so axes are hidden.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    wn : wntr WaterNetworkModel
        Network with node coordinates.
    node_values, link_values : Series, optional
        Values to color (index: node or link names); give one or both.
    label : str, optional
        Colorbar label, with units.
    reverse : bool, optional
        Make low values darkest (use when low values are the concern, e.g. pressure).
    highlight_links : sequence of str, optional
        Links to overlay in the accent color (e.g. proposed improvements).
    highlight_label : str, optional
        Legend label for the highlighted links.
    link_labels : mapping of str to str, optional
        Text placed at the midpoint of selected links (direct labels).
    """
    colors = SEQUENTIAL[::-1] if reverse else SEQUENTIAL
    cmap = LinearSegmentedColormap.from_list("seq", colors)
    xy = {n: tuple(wn.get_node(n).coordinates) for n in wn.node_name_list}

    def segment(link: str) -> list[tuple[float, float]]:
        lk = wn.get_link(link)
        return [xy[lk.start_node_name], xy[lk.end_node_name]]

    all_links = list(wn.link_name_list)
    ax.add_collection(
        LineCollection([segment(k) for k in all_links], colors=GRID, linewidths=1.0, zorder=1)
    )
    mappable: ScalarMappable | None = None
    if link_values is not None:
        names = [k for k in link_values.index if k in xy or k in all_links]
        lc = LineCollection([segment(k) for k in names], cmap=cmap, linewidths=2.2, zorder=2)
        lc.set_array(np.asarray(link_values.loc[names], dtype=float))
        ax.add_collection(lc)
        mappable = lc
    if highlight_links:
        ax.add_collection(
            LineCollection(
                [segment(k) for k in highlight_links],
                colors=CATEGORICAL[1],
                linewidths=3.5,
                zorder=3,
                label=highlight_label or None,
            )
        )
    junctions = list(wn.junction_name_list)
    jx = np.array([xy[n] for n in junctions])
    if node_values is not None:
        names = [n for n in node_values.index if n in xy]
        pts = np.array([xy[n] for n in names])
        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=np.asarray(node_values.loc[names], dtype=float),
            cmap=cmap,
            s=30,
            edgecolors=SURFACE,
            linewidths=0.6,
            zorder=4,
        )
        mappable = sc
    else:
        ax.scatter(jx[:, 0], jx[:, 1], s=6, color=MUTED, zorder=4)
    for kind, names, marker in (
        ("Tank", list(wn.tank_name_list), "s"),
        ("Reservoir", list(wn.reservoir_name_list), "^"),
    ):
        for n in names:
            x, y = xy[n]
            ax.scatter([x], [y], marker=marker, s=70, color=INK, zorder=5)
            ax.annotate(
                f"{kind} {n}",
                (x, y),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=7,
                color=INK_SECONDARY,
            )
    for i, (link, text) in enumerate((link_labels or {}).items()):
        (x0, y0), (x1, y1) = segment(link)
        # Alternate above / below the midpoint so labels on adjacent short pipes separate.
        ax.annotate(
            text,
            ((x0 + x1) / 2, (y0 + y1) / 2),
            xytext=(8, 6 if i % 2 == 0 else -14),
            textcoords="offset points",
            fontsize=7,
            color=INK,
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": SURFACE,
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )
    if mappable is not None:
        cb = ax.figure.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(label)
        cb.outline.set_edgecolor(MUTED)
    if highlight_links and highlight_label:
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
