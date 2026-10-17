"""Computation and Plotly figures for the Streamlit dashboard (``app/streamlit_app.py``).

The app file only lays out widgets; everything it shows is produced here, so it can be tested
without a browser. Values are converted to US customary units for display (psi, ft/s, gpm,
ft, hp), matching the Phase 4 reports.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from wdtk import plotting, units
from wdtk import pumps as pm
from wdtk.analysis import criteria, criticality, fire_flow, water_age
from wdtk.analysis.assumptions import Assumptions
from wdtk.analysis.simulate import (
    load_network,
    net3_path,
    run_eps,
    set_pressure_dependent,
    steady_max_day,
)

HOUR = 3600.0
NETWORKS = {
    "Net3 (study network)": net3_path(),
    "Net1": net3_path().with_name("Net1.inp"),
}
"""Networks offered in the dashboard (EPANET examples bundled with WNTR 1.5.0)."""

LAYERS = {
    "Minimum pressure (psi)": ("node", "min_pressure_psi", True),
    "Maximum velocity (ft/s)": ("link", "max_velocity_fps", False),
    "Maximum water age (h)": ("node", "max_age_h", False),
    "Criticality: demand not served when closed (%)": ("link", "unserved_pct", False),
}
"""Map layers: (element type, column, low values are the concern)."""

N_COLOR_BINS = 5


@dataclass(frozen=True, eq=False)
class NetworkResults:
    """Everything the dashboard shows for one network (display units).

    Attributes
    ----------
    wn : wntr WaterNetworkModel
        The network.
    nodes : DataFrame
        Index junction: ``min_pressure_psi``, ``max_age_h``, ``serves_demand``.
    links : DataFrame
        Index pipe: ``diameter_in``, ``max_velocity_fps``, ``unserved_pct`` (NaN if not
        screened).
    fire : DataFrame
        Index junction: ``available_gpm``, ``required_gpm``, ``deficit_gpm``,
        ``meets_requirement``, ``governing_limit``, ``limiting_junction``.
    """

    wn: Any
    nodes: pd.DataFrame
    links: pd.DataFrame
    fire: pd.DataFrame


def compute_results(network_path: Path, a: Assumptions) -> NetworkResults:
    """Run the dashboard analyses for one network (about 1 s for Net3).

    Baseline EPS (pressure, velocity), water age, max-day fire flow, and criticality, all with
    the methods and assumptions of the Phase 4 study.

    Parameters
    ----------
    network_path : Path
        EPANET input file.
    a : Assumptions
        Study assumptions.

    Returns
    -------
    NetworkResults
        Tables in display units.
    """
    wn = load_network(network_path)
    eps = run_eps(wn, a.eps_duration_s)
    j = criteria.junction_compliance(eps, wn, a.service)
    p = criteria.pipe_compliance(eps, wn, a.service)
    age = water_age.junction_age(
        run_eps(wn, a.water_age.duration_s, quality="AGE"), wn, a.water_age
    )

    snap = steady_max_day(wn, a.max_day_factor)
    set_pressure_dependent(snap, a.pdd)
    fire = fire_flow.fire_flow_table(snap, a.fire_flow)

    pda = copy.deepcopy(wn)
    set_pressure_dependent(pda, a.pdd)
    crit, _ = criticality.criticality_table(pda, a.criticality, a.service)

    nodes = pd.DataFrame(
        {
            "min_pressure_psi": units.head_to_pressure(j["min_pressure_m"], "m", "psi"),
            "max_age_h": age["max_age_s"] / HOUR,
            "serves_demand": j["serves_demand"],
        }
    )
    links = pd.DataFrame(
        {
            "diameter_in": units.length_from_m(p["diameter_m"], "in"),
            "max_velocity_fps": units.length_from_m(p["max_velocity_ms"], "ft"),
            "unserved_pct": crit["unserved_pct"].reindex(p.index),
        }
    )
    fire_us = pd.DataFrame(
        {
            "available_gpm": units.flow_from_m3s(fire["available_m3s"], "gpm"),
            "required_gpm": units.flow_from_m3s(fire["required_m3s"], "gpm"),
            "deficit_gpm": units.flow_from_m3s(fire["deficit_m3s"], "gpm"),
            "meets_requirement": fire["meets_requirement"],
            "governing_limit": fire["governing_limit"],
            "limiting_junction": fire["limiting_junction"],
        }
    )
    return NetworkResults(wn, nodes, links, fire_us)


# --- Network map ----------------------------------------------------------------------------
def bin_edges(values: pd.Series, n_bins: int = N_COLOR_BINS) -> np.ndarray:
    """Equal-width class edges spanning the finite values (for binned link colors).

    Parameters
    ----------
    values : Series
        Values to classify.
    n_bins : int, optional
        Number of classes.

    Returns
    -------
    ndarray
        ``n_bins + 1`` edges, ascending.
    """
    v = values[np.isfinite(values)]
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def _ramp(n: int, reverse: bool) -> list[str]:
    colors = list(plotting.SEQUENTIAL[::-1] if reverse else plotting.SEQUENTIAL)
    idx = np.round(np.linspace(0, len(colors) - 1, n)).astype(int)
    return [colors[i] for i in idx]


def network_map_figure(results: NetworkResults, layer: str) -> go.Figure:
    """Build a Plotly network map colored by one layer, with hover values.

    Node layers color junction markers on a continuous scale. Link layers draw pipes in
    ``N_COLOR_BINS`` equal-width classes (Plotly lines cannot vary color along one trace),
    with invisible midpoint markers carrying the hover text.

    Parameters
    ----------
    results : NetworkResults
        Output of :func:`compute_results`.
    layer : str
        A key of ``LAYERS``.

    Returns
    -------
    plotly Figure
        The map.
    """
    kind, column, reverse = LAYERS[layer]
    wn = results.wn
    xy = {n: tuple(wn.get_node(n).coordinates) for n in wn.node_name_list}
    fig = go.Figure()

    def seg(link: str) -> tuple[tuple[float, float], tuple[float, float]]:
        lk = wn.get_link(link)
        return xy[lk.start_node_name], xy[lk.end_node_name]

    base_x: list[float | None] = []
    base_y: list[float | None] = []
    for link in wn.link_name_list:
        (x0, y0), (x1, y1) = seg(link)
        base_x += [x0, x1, None]
        base_y += [y0, y1, None]
    fig.add_trace(
        go.Scatter(
            x=base_x,
            y=base_y,
            mode="lines",
            hoverinfo="skip",
            line={"color": plotting.GRID, "width": 1},
            showlegend=False,
        )
    )

    if kind == "link":
        values = results.links[column].dropna()
        edges = bin_edges(values)
        colors = _ramp(N_COLOR_BINS, reverse)
        classes = np.clip(
            np.searchsorted(edges, values.to_numpy(), side="right") - 1, 0, N_COLOR_BINS - 1
        )
        for b in range(N_COLOR_BINS):
            members = values.index[classes == b]
            xs: list[float | None] = []
            ys: list[float | None] = []
            for link in members:
                (x0, y0), (x1, y1) = seg(link)
                xs += [x0, x1, None]
                ys += [y0, y1, None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    hoverinfo="skip",
                    line={"color": colors[b], "width": 3},
                    name=f"{edges[b]:.1f} to {edges[b + 1]:.1f}",
                    legendgroup="bins",
                    legendgrouptitle_text=layer if b == 0 else None,
                )
            )
        mids = [tuple(np.mean(seg(k), axis=0)) for k in values.index]
        fig.add_trace(
            go.Scatter(
                x=[m[0] for m in mids],
                y=[m[1] for m in mids],
                mode="markers",
                marker={"size": 10, "opacity": 0},
                showlegend=False,
                customdata=np.column_stack([values.index, values.to_numpy()]),
                hovertemplate="Pipe %{customdata[0]}<br>" + layer + ": %{customdata[1]:.2f}"
                "<extra></extra>",
            )
        )
        junctions = list(wn.junction_name_list)
        fig.add_trace(
            go.Scatter(
                x=[xy[n][0] for n in junctions],
                y=[xy[n][1] for n in junctions],
                mode="markers",
                marker={"size": 4, "color": plotting.MUTED},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
        values = results.nodes.loc[results.nodes["serves_demand"], column].dropna()
        fig.add_trace(
            go.Scatter(
                x=[xy[n][0] for n in values.index],
                y=[xy[n][1] for n in values.index],
                mode="markers",
                showlegend=False,
                marker={
                    "size": 10,
                    "color": values.to_numpy(),
                    "colorscale": [[i / 4, c] for i, c in enumerate(_ramp(5, reverse))],
                    "colorbar": {"title": {"text": layer}},
                    "line": {"width": 1, "color": plotting.SURFACE},
                },
                customdata=np.column_stack([values.index, values.to_numpy()]),
                hovertemplate="Junction %{customdata[0]}<br>" + layer + ": %{customdata[1]:.1f}"
                "<extra></extra>",
            )
        )

    for kind_name, names, symbol in (
        ("Tank", list(wn.tank_name_list), "square"),
        ("Reservoir", list(wn.reservoir_name_list), "triangle-up"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[xy[n][0] for n in names],
                y=[xy[n][1] for n in names],
                mode="markers+text",
                text=[f"{kind_name} {n}" for n in names],
                textposition="top right",
                marker={"symbol": symbol, "size": 12, "color": plotting.INK},
                textfont={"size": 10, "color": plotting.INK_SECONDARY},
                hovertemplate=kind_name + " %{text}<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        plot_bgcolor=plotting.SURFACE,
        paper_bgcolor=plotting.SURFACE,
        height=620,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        legend={"orientation": "v", "x": 1.0, "y": 1.0},
    )
    return fig


def _gpm(q_m3s: Any) -> Any:
    return units.flow_from_m3s(q_m3s, "gpm")


def _ft(x_m: Any) -> Any:
    return units.length_from_m(x_m, "ft")


# --- Pump curve explorer --------------------------------------------------------------------
ILLUSTRATIVE_PUMP = "Phase 2 illustrative pump"


def pump_curves(results: NetworkResults) -> dict[str, pm.PumpCurve]:
    """Pump curves available in the explorer: the network's pumps and the Phase 2 example.

    Network pump coefficients come from WNTR's EPANET-convention fit, which Phase 2 showed
    equal to :mod:`wdtk.pumps` fits.

    Parameters
    ----------
    results : NetworkResults
        Network results (for the network's pumps).

    Returns
    -------
    dict
        Label -> PumpCurve (SI).
    """
    curves = {}
    for name, pump in results.wn.head_pumps():
        a, b, c = pump.get_head_curve_coefficients()
        curves[f"Pump {name}"] = pm.PumpCurve(float(a), float(b), float(c))
    curves[ILLUSTRATIVE_PUMP] = pm.fit_three_point(60.0, 0.10, 45.0, 0.22)
    return curves


@dataclass(frozen=True)
class PumpPoint:
    """Operating point in display units (None fields if the pump cannot deliver)."""

    flow_gpm: float | None
    head_ft: float | None
    hydraulic_hp: float | None
    input_hp: float | None


def pump_explorer(
    curve: pm.PumpCurve,
    speed: float,
    static_head_ft: float,
    length_ft: float,
    diameter_in: float,
    c_factor: float,
    efficiency: float,
) -> tuple[go.Figure, PumpPoint]:
    """Pump curve at a speed ratio, a Hazen-Williams system curve, and the operating point.

    Parameters
    ----------
    curve : PumpCurve
        Full-speed curve (SI).
    speed : float
        Speed ratio N / N_rated.
    static_head_ft, length_ft : float
        System static head and equivalent pipe length, ft.
    diameter_in : float
        Equivalent pipe diameter, in.
    c_factor : float
        Hazen-Williams C.
    efficiency : float
        Wire-to-water efficiency for input power.

    Returns
    -------
    tuple
        (figure in gpm and ft; operating point).
    """
    scaled = pm.scale_curve_speed(curve, speed)
    static_m = units.length_to_m(static_head_ft, "ft")
    length_m = units.length_to_m(length_ft, "ft")
    diameter_m = units.length_to_m(diameter_in, "in")

    def system(q: float) -> float:
        return pm.system_head_hw_m(q, static_m, length_m, diameter_m, c_factor)

    q = np.linspace(0.0, curve.max_flow_m3s, 200)
    q_s = q[q <= scaled.max_flow_m3s]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_gpm(q),
            y=_ft(pm.pump_head_m(curve, q)),
            mode="lines",
            name="Pump at 100% speed",
            line={"color": plotting.MUTED, "dash": "dot", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_gpm(q_s),
            y=_ft(pm.pump_head_m(scaled, q_s)),
            mode="lines",
            name=f"Pump at {speed:.0%} speed",
            line={"color": plotting.CATEGORICAL[0], "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_gpm(q),
            y=_ft(np.array([system(float(x)) for x in q])),
            mode="lines",
            name="System curve",
            line={"color": plotting.INK_SECONDARY, "dash": "dash", "width": 2},
        )
    )
    point = PumpPoint(None, None, None, None)
    try:
        q_op, h_op = pm.operating_point(lambda x: pm.pump_head_m(scaled, x), system)
    except ValueError:
        pass
    else:
        p_w = pm.hydraulic_power_w(q_op, h_op)
        point = PumpPoint(
            _gpm(q_op),
            _ft(h_op),
            units.power_from_w(p_w, "hp"),
            units.power_from_w(p_w / efficiency, "hp"),
        )
        fig.add_trace(
            go.Scatter(
                x=[point.flow_gpm],
                y=[point.head_ft],
                mode="markers",
                name="Operating point",
                marker={
                    "size": 12,
                    "color": plotting.INK,
                    "line": {"width": 2, "color": plotting.SURFACE},
                },
                hovertemplate="%{x:,.0f} gpm at %{y:.1f} ft<extra></extra>",
            )
        )
    fig.update_layout(
        xaxis_title="Flow (gpm)",
        yaxis_title="Head (ft)",
        height=480,
        plot_bgcolor=plotting.SURFACE,
        paper_bgcolor=plotting.SURFACE,
        yaxis={"range": [0, _ft(curve.a_m) * 1.1], "gridcolor": plotting.GRID},
        xaxis={"gridcolor": plotting.GRID},
        margin={"t": 30},
        legend={"x": 1.0, "xanchor": "right", "y": 1.0},
    )
    return fig, point


# --- Fire flow lookup and assumptions -----------------------------------------------------------
def fire_lookup(results: NetworkResults, junction: str) -> dict[str, Any]:
    """Fire flow result for one junction (display units).

    Parameters
    ----------
    results : NetworkResults
        Network results.
    junction : str
        Junction name (must be fire-flow eligible).

    Returns
    -------
    dict
        Columns of the fire table for that junction.
    """
    return dict(results.fire.loc[junction])


def assumptions_table(a: Assumptions) -> pd.DataFrame:
    """Every assumption with its value, units, basis, and status (as written in the file).

    Parameters
    ----------
    a : Assumptions
        Loaded assumptions.

    Returns
    -------
    DataFrame
        Index: dotted name. Columns: ``value``, ``units``, ``basis``, ``status``.
    """
    return pd.DataFrame(
        {
            name: {"value": str(e.value), "units": e.units, "basis": e.basis, "status": e.status}
            for name, e in a.entries.items()
        }
    ).T
