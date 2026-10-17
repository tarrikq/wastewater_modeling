"""4c Pipe sizing: greedy upsizing to remove fire flow and peak-hour pressure deficits.

THIS IS A HEURISTIC, NOT A GLOBAL OPTIMUM. Each iteration tries a one-size increase (to the
next commercial nominal diameter) for every candidate pipe near a deficient junction, and
commits the single change with the largest deficit reduction per dollar. It stops when no
deficit remains, no candidate helps, or the iteration cap is reached. Greedy choices are never
revisited, so a cheaper combination of improvements may exist.

Deficit measure (m of head, summed):

* fire: for each fire-deficient junction, apply the required fire flow in the max-day PDA
  snapshot and take max(0, residual - lowest service pressure);
* peak hour: in the peak-hour snapshot (demand-driven), sum max(0, min service pressure - p)
  over demand junctions.

Cost of a proposed pipe: length x unit cost of the proposed size x (1 + contingency), i.e. a
full replacement. Successive upsizes of the same pipe are scored by their incremental cost.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import pandas as pd

from wdtk.analysis.assumptions import FireFlowSettings, ServiceCriteria, SizingSettings
from wdtk.analysis.fire_flow import eligible_junctions
from wdtk.analysis.simulate import EpanetSession, WaterNetworkModel

SIZE_MATCH_RTOL = 0.05
"""Diameters within 5% of a nominal size count as that size (e.g. 150 mm is 6 in)."""


def steady_snapshot_at(wn: WaterNetworkModel, results: Any, time_s: float) -> WaterNetworkModel:
    """Return a demand-driven steady copy of ``wn`` reproducing an EPS instant.

    Junction demands are set to the EPS demands at ``time_s`` (patterns removed), tank initial
    levels to their EPS levels, and pump and pipe statuses to their EPS statuses. Controls are
    removed so the snapshot cannot switch links.

    Parameters
    ----------
    wn : WaterNetworkModel
        Model that produced ``results``.
    results : wntr SimulationResults
        EPS results.
    time_s : float
        Instant to reproduce, s.

    Returns
    -------
    WaterNetworkModel
        Steady (duration 0) copy.
    """
    out = copy.deepcopy(wn)
    demand = results.node["demand"].loc[time_s]
    for name, j in out.junctions():
        for i, ts in enumerate(j.demand_timeseries_list):
            ts.base_value = float(demand[name]) if i == 0 else 0.0
            ts.pattern_name = None
    head = results.node["head"].loc[time_s]
    for name, t in out.tanks():
        t.init_level = float(head[name]) - t.elevation
    status = results.link["status"].loc[time_s]
    for name in list(out.control_name_list):
        out.remove_control(name)
    for name, link in out.links():
        if link.link_type in ("Pipe", "Pump"):
            link.initial_status = "Open" if float(status[name]) > 0 else "Closed"
    out.options.time.duration = 0
    return out


def next_size_m(diameter_m: float, sizes_m: tuple[float, ...]) -> float | None:
    """Next commercial size above the nominal size of ``diameter_m`` (None if none).

    A diameter within ``SIZE_MATCH_RTOL`` of a nominal size is treated as that size, so a
    150 mm (5.9 in) pipe is a 6 in pipe and its next size is 8 in, not a 1.6% "upsize".

    Parameters
    ----------
    diameter_m : float
        Current diameter, m.
    sizes_m : tuple of float
        Commercial nominal diameters, m, ascending.

    Returns
    -------
    float or None
        Next size, m.
    """
    for d in sizes_m:
        if d > diameter_m * (1.0 + SIZE_MATCH_RTOL):  # strictly the next nominal size
            return d
    return None


def candidate_pipes(wn: WaterNetworkModel, junctions: list[str], hops: int) -> list[str]:
    """Pipes with an end node within ``hops`` links of any of ``junctions``.

    Parameters
    ----------
    wn : WaterNetworkModel
        Model.
    junctions : list of str
        Deficient junctions.
    hops : int
        Search radius in links.

    Returns
    -------
    list of str
        Pipe names (sorted).
    """
    g = wn.to_graph().to_undirected()
    near: set[str] = set()
    for j in junctions:
        near.update(nx.single_source_shortest_path_length(g, j, cutoff=hops))
    return sorted(
        name for name, p in wn.pipes() if p.start_node_name in near or p.end_node_name in near
    )


@dataclass
class _Evaluator:
    fire_session: EpanetSession
    peak_session: EpanetSession
    fire_junctions: list[str]
    service_nodes: list[str]
    fire: FireFlowSettings
    service: ServiceCriteria
    base: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base = {j: self.fire_session.base_demand_m3s(j) for j in self.fire_junctions}

    def set_diameter(self, pipe: str, diameter_m: float) -> None:
        self.fire_session.set_diameter(pipe, diameter_m)
        self.peak_session.set_diameter(pipe, diameter_m)

    def deficits(self) -> tuple[dict[str, float], dict[str, float]]:
        fire_def: dict[str, float] = {}
        for j in self.fire_junctions:
            self.fire_session.set_base_demand(j, self.base[j] + self.fire.required_flow_m3s)
            try:
                p = self.fire_session.solve_steady().loc[self.service_nodes, "pressure_m"]
            finally:
                self.fire_session.set_base_demand(j, self.base[j])
            fire_def[j] = max(0.0, self.fire.residual_pressure_m - float(p.min()))
        p_peak = self.peak_session.solve_steady().loc[self.service_nodes, "pressure_m"]
        short = (self.service.min_pressure_m - p_peak).clip(lower=0.0)
        return fire_def, {n: float(v) for n, v in short.items() if v > 0.0}

    def total(self) -> float:
        fire_def, peak_def = self.deficits()
        return sum(fire_def.values()) + sum(peak_def.values())


def greedy_upsizing(
    fire_snapshot: WaterNetworkModel,
    peak_snapshot: WaterNetworkModel,
    fire_deficient: list[str],
    fire: FireFlowSettings,
    service: ServiceCriteria,
    sizing: SizingSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy benefit-per-cost pipe upsizing (heuristic; see module docstring).

    Parameters
    ----------
    fire_snapshot : WaterNetworkModel
        Max-day PDA snapshot used for fire flow.
    peak_snapshot : WaterNetworkModel
        Peak-hour demand-driven snapshot (see :func:`steady_snapshot_at`).
    fire_deficient : list of str
        Junctions that fail the fire flow requirement before improvements.
    fire : FireFlowSettings
        Fire flow criteria.
    service : ServiceCriteria
        Minimum service pressure for the peak-hour check.
    sizing : SizingSettings
        Sizes, unit costs, contingency, search radius, iteration cap.

    Returns
    -------
    tuple of DataFrame
        (improvements: index pipe; columns ``length_m``, ``existing_diameter_m``,
        ``proposed_diameter_m``, ``cost``;
        history: one row per iteration with the pipe changed, deficit after, and cost).
    """
    proposed: dict[str, float] = {}
    original = {name: p.diameter for name, p in fire_snapshot.pipes()}
    lengths = {name: p.length for name, p in fire_snapshot.pipes()}
    service_nodes = eligible_junctions(fire_snapshot)
    scale = 1.0 + sizing.contingency

    def cost(pipe: str, d: float) -> float:
        return float(lengths[pipe]) * sizing.unit_cost_per_m[d] * scale

    history: list[dict[str, object]] = []
    with EpanetSession(fire_snapshot) as fs, EpanetSession(peak_snapshot) as ps:
        ev = _Evaluator(fs, ps, list(fire_deficient), service_nodes, fire, service)
        fire_def, peak_def = ev.deficits()
        current = sum(fire_def.values()) + sum(peak_def.values())
        history.append({"iteration": 0, "pipe": None, "deficit_m": current, "cost": 0.0})
        for it in range(1, sizing.max_iterations + 1):
            if current <= 0.0:
                break
            targets = [j for j, v in fire_def.items() if v > 0.0] + list(peak_def)
            best: tuple[float, str, float, float] | None = None
            for pipe in candidate_pipes(fire_snapshot, targets, sizing.candidate_hops):
                d_now = proposed.get(pipe, original[pipe])
                d_new = next_size_m(d_now, sizing.diameters_m)
                if d_new is None:
                    continue
                increment = cost(pipe, d_new) - (cost(pipe, d_now) if pipe in proposed else 0.0)
                ev.set_diameter(pipe, d_new)
                trial = ev.total()
                ev.set_diameter(pipe, d_now)
                score = (current - trial) / increment
                if trial < current and (best is None or score > best[0]):
                    best = (score, pipe, d_new, trial)
            if best is None:
                break
            _, pipe, d_new, _ = best
            ev.set_diameter(pipe, d_new)
            proposed[pipe] = d_new
            fire_def, peak_def = ev.deficits()
            current = sum(fire_def.values()) + sum(peak_def.values())
            history.append(
                {
                    "iteration": it,
                    "pipe": pipe,
                    "deficit_m": current,
                    "cost": sum(cost(p, d) for p, d in proposed.items()),
                }
            )

    improvements = pd.DataFrame(
        {
            "length_m": {p: lengths[p] for p in proposed},
            "existing_diameter_m": {p: original[p] for p in proposed},
            "proposed_diameter_m": dict(proposed),
            "cost": {p: cost(p, d) for p, d in proposed.items()},
        }
    )
    return improvements, pd.DataFrame(history)


def apply_improvements(wn: WaterNetworkModel, improvements: pd.DataFrame) -> WaterNetworkModel:
    """Return a copy of ``wn`` with the proposed diameters applied.

    Parameters
    ----------
    wn : WaterNetworkModel
        Model.
    improvements : DataFrame
        Output of :func:`greedy_upsizing` (index pipe, column ``proposed_diameter_m``).

    Returns
    -------
    WaterNetworkModel
        Modified copy.
    """
    out = copy.deepcopy(wn)
    for pipe, row in improvements.iterrows():
        out.get_link(pipe).diameter = float(row["proposed_diameter_m"])
    return out
