"""4b Fire flow analysis: available fire flow at each eligible junction.

Scenario: a max-day steady snapshot (see :func:`wdtk.analysis.simulate.steady_max_day`) in
pressure-dependent demand mode. A fire demand Q_f is added to the base demand of one junction
at a time. Q_f is feasible when all three hold:

1. the fire junction's pressure head is at least the minimum residual;
2. the fire junction receives at least 99.9% of its requested demand (PDA delivery);
3. no other demand junction falls below the minimum residual (system-wide check).

Feasibility decreases monotonically with Q_f, so the available fire flow (the largest
feasible Q_f) is found by bisection to the configured tolerance. The governing limit is the
criterion that fails first just above the available flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wdtk.analysis.assumptions import FireFlowSettings
from wdtk.analysis.simulate import EpanetSession, WaterNetworkModel

DELIVERY_FRACTION = 0.999


@dataclass(frozen=True)
class _Check:
    feasible: bool
    limit: str
    limiting_junction: str | None


def eligible_junctions(wn: WaterNetworkModel) -> list[str]:
    """Junctions eligible for a fire flow test: those with base demand > 0.

    Parameters
    ----------
    wn : WaterNetworkModel
        Model.

    Returns
    -------
    list of str
        Junction names.
    """
    return [
        name
        for name, j in wn.junctions()
        if sum(ts.base_value for ts in j.demand_timeseries_list) > 0.0
    ]


def _check(
    session: EpanetSession,
    junction: str,
    base_m3s: float,
    fire_m3s: float,
    residual_m: float,
    service_nodes: list[str],
) -> _Check:
    requested = base_m3s + fire_m3s
    session.set_base_demand(junction, requested)
    try:
        state = session.solve_steady()
    finally:
        session.set_base_demand(junction, base_m3s)
    if state.at[junction, "pressure_m"] < residual_m:
        return _Check(False, "residual at fire node", junction)
    if state.at[junction, "demand_m3s"] < DELIVERY_FRACTION * requested:
        return _Check(False, "delivery at fire node", junction)
    others = state.loc[[n for n in service_nodes if n != junction], "pressure_m"]
    if len(others) and others.min() < residual_m:
        return _Check(False, "residual elsewhere", str(others.idxmin()))
    return _Check(True, "", None)


def available_fire_flow(
    session: EpanetSession,
    junction: str,
    fire: FireFlowSettings,
    service_nodes: list[str],
) -> tuple[float, str, str | None]:
    """Largest fire flow a junction can supply while meeting the residual criteria.

    Parameters
    ----------
    session : EpanetSession
        Open session on the fire flow snapshot (PDA, max-day demands).
    junction : str
        Fire junction.
    fire : FireFlowSettings
        Residual pressure (m), search bound and tolerance (m^3/s).
    service_nodes : list of str
        Junctions included in the system-wide residual check.

    Returns
    -------
    tuple
        (available flow, m^3/s; governing limit; limiting junction or None).
    """
    base = session.base_demand_m3s(junction)
    residual = fire.residual_pressure_m

    def check(q: float) -> _Check:
        return _check(session, junction, base, q, residual, service_nodes)

    at_zero = check(0.0)
    if not at_zero.feasible:
        return 0.0, f"{at_zero.limit} (before fire flow)", at_zero.limiting_junction
    at_top = check(fire.upper_bound_m3s)
    if at_top.feasible:
        return fire.upper_bound_m3s, "search upper bound", None
    lo, hi, fail = 0.0, fire.upper_bound_m3s, at_top
    while hi - lo > fire.tolerance_m3s:
        mid = 0.5 * (lo + hi)
        c = check(mid)
        if c.feasible:
            lo = mid
        else:
            hi, fail = mid, c
    return lo, fail.limit, fail.limiting_junction


def fire_flow_table(
    snapshot: WaterNetworkModel,
    fire: FireFlowSettings,
    junctions: list[str] | None = None,
) -> pd.DataFrame:
    """Available fire flow at each eligible junction, compared with the requirement.

    Parameters
    ----------
    snapshot : WaterNetworkModel
        Max-day steady snapshot in PDA mode.
    fire : FireFlowSettings
        Fire flow criteria.
    junctions : list of str, optional
        Junctions to test (default: :func:`eligible_junctions`).

    Returns
    -------
    DataFrame
        Index: junction. Columns: ``available_m3s``, ``required_m3s``, ``deficit_m3s``,
        ``meets_requirement``, ``governing_limit``, ``limiting_junction``.
    """
    service = eligible_junctions(snapshot)
    tests = junctions if junctions is not None else service
    rows = {}
    with EpanetSession(snapshot) as session:
        for j in tests:
            q, limit, where = available_fire_flow(session, j, fire, service)
            rows[j] = {
                "available_m3s": q,
                "required_m3s": fire.required_flow_m3s,
                "deficit_m3s": max(fire.required_flow_m3s - q, 0.0),
                "meets_requirement": q >= fire.required_flow_m3s,
                "governing_limit": limit,
                "limiting_junction": where,
            }
    return pd.DataFrame.from_dict(rows, orient="index")
