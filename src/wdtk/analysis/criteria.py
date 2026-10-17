"""4a Baseline extended-period simulation: service-level compliance.

Checks an EPS result against the service criteria in the assumptions file:

* minimum and maximum service pressure at junctions with demand (base demand > 0):
  connection nodes with no demand, such as those next to tanks and pumps, are not service
  points (their extremes are still reported, but not counted as violations);
* maximum pipe velocity.

Counts are in reported time steps (Net3 reports hourly). All values are SI.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from wdtk.analysis.assumptions import ServiceCriteria
from wdtk.analysis.simulate import WaterNetworkModel


def junction_compliance(
    results: Any, wn: WaterNetworkModel, service: ServiceCriteria
) -> pd.DataFrame:
    """Per-junction pressure extremes and steps outside the service range.

    Parameters
    ----------
    results : wntr SimulationResults
        EPS results (pressure head in m).
    wn : WaterNetworkModel
        The simulated model.
    service : ServiceCriteria
        Pressure limits, m of head.

    Returns
    -------
    DataFrame
        Index: junction. Columns: ``elevation_m``, ``base_demand_m3s``, ``serves_demand``,
        ``min_pressure_m``, ``max_pressure_m``, ``steps_below_min`` and ``steps_above_max``
        (both counted only where ``serves_demand``).
    """
    names = list(wn.junction_name_list)
    p = results.node["pressure"][names]
    base = pd.Series(
        {n: sum(ts.base_value for ts in wn.get_node(n).demand_timeseries_list) for n in names}
    )
    serves = base > 0.0
    below = (p < service.min_pressure_m).sum()
    return pd.DataFrame(
        {
            "elevation_m": pd.Series({n: wn.get_node(n).elevation for n in names}),
            "base_demand_m3s": base,
            "serves_demand": serves,
            "min_pressure_m": p.min(),
            "max_pressure_m": p.max(),
            "steps_below_min": below.where(serves, 0).astype(int),
            "steps_above_max": (p > service.max_pressure_m).sum().where(serves, 0).astype(int),
        }
    )


def pipe_compliance(results: Any, wn: WaterNetworkModel, service: ServiceCriteria) -> pd.DataFrame:
    """Per-pipe maximum velocity and steps above the velocity limit.

    Parameters
    ----------
    results : wntr SimulationResults
        EPS results (velocity in m/s).
    wn : WaterNetworkModel
        The simulated model.
    service : ServiceCriteria
        Velocity limit, m/s.

    Returns
    -------
    DataFrame
        Index: pipe. Columns: ``diameter_m``, ``length_m``, ``max_velocity_ms``,
        ``steps_above_max``.
    """
    names = list(wn.pipe_name_list)
    v = results.link["velocity"][names].abs()
    return pd.DataFrame(
        {
            "diameter_m": pd.Series({n: wn.get_link(n).diameter for n in names}),
            "length_m": pd.Series({n: wn.get_link(n).length for n in names}),
            "max_velocity_ms": v.max(),
            "steps_above_max": (v > service.max_velocity_ms).sum().astype(int),
        }
    )


def tank_levels(results: Any, wn: WaterNetworkModel) -> pd.DataFrame:
    """Tank water levels over time, m above tank bottom.

    level = head - elevation (the tank bottom elevation).

    Parameters
    ----------
    results : wntr SimulationResults
        EPS results.
    wn : WaterNetworkModel
        The simulated model.

    Returns
    -------
    DataFrame
        Index: time, s. Columns: tank names.
    """
    names = list(wn.tank_name_list)
    head = results.node["head"][names]
    return head - pd.Series({n: wn.get_node(n).elevation for n in names})


def peak_hour_s(results: Any, wn: WaterNetworkModel) -> float:
    """Time of the highest total junction demand in an EPS result, s.

    Parameters
    ----------
    results : wntr SimulationResults
        EPS results (demand in m^3/s).
    wn : WaterNetworkModel
        The simulated model.

    Returns
    -------
    float
        Simulation time of the peak, s.
    """
    total = results.node["demand"][list(wn.junction_name_list)].sum(axis=1)
    return float(total.idxmax())


def summarize(junctions: pd.DataFrame, pipes: pd.DataFrame) -> dict[str, float]:
    """Network-level compliance summary.

    Parameters
    ----------
    junctions : DataFrame
        Output of :func:`junction_compliance`.
    pipes : DataFrame
        Output of :func:`pipe_compliance`.

    Returns
    -------
    dict
        Counts of non-compliant elements and the extremes found.
    """
    served = junctions[junctions["serves_demand"]]
    return {
        "junctions_checked_min": len(served),
        "junctions_below_min": int((served["steps_below_min"] > 0).sum()),
        "junctions_above_max": int((junctions["steps_above_max"] > 0).sum()),
        "pipes_above_max_velocity": int((pipes["steps_above_max"] > 0).sum()),
        "lowest_service_pressure_m": float(served["min_pressure_m"].min()),
        "highest_pressure_m": float(served["max_pressure_m"].max()),
        "highest_velocity_ms": float(pipes["max_velocity_ms"].max()),
    }
