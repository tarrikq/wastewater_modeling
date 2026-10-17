"""4f Criticality: consequence of losing each large pipe.

Each candidate pipe (diameter at or above the screening threshold) is closed for a full
extended-period run under pressure-dependent demand (PDA). Consequences, relative to the same
PDA run with the pipe open:

* demand not served: sum over junctions and steps of (baseline delivered - delivered) * dt,
  as a volume and as a percentage of baseline delivered volume;
* junctions newly below the minimum service pressure at any step (demand junctions only).

Pipes named in a control action are skipped (the control could reopen them during the run)
and reported as excluded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wdtk.analysis.assumptions import CriticalitySettings, ServiceCriteria
from wdtk.analysis.fire_flow import eligible_junctions
from wdtk.analysis.simulate import EpanetSession, WaterNetworkModel


def controlled_links(wn: WaterNetworkModel) -> set[str]:
    """Links that are the target of any control action.

    Parameters
    ----------
    wn : WaterNetworkModel
        Model.

    Returns
    -------
    set of str
        Link names.
    """
    out: set[str] = set()
    for name in wn.control_name_list:
        for action in wn.get_control(name).actions():
            out.add(action.target()[0].name)
    return out


def _run(
    session: EpanetSession, duration_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    times: list[float] = []
    pressures: list[NDArray[np.float64]] = []
    demands: list[NDArray[np.float64]] = []

    def record(t: float, p: NDArray[np.float64], d: NDArray[np.float64]) -> None:
        times.append(t)
        pressures.append(p)
        demands.append(d)

    session.run_eps(duration_s, record)
    t = np.array(times)
    dt = np.append(np.diff(t), 0.0)  # volume over [t_i, t_i+1); the end instant has zero width
    return dt, np.array(pressures), np.array(demands)


def criticality_table(
    wn_pda: WaterNetworkModel,
    crit: CriticalitySettings,
    service: ServiceCriteria,
) -> tuple[pd.DataFrame, list[str]]:
    """Rank large pipes by the demand left unserved when each is closed.

    Parameters
    ----------
    wn_pda : WaterNetworkModel
        Model with patterns and controls, already in PDA mode.
    crit : CriticalitySettings
        Diameter threshold (m) and run length (s).
    service : ServiceCriteria
        Minimum service pressure (m) for the low-pressure count.

    Returns
    -------
    tuple
        (DataFrame indexed by pipe, sorted by unserved volume descending, with columns
        ``diameter_m``, ``length_m``, ``unserved_m3``, ``unserved_pct``,
        ``junctions_below_min``; list of excluded control-target pipes).
    """
    excluded_targets = controlled_links(wn_pda)
    candidates = [
        name
        for name, p in wn_pda.pipes()
        if p.diameter >= crit.min_diameter_m * (1 - 1e-9) and name not in excluded_targets
    ]
    excluded = sorted(
        name
        for name, p in wn_pda.pipes()
        if p.diameter >= crit.min_diameter_m * (1 - 1e-9) and name in excluded_targets
    )
    service_nodes = set(eligible_junctions(wn_pda))
    rows = {}
    with EpanetSession(wn_pda) as session:
        mask = np.array([n in service_nodes for n in session.junction_names])
        dt0, p0, d0 = _run(session, crit.duration_s)
        base_volume = float((d0 * dt0[:, None]).sum())
        base_low = (p0[:, mask] < service.min_pressure_m).any(axis=0)
        for pipe in candidates:
            session.set_open(pipe, False)
            try:
                dt, p, d = _run(session, crit.duration_s)
            finally:
                session.set_open(pipe, True)
            # Event times can differ from the baseline run, so integrate each run on its own.
            unserved = base_volume - float((d * dt[:, None]).sum())
            low = (p[:, mask] < service.min_pressure_m).any(axis=0)
            link = wn_pda.get_link(pipe)
            rows[pipe] = {
                "diameter_m": link.diameter,
                "length_m": link.length,
                "unserved_m3": max(unserved, 0.0),
                "unserved_pct": 100.0 * max(unserved, 0.0) / base_volume,
                "junctions_below_min": int((low & ~base_low).sum()),
            }
    table = pd.DataFrame.from_dict(rows, orient="index")
    if len(table):
        table = table.sort_values(["unserved_m3", "junctions_below_min"], ascending=False)
    return table, excluded
