"""4d Water age and tank turnover.

Water age comes from an EPANET AGE quality simulation (:func:`wdtk.analysis.simulate.run_eps`
with ``quality="AGE"``), long enough for tank ages to stabilize; ages are reported over a
final window. Tank turnover time is

    turnover = mean stored volume / mean daily fill volume,

with volume = A x level for Net3's cylindrical tanks (A = pi D^2 / 4) and the daily fill
volume the sum of positive volume changes per day over the window.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from wdtk.analysis.assumptions import WaterAgeSettings
from wdtk.analysis.criteria import tank_levels
from wdtk.analysis.simulate import WaterNetworkModel

SECONDS_PER_DAY = 86400.0


def _final_window(frame: pd.DataFrame, window_s: float) -> pd.DataFrame:
    end = float(frame.index.max())
    return frame[frame.index >= end - window_s]


def junction_age(results: Any, wn: WaterNetworkModel, age: WaterAgeSettings) -> pd.DataFrame:
    """Maximum and mean water age at each junction over the final reporting window.

    Parameters
    ----------
    results : wntr SimulationResults
        AGE quality results (age in s).
    wn : WaterNetworkModel
        The simulated model.
    age : WaterAgeSettings
        Threshold and reporting window, s.

    Returns
    -------
    DataFrame
        Index: junction. Columns: ``max_age_s``, ``mean_age_s``, ``above_threshold``.
    """
    q = _final_window(results.node["quality"][list(wn.junction_name_list)], age.report_window_s)
    out = pd.DataFrame({"max_age_s": q.max(), "mean_age_s": q.mean()})
    out["above_threshold"] = out["max_age_s"] > age.threshold_s
    return out


def tank_turnover(results: Any, wn: WaterNetworkModel, window_s: float) -> pd.DataFrame:
    """Tank turnover time and water age over a final window.

    Parameters
    ----------
    results : wntr SimulationResults
        EPS results (with AGE quality for the age columns).
    wn : WaterNetworkModel
        The simulated model.
    window_s : float
        Final window used for the averages, s (whole days recommended).

    Returns
    -------
    DataFrame
        Index: tank. Columns: ``mean_volume_m3``, ``daily_fill_m3``, ``turnover_days``,
        ``level_range_m`` (max - min level in the window), ``max_age_s``.
    """
    levels = _final_window(tank_levels(results, wn), window_s)
    rows = {}
    for name in levels.columns:
        tank = wn.get_node(name)
        area = np.pi * tank.diameter**2 / 4.0
        volume = levels[name] * area
        fill = volume.diff().clip(lower=0.0).sum()
        days = (float(levels.index.max()) - float(levels.index.min())) / SECONDS_PER_DAY
        daily_fill = float(fill) / days if days > 0 else float("nan")
        rows[name] = {
            "mean_volume_m3": float(volume.mean()),
            "daily_fill_m3": daily_fill,
            "turnover_days": float(volume.mean()) / daily_fill if daily_fill > 0 else np.inf,
            "level_range_m": float(levels[name].max() - levels[name].min()),
            "max_age_s": float(_final_window(results.node["quality"][[name]], window_s)[name].max())
            if "quality" in results.node
            else float("nan"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def age_stabilization(results: Any, wn: WaterNetworkModel, window_s: float) -> float:
    """Largest relative change in tank mean age between the last two windows.

    A small value (e.g. below 5%) indicates the simulation was long enough for ages to settle.

    Parameters
    ----------
    results : wntr SimulationResults
        AGE quality results.
    wn : WaterNetworkModel
        The simulated model.
    window_s : float
        Window length, s.

    Returns
    -------
    float
        max over tanks of |mean(last) - mean(previous)| / mean(last), dimensionless.
    """
    q = results.node["quality"][list(wn.tank_name_list)]
    end = float(q.index.max())
    last = q[q.index > end - window_s].mean()
    prev = q[(q.index > end - 2 * window_s) & (q.index <= end - window_s)].mean()
    return float(((last - prev).abs() / last).max())


def set_tank_level_controls(
    wn: WaterNetworkModel, tank: str, low_level_m: float, high_level_m: float
) -> WaterNetworkModel:
    """Return a copy of ``wn`` with a tank's level-control thresholds moved.

    Every control whose condition is "tank level below x" gets ``low_level_m`` and every
    "tank level above x" gets ``high_level_m``; the actions are unchanged. (WNTR exposes no
    public setter for a condition threshold, so the copy's ``_threshold`` is changed.)

    Parameters
    ----------
    wn : WaterNetworkModel
        Model (not modified).
    tank : str
        Tank name.
    low_level_m, high_level_m : float
        New switch levels above the tank bottom, m.

    Returns
    -------
    WaterNetworkModel
        Modified copy.

    Raises
    ------
    ValueError
        If no level control references the tank.
    """
    out = copy.deepcopy(wn)
    changed = 0
    for name in out.control_name_list:
        cond = out.get_control(name).condition
        if type(cond).__name__ != "TankLevelCondition" or cond._source_obj.name != tank:
            continue
        relation = str(cond._relation)
        if "lt" in relation or "<" in relation:
            cond._threshold = low_level_m
        else:
            cond._threshold = high_level_m
        changed += 1
    if not changed:
        raise ValueError(f"No level controls reference tank {tank!r}.")
    return out
