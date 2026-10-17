"""4e Pump energy and cost under a time-of-use tariff.

For each pump and reporting step, with Q the pump flow and H the head gain
(head at the discharge node minus head at the suction node):

    P_in = rho g Q H / eta_ww         (wdtk.pumps.input_power_w; rho = 1000 kg/m^3)
    E    = P_in * dt,   cost = E * price(hour of day)

``dt`` is the time to the next reported step (the last step uses the report interval). The
efficiency is the assumptions' wire-to-water value (Net3: 75% for all pumps).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from wntr.network.base import LinkStatus
from wntr.network.controls import Control, ControlAction, TimeOfDayCondition

from wdtk.analysis.assumptions import EnergySettings
from wdtk.analysis.simulate import WaterNetworkModel
from wdtk.pumps import input_power_w

SECONDS_PER_HOUR = 3600.0
W_PER_KW = 1000.0


def price_per_kwh(time_s: float, energy: EnergySettings) -> float:
    """Tariff price at a simulation time ($/kWh), using hour of day = (t mod 24 h).

    Parameters
    ----------
    time_s : float
        Simulation time, s (t = 0 is midnight).
    energy : EnergySettings
        Tariff.

    Returns
    -------
    float
        Price, $/kWh.
    """
    hour = (time_s / SECONDS_PER_HOUR) % 24.0
    on_peak = energy.on_peak_start_h <= hour < energy.on_peak_end_h
    return energy.on_peak_price_per_kwh if on_peak else energy.off_peak_price_per_kwh


def pump_energy(results: Any, wn: WaterNetworkModel, energy: EnergySettings) -> pd.DataFrame:
    """Per-step pump power, energy, and cost.

    Parameters
    ----------
    results : wntr SimulationResults
        EPS results.
    wn : WaterNetworkModel
        The simulated model.
    energy : EnergySettings
        Tariff and efficiency.

    Returns
    -------
    DataFrame
        Columns: ``time_s``, ``pump``, ``flow_m3s``, ``head_gain_m``, ``power_kw``, ``dt_h``,
        ``energy_kwh``, ``price_per_kwh``, ``cost``.
    """
    times = np.asarray(results.link["flowrate"].index, dtype=float)
    step = np.diff(times)
    dt_s = np.append(step, step[-1] if len(step) else 0.0)
    rows = []
    for name, pump in wn.pumps():
        q = np.clip(results.link["flowrate"][name].to_numpy(dtype=float), 0.0, None)
        head = results.node["head"]
        gain = (head[pump.end_node_name] - head[pump.start_node_name]).to_numpy(dtype=float)
        gain = np.where(q > 0.0, gain, 0.0)
        power_kw = input_power_w(q, gain, energy.pump_efficiency) / W_PER_KW
        prices = np.array([price_per_kwh(t, energy) for t in times])
        e_kwh = power_kw * dt_s / SECONDS_PER_HOUR
        rows.append(
            pd.DataFrame(
                {
                    "time_s": times,
                    "pump": name,
                    "flow_m3s": q,
                    "head_gain_m": gain,
                    "power_kw": power_kw,
                    "dt_h": dt_s / SECONDS_PER_HOUR,
                    "energy_kwh": e_kwh,
                    "price_per_kwh": prices,
                    "cost": e_kwh * prices,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def energy_summary(steps: pd.DataFrame, energy: EnergySettings) -> pd.DataFrame:
    """Energy and cost totals per pump.

    Parameters
    ----------
    steps : DataFrame
        Output of :func:`pump_energy`.
    energy : EnergySettings
        Tariff (to classify on-peak energy).

    Returns
    -------
    DataFrame
        Index: pump. Columns: ``energy_kwh``, ``cost``, ``on_peak_fraction``,
        ``average_price_per_kwh``, ``run_hours``.
    """
    on_peak = steps["price_per_kwh"] == energy.on_peak_price_per_kwh
    g = steps.assign(on_peak_kwh=steps["energy_kwh"].where(on_peak, 0.0)).groupby("pump")
    total = g["energy_kwh"].sum()
    return pd.DataFrame(
        {
            "energy_kwh": total,
            "cost": g["cost"].sum(),
            "on_peak_fraction": g["on_peak_kwh"].sum() / total.where(total > 0),
            "average_price_per_kwh": g["cost"].sum() / total.where(total > 0),
            "run_hours": steps.assign(on=(steps["flow_m3s"] > 0) * steps["dt_h"])
            .groupby("pump")["on"]
            .sum(),
        }
    )


def timed_run_hours(wn: WaterNetworkModel, pump: str) -> float:
    """Daily run length of a pump driven by simulation-time controls, h.

    Pairs each "open" time with the next "close" time among the pump's time controls and
    returns the first pair's duration (Net3's pump 10 repeats the same daily window).

    Parameters
    ----------
    wn : WaterNetworkModel
        Model.
    pump : str
        Pump name.

    Returns
    -------
    float
        Run length, h.
    """
    events = []
    for name in wn.control_name_list:
        ctrl = wn.get_control(name)
        cond = ctrl.condition
        if type(cond).__name__ != "SimTimeCondition":
            continue
        for action in ctrl.actions():
            target, attr = action.target()
            if target.name == pump and attr == "status":
                events.append((float(cond._threshold), int(action._value)))
    events.sort()
    opens = [t for t, v in events if v == LinkStatus.Open]
    closes = [t for t, v in events if v == LinkStatus.Closed]
    if not opens or not closes:
        raise ValueError(f"Pump {pump!r} has no open/close time controls.")
    first_close = min(t for t in closes if t > opens[0])
    return (first_close - opens[0]) / SECONDS_PER_HOUR


def reschedule_timed_pump(
    wn: WaterNetworkModel, pump: str, start_h: float, run_h: float
) -> WaterNetworkModel:
    """Return a copy with a pump's time controls replaced by a daily clock-time window.

    The pump opens at ``start_h`` and closes ``run_h`` hours later every day (EPANET clock-
    time controls). Its initial status is open if the window covers midnight.

    Parameters
    ----------
    wn : WaterNetworkModel
        Model (not modified).
    pump : str
        Pump name.
    start_h : float
        Daily start time, hours after midnight.
    run_h : float
        Daily run length, h (0 < run_h < 24).

    Returns
    -------
    WaterNetworkModel
        Modified copy.
    """
    if not 0.0 < run_h < 24.0:
        raise ValueError("run_h must be between 0 and 24 hours.")
    out = copy.deepcopy(wn)
    link = out.get_link(pump)
    for name in list(out.control_name_list):
        ctrl = out.get_control(name)
        if type(ctrl.condition).__name__ not in ("SimTimeCondition", "TimeOfDayCondition"):
            continue
        if any(a.target()[0].name == pump for a in ctrl.actions()):
            out.remove_control(name)
    stop_h = (start_h + run_h) % 24.0
    for tag, hour, status in (
        ("open", start_h, LinkStatus.Open),
        ("close", stop_h, LinkStatus.Closed),
    ):
        # WNTR's docstring says a float threshold is in hours, but a float is parsed as
        # seconds; an "hh:mm" string is unambiguous.
        minutes = round(hour * 60.0)
        cond = TimeOfDayCondition(out, "=", f"{minutes // 60:02d}:{minutes % 60:02d}")
        out.add_control(f"{pump}_{tag}_daily", Control(cond, ControlAction(link, "status", status)))
    wraps = start_h + run_h > 24.0 or start_h == 0.0
    link.initial_status = LinkStatus.Open if wraps else LinkStatus.Closed
    return out
