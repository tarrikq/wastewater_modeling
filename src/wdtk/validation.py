"""EPANET reference runs (through WNTR) used to validate wdtk, shared by tests and scripts.

Every function here produces a *reference* value from EPANET 2.2; the from-scratch results
they are compared with come from :mod:`wdtk.pumps`, :mod:`wdtk.headloss`, and
:mod:`wdtk.network`.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import wntr
from scipy.optimize import brentq

from wdtk import headloss as hl
from wdtk import pumps as pm
from wdtk.network.io import write_inp
from wdtk.network.model import Junction, Network, Pipe, Reservoir

ACCURACY = 1e-6
"""EPANET convergence accuracy used for validation (tighter than its 1e-3 default)."""


def run_epanet_steady(
    model: Any, workdir: str | Path | None = None
) -> tuple[dict[str, float], dict[str, float]]:
    """Run EPANET once (steady state, demand-driven) and return heads and flows.

    Parameters
    ----------
    model : wntr WaterNetworkModel or path to an .inp file
        Model to run (a path is loaded with WNTR).
    workdir : str or Path, optional
        Directory for EPANET's scratch files (default: temporary).

    Returns
    -------
    tuple of dict
        (node heads, m; link flows, m^3/s).
    """
    wn = model if not isinstance(model, (str, Path)) else wntr.network.WaterNetworkModel(str(model))
    wn.options.hydraulic.accuracy = ACCURACY
    wn.options.hydraulic.demand_model = "DD"
    wn.options.time.duration = 0
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(workdir or tmp) / "validation"
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(prefix))
    return res.node["head"].loc[0].to_dict(), res.link["flowrate"].loc[0].to_dict()


# --- Phase 2: single pump between two reservoirs ---------------------------------------------
PUMP_CASE = {
    "r1_head_m": 10.0,
    "r2_head_m": 40.0,
    "length_m": 2000.0,
    "diameter_m": 0.3,
    "c_factor": 120.0,
}
"""Validation model: R1 -> pump -> J1 -> pipe -> R2 (arbitrary test inputs, SI)."""


def single_pump_model(points: list[tuple[float, float]], speed: float = 1.0) -> Any:
    """WNTR model of the Phase 2 validation case with the given pump curve points (SI).

    Parameters
    ----------
    points : list of (flow m^3/s, head m)
        EPANET pump curve points.
    speed : float, optional
        Pump speed setting (default 1.0).

    Returns
    -------
    wntr WaterNetworkModel
        The model.
    """
    c = PUMP_CASE
    wn = wntr.network.WaterNetworkModel()
    wn.options.hydraulic.headloss = "H-W"
    wn.add_reservoir("R1", base_head=c["r1_head_m"])
    wn.add_reservoir("R2", base_head=c["r2_head_m"])
    wn.add_junction("J1", base_demand=0.0, elevation=0.0)
    wn.add_curve("C1", "HEAD", points)
    wn.add_pump("P1", "R1", "J1", "HEAD", "C1", speed=speed)
    wn.add_pipe(
        "L1", "J1", "R2", length=c["length_m"], diameter=c["diameter_m"], roughness=c["c_factor"]
    )
    return wn


def pump_case_wdtk(curve: pm.PumpCurve) -> tuple[float, float]:
    """Return the wdtk operating point (flow m^3/s, pump head m) for the Phase 2 case."""
    c = PUMP_CASE

    def system(q: float) -> float:
        return pm.system_head_hw_m(
            q, c["r2_head_m"] - c["r1_head_m"], c["length_m"], c["diameter_m"], c["c_factor"]
        )

    return pm.operating_point(lambda q: pm.pump_head_m(curve, q), system)


def pump_case_epanet(points: list[tuple[float, float]], speed: float = 1.0) -> tuple[float, float]:
    """EPANET operating point (flow m^3/s, pump head gain m) for the Phase 2 validation case."""
    heads, flows = run_epanet_steady(single_pump_model(points, speed))
    return flows["P1"], heads["J1"] - heads["R1"]


def pump_validation_table() -> pd.DataFrame:
    """Phase 2 comparison: wdtk vs EPANET operating points for four pump cases.

    Returns
    -------
    DataFrame
        Columns: ``case``, ``flow_wdtk_Ls``, ``flow_epanet_Ls``, ``flow_diff_pct``,
        ``head_wdtk_m``, ``head_epanet_m``, ``head_diff_pct``.
    """
    single = [(0.10, 50.0)]
    three = [(0.0, 60.0), (0.10, 45.0), (0.22, 0.0)]
    (_, h1), (q2, h2), (q3, h3) = three
    base3 = pm.fit_three_point(h1, q2, h2, q3, h3)
    cases = [
        ("single-point curve", pm.fit_single_point(*single[0]), single, 1.0),
        ("three-point curve", base3, three, 1.0),
        ("three-point, speed 0.9", pm.scale_curve_speed(base3, 0.9), three, 0.9),
        ("three-point, speed 0.8", pm.scale_curve_speed(base3, 0.8), three, 0.8),
    ]
    rows = []
    for name, curve, points, speed in cases:
        q, h = pump_case_wdtk(curve)
        qe, he = pump_case_epanet(points, speed)
        rows.append(
            {
                "case": name,
                "flow_wdtk_Ls": q * 1000,
                "flow_epanet_Ls": qe * 1000,
                "flow_diff_pct": 100 * (q / qe - 1),
                "head_wdtk_m": h,
                "head_epanet_m": he,
                "head_diff_pct": 100 * (h / he - 1),
            }
        )
    return pd.DataFrame(rows)


# --- Phase 3: EPANET Darcy-Weisbach probe ------------------------------------------------------
DW_PROBES = [  # (head m, length m, diameter m, roughness m): turbulent, Re ~5e3 to ~1.6e6
    (50.0, 1000.0, 0.3, 3e-3),
    (50.0, 1000.0, 0.3, 1e-3),
    (5.0, 1000.0, 0.3, 1e-4),
    (5.0, 1000.0, 0.3, 1e-5),
    (0.5, 1000.0, 0.1, 1e-6),
    (0.05, 1000.0, 0.1, 1e-6),
    (20.0, 2000.0, 0.6, 2e-4),
]


def dw_probe_epanet(head_m: float, length_m: float, diameter_m: float, eps_m: float) -> float:
    """EPANET flow (m^3/s) through a D-W pipe between two reservoirs (split at one junction)."""
    net = Network(
        junctions=(Junction("J", 0.0, 0.0),),  # EPANET needs at least one junction
        reservoirs=(Reservoir("A", head_m), Reservoir("B", 0.0)),
        pipes=(
            Pipe("P1", "A", "J", length_m / 2, diameter_m, eps_m),
            Pipe("P2", "J", "B", length_m / 2, diameter_m, eps_m),
        ),
        headloss="D-W",
    )
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "probe.inp"
        write_inp(net, inp)
        _, flows = run_epanet_steady(inp, tmp)
    return flows["P1"]


def dw_probe_wdtk(head_m: float, length_m: float, diameter_m: float, eps_m: float) -> float:
    """Return the wdtk flow (m^3/s) for the same pipe: Swamee-Jain, EPANET constants."""

    def residual(q: float) -> float:
        return (
            hl.darcy_weisbach_headloss_m(
                q,
                length_m,
                diameter_m,
                eps_m,
                hl.EPANET_KINEMATIC_VISCOSITY_M2S,
                "swamee-jain",
                g_ms2=hl.EPANET_DW_GRAVITY_MS2,
            )
            - head_m
        )

    return float(brentq(residual, 1e-9, 10.0, xtol=1e-15, rtol=1e-15))


def dw_probe_table() -> pd.DataFrame:
    """EPANET D-W probe: flow from EPANET vs wdtk with the identified constants.

    Returns
    -------
    DataFrame
        Columns: ``head_m``, ``length_m``, ``diameter_m``, ``rel_roughness``, ``reynolds``,
        ``flow_epanet_Ls``, ``flow_wdtk_Ls``, ``diff_pct``.
    """
    rows = []
    for h, length, d, eps in DW_PROBES:
        qe = dw_probe_epanet(h, length, d, eps)
        q = dw_probe_wdtk(h, length, d, eps)
        re = q / (math.pi * d**2 / 4) * d / hl.EPANET_KINEMATIC_VISCOSITY_M2S
        rows.append(
            {
                "head_m": h,
                "length_m": length,
                "diameter_m": d,
                "rel_roughness": eps / d,
                "reynolds": re,
                "flow_epanet_Ls": qe * 1000,
                "flow_wdtk_Ls": q * 1000,
                "diff_pct": 100 * (q / qe - 1),
            }
        )
    return pd.DataFrame(rows)
