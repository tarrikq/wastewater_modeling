"""Centrifugal pumps: head curves, system curves, operating points, power, and NPSH.

Pump head curves use the EPANET form H = a - b Q^c (H in m, Q in m^3/s). Curve fitting
reproduces EPANET's conventions as implemented in WNTR 1.5.0,
``wntr/network/elements.py``, ``HeadPump.get_head_curve_coefficients`` (commented there as
replicating EPANET):

* one-point curve (design Q_d, H_d): a = 4/3 H_d, b = 1/3 H_d / Q_d^2, c = 2, i.e. shutoff
  head 1.33 H_d and maximum flow 2 Q_d;
* three-point curve with the first point at zero flow: the exact interpolant through the
  three points (closed form below).

Pump flow is non-negative (EPANET pumps carry an implicit check valve). Variable speed uses
the affinity laws at a speed ratio s = N / N_rated: Q scales with s, H with s^2, power with s^3.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from wdtk import headloss as hl
from wdtk._arrays import F, FloatOrArray
from wdtk._arrays import like as _like
from wdtk._arrays import require_nonnegative as _require_nonnegative
from wdtk._arrays import require_positive as _require_positive
from wdtk.units import DEFAULT_DENSITY_KGM3, G_STANDARD_MS2, pressure_kpa_to_head_m

HeadFunction = Callable[[float], float]


@dataclass(frozen=True)
class PumpCurve:
    """Pump head curve H = a - b Q^c.

    A plain data record; all behavior lives in module functions.

    Attributes
    ----------
    a_m : float
        Shutoff head (head at zero flow), m. Positive.
    b : float
        Curve coefficient, m / (m^3/s)^c. Positive.
    c : float
        Curve exponent, dimensionless. Positive.
    """

    a_m: float
    b: float
    c: float

    def __post_init__(self) -> None:
        """Validate that all coefficients are positive."""
        for name in ("a_m", "b", "c"):
            _require_positive(name, getattr(self, name))

    @property
    def max_flow_m3s(self) -> float:
        """Flow at which the curve reaches zero head, (a/b)^(1/c), m^3/s."""
        return float((self.a_m / self.b) ** (1.0 / self.c))


# --- Curve fitting ---------------------------------------------------------------------------
def fit_single_point(design_flow_m3s: float, design_head_m: float) -> PumpCurve:
    """Fit a pump curve to a single design point, EPANET convention.

    a = (4/3) H_d, b = (1/3) H_d / Q_d^2, c = 2, which passes through (Q_d, H_d) with shutoff
    head 4/3 H_d and zero head at 2 Q_d.

    Parameters
    ----------
    design_flow_m3s : float
        Design (rated) flow, m^3/s. Positive.
    design_head_m : float
        Design (rated) head, m. Positive.

    Returns
    -------
    PumpCurve
        Fitted curve.
    """
    _require_positive("design_flow_m3s", design_flow_m3s)
    _require_positive("design_head_m", design_head_m)
    return PumpCurve(
        a_m=4.0 / 3.0 * design_head_m,
        b=design_head_m / (3.0 * design_flow_m3s**2),
        c=2.0,
    )


def fit_three_point(
    shutoff_head_m: float,
    design_flow_m3s: float,
    design_head_m: float,
    max_flow_m3s: float,
    max_flow_head_m: float = 0.0,
) -> PumpCurve:
    """Fit a pump curve exactly through three points, the first at zero flow.

    With points (0, h1), (q2, h2), (q3, h3), H = a - b Q^c gives

        a = h1,  c = ln((h1 - h3) / (h1 - h2)) / ln(q3 / q2),  b = (h1 - h2) / q2^c.

    Parameters
    ----------
    shutoff_head_m : float
        Head at zero flow h1, m.
    design_flow_m3s, design_head_m : float
        Design point (q2, h2), m^3/s and m.
    max_flow_m3s, max_flow_head_m : float
        High-flow point (q3, h3), m^3/s and m (head defaults to 0, i.e. maximum flow).

    Returns
    -------
    PumpCurve
        Fitted curve.

    Raises
    ------
    ValueError
        Unless 0 < q2 < q3 and h1 > h2 > h3 >= 0.
    """
    q2, q3 = design_flow_m3s, max_flow_m3s
    h1, h2, h3 = shutoff_head_m, design_head_m, max_flow_head_m
    if not (0.0 < q2 < q3 and h1 > h2 > h3 >= 0.0):
        raise ValueError("Three-point curve needs 0 < q2 < q3 and h1 > h2 > h3 >= 0.")
    c = math.log((h1 - h3) / (h1 - h2)) / math.log(q3 / q2)
    return PumpCurve(a_m=h1, b=(h1 - h2) / q2**c, c=c)


# --- Evaluating curves -----------------------------------------------------------------------
def pump_head_m(curve: PumpCurve, flow_m3s: F) -> F:
    """Pump head at a given flow.

    H = a - b Q^c. Beyond ``curve.max_flow_m3s`` the result is negative (extrapolation).

    Parameters
    ----------
    curve : PumpCurve
        Pump curve.
    flow_m3s : float or ndarray
        Flow, m^3/s. Must be non-negative.

    Returns
    -------
    float or ndarray
        Head added by the pump, m.
    """
    _require_nonnegative("flow_m3s", flow_m3s)
    q = np.asarray(flow_m3s, dtype=np.float64)
    return _like(flow_m3s, curve.a_m - curve.b * q**curve.c, "flow_m3s")


def pump_flow_m3s(curve: PumpCurve, head_m: F) -> F:
    """Pump flow at a given head (inverse of :func:`pump_head_m`).

    Q = ((a - H) / b)^(1/c) for H < a; Q = 0 for H >= a (check valve closed).

    Parameters
    ----------
    curve : PumpCurve
        Pump curve.
    head_m : float or ndarray
        Head across the pump, m.

    Returns
    -------
    float or ndarray
        Flow, m^3/s (non-negative).
    """
    h = np.asarray(head_m, dtype=np.float64)
    deficit = np.maximum(curve.a_m - h, 0.0)
    return _like(head_m, (deficit / curve.b) ** (1.0 / curve.c), "head_m")


# --- Affinity laws ---------------------------------------------------------------------------
def scale_curve_speed(curve: PumpCurve, speed_ratio: float) -> PumpCurve:
    """Pump curve at a different speed, by the affinity laws.

    Each point (Q, H) maps to (s Q, s^2 H), so H'(Q') = s^2 a - b s^(2-c) Q'^c:
    a' = s^2 a, b' = s^(2-c) b, c' = c.

    Parameters
    ----------
    curve : PumpCurve
        Curve at the reference speed.
    speed_ratio : float
        s = N / N_ref, dimensionless. Positive.

    Returns
    -------
    PumpCurve
        Curve at the new speed.

    Notes
    -----
    The affinity laws assume geometric and kinematic similarity, and constant efficiency
    along affinity parabolas; accuracy falls off at large speed reductions.
    """
    _require_positive("speed_ratio", speed_ratio)
    s = speed_ratio
    return PumpCurve(a_m=s**2 * curve.a_m, b=s ** (2.0 - curve.c) * curve.b, c=curve.c)


def affinity_point(flow_m3s: float, head_m: float, speed_ratio: float) -> tuple[float, float]:
    """Map an operating point to a new speed with the affinity laws.

    Q2 = s Q1, H2 = s^2 H1 (and power P2 = s^3 P1, see :func:`hydraulic_power_w`).

    Parameters
    ----------
    flow_m3s : float
        Flow at the reference speed, m^3/s.
    head_m : float
        Head at the reference speed, m.
    speed_ratio : float
        s = N / N_ref, dimensionless. Positive.

    Returns
    -------
    tuple of float
        (flow, m^3/s; head, m) at the new speed.
    """
    _require_positive("speed_ratio", speed_ratio)
    return speed_ratio * flow_m3s, speed_ratio**2 * head_m


# --- Pumps in series and parallel ------------------------------------------------------------
def series_head_m(curves: Sequence[PumpCurve], flow_m3s: float) -> float:
    """Head of pumps in series: the same flow passes each pump, heads add.

    H(Q) = sum_i H_i(Q).

    Parameters
    ----------
    curves : sequence of PumpCurve
        Pump curves.
    flow_m3s : float
        Flow through the train, m^3/s. Non-negative.

    Returns
    -------
    float
        Total head, m.
    """
    return float(sum(pump_head_m(c, flow_m3s) for c in curves))


def parallel_head_m(curves: Sequence[PumpCurve], flow_m3s: float) -> float:
    """Head of pumps in parallel: a common head, flows add.

    Solves sum_i Q_i(H) = Q for H with ``brentq``. A pump whose shutoff head is below H
    delivers no flow (its check valve is closed).

    Parameters
    ----------
    curves : sequence of PumpCurve
        Pump curves.
    flow_m3s : float
        Total flow, m^3/s. Non-negative.

    Returns
    -------
    float
        Common head, m.
    """
    _require_nonnegative("flow_m3s", flow_m3s)
    h_hi = max(c.a_m for c in curves)
    if flow_m3s == 0.0:
        return h_hi

    def excess(h: float) -> float:
        return float(sum(pump_flow_m3s(c, h) for c in curves)) - flow_m3s

    step = h_hi
    h_lo = h_hi - step
    while excess(h_lo) < 0.0:
        step *= 2.0
        h_lo = h_hi - step
    return float(brentq(excess, h_lo, h_hi, xtol=1e-13, rtol=1e-15))


# --- System curves ---------------------------------------------------------------------------
def system_head_hw_m(
    flow_m3s: F,
    static_head_m: float,
    length_m: float,
    diameter_m: float,
    c_factor: float,
    minor_k: float = 0.0,
) -> F:
    """System head curve with Hazen-Williams friction.

    H_sys(Q) = H_static + h_f,HW(Q) + K V|V| / (2g).

    Parameters
    ----------
    flow_m3s : float or ndarray
        Flow, m^3/s.
    static_head_m : float
        Static lift (downstream minus upstream hydraulic grade at zero flow), m.
    length_m, diameter_m : float
        Equivalent pipe length and internal diameter, m.
    c_factor : float
        Hazen-Williams C, dimensionless.
    minor_k : float, optional
        Sum of minor loss coefficients referenced to the pipe velocity (default 0).

    Returns
    -------
    float or ndarray
        Head the pump must supply, m.
    """
    friction = hl.hazen_williams_headloss_m(flow_m3s, length_m, diameter_m, c_factor)
    minor = hl.minor_loss_m(hl.velocity_ms(flow_m3s, diameter_m), minor_k)
    return _like(flow_m3s, static_head_m + np.asarray(friction) + np.asarray(minor), "flow_m3s")


def system_head_dw_m(
    flow_m3s: F,
    static_head_m: float,
    length_m: float,
    diameter_m: float,
    roughness_m: float,
    kinematic_viscosity_m2s: float,
    minor_k: float = 0.0,
) -> F:
    """System head curve with Darcy-Weisbach friction.

    H_sys(Q) = H_static + f (L/D) V|V| / (2g) + K V|V| / (2g).

    Parameters
    ----------
    flow_m3s : float or ndarray
        Flow, m^3/s.
    static_head_m : float
        Static lift, m.
    length_m, diameter_m : float
        Equivalent pipe length and internal diameter, m.
    roughness_m : float
        Absolute roughness, m.
    kinematic_viscosity_m2s : float
        Kinematic viscosity, m^2/s.
    minor_k : float, optional
        Sum of minor loss coefficients referenced to the pipe velocity (default 0).

    Returns
    -------
    float or ndarray
        Head the pump must supply, m.
    """
    friction = hl.darcy_weisbach_headloss_m(
        flow_m3s, length_m, diameter_m, roughness_m, kinematic_viscosity_m2s
    )
    minor = hl.minor_loss_m(hl.velocity_ms(flow_m3s, diameter_m), minor_k)
    return _like(flow_m3s, static_head_m + np.asarray(friction) + np.asarray(minor), "flow_m3s")


# --- Operating point -------------------------------------------------------------------------
def operating_point(pump_head: HeadFunction, system_head: HeadFunction) -> tuple[float, float]:
    """Intersection of a pump (or pump combination) curve with a system curve.

    Solves H_pump(Q) = H_sys(Q) for Q > 0 with ``brentq``. H_pump must decrease and H_sys
    increase with Q, so the intersection is unique.

    Parameters
    ----------
    pump_head : callable
        Q (m^3/s) -> pump head (m), e.g. ``lambda q: pump_head_m(curve, q)``.
    system_head : callable
        Q (m^3/s) -> required system head (m).

    Returns
    -------
    tuple of float
        (flow, m^3/s; head, m) at the operating point.

    Raises
    ------
    ValueError
        If the pump cannot overcome the static head (no flow).
    """

    def surplus(q: float) -> float:
        return pump_head(q) - system_head(q)

    if surplus(0.0) <= 0.0:
        raise ValueError("Pump shutoff head does not exceed the static head: no flow.")
    q_hi = 1e-3
    while surplus(q_hi) > 0.0:
        q_hi *= 2.0
        if q_hi > 1e6:
            raise ValueError("Could not bracket the operating point; check the curves.")
    q = float(brentq(surplus, 0.0, q_hi, xtol=1e-15, rtol=1e-15))
    return q, pump_head(q)


# --- Power -----------------------------------------------------------------------------------
def hydraulic_power_w(
    flow_m3s: F, head_m: FloatOrArray, density_kgm3: float = DEFAULT_DENSITY_KGM3
) -> F:
    """Hydraulic (water) power delivered to the fluid.

    P_w = rho g Q H.

    Parameters
    ----------
    flow_m3s : float or ndarray
        Flow, m^3/s.
    head_m : float or ndarray
        Head added, m.
    density_kgm3 : float, optional
        Water density, kg/m^3 (default 1000; see ``wdtk.fluid`` for temperature effects).

    Returns
    -------
    float or ndarray
        Hydraulic power, W.
    """
    _require_positive("density_kgm3", density_kgm3)
    p = density_kgm3 * G_STANDARD_MS2 * np.asarray(flow_m3s) * np.asarray(head_m)
    return _like(flow_m3s, p, "flow_m3s")


def _require_efficiency(name: str, value: FloatOrArray) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must be in (0, 1]; got {value!r}.")


def brake_power_w(
    flow_m3s: F,
    head_m: FloatOrArray,
    pump_efficiency: FloatOrArray,
    density_kgm3: float = DEFAULT_DENSITY_KGM3,
) -> F:
    """Brake (shaft) power required by the pump.

    P_b = P_w / eta_p = rho g Q H / eta_p.

    Parameters
    ----------
    flow_m3s : float or ndarray
        Flow, m^3/s.
    head_m : float or ndarray
        Head added, m.
    pump_efficiency : float or ndarray
        Pump efficiency eta_p, in (0, 1].
    density_kgm3 : float, optional
        Water density, kg/m^3 (default 1000).

    Returns
    -------
    float or ndarray
        Shaft power, W.
    """
    _require_efficiency("pump_efficiency", pump_efficiency)
    p = np.asarray(hydraulic_power_w(flow_m3s, head_m, density_kgm3)) / np.asarray(pump_efficiency)
    return _like(flow_m3s, p, "flow_m3s")


def wire_to_water_efficiency(
    pump_efficiency: float, motor_efficiency: float, drive_efficiency: float = 1.0
) -> float:
    """Overall wire-to-water efficiency of a pumping unit.

    eta_ww = eta_pump * eta_motor * eta_drive.

    Parameters
    ----------
    pump_efficiency, motor_efficiency : float
        Component efficiencies, in (0, 1].
    drive_efficiency : float, optional
        Variable-frequency drive efficiency, in (0, 1] (default 1, i.e. no drive).

    Returns
    -------
    float
        Wire-to-water efficiency, in (0, 1].
    """
    for name, value in [
        ("pump_efficiency", pump_efficiency),
        ("motor_efficiency", motor_efficiency),
        ("drive_efficiency", drive_efficiency),
    ]:
        _require_efficiency(name, value)
    return pump_efficiency * motor_efficiency * drive_efficiency


def input_power_w(
    flow_m3s: F,
    head_m: FloatOrArray,
    wire_to_water_eff: FloatOrArray,
    density_kgm3: float = DEFAULT_DENSITY_KGM3,
) -> F:
    """Electrical input power of a pumping unit.

    P_in = rho g Q H / eta_ww.

    Parameters
    ----------
    flow_m3s : float or ndarray
        Flow, m^3/s.
    head_m : float or ndarray
        Head added, m.
    wire_to_water_eff : float or ndarray
        Wire-to-water efficiency, in (0, 1]; see :func:`wire_to_water_efficiency`.
    density_kgm3 : float, optional
        Water density, kg/m^3 (default 1000).

    Returns
    -------
    float or ndarray
        Electrical power, W.
    """
    return brake_power_w(flow_m3s, head_m, wire_to_water_eff, density_kgm3)


# --- Net positive suction head ---------------------------------------------------------------
def npsh_available_m(
    atmospheric_pressure_kpa: float,
    vapor_pressure_kpa: float,
    suction_static_head_m: float,
    suction_headloss_m: float,
    density_kgm3: float = DEFAULT_DENSITY_KGM3,
) -> float:
    """Net positive suction head available at the pump inlet.

    NPSHa = (p_atm - p_v) / (rho g) + z_s - h_f,s

    for a pump drawing from a free surface open to the atmosphere.

    Parameters
    ----------
    atmospheric_pressure_kpa : float
        Absolute atmospheric pressure at the site, kPa. Site-specific; supply it.
    vapor_pressure_kpa : float
        Absolute vapor pressure of the water, kPa (``wdtk.fluid.vapor_pressure_kpa``).
    suction_static_head_m : float
        Elevation of the supply free surface above the pump impeller eye, m. Negative for a
        suction lift.
    suction_headloss_m : float
        Friction and minor losses in the suction piping, m. Non-negative.
    density_kgm3 : float, optional
        Water density, kg/m^3 (default 1000).

    Returns
    -------
    float
        NPSH available, m.
    """
    _require_positive("atmospheric_pressure_kpa", atmospheric_pressure_kpa)
    _require_nonnegative("vapor_pressure_kpa", vapor_pressure_kpa)
    _require_nonnegative("suction_headloss_m", suction_headloss_m)
    pressure_head = pressure_kpa_to_head_m(
        atmospheric_pressure_kpa - vapor_pressure_kpa, density_kgm3
    )
    return pressure_head + suction_static_head_m - suction_headloss_m


def npsh_margin_m(npsh_available_m: float, npsh_required_m: float) -> float:
    """NPSH margin, NPSHa - NPSHr, m.

    Parameters
    ----------
    npsh_available_m : float
        NPSH available, m.
    npsh_required_m : float
        NPSH required by the pump at the operating flow (manufacturer data), m.

    Returns
    -------
    float
        Margin, m. Negative means cavitation is expected.
    """
    return npsh_available_m - npsh_required_m


def npsh_check(
    npsh_available_m: float, npsh_required_m: float, required_margin_m: float = 0.0
) -> bool:
    """Check NPSHa >= NPSHr + required margin.

    Parameters
    ----------
    npsh_available_m : float
        NPSH available, m.
    npsh_required_m : float
        NPSH required at the operating flow, m.
    required_margin_m : float, optional
        Design margin above NPSHr, m (default 0). A project design criterion, not a
        physical constant.

    Returns
    -------
    bool
        True if the margin is met.
    """
    _require_nonnegative("required_margin_m", required_margin_m)
    return npsh_margin_m(npsh_available_m, npsh_required_m) >= required_margin_m
