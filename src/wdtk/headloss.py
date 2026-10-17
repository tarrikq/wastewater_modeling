"""Head loss in a single pressurized pipe: Hazen-Williams, Darcy-Weisbach, minor losses.

All functions use SI units (m, s, m^3/s, m of head) except
:func:`hazen_williams_headloss_ft`, which is the US customary (ft, cfs) form.

Sign convention: head loss carries the sign of the flow, h = r * Q * |Q|^(n - 1), so a
reversed flow gives a negative head loss (as needed by network solvers). Reynolds number
and friction factor use |V|.

Array support: the first argument may be a float or a float64 array; the other arguments
may be floats or arrays that broadcast to the first argument's shape. The result has the
type and shape of the first argument. Broadcasting to a larger shape raises ``ValueError``.

Reference for the Hazen-Williams constants and the laminar/turbulent Reynolds number limits
(verified by the project author):

    Bombardelli, F. A., and Garcia, M. H. (2003). "Hydraulic Design of Large-Diameter
    Pipes." Journal of Hydraulic Engineering, 129(11), 839-846.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from wdtk.units import FT_M, G_STANDARD_MS2

F = TypeVar("F", float, NDArray[np.float64])
FloatOrArray = float | NDArray[np.float64]
TurbulentMethod = Literal["colebrook", "swamee-jain"]

# --- Hazen-Williams constants ----------------------------------------------------------------
HW_K_US = 1.318
"""Hazen-Williams velocity-form constant for ft-s units: V = 1.318 C R^0.63 S^0.54.

Units ft^0.37/s. This is the defining constant of the formula; it and the exponents
0.63 and 0.54 are as given in Bombardelli and Garcia (2003), see module docstring.
"""

HW_EXP_R = 0.63
"""Hazen-Williams hydraulic radius exponent (defining constant)."""

HW_EXP_S = 0.54
"""Hazen-Williams slope exponent (defining constant)."""

HW_K_SI = HW_K_US * FT_M ** (1.0 - HW_EXP_R)
"""Hazen-Williams velocity-form constant for SI units (m^0.37/s), derived exactly.

V[m/s] = FT_M * k_US * C * (R[m] / FT_M)^0.63 * S^0.54 = k_US * FT_M^0.37 * C * R^0.63 * S^0.54.
"""

HW_FLOW_EXPONENT = 1.0 / HW_EXP_S
"""Exponent n in h = r Q^n for Hazen-Williams (1/0.54, about 1.852)."""

HW_DIAMETER_EXPONENT = (2.0 + HW_EXP_R) / HW_EXP_S
"""Exponent m in h proportional to D^-m for Hazen-Williams (2.63/0.54, about 4.870)."""

# --- Darcy-Weisbach constants ----------------------------------------------------------------
RE_LAMINAR_MAX = 2000.0
"""Upper Reynolds number of the laminar regime (Bombardelli and Garcia 2003)."""

RE_TURBULENT_MIN = 4000.0
"""Lower Reynolds number of the turbulent regime (Bombardelli and Garcia 2003)."""

_COLEBROOK_X_BRACKET = (1e-6, 1e3)  # x = 1/sqrt(f); contains every root when eps/D < 3.7


# --- Helpers ---------------------------------------------------------------------------------
def _like(template: F, value: NDArray[np.float64] | float, name: str) -> F:
    """Return ``value`` with the type and shape of ``template``."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != np.shape(template):
        raise ValueError(
            f"Arguments broadcast to shape {arr.shape}, larger than {name} shape "
            f"{np.shape(template)}; pass {name} as an array of the full shape."
        )
    result: float | NDArray[np.float64] = float(arr) if arr.ndim == 0 else arr
    return cast(F, result)


def _require_positive(name: str, value: FloatOrArray) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and positive; got {value!r}.")


def _require_nonnegative(name: str, value: FloatOrArray) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")


def _solve_positive(func: Callable[[float], float], target: float, x0: float) -> float:
    """Solve ``func(x) = target`` for x > 0, where func is positive and monotonic in x.

    Works in log space, log(func(e^u)) - log(target) = 0, so the result has full relative
    precision whatever the magnitude of x. Starting from ``x0``, the bracket is grown by
    factors of 2 in the direction of the root only, so ``func`` is never evaluated far beyond
    the solution on the wrong side (e.g. diameters smaller than the roughness height).
    """

    def g(u: float) -> float:
        return math.log(func(math.exp(u))) - math.log(target)

    step = math.log(2.0)
    u0 = math.log(x0)
    g0 = g(u0)
    if g0 == 0.0:
        return x0
    increasing = g(u0 + step) > g0
    direction = step if (g0 < 0.0) == increasing else -step
    u_far, g_far = u0, g0
    for _ in range(200):
        u_far += direction
        g_far = g(u_far)
        if g0 * g_far <= 0.0:
            break
    else:
        raise ValueError("Could not bracket a solution; check the inputs.")
    lo, hi = sorted((u_far - direction, u_far))
    u = brentq(g, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)
    return math.exp(float(u))


# --- Kinematics ------------------------------------------------------------------------------
def area_m2(diameter_m: F) -> F:
    """Cross-sectional area of a full circular pipe.

    A = pi * D^2 / 4.

    Parameters
    ----------
    diameter_m : float or ndarray
        Internal diameter, m. Must be positive.

    Returns
    -------
    float or ndarray
        Flow area, m^2.
    """
    _require_positive("diameter_m", diameter_m)
    return _like(diameter_m, np.pi * np.asarray(diameter_m) ** 2 / 4.0, "diameter_m")


def velocity_ms(flow_m3s: F, diameter_m: FloatOrArray) -> F:
    """Mean velocity in a full circular pipe.

    V = Q / A = 4 Q / (pi D^2).

    Parameters
    ----------
    flow_m3s : float or ndarray
        Volumetric flow rate, m^3/s. May be negative (reversed flow).
    diameter_m : float or ndarray
        Internal diameter, m. Must be positive.

    Returns
    -------
    float or ndarray
        Mean velocity, m/s, with the sign of the flow.
    """
    _require_positive("diameter_m", diameter_m)
    v = 4.0 * np.asarray(flow_m3s) / (np.pi * np.asarray(diameter_m) ** 2)
    return _like(flow_m3s, v, "flow_m3s")


def reynolds_number(velocity_ms: F, diameter_m: FloatOrArray, kinematic_viscosity_m2s: float) -> F:
    """Reynolds number of pipe flow.

    Re = |V| D / nu.

    Parameters
    ----------
    velocity_ms : float or ndarray
        Mean velocity, m/s. The sign is ignored.
    diameter_m : float or ndarray
        Internal diameter, m. Must be positive.
    kinematic_viscosity_m2s : float
        Kinematic viscosity, m^2/s (see ``wdtk.fluid.kinematic_viscosity_m2s``). Must be
        positive.

    Returns
    -------
    float or ndarray
        Reynolds number, dimensionless (non-negative).
    """
    _require_positive("diameter_m", diameter_m)
    _require_positive("kinematic_viscosity_m2s", kinematic_viscosity_m2s)
    re = np.abs(np.asarray(velocity_ms)) * np.asarray(diameter_m) / kinematic_viscosity_m2s
    return _like(velocity_ms, re, "velocity_ms")


# --- Hazen-Williams --------------------------------------------------------------------------
def _hw_conveyance(diameter: FloatOrArray, c_factor: FloatOrArray, k: float) -> NDArray[np.float64]:
    """Return K_c such that Q = K_c * S^0.54, from V = k C R^0.63 S^0.54 with R = D/4."""
    d = np.asarray(diameter, dtype=np.float64)
    return k * np.asarray(c_factor) * (np.pi * d**2 / 4.0) * (d / 4.0) ** HW_EXP_R


def _hw_headloss(
    flow: F, length: FloatOrArray, diameter: FloatOrArray, c_factor: FloatOrArray, k: float
) -> F:
    _require_positive("length", length)
    _require_positive("diameter", diameter)
    _require_positive("c_factor", c_factor)
    q = np.asarray(flow, dtype=np.float64)
    slope = (np.abs(q) / _hw_conveyance(diameter, c_factor, k)) ** HW_FLOW_EXPONENT
    return _like(flow, np.sign(q) * slope * np.asarray(length), "flow")


def hazen_williams_headloss_m(
    flow_m3s: F, length_m: FloatOrArray, diameter_m: FloatOrArray, c_factor: FloatOrArray
) -> F:
    """Friction head loss by Hazen-Williams, SI units.

    From the defining velocity form V = k C R^0.63 S^0.54 with R = D/4 and S = h/L:

        h = L * (|Q| / (k C (pi D^2/4) (D/4)^0.63))^(1/0.54) * sign(Q),

    i.e. h proportional to Q^1.852 / (C^1.852 D^4.870), with k = ``HW_K_SI``.

    Parameters
    ----------
    flow_m3s : float or ndarray
        Flow rate, m^3/s. May be negative.
    length_m : float or ndarray
        Pipe length, m. Must be positive.
    diameter_m : float or ndarray
        Internal diameter, m. Must be positive.
    c_factor : float or ndarray
        Hazen-Williams roughness coefficient C, dimensionless. Must be positive.

    Returns
    -------
    float or ndarray
        Friction head loss, m, with the sign of the flow.

    Notes
    -----
    Empirical formula, valid for water at ordinary temperatures in turbulent flow. It has
    no viscosity term, so it cannot represent laminar flow or other fluids.
    """
    return _hw_headloss(flow_m3s, length_m, diameter_m, c_factor, HW_K_SI)


def hazen_williams_headloss_ft(
    flow_cfs: F, length_ft: FloatOrArray, diameter_ft: FloatOrArray, c_factor: FloatOrArray
) -> F:
    """Friction head loss by Hazen-Williams, US customary units (ft, cfs).

    Same form as :func:`hazen_williams_headloss_m` with k = ``HW_K_US`` = 1.318
    (ft^0.37/s); equivalent to h = 4.73 L Q^1.852 / (C^1.852 D^4.87) in ft and cfs.

    Parameters
    ----------
    flow_cfs : float or ndarray
        Flow rate, ft^3/s. May be negative.
    length_ft : float or ndarray
        Pipe length, ft. Must be positive.
    diameter_ft : float or ndarray
        Internal diameter, ft. Must be positive.
    c_factor : float or ndarray
        Hazen-Williams roughness coefficient C, dimensionless. Must be positive.

    Returns
    -------
    float or ndarray
        Friction head loss, ft, with the sign of the flow.
    """
    return _hw_headloss(flow_cfs, length_ft, diameter_ft, c_factor, HW_K_US)


def hazen_williams_c_factor(
    flow_m3s: float, length_m: float, diameter_m: float, headloss_m: float
) -> float:
    """Back-calculate the Hazen-Williams C that reproduces an observed head loss.

    From V = k C R^0.63 S^0.54: C = |Q| / (k (pi D^2/4) (D/4)^0.63 (h/L)^0.54).

    Parameters
    ----------
    flow_m3s : float
        Flow rate, m^3/s. Must be non-zero; its sign is ignored.
    length_m : float
        Pipe length, m. Must be positive.
    diameter_m : float
        Internal diameter, m. Must be positive.
    headloss_m : float
        Friction head loss over ``length_m``, m. Must be positive.

    Returns
    -------
    float
        Hazen-Williams C, dimensionless.
    """
    _require_positive("|flow_m3s|", abs(flow_m3s))
    _require_positive("length_m", length_m)
    _require_positive("headloss_m", headloss_m)
    _require_positive("diameter_m", diameter_m)
    unit_c = float(_hw_conveyance(diameter_m, 1.0, HW_K_SI))
    return abs(flow_m3s) / (unit_c * float((headloss_m / length_m) ** HW_EXP_S))


# --- Darcy-Weisbach friction factor ----------------------------------------------------------
def friction_factor_laminar(reynolds: F) -> F:
    """Darcy friction factor for laminar flow (Hagen-Poiseuille).

    f = 64 / Re.

    Parameters
    ----------
    reynolds : float or ndarray
        Reynolds number, dimensionless. Must be positive. Physically valid for
        Re <= ``RE_LAMINAR_MAX``; this function does not enforce the upper limit.

    Returns
    -------
    float or ndarray
        Darcy friction factor, dimensionless.
    """
    _require_positive("reynolds", reynolds)
    return _like(reynolds, 64.0 / np.asarray(reynolds, dtype=np.float64), "reynolds")


def _colebrook_scalar(re: float, rr: float) -> float:
    a = rr / 3.7
    b = 2.51 / re

    def g(x: float) -> float:
        return x + 2.0 * math.log10(a + b * x)

    x = float(brentq(g, *_COLEBROOK_X_BRACKET, xtol=1e-14, rtol=1e-15))
    return 1.0 / (x * x)


def friction_factor_colebrook(reynolds: F, relative_roughness: FloatOrArray) -> F:
    """Darcy friction factor from the Colebrook-White equation (turbulent flow).

    1/sqrt(f) = -2 log10( (eps/D)/3.7 + 2.51/(Re sqrt(f)) ),

    solved for x = 1/sqrt(f) with ``scipy.optimize.brentq``. The residual
    x + 2 log10(a + b x) is strictly increasing in x, so the root is unique.

    Parameters
    ----------
    reynolds : float or ndarray
        Reynolds number, dimensionless. Must be >= ``RE_TURBULENT_MIN``.
    relative_roughness : float or ndarray
        Relative roughness eps/D, dimensionless. Must be non-negative and below 3.7 (the
        equation has no real solution beyond that; pipe values are orders of magnitude
        smaller).

    Returns
    -------
    float or ndarray
        Darcy friction factor, dimensionless.

    Raises
    ------
    ValueError
        If Re is below the turbulent limit or roughness is out of range.
    """
    re = np.asarray(reynolds, dtype=np.float64)
    if np.any(~np.isfinite(re)) or np.any(re < RE_TURBULENT_MIN):
        raise ValueError(
            f"Colebrook-White applies to turbulent flow, Re >= {RE_TURBULENT_MIN:g}; "
            f"got {reynolds!r}. Use friction_factor() for the full range."
        )
    _require_nonnegative("relative_roughness", relative_roughness)
    if np.any(np.asarray(relative_roughness) >= 3.7):
        # The right-hand side is negative when (eps/D)/3.7 >= 1, so no real f exists.
        raise ValueError(
            f"Colebrook-White has no solution for eps/D >= 3.7; got {relative_roughness!r}."
        )
    re_b, rr_b = np.broadcast_arrays(re, np.asarray(relative_roughness, dtype=np.float64))
    f = np.array(
        [_colebrook_scalar(float(r), float(e)) for r, e in zip(re_b.flat, rr_b.flat, strict=True)]
    ).reshape(re_b.shape)
    return _like(reynolds, f, "reynolds")


def friction_factor_swamee_jain(reynolds: F, relative_roughness: FloatOrArray) -> F:
    """Darcy friction factor from the Swamee-Jain explicit approximation.

    f = 0.25 / [ log10( (eps/D)/3.7 + 5.74/Re^0.9 ) ]^2.

    Parameters
    ----------
    reynolds : float or ndarray
        Reynolds number, dimensionless. Must be positive; intended for turbulent flow.
    relative_roughness : float or ndarray
        Relative roughness eps/D, dimensionless. Must be non-negative.

    Returns
    -------
    float or ndarray
        Darcy friction factor, dimensionless.

    Notes
    -----
    Measured in this project's tests against :func:`friction_factor_colebrook` over
    5e3 <= Re <= 1e8 and 1e-6 <= eps/D <= 1e-2: the largest relative deviation is about
    2.8%, at Re = 5e3, eps/D = 1e-2 (the low-Re, high-roughness corner). Deviations exceed
    1% only in that corner region.
    """
    _require_positive("reynolds", reynolds)
    _require_nonnegative("relative_roughness", relative_roughness)
    re = np.asarray(reynolds, dtype=np.float64)
    arg = np.asarray(relative_roughness) / 3.7 + 5.74 / re**0.9
    return _like(reynolds, 0.25 / np.log10(arg) ** 2, "reynolds")


def friction_factor(
    reynolds: F,
    relative_roughness: FloatOrArray,
    turbulent: TurbulentMethod = "colebrook",
) -> F:
    """Darcy friction factor over the full Reynolds number range.

    * Re <= ``RE_LAMINAR_MAX`` (2000): f = 64/Re.
    * Re >= ``RE_TURBULENT_MIN`` (4000): the chosen turbulent formula.
    * In between: linear interpolation in Re between f_lam(2000) and f_turb(4000, eps/D).

    The transitional interpolation is a modeling assumption that keeps f continuous for
    iterative solvers; flow in this range is physically unsteady and f is not well defined.

    Parameters
    ----------
    reynolds : float or ndarray
        Reynolds number, dimensionless. Must be positive.
    relative_roughness : float or ndarray
        Relative roughness eps/D, dimensionless. Must be non-negative.
    turbulent : {"colebrook", "swamee-jain"}, optional
        Turbulent friction factor formula (default ``"colebrook"``).

    Returns
    -------
    float or ndarray
        Darcy friction factor, dimensionless.
    """
    _require_positive("reynolds", reynolds)
    _require_nonnegative("relative_roughness", relative_roughness)
    turb_func = {
        "colebrook": friction_factor_colebrook,
        "swamee-jain": friction_factor_swamee_jain,
    }.get(turbulent)
    if turb_func is None:
        raise ValueError(f"Unknown turbulent method {turbulent!r}.")

    re, rr = np.broadcast_arrays(
        np.asarray(reynolds, dtype=np.float64), np.asarray(relative_roughness, dtype=np.float64)
    )
    f = np.empty(re.shape, dtype=np.float64)
    lam = re <= RE_LAMINAR_MAX
    turb = re >= RE_TURBULENT_MIN
    trans = ~lam & ~turb
    f[lam] = 64.0 / re[lam]
    if np.any(turb):
        f[turb] = turb_func(re[turb], rr[turb])
    if np.any(trans):
        f_lo = 64.0 / RE_LAMINAR_MAX
        f_hi = turb_func(np.full(np.count_nonzero(trans), RE_TURBULENT_MIN), rr[trans])
        w = (re[trans] - RE_LAMINAR_MAX) / (RE_TURBULENT_MIN - RE_LAMINAR_MAX)
        f[trans] = f_lo + w * (f_hi - f_lo)
    return _like(reynolds, f, "reynolds")


def darcy_weisbach_headloss_m(
    flow_m3s: F,
    length_m: FloatOrArray,
    diameter_m: FloatOrArray,
    roughness_m: FloatOrArray,
    kinematic_viscosity_m2s: float,
    turbulent: TurbulentMethod = "colebrook",
    g_ms2: float = G_STANDARD_MS2,
) -> F:
    """Friction head loss by Darcy-Weisbach.

    h = f (L/D) V|V| / (2g), with f from :func:`friction_factor` at Re = |V| D / nu and
    relative roughness eps/D. Zero flow gives zero head loss.

    Parameters
    ----------
    flow_m3s : float or ndarray
        Flow rate, m^3/s. May be negative.
    length_m : float or ndarray
        Pipe length, m. Must be positive.
    diameter_m : float or ndarray
        Internal diameter, m. Must be positive.
    roughness_m : float or ndarray
        Absolute (equivalent sand-grain) roughness eps, m. Must be non-negative.
    kinematic_viscosity_m2s : float
        Kinematic viscosity, m^2/s. Must be positive.
    turbulent : {"colebrook", "swamee-jain"}, optional
        Turbulent friction factor formula (default ``"colebrook"``).
    g_ms2 : float, optional
        Gravitational acceleration, m/s^2 (default: standard gravity).

    Returns
    -------
    float or ndarray
        Friction head loss, m, with the sign of the flow.
    """
    _require_positive("length_m", length_m)
    _require_nonnegative("roughness_m", roughness_m)
    q = np.asarray(flow_m3s, dtype=np.float64)
    v = np.asarray(velocity_ms(q, diameter_m))
    re = np.asarray(reynolds_number(np.abs(v), diameter_m, kinematic_viscosity_m2s))
    d, rr = np.broadcast_arrays(
        np.asarray(diameter_m, dtype=np.float64),
        np.asarray(roughness_m, dtype=np.float64) / np.asarray(diameter_m, dtype=np.float64),
    )
    re, d, rr, length = np.broadcast_arrays(re, d, rr, np.asarray(length_m, dtype=np.float64))
    v = np.broadcast_to(v, re.shape)

    h = np.zeros(re.shape, dtype=np.float64)
    moving = re > 0.0
    if np.any(moving):
        f = friction_factor(re[moving], rr[moving], turbulent)
        h[moving] = f * length[moving] / d[moving] * v[moving] * np.abs(v[moving]) / (2.0 * g_ms2)
    return _like(flow_m3s, h, "flow_m3s")


# --- Minor losses ----------------------------------------------------------------------------
def minor_loss_m(velocity_ms: F, k: float | Sequence[float], g_ms2: float = G_STANDARD_MS2) -> F:
    """Minor (local) head loss from fittings, valves, entrances, and exits.

    h_m = (sum K) V|V| / (2g).

    Parameters
    ----------
    velocity_ms : float or ndarray
        Mean velocity at the reference section, m/s. May be negative.
    k : float or sequence of float
        Loss coefficient(s) K, dimensionless, referenced to ``velocity_ms``; summed if a
        sequence. Each must be non-negative.
    g_ms2 : float, optional
        Gravitational acceleration, m/s^2 (default: standard gravity).

    Returns
    -------
    float or ndarray
        Minor head loss, m, with the sign of the velocity.
    """
    k_arr = np.atleast_1d(np.asarray(k, dtype=np.float64))
    _require_nonnegative("k", k_arr)
    v = np.asarray(velocity_ms, dtype=np.float64)
    return _like(velocity_ms, float(k_arr.sum()) * v * np.abs(v) / (2.0 * g_ms2), "velocity_ms")


# --- Inverse problems ------------------------------------------------------------------------
def diameter_for_velocity_m(flow_m3s: F, velocity_ms: FloatOrArray) -> F:
    """Diameter that carries a flow at a given mean velocity.

    D = sqrt(4 |Q| / (pi V)).

    Parameters
    ----------
    flow_m3s : float or ndarray
        Flow rate, m^3/s. Must be non-zero; its sign is ignored.
    velocity_ms : float or ndarray
        Target mean velocity, m/s. Must be positive.

    Returns
    -------
    float or ndarray
        Internal diameter, m.
    """
    q = np.abs(np.asarray(flow_m3s, dtype=np.float64))
    _require_positive("|flow_m3s|", q)
    _require_positive("velocity_ms", velocity_ms)
    return _like(flow_m3s, np.sqrt(4.0 * q / (np.pi * np.asarray(velocity_ms))), "flow_m3s")


def hazen_williams_diameter_m(
    flow_m3s: F, length_m: FloatOrArray, headloss_m: FloatOrArray, c_factor: FloatOrArray
) -> F:
    """Diameter that gives a target Hazen-Williams friction head loss.

    Inverting Q = k C (pi/4) 4^-0.63 D^2.63 S^0.54 with S = h/L:

        D = ( |Q| / (k C (pi/4) 4^-0.63 S^0.54) )^(1/2.63).

    Parameters
    ----------
    flow_m3s : float or ndarray
        Flow rate, m^3/s. Must be non-zero; its sign is ignored.
    length_m : float or ndarray
        Pipe length, m. Must be positive.
    headloss_m : float or ndarray
        Target friction head loss, m. Must be positive.
    c_factor : float or ndarray
        Hazen-Williams C, dimensionless. Must be positive.

    Returns
    -------
    float or ndarray
        Internal diameter, m.
    """
    q = np.abs(np.asarray(flow_m3s, dtype=np.float64))
    _require_positive("|flow_m3s|", q)
    _require_positive("length_m", length_m)
    _require_positive("headloss_m", headloss_m)
    _require_positive("c_factor", c_factor)
    slope = np.asarray(headloss_m) / np.asarray(length_m)
    coef = HW_K_SI * np.asarray(c_factor) * (np.pi / 4.0) * 4.0**-HW_EXP_R * slope**HW_EXP_S
    return _like(flow_m3s, (q / coef) ** (1.0 / (2.0 + HW_EXP_R)), "flow_m3s")


def hazen_williams_capacity_m3s(
    available_head_m: float,
    length_m: float,
    diameter_m: float,
    c_factor: float,
    minor_k: float = 0.0,
) -> float:
    """Flow a pipe carries for a given available head, by Hazen-Williams.

    Solves H = h_f(Q) + K V^2/(2g). Without minor losses this is closed form,
    Q = k C (pi D^2/4) (D/4)^0.63 (H/L)^0.54; otherwise it is solved with ``brentq``.

    Parameters
    ----------
    available_head_m : float
        Head available to overcome friction and minor losses, m. Must be positive.
    length_m : float
        Pipe length, m. Must be positive.
    diameter_m : float
        Internal diameter, m. Must be positive.
    c_factor : float
        Hazen-Williams C, dimensionless. Must be positive.
    minor_k : float, optional
        Sum of minor loss coefficients referenced to the pipe velocity (default 0).

    Returns
    -------
    float
        Flow rate, m^3/s (positive).
    """
    _require_positive("available_head_m", available_head_m)
    _require_positive("length_m", length_m)
    _require_positive("c_factor", c_factor)
    _require_nonnegative("minor_k", minor_k)
    q_friction_only = float(_hw_conveyance(diameter_m, c_factor, HW_K_SI)) * float(
        (available_head_m / length_m) ** HW_EXP_S
    )
    if minor_k == 0.0:
        return q_friction_only

    def total_head(q: float) -> float:
        hf = hazen_williams_headloss_m(q, length_m, diameter_m, c_factor)
        return hf + minor_loss_m(velocity_ms(q, diameter_m), minor_k)

    return _solve_positive(total_head, available_head_m, q_friction_only)


def darcy_weisbach_diameter_m(
    flow_m3s: float,
    length_m: float,
    headloss_m: float,
    roughness_m: float,
    kinematic_viscosity_m2s: float,
    turbulent: TurbulentMethod = "colebrook",
) -> float:
    """Diameter that gives a target Darcy-Weisbach friction head loss.

    Solves h_f(D) = h for D with ``brentq`` (in log D). Head loss decreases monotonically
    with D, so the solution is unique.

    Parameters
    ----------
    flow_m3s : float
        Flow rate, m^3/s. Must be non-zero; its sign is ignored.
    length_m : float
        Pipe length, m. Must be positive.
    headloss_m : float
        Target friction head loss, m. Must be positive.
    roughness_m : float
        Absolute roughness eps, m. Must be non-negative.
    kinematic_viscosity_m2s : float
        Kinematic viscosity, m^2/s. Must be positive.
    turbulent : {"colebrook", "swamee-jain"}, optional
        Turbulent friction factor formula (default ``"colebrook"``).

    Returns
    -------
    float
        Internal diameter, m.
    """
    q = abs(flow_m3s)
    _require_positive("|flow_m3s|", q)
    _require_positive("headloss_m", headloss_m)

    def headloss(d: float) -> float:
        return darcy_weisbach_headloss_m(
            q, length_m, d, roughness_m, kinematic_viscosity_m2s, turbulent
        )

    return _solve_positive(headloss, headloss_m, x0=diameter_for_velocity_m(q, 1.0))


def darcy_weisbach_capacity_m3s(
    available_head_m: float,
    length_m: float,
    diameter_m: float,
    roughness_m: float,
    kinematic_viscosity_m2s: float,
    minor_k: float = 0.0,
    turbulent: TurbulentMethod = "colebrook",
) -> float:
    """Flow a pipe carries for a given available head, by Darcy-Weisbach.

    Solves H = f(Q) (L/D) V^2/(2g) + K V^2/(2g) for Q with ``brentq`` (in log Q). Total
    head loss increases monotonically with Q, so the solution is unique.

    Parameters
    ----------
    available_head_m : float
        Head available to overcome friction and minor losses, m. Must be positive.
    length_m : float
        Pipe length, m. Must be positive.
    diameter_m : float
        Internal diameter, m. Must be positive.
    roughness_m : float
        Absolute roughness eps, m. Must be non-negative.
    kinematic_viscosity_m2s : float
        Kinematic viscosity, m^2/s. Must be positive.
    minor_k : float, optional
        Sum of minor loss coefficients referenced to the pipe velocity (default 0).
    turbulent : {"colebrook", "swamee-jain"}, optional
        Turbulent friction factor formula (default ``"colebrook"``).

    Returns
    -------
    float
        Flow rate, m^3/s (positive).
    """
    _require_positive("available_head_m", available_head_m)
    _require_nonnegative("minor_k", minor_k)

    def total_head(q: float) -> float:
        hf = darcy_weisbach_headloss_m(
            q, length_m, diameter_m, roughness_m, kinematic_viscosity_m2s, turbulent
        )
        return hf + minor_loss_m(velocity_ms(q, diameter_m), minor_k)

    return _solve_positive(total_head, available_head_m, x0=float(area_m2(diameter_m)))
