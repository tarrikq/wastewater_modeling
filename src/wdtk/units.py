"""Unit conversions and physical constants.

All internal calculations in ``wdtk`` use SI: m, s, m^3/s, m of head, and kPa where a
pressure is needed. This module is the only place unit factors are defined.

Every factor below converts *to* SI and is exact by definition (international yard and
pound of 1959, standard gravity of the 3rd CGPM, 1901) or derived arithmetically from exact
definitions. No empirically measured values appear here.

All conversion functions accept a Python ``float`` or a NumPy ``float64`` array and return
the same type. Conversions are linear, so they are valid for any finite input, including
negative values (e.g. reversed flow).
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

F = TypeVar("F", float, NDArray[np.float64])

# --- Exact base definitions ---------------------------------------------------------------
G_STANDARD_MS2 = 9.80665
"""Standard acceleration of gravity, m/s^2 (exact by definition)."""

FT_M = 0.3048
"""International foot, m (exact)."""

IN_M = 0.0254
"""International inch, m (exact)."""

LB_KG = 0.45359237
"""International avoirdupois pound mass, kg (exact)."""

US_GAL_M3 = 231.0 * IN_M**3
"""US liquid gallon (231 in^3), m^3 (exact)."""

LBF_N = LB_KG * G_STANDARD_MS2
"""Pound-force, N (exact): 1 lbm under standard gravity."""

# --- Derived factors ----------------------------------------------------------------------
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_DAY = 86_400.0

GPM_M3S = US_GAL_M3 / SECONDS_PER_MINUTE
"""US gallon per minute, m^3/s."""

MGD_M3S = 1.0e6 * US_GAL_M3 / SECONDS_PER_DAY
"""Million US gallons per day, m^3/s."""

CFS_M3S = FT_M**3
"""Cubic foot per second, m^3/s."""

LPS_M3S = 1.0e-3
"""Litre per second, m^3/s."""

PSI_PA = LBF_N / IN_M**2
"""Pound-force per square inch, Pa."""

DEFAULT_DENSITY_KGM3 = 1000.0
"""Conventional water density, kg/m^3, used for pressure-head conversions.

This is a stated convention, not a property value. With standard gravity it gives
1 ft of head = 0.4335 psi (1 psi = 2.307 ft). Pass a temperature-dependent value from
``wdtk.fluid`` where the distinction matters.
"""

FLOW_TO_M3S: dict[str, float] = {
    "m3/s": 1.0,
    "L/s": LPS_M3S,
    "gpm": GPM_M3S,
    "MGD": MGD_M3S,
    "cfs": CFS_M3S,
}
"""Multiply a flow in the keyed unit by this factor to get m^3/s."""

LENGTH_TO_M: dict[str, float] = {
    "m": 1.0,
    "mm": 1.0e-3,
    "ft": FT_M,
    "in": IN_M,
}
"""Multiply a length in the keyed unit by this factor to get m."""

PRESSURE_TO_KPA: dict[str, float] = {
    "kPa": 1.0,
    "Pa": 1.0e-3,
    "psi": PSI_PA * 1.0e-3,
}
"""Multiply a pressure in the keyed unit by this factor to get kPa."""


def _factor(table: dict[str, float], unit: str, quantity: str) -> float:
    """Look up a conversion factor, failing loudly on an unknown unit."""
    try:
        return table[unit]
    except KeyError:
        valid = ", ".join(repr(u) for u in table)
        raise ValueError(f"Unknown {quantity} unit {unit!r}; expected one of {valid}.") from None


def flow_to_m3s(value: F, unit: str) -> F:
    """Convert a volumetric flow rate to SI.

    Q_m3s = Q_unit * k_unit, where k_unit is ``FLOW_TO_M3S[unit]``.

    Parameters
    ----------
    value : float or ndarray
        Flow rate expressed in ``unit``.
    unit : str
        One of ``"m3/s"``, ``"L/s"``, ``"gpm"`` (US gal/min), ``"MGD"`` (million US gal/day),
        ``"cfs"`` (ft^3/s).

    Returns
    -------
    float or ndarray
        Flow rate, m^3/s.

    Raises
    ------
    ValueError
        If ``unit`` is not recognized.
    """
    return value * _factor(FLOW_TO_M3S, unit, "flow")


def flow_from_m3s(flow_m3s: F, unit: str) -> F:
    """Convert a volumetric flow rate from SI.

    Q_unit = Q_m3s / k_unit, where k_unit is ``FLOW_TO_M3S[unit]``.

    Parameters
    ----------
    flow_m3s : float or ndarray
        Flow rate, m^3/s.
    unit : str
        Target unit; see :func:`flow_to_m3s`.

    Returns
    -------
    float or ndarray
        Flow rate expressed in ``unit``.

    Raises
    ------
    ValueError
        If ``unit`` is not recognized.
    """
    return flow_m3s / _factor(FLOW_TO_M3S, unit, "flow")


def length_to_m(value: F, unit: str) -> F:
    """Convert a length to SI.

    L_m = L_unit * k_unit, where k_unit is ``LENGTH_TO_M[unit]``.

    Parameters
    ----------
    value : float or ndarray
        Length expressed in ``unit``.
    unit : str
        One of ``"m"``, ``"mm"``, ``"ft"``, ``"in"``.

    Returns
    -------
    float or ndarray
        Length, m.

    Raises
    ------
    ValueError
        If ``unit`` is not recognized.
    """
    return value * _factor(LENGTH_TO_M, unit, "length")


def length_from_m(length_m: F, unit: str) -> F:
    """Convert a length from SI.

    L_unit = L_m / k_unit, where k_unit is ``LENGTH_TO_M[unit]``.

    Parameters
    ----------
    length_m : float or ndarray
        Length, m.
    unit : str
        Target unit; see :func:`length_to_m`.

    Returns
    -------
    float or ndarray
        Length expressed in ``unit``.

    Raises
    ------
    ValueError
        If ``unit`` is not recognized.
    """
    return length_m / _factor(LENGTH_TO_M, unit, "length")


def pressure_to_kpa(value: F, unit: str) -> F:
    """Convert a pressure to kPa.

    p_kPa = p_unit * k_unit, where k_unit is ``PRESSURE_TO_KPA[unit]``.

    Parameters
    ----------
    value : float or ndarray
        Pressure expressed in ``unit`` (gauge or absolute; the conversion is the same).
    unit : str
        One of ``"kPa"``, ``"Pa"``, ``"psi"``. For pressures expressed as a height of water
        column, use :func:`head_to_pressure`.

    Returns
    -------
    float or ndarray
        Pressure, kPa.

    Raises
    ------
    ValueError
        If ``unit`` is not recognized.
    """
    return value * _factor(PRESSURE_TO_KPA, unit, "pressure")


def pressure_from_kpa(pressure_kpa: F, unit: str) -> F:
    """Convert a pressure from kPa.

    p_unit = p_kPa / k_unit, where k_unit is ``PRESSURE_TO_KPA[unit]``.

    Parameters
    ----------
    pressure_kpa : float or ndarray
        Pressure, kPa.
    unit : str
        Target unit; see :func:`pressure_to_kpa`.

    Returns
    -------
    float or ndarray
        Pressure expressed in ``unit``.

    Raises
    ------
    ValueError
        If ``unit`` is not recognized.
    """
    return pressure_kpa / _factor(PRESSURE_TO_KPA, unit, "pressure")


def _check_positive(name: str, value: float) -> None:
    if not value > 0.0:
        raise ValueError(f"{name} must be positive; got {value!r}.")


def head_m_to_pressure_kpa(
    head_m: F,
    density_kgm3: float = DEFAULT_DENSITY_KGM3,
    g_ms2: float = G_STANDARD_MS2,
) -> F:
    """Convert a pressure head (height of water column) to pressure.

    Hydrostatic relation: p = rho * g * h, with p in Pa, then divided by 1000 for kPa.

    Parameters
    ----------
    head_m : float or ndarray
        Pressure head, m of water.
    density_kgm3 : float, optional
        Water density, kg/m^3 (default: the 1000 kg/m^3 convention, see
        ``DEFAULT_DENSITY_KGM3``). Must be positive.
    g_ms2 : float, optional
        Gravitational acceleration, m/s^2 (default: standard gravity). Must be positive.

    Returns
    -------
    float or ndarray
        Pressure, kPa.

    Raises
    ------
    ValueError
        If ``density_kgm3`` or ``g_ms2`` is not positive.
    """
    _check_positive("density_kgm3", density_kgm3)
    _check_positive("g_ms2", g_ms2)
    return head_m * (density_kgm3 * g_ms2 * 1.0e-3)


def pressure_kpa_to_head_m(
    pressure_kpa: F,
    density_kgm3: float = DEFAULT_DENSITY_KGM3,
    g_ms2: float = G_STANDARD_MS2,
) -> F:
    """Convert a pressure to pressure head (height of water column).

    Hydrostatic relation: h = p / (rho * g), with p converted from kPa to Pa.

    Parameters
    ----------
    pressure_kpa : float or ndarray
        Pressure, kPa.
    density_kgm3 : float, optional
        Water density, kg/m^3 (default: the 1000 kg/m^3 convention). Must be positive.
    g_ms2 : float, optional
        Gravitational acceleration, m/s^2 (default: standard gravity). Must be positive.

    Returns
    -------
    float or ndarray
        Pressure head, m of water.

    Raises
    ------
    ValueError
        If ``density_kgm3`` or ``g_ms2`` is not positive.
    """
    _check_positive("density_kgm3", density_kgm3)
    _check_positive("g_ms2", g_ms2)
    return pressure_kpa / (density_kgm3 * g_ms2 * 1.0e-3)


def pressure_to_head(
    value: F,
    unit: str,
    head_unit: str = "m",
    density_kgm3: float = DEFAULT_DENSITY_KGM3,
) -> F:
    """Convert a pressure in any supported unit to pressure head in any length unit.

    h = p / (rho * g), using standard gravity; e.g. psi -> ft of water.

    Parameters
    ----------
    value : float or ndarray
        Pressure expressed in ``unit``.
    unit : str
        Pressure unit; see :func:`pressure_to_kpa`.
    head_unit : str, optional
        Length unit of the returned head; see :func:`length_to_m` (default ``"m"``).
    density_kgm3 : float, optional
        Water density, kg/m^3 (default: the 1000 kg/m^3 convention). Must be positive.

    Returns
    -------
    float or ndarray
        Pressure head, in ``head_unit`` of water.

    Raises
    ------
    ValueError
        If a unit is not recognized or ``density_kgm3`` is not positive.
    """
    head_m = pressure_kpa_to_head_m(pressure_to_kpa(value, unit), density_kgm3)
    return length_from_m(head_m, head_unit)


def head_to_pressure(
    value: F,
    head_unit: str,
    unit: str,
    density_kgm3: float = DEFAULT_DENSITY_KGM3,
) -> F:
    """Convert a pressure head in any length unit to pressure in any supported unit.

    p = rho * g * h, using standard gravity; e.g. ft of water -> psi.

    Parameters
    ----------
    value : float or ndarray
        Pressure head, in ``head_unit`` of water.
    head_unit : str
        Length unit of ``value``; see :func:`length_to_m`.
    unit : str
        Pressure unit of the result; see :func:`pressure_to_kpa`.
    density_kgm3 : float, optional
        Water density, kg/m^3 (default: the 1000 kg/m^3 convention). Must be positive.

    Returns
    -------
    float or ndarray
        Pressure expressed in ``unit``.

    Raises
    ------
    ValueError
        If a unit is not recognized or ``density_kgm3`` is not positive.
    """
    pressure_kpa = head_m_to_pressure_kpa(length_to_m(value, head_unit), density_kgm3)
    return pressure_from_kpa(pressure_kpa, unit)
