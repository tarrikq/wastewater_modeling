"""Physical properties of liquid water as a function of temperature.

Properties are interpolated from a tabulated data set shipped with the package
(``wdtk/data/water_properties.csv``). The table's source is recorded in a ``# source:``
comment line in that file and exposed through :func:`table_source`.

Interpolation uses a cubic spline (``scipy.interpolate.CubicSpline``, not-a-knot end
conditions), which passes exactly through every tabulated point. Values are never
extrapolated. Accepted temperatures are the intersection of the table's range and
``VALID_RANGE_C`` (0 to 40 degC, which covers potable distribution systems); anything
outside raises ``ValueError``.
"""

from __future__ import annotations

import csv
import functools
import io
from dataclasses import dataclass
from importlib import resources
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from wdtk.units import G_STANDARD_MS2

F = TypeVar("F", float, NDArray[np.float64])

VALID_RANGE_C: tuple[float, float] = (0.0, 40.0)
"""Intended temperature range of the property functions, degC."""

_COLUMNS = ("temperature_c", "density_kgm3", "kinematic_viscosity_m2s", "vapor_pressure_kpa")
_SOURCE_PREFIX = "# source:"
_MIN_POINTS = 4


@dataclass(frozen=True, eq=False)
class PropertyTable:
    """Tabulated water properties.

    Attributes
    ----------
    temperature_c : ndarray
        Temperatures, degC, strictly increasing.
    density_kgm3 : ndarray
        Density at each temperature, kg/m^3.
    kinematic_viscosity_m2s : ndarray
        Kinematic viscosity at each temperature, m^2/s.
    vapor_pressure_kpa : ndarray
        Saturation vapor pressure (absolute) at each temperature, kPa.
    source : str
        Citation for the tabulated values.
    """

    temperature_c: NDArray[np.float64]
    density_kgm3: NDArray[np.float64]
    kinematic_viscosity_m2s: NDArray[np.float64]
    vapor_pressure_kpa: NDArray[np.float64]
    source: str


@dataclass(frozen=True, eq=False)
class _Interpolants:
    t_min_c: float
    t_max_c: float
    density: CubicSpline
    kinematic_viscosity: CubicSpline
    vapor_pressure: CubicSpline


def parse_property_table(text: str) -> PropertyTable:
    """Parse a water property table from CSV text.

    Expected format: optional ``#`` comment lines (one of which should be
    ``# source: <citation>``), then a header row with the columns ``temperature_c``,
    ``density_kgm3``, ``kinematic_viscosity_m2s``, ``vapor_pressure_kpa``, then one row per
    temperature.

    Parameters
    ----------
    text : str
        CSV file contents.

    Returns
    -------
    PropertyTable
        Parsed table, with rows in the file order.

    Raises
    ------
    ValueError
        If columns are missing, there are fewer than four rows, temperatures are not
        strictly increasing, any property is non-positive, or no source line is present.
    """
    source = ""
    data_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(_SOURCE_PREFIX):
            source = stripped[len(_SOURCE_PREFIX) :].strip()
        elif stripped and not stripped.startswith("#"):
            data_lines.append(stripped)

    if not source:
        raise ValueError("Property table must include a '# source: <citation>' line.")

    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
    missing = [c for c in _COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"Property table is missing columns: {missing}.")

    rows = list(reader)
    if len(rows) < _MIN_POINTS:
        raise ValueError(f"Property table needs at least {_MIN_POINTS} rows; got {len(rows)}.")

    cols = {c: np.array([float(r[c]) for r in rows], dtype=np.float64) for c in _COLUMNS}
    if np.any(np.diff(cols["temperature_c"]) <= 0.0):
        raise ValueError("Property table temperatures must be strictly increasing.")
    for c in _COLUMNS[1:]:
        if np.any(cols[c] <= 0.0):
            raise ValueError(f"Property table column {c!r} must be positive.")

    return PropertyTable(
        temperature_c=cols["temperature_c"],
        density_kgm3=cols["density_kgm3"],
        kinematic_viscosity_m2s=cols["kinematic_viscosity_m2s"],
        vapor_pressure_kpa=cols["vapor_pressure_kpa"],
        source=source,
    )


@functools.cache
def packaged_table() -> PropertyTable:
    """Load the water property table shipped with the package.

    Returns
    -------
    PropertyTable
        Contents of ``wdtk/data/water_properties.csv``.

    Raises
    ------
    FileNotFoundError
        If the data file has not been provided.
    """
    path = resources.files("wdtk").joinpath("data", "water_properties.csv")
    if not path.is_file():
        raise FileNotFoundError(
            "wdtk/data/water_properties.csv is missing; supply a sourced property table."
        )
    return parse_property_table(path.read_text(encoding="utf-8"))


def table_source() -> str:
    """Return the citation of the packaged property table.

    Returns
    -------
    str
        The text following ``# source:`` in the data file.
    """
    return packaged_table().source


def _build_interpolants(table: PropertyTable) -> _Interpolants:
    return _Interpolants(
        t_min_c=max(float(table.temperature_c[0]), VALID_RANGE_C[0]),
        t_max_c=min(float(table.temperature_c[-1]), VALID_RANGE_C[1]),
        density=CubicSpline(table.temperature_c, table.density_kgm3),
        kinematic_viscosity=CubicSpline(table.temperature_c, table.kinematic_viscosity_m2s),
        vapor_pressure=CubicSpline(table.temperature_c, table.vapor_pressure_kpa),
    )


@functools.cache
def _packaged_interpolants() -> _Interpolants:
    return _build_interpolants(packaged_table())


def _interpolants(table: PropertyTable | None) -> _Interpolants:
    return _packaged_interpolants() if table is None else _build_interpolants(table)


def _evaluate(spline: CubicSpline, interp: _Interpolants, temperature_c: F) -> F:
    t = np.asarray(temperature_c, dtype=np.float64)
    if np.any(~np.isfinite(t)) or np.any(t < interp.t_min_c) or np.any(t > interp.t_max_c):
        raise ValueError(
            f"Temperature outside tabulated range [{interp.t_min_c}, {interp.t_max_c}] degC: "
            f"{temperature_c!r}."
        )
    out = np.asarray(spline(t), dtype=np.float64)
    result: float | NDArray[np.float64] = float(out) if out.ndim == 0 else out
    return cast(F, result)


def density_kgm3(temperature_c: F, table: PropertyTable | None = None) -> F:
    """Density of liquid water at atmospheric pressure.

    rho(T) is a cubic-spline interpolant through tabulated values (see module docstring).

    Parameters
    ----------
    temperature_c : float or ndarray
        Water temperature, degC. Must lie within the table's range (intended 0 to 40).
    table : PropertyTable, optional
        Property table to interpolate. Defaults to the packaged table.

    Returns
    -------
    float or ndarray
        Density, kg/m^3.

    Raises
    ------
    ValueError
        If any temperature is outside the tabulated range or not finite.
    """
    interp = _interpolants(table)
    return _evaluate(interp.density, interp, temperature_c)


def kinematic_viscosity_m2s(temperature_c: F, table: PropertyTable | None = None) -> F:
    """Kinematic viscosity of liquid water at atmospheric pressure.

    nu(T) is a cubic-spline interpolant through tabulated values (see module docstring).

    Parameters
    ----------
    temperature_c : float or ndarray
        Water temperature, degC. Must lie within the table's range (intended 0 to 40).
    table : PropertyTable, optional
        Property table to interpolate. Defaults to the packaged table.

    Returns
    -------
    float or ndarray
        Kinematic viscosity, m^2/s.

    Raises
    ------
    ValueError
        If any temperature is outside the tabulated range or not finite.
    """
    interp = _interpolants(table)
    return _evaluate(interp.kinematic_viscosity, interp, temperature_c)


def dynamic_viscosity_pas(temperature_c: F, table: PropertyTable | None = None) -> F:
    """Dynamic (absolute) viscosity of liquid water.

    mu = rho * nu, using the interpolated density and kinematic viscosity.

    Parameters
    ----------
    temperature_c : float or ndarray
        Water temperature, degC. Must lie within the table's range (intended 0 to 40).
    table : PropertyTable, optional
        Property table to interpolate. Defaults to the packaged table.

    Returns
    -------
    float or ndarray
        Dynamic viscosity, Pa*s.

    Raises
    ------
    ValueError
        If any temperature is outside the tabulated range or not finite.
    """
    return density_kgm3(temperature_c, table) * kinematic_viscosity_m2s(temperature_c, table)


def vapor_pressure_kpa(temperature_c: F, table: PropertyTable | None = None) -> F:
    """Saturation vapor pressure of water (absolute).

    p_v(T) is a cubic-spline interpolant through tabulated values (see module docstring).
    Used for net positive suction head: NPSHa includes -p_v / (rho g).

    Parameters
    ----------
    temperature_c : float or ndarray
        Water temperature, degC. Must lie within the table's range (intended 0 to 40).
    table : PropertyTable, optional
        Property table to interpolate. Defaults to the packaged table.

    Returns
    -------
    float or ndarray
        Vapor pressure, kPa absolute.

    Raises
    ------
    ValueError
        If any temperature is outside the tabulated range or not finite.
    """
    interp = _interpolants(table)
    return _evaluate(interp.vapor_pressure, interp, temperature_c)


def specific_weight_nm3(
    temperature_c: F,
    g_ms2: float = G_STANDARD_MS2,
    table: PropertyTable | None = None,
) -> F:
    """Specific weight of liquid water.

    gamma = rho * g.

    Parameters
    ----------
    temperature_c : float or ndarray
        Water temperature, degC. Must lie within the table's range (intended 0 to 40).
    g_ms2 : float, optional
        Gravitational acceleration, m/s^2 (default: standard gravity).
    table : PropertyTable, optional
        Property table to interpolate. Defaults to the packaged table.

    Returns
    -------
    float or ndarray
        Specific weight, N/m^3.

    Raises
    ------
    ValueError
        If any temperature is outside the tabulated range or not finite.
    """
    return density_kgm3(temperature_c, table) * g_ms2
