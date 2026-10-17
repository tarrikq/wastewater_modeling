"""Hazen-Williams vs Darcy-Weisbach comparison (Phase 1 figures).

Matches HW to DW at one design point by back-calculating C, then measures how far the two
formulas drift apart across diameter and velocity. Matching at one point avoids assuming a
correspondence between a C-factor and a roughness height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wdtk import headloss as hl


@dataclass(frozen=True)
class RatioGrid:
    """h_HW / h_DW over a diameter x velocity grid, with HW calibrated at one point.

    Attributes
    ----------
    diameter_m : ndarray
        Diameters, m (grid columns).
    velocity_ms : ndarray
        Velocities, m/s (grid rows).
    ratio : ndarray
        h_HW / h_DW, dimensionless, shape (n_velocity, n_diameter).
    c_calibrated : float
        Hazen-Williams C that matches DW at the calibration point.
    """

    diameter_m: NDArray[np.float64]
    velocity_ms: NDArray[np.float64]
    ratio: NDArray[np.float64]
    c_calibrated: float


def calibrated_c(
    diameter_m: float, velocity_ms: float, roughness_m: float, nu_m2s: float, length_m: float
) -> float:
    """C that makes Hazen-Williams reproduce the Darcy-Weisbach head loss at one point.

    Parameters
    ----------
    diameter_m, velocity_ms : float
        Calibration point, m and m/s.
    roughness_m : float
        Absolute roughness for DW, m.
    nu_m2s : float
        Kinematic viscosity, m^2/s.
    length_m : float
        Pipe length, m (both formulas are linear in L, so any positive value works).

    Returns
    -------
    float
        Hazen-Williams C, dimensionless.
    """
    q = velocity_ms * float(hl.area_m2(diameter_m))
    h = hl.darcy_weisbach_headloss_m(q, length_m, diameter_m, roughness_m, nu_m2s)
    return hl.hazen_williams_c_factor(q, length_m, diameter_m, h)


def ratio_grid(
    diameter_m: NDArray[np.float64],
    velocity_ms: NDArray[np.float64],
    roughness_m: float,
    nu_m2s: float,
    calibration: tuple[float, float],
    length_m: float = 1000.0,
) -> RatioGrid:
    """Ratio h_HW / h_DW across a grid, with C calibrated at ``calibration`` (D m, V m/s).

    Parameters
    ----------
    diameter_m, velocity_ms : ndarray
        Grid axes, m and m/s.
    roughness_m : float
        Absolute roughness, m.
    nu_m2s : float
        Kinematic viscosity, m^2/s.
    calibration : tuple of float
        (diameter m, velocity m/s) where HW and DW agree.
    length_m : float, optional
        Pipe length, m (does not affect the ratio).

    Returns
    -------
    RatioGrid
        The grid and the calibrated C.
    """
    c = calibrated_c(calibration[0], calibration[1], roughness_m, nu_m2s, length_m)
    d, v = np.meshgrid(diameter_m, velocity_ms)
    q = v * np.pi * d**2 / 4.0
    h_dw = hl.darcy_weisbach_headloss_m(q, length_m, d, roughness_m, nu_m2s)
    h_hw = hl.hazen_williams_headloss_m(q, length_m, d, c)
    return RatioGrid(np.asarray(diameter_m), np.asarray(velocity_ms), h_hw / h_dw, c)


def equivalent_c(
    velocity_ms: NDArray[np.float64],
    diameter_m: float,
    roughness_m: float,
    nu_m2s: float,
    length_m: float = 1000.0,
) -> NDArray[np.float64]:
    """C that reproduces the DW head loss at each velocity for one pipe.

    If Hazen-Williams were physically exact, this would be constant for a given pipe.

    Parameters
    ----------
    velocity_ms : ndarray
        Velocities, m/s.
    diameter_m : float
        Diameter, m.
    roughness_m : float
        Absolute roughness, m.
    nu_m2s : float
        Kinematic viscosity, m^2/s.
    length_m : float, optional
        Pipe length, m (does not affect C).

    Returns
    -------
    ndarray
        Equivalent C at each velocity, dimensionless.
    """
    return np.array(
        [calibrated_c(diameter_m, float(v), roughness_m, nu_m2s, length_m) for v in velocity_ms]
    )
