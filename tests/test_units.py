"""Tests for wdtk.units.

Expected values are either exact definitions recomputed here or self-consistency checks
(round trips); no external reference values are used.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from wdtk import units

REL = 1e-12

finite = st.floats(
    min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False, allow_subnormal=False
)
density = st.floats(min_value=900.0, max_value=1100.0)


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL, abs_tol=0.0)


# --- Round trips ----------------------------------------------------------------------------


@pytest.mark.parametrize("unit", list(units.FLOW_TO_M3S))
@given(x=finite)
def test_flow_round_trip(unit: str, x: float) -> None:
    assert close(units.flow_from_m3s(units.flow_to_m3s(x, unit), unit), x)


@pytest.mark.parametrize("unit", list(units.LENGTH_TO_M))
@given(x=finite)
def test_length_round_trip(unit: str, x: float) -> None:
    assert close(units.length_from_m(units.length_to_m(x, unit), unit), x)


@pytest.mark.parametrize("unit", list(units.PRESSURE_TO_KPA))
@given(x=finite)
def test_pressure_round_trip(unit: str, x: float) -> None:
    assert close(units.pressure_from_kpa(units.pressure_to_kpa(x, unit), unit), x)


@given(x=finite, rho=density)
def test_head_pressure_si_round_trip(x: float, rho: float) -> None:
    p = units.head_m_to_pressure_kpa(x, rho)
    assert close(units.pressure_kpa_to_head_m(p, rho), x)


@pytest.mark.parametrize("pressure_unit", list(units.PRESSURE_TO_KPA))
@pytest.mark.parametrize("head_unit", ["m", "ft"])
@given(x=finite, rho=density)
def test_pressure_head_round_trip(pressure_unit: str, head_unit: str, x: float, rho: float) -> None:
    h = units.pressure_to_head(x, pressure_unit, head_unit, rho)
    assert close(units.head_to_pressure(h, head_unit, pressure_unit, rho), x)


@given(x=finite)
def test_flow_cross_unit_chain(x: float) -> None:
    """gpm -> cfs -> MGD -> L/s -> gpm returns the starting value."""
    cfs = units.flow_from_m3s(units.flow_to_m3s(x, "gpm"), "cfs")
    mgd = units.flow_from_m3s(units.flow_to_m3s(cfs, "cfs"), "MGD")
    lps = units.flow_from_m3s(units.flow_to_m3s(mgd, "MGD"), "L/s")
    gpm = units.flow_from_m3s(units.flow_to_m3s(lps, "L/s"), "gpm")
    assert close(gpm, x)


# --- Exact-definition identities -------------------------------------------------------------


def test_foot_is_twelve_inches() -> None:
    assert close(units.length_from_m(units.length_to_m(1.0, "ft"), "in"), 12.0)


def test_mgd_in_gpm() -> None:
    # 1 MGD = 1e6 gal / 1440 min
    assert close(units.flow_from_m3s(units.flow_to_m3s(1.0, "MGD"), "gpm"), 1e6 / 1440.0)


def test_cfs_in_gpm() -> None:
    # 1 ft^3 = 1728 in^3 and 1 gal = 231 in^3, so 1 cfs = 60 * 1728 / 231 gpm
    assert close(units.flow_from_m3s(units.flow_to_m3s(1.0, "cfs"), "gpm"), 60 * 1728 / 231)


def test_psi_definition() -> None:
    lbf_n = 0.45359237 * 9.80665
    assert close(units.pressure_to_kpa(1.0, "psi"), lbf_n / 0.0254**2 / 1000.0)


def test_head_hydrostatic() -> None:
    # p = rho g h for 10 m of head at rho = 1000 kg/m^3, standard gravity
    assert close(units.head_m_to_pressure_kpa(10.0), 1000.0 * 9.80665 * 10.0 / 1000.0)


def test_psi_to_ft_head_matches_composition() -> None:
    expected_ft = (units.PSI_PA / (1000.0 * units.G_STANDARD_MS2)) / units.FT_M
    assert close(units.pressure_to_head(1.0, "psi", "ft"), expected_ft)


# --- Arrays and errors -----------------------------------------------------------------------


def test_array_input_preserves_shape() -> None:
    q = np.linspace(0.0, 1000.0, 12).reshape(3, 4)
    out = units.flow_to_m3s(q, "gpm")
    assert out.shape == q.shape
    np.testing.assert_allclose(units.flow_from_m3s(out, "gpm"), q, rtol=REL)


@pytest.mark.parametrize(
    "func",
    [
        units.flow_to_m3s,
        units.flow_from_m3s,
        units.length_to_m,
        units.length_from_m,
        units.pressure_to_kpa,
        units.pressure_from_kpa,
    ],
)
def test_unknown_unit_raises(func: object) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        func(1.0, "furlongs")  # type: ignore[operator]


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_density_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="density"):
        units.head_m_to_pressure_kpa(1.0, density_kgm3=bad)
    with pytest.raises(ValueError, match="g_ms2"):
        units.pressure_kpa_to_head_m(1.0, g_ms2=bad)
