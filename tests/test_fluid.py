"""Tests for wdtk.fluid.

Two groups:

* Interpolation machinery, tested on a *synthetic* table generated from closed-form
  polynomials in the test itself. These are not water properties; they only check that the
  spline, range checks, and derived quantities behave correctly.
* The packaged table (user-supplied, sourced data), tested for exact reproduction of its
  rows and basic physical behavior. Skipped until the data file exists.
"""

import numpy as np
import pytest

from wdtk import fluid
from wdtk.units import G_STANDARD_MS2

# --- Synthetic table from closed-form polynomials ----------------------------------------------
# A cubic spline with not-a-knot end conditions reproduces any cubic polynomial exactly, so
# interpolated values between nodes can be checked against the generating function.

T_NODES = np.arange(0.0, 41.0, 5.0)


def synth_rho(t: np.ndarray) -> np.ndarray:
    return 1000.0 - 0.005 * (t - 4.0) ** 2 + 1e-5 * t**3


def synth_nu(t: np.ndarray) -> np.ndarray:
    return 1.8e-6 - 5e-8 * t + 6e-10 * t**2 - 2e-12 * t**3


def synth_pv(t: np.ndarray) -> np.ndarray:
    return 0.6 + 0.04 * t + 1e-3 * t**2 + 2e-5 * t**3


HEADER = "temperature_c,density_kgm3,kinematic_viscosity_m2s,vapor_pressure_kpa"


def synth_csv() -> str:
    lines = [
        "# synthetic test table",
        "# source: closed-form polynomials in tests/test_fluid.py",
        HEADER,
    ]
    lines += [
        ",".join(repr(float(x)) for x in (t, synth_rho(t), synth_nu(t), synth_pv(t)))
        for t in T_NODES
    ]
    return "\n".join(lines)


@pytest.fixture
def table() -> fluid.PropertyTable:
    return fluid.parse_property_table(synth_csv())


def test_parse_reads_source_and_rows(table: fluid.PropertyTable) -> None:
    assert table.source == "closed-form polynomials in tests/test_fluid.py"
    np.testing.assert_array_equal(table.temperature_c, T_NODES)


def test_spline_reproduces_cubic_between_nodes(table: fluid.PropertyTable) -> None:
    t = np.linspace(0.0, 40.0, 97)
    np.testing.assert_allclose(fluid.density_kgm3(t, table), synth_rho(t), rtol=1e-12)
    np.testing.assert_allclose(fluid.kinematic_viscosity_m2s(t, table), synth_nu(t), rtol=1e-10)
    np.testing.assert_allclose(fluid.vapor_pressure_kpa(t, table), synth_pv(t), rtol=1e-12)


def test_scalar_in_scalar_out(table: fluid.PropertyTable) -> None:
    rho = fluid.density_kgm3(12.5, table)
    assert isinstance(rho, float)
    assert rho == pytest.approx(float(synth_rho(np.float64(12.5))), rel=1e-12)


def test_derived_properties(table: fluid.PropertyTable) -> None:
    t = np.array([0.0, 7.3, 21.0, 40.0])
    mu = fluid.dynamic_viscosity_pas(t, table)
    np.testing.assert_allclose(mu, synth_rho(t) * synth_nu(t), rtol=1e-10)
    gamma = fluid.specific_weight_nm3(t, table=table)
    np.testing.assert_allclose(gamma, synth_rho(t) * G_STANDARD_MS2, rtol=1e-12)


@pytest.mark.parametrize("bad", [-0.1, 40.1, float("nan"), float("inf")])
def test_out_of_range_raises(table: fluid.PropertyTable, bad: float) -> None:
    with pytest.raises(ValueError, match="outside tabulated range"):
        fluid.density_kgm3(bad, table)


def test_array_with_one_bad_value_raises(table: fluid.PropertyTable) -> None:
    with pytest.raises(ValueError):
        fluid.kinematic_viscosity_m2s(np.array([10.0, 50.0]), table)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (f"{HEADER}\n0,1,1,1\n1,1,1,1\n2,1,1,1\n3,1,1,1", "source"),
        ("# source: x\ntemperature_c,density_kgm3\n0,1\n1,1\n2,1\n3,1", "missing columns"),
        (f"# source: x\n{HEADER}\n0,1,1,1\n1,1,1,1", "at least"),
        (f"# source: x\n{HEADER}\n0,1,1,1\n2,1,1,1\n1,1,1,1\n3,1,1,1", "strictly increasing"),
        (f"# source: x\n{HEADER}\n0,1,1,1\n1,1,1,0\n2,1,1,1\n3,1,1,1", "positive"),
    ],
)
def test_parse_rejects_bad_tables(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fluid.parse_property_table(text)


# --- Packaged (sourced) table ----------------------------------------------------------------


def _packaged_or_skip() -> fluid.PropertyTable:
    try:
        return fluid.packaged_table()
    except FileNotFoundError:
        pytest.skip("wdtk/data/water_properties.csv not yet provided")


def test_packaged_table_has_source() -> None:
    _packaged_or_skip()
    assert fluid.table_source()


def test_packaged_table_covers_valid_range() -> None:
    tbl = _packaged_or_skip()
    assert tbl.temperature_c[0] <= fluid.VALID_RANGE_C[0]
    assert tbl.temperature_c[-1] >= fluid.VALID_RANGE_C[1]


def test_packaged_spline_reproduces_rows() -> None:
    tbl = _packaged_or_skip()
    lo, hi = fluid.VALID_RANGE_C
    keep = (tbl.temperature_c >= lo) & (tbl.temperature_c <= hi)
    t = tbl.temperature_c[keep]
    np.testing.assert_allclose(fluid.density_kgm3(t), tbl.density_kgm3[keep], rtol=1e-12)
    np.testing.assert_allclose(
        fluid.kinematic_viscosity_m2s(t), tbl.kinematic_viscosity_m2s[keep], rtol=1e-12
    )


def test_packaged_rejects_temperature_above_valid_range() -> None:
    tbl = _packaged_or_skip()
    assert tbl.temperature_c[-1] > fluid.VALID_RANGE_C[1]  # table extends past 40 degC
    with pytest.raises(ValueError):
        fluid.density_kgm3(tbl.temperature_c[-1])


def test_packaged_density_peaks_below_10c() -> None:
    _packaged_or_skip()
    t = np.linspace(*fluid.VALID_RANGE_C, 4001)
    t_peak = t[np.argmax(fluid.density_kgm3(t))]
    assert 0.0 < t_peak < 10.0


def test_packaged_viscosity_decreases_with_temperature() -> None:
    _packaged_or_skip()
    t = np.linspace(*fluid.VALID_RANGE_C, 4001)
    assert np.all(np.diff(fluid.kinematic_viscosity_m2s(t)) < 0.0)


def test_packaged_vapor_pressure_reproduces_rows_and_increases() -> None:
    tbl = _packaged_or_skip()
    keep = tbl.temperature_c <= fluid.VALID_RANGE_C[1]
    np.testing.assert_allclose(
        fluid.vapor_pressure_kpa(tbl.temperature_c[keep]), tbl.vapor_pressure_kpa[keep], rtol=1e-12
    )
    t = np.linspace(*fluid.VALID_RANGE_C, 4001)
    assert np.all(np.diff(fluid.vapor_pressure_kpa(t)) > 0.0)
