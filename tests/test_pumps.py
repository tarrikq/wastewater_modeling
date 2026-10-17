"""Unit tests for wdtk.pumps (closed form and self-consistency; inputs are arbitrary)."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from wdtk import fluid
from wdtk import headloss as hl
from wdtk import pumps as pm
from wdtk.units import G_STANDARD_MS2 as G

flows = st.floats(0.01, 0.5)
heads = st.floats(5.0, 150.0)
speeds = st.floats(0.3, 1.5)


@pytest.fixture
def curve() -> pm.PumpCurve:
    return pm.fit_three_point(60.0, 0.10, 45.0, 0.22)


# --- Fitting ---------------------------------------------------------------------------------


@given(q=flows, h=heads)
def test_single_point_fit(q: float, h: float) -> None:
    c = pm.fit_single_point(q, h)
    assert pm.pump_head_m(c, q) == pytest.approx(h, rel=1e-12)
    assert c.a_m == pytest.approx(4.0 / 3.0 * h, rel=1e-15)
    assert c.c == 2.0
    assert c.max_flow_m3s == pytest.approx(2.0 * q, rel=1e-12)


@given(
    h1=st.floats(20.0, 150.0),
    f2=st.floats(0.1, 0.9),
    f3=st.floats(0.0, 0.9),
    q2=st.floats(0.01, 0.5),
    r=st.floats(1.2, 4.0),
)
def test_three_point_fit_passes_through_points(
    h1: float, f2: float, f3: float, q2: float, r: float
) -> None:
    h2 = f2 * h1
    h3 = f3 * h2
    q3 = r * q2
    c = pm.fit_three_point(h1, q2, h2, q3, h3)
    assert pm.pump_head_m(c, 0.0) == pytest.approx(h1, rel=1e-12)
    assert pm.pump_head_m(c, q2) == pytest.approx(h2, rel=1e-9, abs=1e-9)
    assert pm.pump_head_m(c, q3) == pytest.approx(h3, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(
    "args",
    [(60.0, 0.1, 45.0, 0.1), (60.0, 0.1, 65.0, 0.2), (60.0, 0.1, 45.0, 0.2, 50.0), (60, 0, 45, 1)],
)
def test_three_point_fit_rejects_bad_points(args: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        pm.fit_three_point(*args)


def test_curve_rejects_nonpositive_coefficients() -> None:
    with pytest.raises(ValueError):
        pm.PumpCurve(a_m=50.0, b=-1.0, c=2.0)


@given(q=st.floats(0.0, 0.22))
def test_pump_flow_inverts_pump_head(q: float) -> None:
    c = pm.fit_three_point(60.0, 0.10, 45.0, 0.22)
    assert pm.pump_flow_m3s(c, pm.pump_head_m(c, q)) == pytest.approx(q, rel=1e-9, abs=1e-12)


def test_pump_flow_is_zero_above_shutoff(curve: pm.PumpCurve) -> None:
    h = np.array([curve.a_m, curve.a_m + 10.0])
    np.testing.assert_array_equal(pm.pump_flow_m3s(curve, h), [0.0, 0.0])


# --- Affinity --------------------------------------------------------------------------------


@given(q=st.floats(0.0, 0.2), s=speeds)
def test_scaled_curve_obeys_affinity_laws(q: float, s: float) -> None:
    c = pm.fit_three_point(60.0, 0.10, 45.0, 0.22)
    q2, h2 = pm.affinity_point(q, pm.pump_head_m(c, q), s)
    assert pm.pump_head_m(pm.scale_curve_speed(c, s), q2) == pytest.approx(h2, rel=1e-9, abs=1e-9)


def test_affinity_power_scales_with_cube(curve: pm.PumpCurve) -> None:
    q, s = 0.12, 0.8
    q2, h2 = pm.affinity_point(q, pm.pump_head_m(curve, q), s)
    p1 = pm.hydraulic_power_w(q, pm.pump_head_m(curve, q))
    assert pm.hydraulic_power_w(q2, h2) == pytest.approx(s**3 * p1, rel=1e-12)


# --- Series and parallel ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_identical_pumps_in_series_and_parallel(curve: pm.PumpCurve, n: int) -> None:
    for q in (0.0, 0.05, 0.15):
        assert pm.series_head_m([curve] * n, q) == pytest.approx(n * pm.pump_head_m(curve, q))
        assert pm.parallel_head_m([curve] * n, n * q) == pytest.approx(
            pm.pump_head_m(curve, q), rel=1e-10
        )


def test_parallel_unequal_pumps_share_head(curve: pm.PumpCurve) -> None:
    small = pm.fit_single_point(0.05, 40.0)
    h = pm.parallel_head_m([curve, small], 0.2)
    total = pm.pump_flow_m3s(curve, h) + pm.pump_flow_m3s(small, h)
    assert total == pytest.approx(0.2, rel=1e-10)


def test_parallel_weak_pump_delivers_nothing_above_its_shutoff(curve: pm.PumpCurve) -> None:
    weak = pm.fit_single_point(0.05, 20.0)  # shutoff 26.7 m, below the strong pump's head
    h = pm.parallel_head_m([curve, weak], 0.05)
    assert h > weak.a_m
    assert pm.pump_flow_m3s(weak, h) == 0.0


# --- System curves and operating point --------------------------------------------------------


def test_system_curves_closed_form() -> None:
    q, h0, length, d = 0.1, 25.0, 1500.0, 0.3
    v = q / (math.pi * d**2 / 4)
    hw = 25.0 + hl.hazen_williams_headloss_m(q, length, d, 120.0) + 3.0 * v**2 / (2 * G)
    assert pm.system_head_hw_m(q, h0, length, d, 120.0, minor_k=3.0) == pytest.approx(hw)
    nu = 1e-6
    dw = 25.0 + hl.darcy_weisbach_headloss_m(q, length, d, 1e-4, nu) + 3.0 * v**2 / (2 * G)
    assert pm.system_head_dw_m(q, h0, length, d, 1e-4, nu, minor_k=3.0) == pytest.approx(dw)
    assert pm.system_head_hw_m(0.0, h0, length, d, 120.0) == h0


def test_operating_point_satisfies_both_curves(curve: pm.PumpCurve) -> None:
    def system(q: float) -> float:
        return pm.system_head_hw_m(q, 20.0, 2000.0, 0.3, 120.0, minor_k=5.0)

    q, h = pm.operating_point(lambda x: pm.pump_head_m(curve, x), system)
    assert h == pytest.approx(system(q), rel=1e-9)
    assert h == pytest.approx(pm.pump_head_m(curve, q), rel=1e-12)


def test_operating_point_closed_form_quadratic() -> None:
    # Pump H = a - b Q^2 against system H = h0 + k Q^2  =>  Q = sqrt((a - h0)/(b + k))
    curve = pm.PumpCurve(a_m=80.0, b=2000.0, c=2.0)
    h0, k = 30.0, 1500.0
    q, h = pm.operating_point(lambda x: pm.pump_head_m(curve, x), lambda x: h0 + k * x**2)
    assert q == pytest.approx(math.sqrt((80.0 - h0) / (2000.0 + k)), rel=1e-12)
    assert h == pytest.approx(h0 + k * q**2, rel=1e-12)


def test_operating_point_no_flow_raises(curve: pm.PumpCurve) -> None:
    with pytest.raises(ValueError, match="no flow"):
        pm.operating_point(lambda x: pm.pump_head_m(curve, x), lambda x: 100.0 + x)


# --- Power -----------------------------------------------------------------------------------


def test_power_relations() -> None:
    q, h, rho = 0.1, 50.0, 998.0
    pw = pm.hydraulic_power_w(q, h, rho)
    assert pw == pytest.approx(rho * G * q * h, rel=1e-15)
    assert pm.brake_power_w(q, h, 0.8, rho) == pytest.approx(pw / 0.8, rel=1e-15)
    eta = pm.wire_to_water_efficiency(0.8, 0.95, 0.97)
    assert eta == pytest.approx(0.8 * 0.95 * 0.97, rel=1e-15)
    assert pm.input_power_w(q, h, eta, rho) == pytest.approx(pw / eta, rel=1e-15)
    q_arr = np.array([0.05, 0.1])
    np.testing.assert_allclose(pm.hydraulic_power_w(q_arr, h), 1000.0 * G * q_arr * h)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.2])
def test_efficiency_out_of_range_raises(bad: float) -> None:
    with pytest.raises(ValueError):
        pm.brake_power_w(0.1, 50.0, bad)


# --- NPSH ------------------------------------------------------------------------------------


def test_npsh_available_closed_form() -> None:
    p_atm, t = 95.0, 20.0
    pv = fluid.vapor_pressure_kpa(t)
    rho = fluid.density_kgm3(t)
    expected = (p_atm - pv) * 1000.0 / (rho * G) + (-3.0) - 0.8
    assert pm.npsh_available_m(p_atm, pv, -3.0, 0.8, rho) == pytest.approx(expected, rel=1e-12)


def test_npsh_margin_and_check() -> None:
    assert pm.npsh_margin_m(6.0, 4.5) == pytest.approx(1.5)
    assert pm.npsh_check(6.0, 4.5)
    assert pm.npsh_check(6.0, 4.5, required_margin_m=1.5)
    assert not pm.npsh_check(6.0, 4.5, required_margin_m=1.6)
    assert not pm.npsh_check(4.0, 4.5)
