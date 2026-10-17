"""Tests for wdtk.headloss.

Expected values are closed-form results computed in the test, or self-consistency checks
(round trips, cross-unit agreement, governing-equation residuals). Fluid and pipe inputs
are arbitrary test inputs, not reference values.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wdtk import headloss as hl
from wdtk import units
from wdtk.units import G_STANDARD_MS2 as G

NU = 1.0e-6  # arbitrary test viscosity, m^2/s


def log_uniform(lo: float, hi: float) -> st.SearchStrategy[float]:
    return st.floats(math.log10(lo), math.log10(hi)).map(lambda x: 10.0**x)


flows = log_uniform(1e-4, 5.0)
diameters = log_uniform(0.05, 2.0)
lengths = log_uniform(10.0, 5000.0)
c_factors = st.floats(60.0, 150.0)
roughnesses = st.one_of(st.just(0.0), log_uniform(1e-6, 3e-3))
minor_ks = st.one_of(st.just(0.0), st.floats(0.1, 50.0))

RTOL = 1e-9


# --- Kinematics ------------------------------------------------------------------------------


@given(q=flows, d=diameters)
def test_diameter_for_velocity_round_trip(q: float, d: float) -> None:
    v = hl.velocity_ms(q, d)
    assert hl.diameter_for_velocity_m(q, v) == pytest.approx(d, rel=RTOL)


def test_velocity_and_reynolds_closed_form() -> None:
    q, d = 0.05, 0.2
    v = q / (math.pi * d**2 / 4)
    assert hl.area_m2(d) == pytest.approx(math.pi * d**2 / 4, rel=1e-15)
    assert hl.velocity_ms(q, d) == pytest.approx(v, rel=1e-15)
    assert hl.reynolds_number(-v, d, NU) == pytest.approx(v * d / NU, rel=1e-15)


# --- Hazen-Williams --------------------------------------------------------------------------


@given(q=flows, d=diameters, length=lengths, c=c_factors)
def test_hazen_williams_si_and_us_forms_agree(q: float, d: float, length: float, c: float) -> None:
    h_si = hl.hazen_williams_headloss_m(q, length, d, c)
    h_us_ft = hl.hazen_williams_headloss_ft(
        units.flow_from_m3s(q, "cfs"),
        units.length_from_m(length, "ft"),
        units.length_from_m(d, "ft"),
        c,
    )
    assert units.length_to_m(h_us_ft, "ft") == pytest.approx(h_si, rel=1e-12)


def test_hazen_williams_si_constant_derivation() -> None:
    # V_SI = FT_M * k_US * C * (R_m / FT_M)^0.63 * S^0.54  =>  k_SI = k_US * FT_M^0.37
    r_m, s, c = 0.1, 0.002, 120.0
    v_from_us = units.FT_M * hl.HW_K_US * c * (r_m / units.FT_M) ** 0.63 * s**0.54
    v_si = hl.HW_K_SI * c * r_m**0.63 * s**0.54
    assert v_si == pytest.approx(v_from_us, rel=1e-14)


def test_hazen_williams_matches_velocity_form() -> None:
    q, length, d, c = 0.08, 500.0, 0.25, 110.0
    h = hl.hazen_williams_headloss_m(q, length, d, c)
    v = q / (math.pi * d**2 / 4)
    v_formula = hl.HW_K_SI * c * (d / 4) ** 0.63 * (h / length) ** 0.54
    assert v_formula == pytest.approx(v, rel=1e-12)


def test_hazen_williams_scaling() -> None:
    args = dict(length_m=300.0, c_factor=130.0)
    h1 = hl.hazen_williams_headloss_m(0.03, diameter_m=0.2, **args)
    h2 = hl.hazen_williams_headloss_m(0.06, diameter_m=0.2, **args)
    assert h2 / h1 == pytest.approx(2.0 ** (1 / 0.54), rel=1e-12)
    h3 = hl.hazen_williams_headloss_m(0.03, diameter_m=0.4, **args)
    assert h1 / h3 == pytest.approx(2.0 ** (2.63 / 0.54), rel=1e-12)


@given(q=flows, d=diameters, length=lengths, c=c_factors)
def test_hazen_williams_diameter_round_trip(q: float, d: float, length: float, c: float) -> None:
    h = hl.hazen_williams_headloss_m(q, length, d, c)
    assert hl.hazen_williams_diameter_m(q, length, h, c) == pytest.approx(d, rel=RTOL)


@given(q=flows, d=diameters, length=lengths, c=c_factors)
def test_hazen_williams_c_factor_round_trip(q: float, d: float, length: float, c: float) -> None:
    h = hl.hazen_williams_headloss_m(q, length, d, c)
    assert hl.hazen_williams_c_factor(q, length, d, h) == pytest.approx(c, rel=RTOL)


@given(q=flows, d=diameters, length=lengths, c=c_factors, k=minor_ks)
def test_hazen_williams_capacity_round_trip(
    q: float, d: float, length: float, c: float, k: float
) -> None:
    head = hl.hazen_williams_headloss_m(q, length, d, c) + hl.minor_loss_m(hl.velocity_ms(q, d), k)
    assert hl.hazen_williams_capacity_m3s(head, length, d, c, minor_k=k) == pytest.approx(
        q, rel=RTOL
    )


# --- Friction factor -------------------------------------------------------------------------

re_turbulent = log_uniform(5e3, 1e8)
rel_rough = log_uniform(1e-6, 1e-2)
SWAMEE_JAIN_TOL = 0.03  # measured worst case ~2.83% over this range (see docstring)


@given(re=re_turbulent, rr=rel_rough)
def test_swamee_jain_within_tolerance_of_colebrook(re: float, rr: float) -> None:
    f_cw = hl.friction_factor_colebrook(re, rr)
    f_sj = hl.friction_factor_swamee_jain(re, rr)
    assert abs(f_sj - f_cw) / f_cw <= SWAMEE_JAIN_TOL


def test_swamee_jain_worst_case_on_grid() -> None:
    re, rr = np.meshgrid(np.logspace(math.log10(5e3), 8, 60), np.logspace(-6, -2, 60))
    f_cw = hl.friction_factor_colebrook(re, rr)
    err = np.abs(hl.friction_factor_swamee_jain(re, rr) - f_cw) / f_cw
    assert err.max() <= SWAMEE_JAIN_TOL
    i = np.unravel_index(np.argmax(err), err.shape)
    assert re[i] == pytest.approx(5e3) and rr[i] == pytest.approx(1e-2)  # worst at the corner


@given(re=re_turbulent, rr=st.one_of(st.just(0.0), rel_rough))
def test_colebrook_satisfies_equation(re: float, rr: float) -> None:
    x = 1.0 / math.sqrt(hl.friction_factor_colebrook(re, rr))
    assert x + 2.0 * math.log10(rr / 3.7 + 2.51 * x / re) == pytest.approx(0.0, abs=1e-12)


def test_laminar_friction_factor() -> None:
    re = np.array([10.0, 500.0, 1999.0])
    np.testing.assert_allclose(hl.friction_factor_laminar(re), 64.0 / re, rtol=1e-15)
    np.testing.assert_allclose(hl.friction_factor(re, 1e-3), 64.0 / re, rtol=1e-15)


@pytest.mark.parametrize("method", ["colebrook", "swamee-jain"])
def test_friction_factor_continuous_at_regime_limits(method: hl.TurbulentMethod) -> None:
    rr, d = 1e-3, 1e-9
    for limit in (hl.RE_LAMINAR_MAX, hl.RE_TURBULENT_MIN):
        below = hl.friction_factor(limit * (1 - d), rr, method)
        above = hl.friction_factor(limit * (1 + d), rr, method)
        assert below == pytest.approx(above, rel=1e-6)


def test_transitional_interpolation_midpoint() -> None:
    rr = 1e-4
    f_mid = hl.friction_factor(3000.0, rr)
    expected = 0.5 * (64.0 / 2000.0 + hl.friction_factor_colebrook(4000.0, rr))
    assert f_mid == pytest.approx(expected, rel=1e-12)


# --- Darcy-Weisbach --------------------------------------------------------------------------


def test_laminar_darcy_weisbach_is_hagen_poiseuille() -> None:
    d, length = 0.05, 100.0
    q = 1e-5  # Re = 4Q/(pi D nu) ~ 255
    v = q / (math.pi * d**2 / 4)
    assert v * d / NU < hl.RE_LAMINAR_MAX
    expected = 32.0 * NU * length * v / (G * d**2)
    assert hl.darcy_weisbach_headloss_m(q, length, d, 1e-4, NU) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("eps", [0.0, 1.5e-6, 2.6e-4, 3e-3])
def test_darcy_weisbach_capacity_matches_explicit_colebrook(eps: float) -> None:
    # With h known, Colebrook is explicit in V:
    #   V = -2 sqrt(2gDS) log10(eps/(3.7D) + 2.51 nu / (D sqrt(2gDS))),  S = h/L
    length, d, h = 1000.0, 0.3, 5.0
    root = math.sqrt(2.0 * G * d * h / length)
    v = -2.0 * root * math.log10(eps / (3.7 * d) + 2.51 * NU / (d * root))
    assert v * d / NU > hl.RE_TURBULENT_MIN
    q_expected = v * math.pi * d**2 / 4
    assert hl.darcy_weisbach_capacity_m3s(h, length, d, eps, NU) == pytest.approx(
        q_expected, rel=1e-10
    )


@settings(max_examples=50, deadline=None)
@given(q=flows, d=diameters, length=lengths, eps=roughnesses)
def test_darcy_weisbach_diameter_round_trip(q: float, d: float, length: float, eps: float) -> None:
    h = hl.darcy_weisbach_headloss_m(q, length, d, eps, NU)
    assert hl.darcy_weisbach_diameter_m(q, length, h, eps, NU) == pytest.approx(d, rel=RTOL)


def test_darcy_weisbach_diameter_regression_large_pipe_low_flow() -> None:
    # Hypothesis-found case: the root (D = 1 m) lies far above the starting guess, and a
    # two-sided bracket search used to probe diameters below the roughness height.
    q, d, length, eps = 0.001, 1.0, 10.0, 0.001
    h = hl.darcy_weisbach_headloss_m(q, length, d, eps, NU)
    assert hl.darcy_weisbach_diameter_m(q, length, h, eps, NU) == pytest.approx(d, rel=RTOL)


@settings(max_examples=50, deadline=None)
@given(q=log_uniform(1e-7, 5.0), d=diameters, length=lengths, eps=roughnesses, k=minor_ks)
def test_darcy_weisbach_capacity_round_trip(
    q: float, d: float, length: float, eps: float, k: float
) -> None:
    """Covers laminar, transitional, and turbulent flow (Re from ~0.06 to ~1e8)."""
    head = hl.darcy_weisbach_headloss_m(q, length, d, eps, NU) + hl.minor_loss_m(
        hl.velocity_ms(q, d), k
    )
    assert hl.darcy_weisbach_capacity_m3s(head, length, d, eps, NU, minor_k=k) == pytest.approx(
        q, rel=RTOL
    )


def test_swamee_jain_option_in_darcy_weisbach() -> None:
    q, length, d, eps = 0.1, 500.0, 0.3, 2.6e-4
    v = q / (math.pi * d**2 / 4)
    f = hl.friction_factor_swamee_jain(v * d / NU, eps / d)
    expected = f * length / d * v**2 / (2 * G)
    assert hl.darcy_weisbach_headloss_m(q, length, d, eps, NU, "swamee-jain") == pytest.approx(
        expected, rel=1e-12
    )


# --- Minor losses ----------------------------------------------------------------------------


def test_minor_loss_closed_form_and_sum() -> None:
    v = 2.0
    assert hl.minor_loss_m(v, 0.5) == pytest.approx(0.5 * v**2 / (2 * G), rel=1e-15)
    assert hl.minor_loss_m(v, [0.5, 0.3, 1.0]) == pytest.approx(1.8 * v**2 / (2 * G), rel=1e-15)


# --- Sign convention, arrays, errors ---------------------------------------------------------


def test_headloss_is_antisymmetric_in_flow() -> None:
    q = np.array([-0.2, -1e-6, 0.0, 1e-6, 0.2])
    h_hw = hl.hazen_williams_headloss_m(q, 100.0, 0.3, 120.0)
    h_dw = hl.darcy_weisbach_headloss_m(q, 100.0, 0.3, 1e-4, NU)
    for h in (h_hw, h_dw):
        assert h.shape == q.shape
        np.testing.assert_allclose(h, -h[::-1], rtol=1e-15)
        assert h[2] == 0.0
    v = hl.velocity_ms(q, 0.3)
    np.testing.assert_allclose(hl.minor_loss_m(v, 2.0), -hl.minor_loss_m(-v, 2.0))


def test_array_arguments_broadcast_to_first_argument() -> None:
    q = np.full(3, 0.05)
    d = np.array([0.2, 0.25, 0.3])
    h = hl.darcy_weisbach_headloss_m(q, 100.0, d, 1e-4, NU)
    expected = [hl.darcy_weisbach_headloss_m(0.05, 100.0, float(di), 1e-4, NU) for di in d]
    np.testing.assert_allclose(h, expected, rtol=1e-15)
    with pytest.raises(ValueError, match="broadcast"):
        hl.darcy_weisbach_headloss_m(0.05, 100.0, d, 1e-4, NU)


@pytest.mark.parametrize(
    "call",
    [
        lambda: hl.friction_factor_colebrook(3000.0, 1e-4),
        lambda: hl.friction_factor_colebrook(1e5, 3.7),
        lambda: hl.friction_factor(1e5, -1e-4),
        lambda: hl.friction_factor(1e5, 1e-4, "moody"),  # type: ignore[arg-type]
        lambda: hl.velocity_ms(0.1, 0.0),
        lambda: hl.hazen_williams_headloss_m(0.1, 100.0, 0.3, -5.0),
        lambda: hl.hazen_williams_capacity_m3s(-1.0, 100.0, 0.3, 120.0),
        lambda: hl.darcy_weisbach_diameter_m(0.0, 100.0, 1.0, 1e-4, NU),
        lambda: hl.minor_loss_m(1.0, [0.5, -0.1]),
        lambda: hl.reynolds_number(1.0, 0.3, 0.0),
    ],
)
def test_invalid_inputs_raise(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]
