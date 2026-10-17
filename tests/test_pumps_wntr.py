"""Validate wdtk.pumps against EPANET 2.2 (through WNTR); models in wdtk.validation.

Model: reservoir R1 -> pump P1 -> junction J1 -> pipe L1 -> reservoir R2, steady state,
Hazen-Williams, demand-driven, no minor losses. Acceptance (PLAN.md Phase 2): 0.5% in flow
and head.
"""

import pytest

from wdtk import pumps as pm
from wdtk import validation as v

TOL = 0.005  # 0.5 %

SINGLE_POINT = [(0.10, 50.0)]
THREE_POINT = [(0.0, 60.0), (0.10, 45.0), (0.22, 0.0)]


def three_point_curve() -> pm.PumpCurve:
    (_, h1), (q2, h2), (q3, h3) = THREE_POINT
    return pm.fit_three_point(h1, q2, h2, q3, h3)


def assert_close(ours: tuple[float, float], epanet: tuple[float, float]) -> None:
    (q, h), (q_ep, h_ep) = ours, epanet
    assert abs(q / q_ep - 1.0) < TOL, f"flow: ours {q:.6g}, EPANET {q_ep:.6g}"
    assert abs(h / h_ep - 1.0) < TOL, f"head: ours {h:.6g}, EPANET {h_ep:.6g}"


def test_single_point_curve_operating_point() -> None:
    curve = pm.fit_single_point(*SINGLE_POINT[0])
    assert_close(v.pump_case_wdtk(curve), v.pump_case_epanet(SINGLE_POINT))


def test_three_point_curve_operating_point() -> None:
    assert_close(v.pump_case_wdtk(three_point_curve()), v.pump_case_epanet(THREE_POINT))


@pytest.mark.parametrize("speed", [0.8, 0.9])
def test_reduced_speed_matches_affinity_prediction(speed: float) -> None:
    curve = pm.scale_curve_speed(three_point_curve(), speed)
    assert_close(v.pump_case_wdtk(curve), v.pump_case_epanet(THREE_POINT, speed))


@pytest.mark.parametrize("points", [SINGLE_POINT, THREE_POINT])
def test_fitted_coefficients_match_wntr(points: list[tuple[float, float]]) -> None:
    a, b, c = v.single_pump_model(points).get_link("P1").get_head_curve_coefficients()
    curve = pm.fit_single_point(*points[0]) if len(points) == 1 else three_point_curve()
    assert (curve.a_m, curve.b, curve.c) == pytest.approx((a, b, c), rel=1e-6)


def test_validation_table_within_tolerance() -> None:
    table = v.pump_validation_table()
    assert len(table) == 4
    assert (table["flow_diff_pct"].abs() < 100 * TOL).all()
    assert (table["head_diff_pct"].abs() < 100 * TOL).all()
