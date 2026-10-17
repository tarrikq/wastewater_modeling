"""Validate wdtk.pumps against EPANET 2.2, run through WNTR's EpanetSimulator.

Model: reservoir R1 -> pump P1 -> junction J1 -> pipe L1 -> reservoir R2, steady state,
Hazen-Williams, demand-driven, no minor losses. Pipe and pump values are arbitrary test
inputs. Acceptance tolerance (PLAN.md Phase 2): 0.5% in flow and head.
"""

from pathlib import Path

import pytest
import wntr

from wdtk import pumps as pm

TOL = 0.005  # 0.5 %

R1_HEAD_M, R2_HEAD_M = 10.0, 40.0
LENGTH_M, DIAMETER_M, C_FACTOR = 2000.0, 0.3, 120.0

SINGLE_POINT = [(0.10, 50.0)]
THREE_POINT = [(0.0, 60.0), (0.10, 45.0), (0.22, 0.0)]


def build_model(
    points: list[tuple[float, float]], speed: float = 1.0
) -> wntr.network.WaterNetworkModel:
    wn = wntr.network.WaterNetworkModel()
    wn.options.hydraulic.headloss = "H-W"
    wn.options.hydraulic.demand_model = "DD"
    wn.options.hydraulic.accuracy = 1e-6  # tighter than the 1e-3 default
    wn.options.time.duration = 0
    wn.add_reservoir("R1", base_head=R1_HEAD_M)
    wn.add_reservoir("R2", base_head=R2_HEAD_M)
    wn.add_junction("J1", base_demand=0.0, elevation=0.0)
    wn.add_curve("C1", "HEAD", points)
    wn.add_pump("P1", "R1", "J1", "HEAD", "C1", speed=speed)
    wn.add_pipe("L1", "J1", "R2", length=LENGTH_M, diameter=DIAMETER_M, roughness=C_FACTOR)
    return wn


def run_epanet(wn: wntr.network.WaterNetworkModel, tmp_path: Path) -> tuple[float, float]:
    """Return (pump flow m3/s, pump head gain m) from EPANET."""
    res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(tmp_path / "model"))
    flow = float(res.link["flowrate"].loc[0, "P1"])
    head_gain = float(res.node["head"].loc[0, "J1"] - res.node["head"].loc[0, "R1"])
    return flow, head_gain


def ours(curve: pm.PumpCurve) -> tuple[float, float]:
    def system(q: float) -> float:
        return pm.system_head_hw_m(q, R2_HEAD_M - R1_HEAD_M, LENGTH_M, DIAMETER_M, C_FACTOR)

    return pm.operating_point(lambda q: pm.pump_head_m(curve, q), system)


def assert_close(ours_qh: tuple[float, float], epanet_qh: tuple[float, float]) -> None:
    (q, h), (q_ep, h_ep) = ours_qh, epanet_qh
    assert abs(q / q_ep - 1.0) < TOL, f"flow: ours {q:.6g}, EPANET {q_ep:.6g}"
    assert abs(h / h_ep - 1.0) < TOL, f"head: ours {h:.6g}, EPANET {h_ep:.6g}"


def test_single_point_curve_operating_point(tmp_path: Path) -> None:
    curve = pm.fit_single_point(*SINGLE_POINT[0])
    assert_close(ours(curve), run_epanet(build_model(SINGLE_POINT), tmp_path))


def test_three_point_curve_operating_point(tmp_path: Path) -> None:
    (_, h1), (q2, h2), (q3, h3) = THREE_POINT
    curve = pm.fit_three_point(h1, q2, h2, q3, h3)
    assert_close(ours(curve), run_epanet(build_model(THREE_POINT), tmp_path))


@pytest.mark.parametrize("speed", [0.8, 0.9])
def test_reduced_speed_matches_affinity_prediction(tmp_path: Path, speed: float) -> None:
    (_, h1), (q2, h2), (q3, h3) = THREE_POINT
    curve = pm.scale_curve_speed(pm.fit_three_point(h1, q2, h2, q3, h3), speed)
    assert_close(ours(curve), run_epanet(build_model(THREE_POINT, speed=speed), tmp_path))


@pytest.mark.filterwarnings("ignore:Covariance of the parameters could not be estimated")
@pytest.mark.parametrize("points", [SINGLE_POINT, THREE_POINT])
def test_fitted_coefficients_match_wntr(points: list[tuple[float, float]]) -> None:
    pump = build_model(points).get_link("P1")
    a, b, c = pump.get_head_curve_coefficients()
    if len(points) == 1:
        curve = pm.fit_single_point(*points[0])
    else:
        (_, h1), (q2, h2), (q3, h3) = points
        curve = pm.fit_three_point(h1, q2, h2, q3, h3)
    assert (curve.a_m, curve.b, curve.c) == pytest.approx((a, b, c), rel=1e-6)
