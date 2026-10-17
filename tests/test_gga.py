"""Unit tests for the GGA and Hardy Cross solvers (closed form and self-consistency)."""

from pathlib import Path

import numpy as np
import pytest

from wdtk import headloss as hl
from wdtk import pumps as pm
from wdtk.network.gga import solve_gga
from wdtk.network.hardy_cross import find_loops, solve_hardy_cross
from wdtk.network.hydraulics import (
    HydraulicOptions,
    continuity_residual_m3s,
    energy_residual_m,
)
from wdtk.network.io import read_inp
from wdtk.network.model import Junction, Network, Pipe, Pump, Reservoir

NETWORKS = Path(__file__).resolve().parents[1] / "data" / "networks"
C = 120.0


def series_network(minor_k: float = 0.0) -> Network:
    return Network(
        junctions=(Junction("J", 0.0, 0.0),),
        reservoirs=(Reservoir("R1", 50.0), Reservoir("R2", 30.0)),
        pipes=(Pipe("P1", "R1", "J", 800.0, 0.3, C, minor_k), Pipe("P2", "J", "R2", 700.0, 0.3, C)),
    )


@pytest.mark.parametrize("solver", [solve_gga, solve_hardy_cross])
def test_series_pipes_match_closed_form_capacity(solver: object) -> None:
    res = solver(series_network(), convention="exact", accuracy=1e-12)  # type: ignore[operator]
    q = hl.hazen_williams_capacity_m3s(20.0, 1500.0, 0.3, C)
    np.testing.assert_allclose(res.flows_m3s, [q, q], rtol=1e-10)
    h_p1 = hl.hazen_williams_headloss_m(q, 800.0, 0.3, C)
    assert res.head("J") == pytest.approx(50.0 - h_p1, rel=1e-10)


def test_minor_losses_included() -> None:
    res = solve_gga(series_network(minor_k=10.0), convention="exact", accuracy=1e-12)
    q = res.flow("P1")
    v = q / (np.pi * 0.3**2 / 4)
    total = hl.hazen_williams_headloss_m(q, 1500.0, 0.3, C) + hl.minor_loss_m(v, 10.0)
    assert total == pytest.approx(20.0, rel=1e-10)


def test_parallel_pipes_split_by_conveyance() -> None:
    net = Network(
        junctions=(Junction("J", 0.0, 0.2),),
        reservoirs=(Reservoir("R", 80.0),),
        pipes=(Pipe("A", "R", "J", 1000.0, 0.3, C), Pipe("B", "R", "J", 1000.0, 0.2, C)),
    )
    res = solve_gga(net, convention="exact", accuracy=1e-12)
    qa, qb = res.flow("A"), res.flow("B")
    assert qa + qb == pytest.approx(0.2, rel=1e-12)
    # Equal head loss in both branches; with h ~ Q^n / D^m, Qa/Qb = (Da/Db)^(m/n)
    expected_ratio = (0.3 / 0.2) ** (hl.HW_DIAMETER_EXPONENT / hl.HW_FLOW_EXPONENT)
    assert qa / qb == pytest.approx(expected_ratio, rel=1e-9)


@pytest.mark.parametrize(
    "name", ["single_loop_hw", "two_loop_hw", "net1_steady_hw", "single_loop_dw", "two_loop_dw"]
)
def test_gga_satisfies_both_equation_sets(name: str) -> None:
    net = read_inp(NETWORKS / f"{name}.inp")
    res = solve_gga(net, accuracy=1e-10)
    opts = HydraulicOptions()
    assert np.max(np.abs(continuity_residual_m3s(net, res.flows_m3s))) < 1e-10
    assert np.max(np.abs(energy_residual_m(net, res.flows_m3s, res.heads_m, opts))) < 1e-6
    rel = [r.relative_flow_change for r in res.history]
    assert rel[-1] < 1e-10 and res.converged


def test_gga_pump_matches_phase2_operating_point() -> None:
    curve_pts = ((0.0, 60.0), (0.10, 45.0), (0.22, 0.0))
    net = Network(
        junctions=(Junction("J", 0.0, 0.0),),
        reservoirs=(Reservoir("R1", 10.0), Reservoir("R2", 40.0)),
        pipes=(Pipe("L", "J", "R2", 2000.0, 0.3, C),),
        pumps=(Pump("PU", "R1", "J", curve_pts),),
    )
    res = solve_gga(net, convention="exact", accuracy=1e-12)
    curve = net.pumps[0].curve
    q, h = pm.operating_point(
        lambda x: pm.pump_head_m(curve, x),
        lambda x: pm.system_head_hw_m(x, 30.0, 2000.0, 0.3, C),
    )
    assert res.flow("PU") == pytest.approx(q, rel=1e-9)
    assert res.head("J") - 10.0 == pytest.approx(h, rel=1e-9)


def test_gga_reverse_pump_flow_raises() -> None:
    net = Network(
        junctions=(Junction("J", 0.0, 0.0),),
        reservoirs=(Reservoir("R1", 10.0), Reservoir("R2", 100.0)),
        pipes=(Pipe("L", "J", "R2", 200.0, 0.3, C),),
        pumps=(Pump("PU", "R1", "J", ((0.1, 20.0),)),),
    )
    with pytest.raises(RuntimeError, match="reverse flow"):
        solve_gga(net)


def test_gga_nonconvergence_raises() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_gga(read_inp(NETWORKS / "two_loop_hw.inp"), max_iter=1)


def test_darcy_weisbach_exact_convention_needs_viscosity() -> None:
    with pytest.raises(ValueError, match="viscosity"):
        solve_gga(read_inp(NETWORKS / "two_loop_dw.inp"), convention="exact")


def test_zero_flow_link_converges() -> None:
    # Symmetric network: the cross pipe between J1 and J2 carries no flow at the solution.
    net = Network(
        junctions=(Junction("J1", 0.0, 0.05), Junction("J2", 0.0, 0.05)),
        reservoirs=(Reservoir("R", 50.0),),
        pipes=(
            Pipe("A", "R", "J1", 500.0, 0.25, C),
            Pipe("B", "R", "J2", 500.0, 0.25, C),
            Pipe("X", "J1", "J2", 300.0, 0.15, C),
        ),
    )
    for solver in (solve_gga, solve_hardy_cross):
        res = solver(net, accuracy=1e-10)
        assert abs(res.flow("X")) < 1e-9
        assert res.head("J1") == pytest.approx(res.head("J2"), abs=1e-9)


# --- Hardy Cross -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["single_loop_hw", "two_loop_hw", "single_loop_dw", "two_loop_dw"])
def test_hardy_cross_matches_gga(name: str) -> None:
    net = read_inp(NETWORKS / f"{name}.inp")
    g = solve_gga(net, accuracy=1e-10)
    h = solve_hardy_cross(net, accuracy=1e-10)
    np.testing.assert_allclose(h.heads_m, g.heads_m, atol=1e-6)
    np.testing.assert_allclose(h.flows_m3s, g.flows_m3s, atol=1e-9)
    assert max(r.max_continuity_residual_m3s for r in h.history) < 1e-12  # continuity kept


def test_hardy_cross_two_reservoir_pseudo_loop() -> None:
    net = Network(
        junctions=(Junction("J1", 0.0, 0.03), Junction("J2", 0.0, 0.04)),
        reservoirs=(Reservoir("RA", 60.0), Reservoir("RB", 52.0)),
        pipes=(
            Pipe("P1", "RA", "J1", 600.0, 0.25, C),
            Pipe("P2", "J1", "J2", 500.0, 0.2, C),
            Pipe("P3", "J2", "RB", 400.0, 0.2, C),
        ),
    )
    loops = find_loops(net)
    assert len(loops) == 1 and abs(loops[0].head_offset_m) == pytest.approx(8.0)
    g = solve_gga(net, accuracy=1e-12)
    h = solve_hardy_cross(net, accuracy=1e-12)
    np.testing.assert_allclose(h.flows_m3s, g.flows_m3s, atol=1e-9)
    np.testing.assert_allclose(h.heads_m, g.heads_m, atol=1e-6)


def test_hardy_cross_scope_limits() -> None:
    with pytest.raises(NotImplementedError):
        solve_hardy_cross(read_inp(NETWORKS / "net1_steady_hw.inp"))
    net = Network(
        junctions=(Junction("J", 0.0, 0.01),),
        reservoirs=(Reservoir("RA", 60.0), Reservoir("RB", 50.0)),
        pipes=(Pipe("P1", "RA", "J", 100.0, 0.2, C), Pipe("PX", "RA", "RB", 100.0, 0.2, C)),
    )
    with pytest.raises(NotImplementedError):
        solve_hardy_cross(net)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_hardy_cross(read_inp(NETWORKS / "two_loop_hw.inp"), max_iter=2)
