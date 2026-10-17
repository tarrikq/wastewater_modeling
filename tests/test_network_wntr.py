"""Validate the wdtk network solvers against EPANET 2.2 (through WNTR).

Each test network in ``data/networks`` is read by ``wdtk.network.io.read_inp`` and by WNTR,
so both solvers see the same file. Acceptance (PLAN.md Phase 3; flow floor agreed with the
project author): heads within 0.01 m; flows within 0.1% or 1e-6 m^3/s.
"""

from pathlib import Path

import pytest

from wdtk import headloss as hl
from wdtk import validation as v
from wdtk.network.gga import solve_gga
from wdtk.network.io import read_inp

NETWORKS = Path(__file__).resolve().parents[1] / "data" / "networks"
HEAD_TOL_M = 0.01
FLOW_REL_TOL = 1e-3
FLOW_ABS_TOL_M3S = 1e-6


@pytest.mark.parametrize(
    "name", ["single_loop_hw", "two_loop_hw", "net1_steady_hw", "single_loop_dw", "two_loop_dw"]
)
def test_gga_matches_epanet(name: str, tmp_path: Path) -> None:
    inp = NETWORKS / f"{name}.inp"
    ours = solve_gga(read_inp(inp))
    heads, flows = v.run_epanet_steady(inp, tmp_path)
    for j in ours.junction_ids:
        assert abs(ours.head(j) - heads[j]) <= HEAD_TOL_M, f"head at {j}"
    for lk in ours.link_ids:
        dq = abs(ours.flow(lk) - flows[lk])
        assert dq <= max(FLOW_REL_TOL * abs(flows[lk]), FLOW_ABS_TOL_M3S), f"flow in {lk}"


# --- Darcy-Weisbach probe: identifies EPANET's D-W constants empirically -----------------------
# Cited by wdtk.headloss.EPANET_DW_GRAVITY_MS2 / EPANET_KINEMATIC_VISCOSITY_M2S.


@pytest.mark.parametrize(("head", "length", "diameter", "eps"), v.DW_PROBES)
def test_epanet_darcy_weisbach_is_swamee_jain_with_epanet_constants(
    head: float, length: float, diameter: float, eps: float
) -> None:
    q_epanet = v.dw_probe_epanet(head, length, diameter, eps)
    q_ours = v.dw_probe_wdtk(head, length, diameter, eps)
    re = (
        q_ours
        / (3.141592653589793 * diameter**2 / 4)
        * diameter
        / (hl.EPANET_KINEMATIC_VISCOSITY_M2S)
    )
    assert re > hl.RE_TURBULENT_MIN
    assert q_ours == pytest.approx(q_epanet, rel=2e-5)  # EPANET reports single precision
