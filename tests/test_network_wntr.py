"""Validate the wdtk network solvers against EPANET 2.2 (through WNTR's EpanetSimulator).

Each test network in ``data/networks`` is read by ``wdtk.network.io.read_inp`` and by WNTR,
so both solvers see the same file. Acceptance (PLAN.md Phase 3; flow floor agreed with the
project author): heads within 0.01 m; flows within 0.1% or 1e-6 m^3/s.
"""

import math
from pathlib import Path

import pytest
import wntr
from scipy.optimize import brentq

from wdtk import headloss as hl
from wdtk.network.gga import solve_gga
from wdtk.network.io import read_inp, write_inp
from wdtk.network.model import Junction, Network, Pipe, Reservoir

NETWORKS = Path(__file__).resolve().parents[1] / "data" / "networks"
HEAD_TOL_M = 0.01
FLOW_REL_TOL = 1e-3
FLOW_ABS_TOL_M3S = 1e-6


def run_epanet(inp: Path, tmp_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Return (junction/reservoir heads m, link flows m^3/s) from EPANET at accuracy 1e-6."""
    wn = wntr.network.WaterNetworkModel(str(inp))
    wn.options.hydraulic.accuracy = 1e-6
    wn.options.hydraulic.demand_model = "DD"
    wn.options.time.duration = 0
    res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(tmp_path / inp.stem))
    return res.node["head"].loc[0].to_dict(), res.link["flowrate"].loc[0].to_dict()


@pytest.mark.parametrize(
    "name", ["single_loop_hw", "two_loop_hw", "net1_steady_hw", "single_loop_dw", "two_loop_dw"]
)
def test_gga_matches_epanet(name: str, tmp_path: Path) -> None:
    inp = NETWORKS / f"{name}.inp"
    ours = solve_gga(read_inp(inp))
    heads, flows = run_epanet(inp, tmp_path)
    for j in ours.junction_ids:
        assert abs(ours.head(j) - heads[j]) <= HEAD_TOL_M, f"head at {j}"
    for lk in ours.link_ids:
        dq = abs(ours.flow(lk) - flows[lk])
        assert dq <= max(FLOW_REL_TOL * abs(flows[lk]), FLOW_ABS_TOL_M3S), f"flow in {lk}"


# --- Darcy-Weisbach probe: identifies EPANET's D-W constants empirically -----------------------
# Cited by wdtk.headloss.EPANET_DW_GRAVITY_MS2 / EPANET_KINEMATIC_VISCOSITY_M2S.

PROBES = [  # (head m, length m, diameter m, roughness m): turbulent, Re ~5e3 to ~1.6e6
    (50.0, 1000.0, 0.3, 3e-3),
    (50.0, 1000.0, 0.3, 1e-3),
    (5.0, 1000.0, 0.3, 1e-4),
    (5.0, 1000.0, 0.3, 1e-5),
    (0.5, 1000.0, 0.1, 1e-6),
    (0.05, 1000.0, 0.1, 1e-6),
    (20.0, 2000.0, 0.6, 2e-4),
]


@pytest.mark.parametrize(("head", "length", "diameter", "eps"), PROBES)
def test_epanet_darcy_weisbach_is_swamee_jain_with_epanet_constants(
    tmp_path: Path, head: float, length: float, diameter: float, eps: float
) -> None:
    net = Network(
        junctions=(Junction("J", 0.0, 0.0),),  # EPANET needs at least one junction
        reservoirs=(Reservoir("A", head), Reservoir("B", 0.0)),
        pipes=(
            Pipe("P1", "A", "J", length / 2, diameter, eps),
            Pipe("P2", "J", "B", length / 2, diameter, eps),
        ),
        headloss="D-W",
    )
    inp = tmp_path / "probe.inp"
    write_inp(net, inp)
    _, flows = run_epanet(inp, tmp_path)
    q_epanet = flows["P1"]

    def h_ours(q: float) -> float:
        return (
            hl.darcy_weisbach_headloss_m(
                q,
                length,
                diameter,
                eps,
                hl.EPANET_KINEMATIC_VISCOSITY_M2S,
                "swamee-jain",
                g_ms2=hl.EPANET_DW_GRAVITY_MS2,
            )
            - head
        )

    q_ours = brentq(h_ours, 1e-9, 10.0, xtol=1e-15, rtol=1e-15)
    re = q_ours / (math.pi * diameter**2 / 4) * diameter / hl.EPANET_KINEMATIC_VISCOSITY_M2S
    assert re > hl.RE_TURBULENT_MIN
    assert q_ours == pytest.approx(q_epanet, rel=2e-5)  # EPANET reports single precision
