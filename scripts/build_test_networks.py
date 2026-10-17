"""Build the Phase 3 validation networks and write them to ``data/networks/``.

Networks (all steady state, demand-driven, reservoirs only):

* ``single_loop``: one reservoir, a feed main, and a four-junction loop.
* ``two_loop``: one reservoir and a 2 x 3 grid of junctions forming two loops.
* ``net1_steady``: EPANET's Net1 (as bundled with WNTR 1.5.0,
  ``wntr/library/networks/Net1.inp``) reduced to steady state: tank 2 replaced by a reservoir
  at its initial head (elevation + initial level), base demands only (no patterns), and no
  controls, quality, or energy data. The pump and its curve are kept.

The looped networks are written in a Hazen-Williams and a Darcy-Weisbach variant. Pipe
sizes, lengths, demands, and elevations of the first two are illustrative inputs chosen for
this project. Each file is written with :func:`wdtk.network.io.write_inp`, so the same file
is read by the wdtk solvers and by EPANET (through WNTR).

Usage: ``uv run python scripts/build_test_networks.py``
"""

from pathlib import Path

import wntr

from wdtk.network.io import write_inp
from wdtk.network.model import Junction, Network, Pipe, Pump, Reservoir

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "networks"
NET1_SOURCE = Path(wntr.__file__).parent / "library" / "networks" / "Net1.inp"

HW_C = 120.0
DW_ROUGHNESS_M = 0.1e-3  # cement-lined ductile iron (project author)


def _with_formula(net: Network, headloss: str) -> Network:
    if headloss == "H-W":
        return net
    pipes = tuple(
        Pipe(p.id, p.start, p.end, p.length_m, p.diameter_m, DW_ROUGHNESS_M, p.minor_k, p.closed)
        for p in net.pipes
    )
    return Network(
        net.junctions,
        net.reservoirs,
        pipes,
        net.pumps,
        "D-W",
        net.coordinates,
        net.title.replace("Hazen-Williams", "Darcy-Weisbach"),
    )


def single_loop() -> Network:
    """Return the single-loop network (Hazen-Williams)."""
    junctions = (
        Junction("J1", 50.0, 0.000),
        Junction("J2", 48.0, 0.030),
        Junction("J3", 45.0, 0.045),
        Junction("J4", 47.0, 0.025),
    )
    pipes = (
        Pipe("P0", "R1", "J1", 1500.0, 0.40, HW_C),
        Pipe("P1", "J1", "J2", 800.0, 0.30, HW_C),
        Pipe("P2", "J2", "J3", 900.0, 0.20, HW_C),
        Pipe("P3", "J3", "J4", 800.0, 0.25, HW_C),
        Pipe("P4", "J4", "J1", 900.0, 0.30, HW_C),
    )
    coords = {
        "R1": (0.0, 1.0),
        "J1": (1.0, 1.0),
        "J2": (2.0, 1.0),
        "J3": (2.0, 0.0),
        "J4": (1.0, 0.0),
    }
    return Network(
        junctions,
        (Reservoir("R1", 100.0),),
        pipes,
        (),
        "H-W",
        coords,
        "Single loop, Hazen-Williams (wdtk test network)",
    )


def two_loop() -> Network:
    """Return the two-loop network (Hazen-Williams)."""
    junctions = (
        Junction("J1", 150.0, 0.010),
        Junction("J2", 155.0, 0.030),
        Junction("J3", 160.0, 0.025),
        Junction("J4", 145.0, 0.035),
        Junction("J5", 150.0, 0.040),
        Junction("J6", 152.0, 0.030),
    )
    pipes = (
        Pipe("P1", "R1", "J1", 1000.0, 0.45, HW_C),
        Pipe("P2", "J1", "J2", 1000.0, 0.30, HW_C),
        Pipe("P3", "J2", "J3", 1000.0, 0.20, HW_C),
        Pipe("P4", "J1", "J4", 1000.0, 0.35, HW_C),
        Pipe("P5", "J2", "J5", 1000.0, 0.15, HW_C),
        Pipe("P6", "J3", "J6", 1000.0, 0.15, HW_C),
        Pipe("P7", "J4", "J5", 1000.0, 0.25, HW_C),
        Pipe("P8", "J5", "J6", 1000.0, 0.20, HW_C),
    )
    coords = {
        "R1": (0.0, 1.0),
        "J1": (1.0, 1.0),
        "J2": (2.0, 1.0),
        "J3": (3.0, 1.0),
        "J4": (1.0, 0.0),
        "J5": (2.0, 0.0),
        "J6": (3.0, 0.0),
    }
    return Network(
        junctions,
        (Reservoir("R1", 210.0),),
        pipes,
        (),
        "H-W",
        coords,
        "Two loop, Hazen-Williams (wdtk test network)",
    )


def net1_steady() -> Network:
    """Return Net1 reduced to steady state with reservoirs only (Hazen-Williams)."""
    wn = wntr.network.WaterNetworkModel(str(NET1_SOURCE))
    junctions = tuple(
        Junction(name, j.elevation, j.demand_timeseries_list[0].base_value)
        for name, j in wn.junctions()
    )
    reservoirs = [Reservoir(name, r.base_head) for name, r in wn.reservoirs()]
    reservoirs += [Reservoir(name, t.elevation + t.init_level) for name, t in wn.tanks()]
    pipes = tuple(
        Pipe(
            name,
            p.start_node_name,
            p.end_node_name,
            p.length,
            p.diameter,
            p.roughness,
            p.minor_loss,
            closed=str(p.initial_status).upper() == "CLOSED",
        )
        for name, p in wn.pipes()
    )
    pumps = tuple(
        Pump(
            name,
            p.start_node_name,
            p.end_node_name,
            tuple((float(q), float(h)) for q, h in p.get_pump_curve().points),
        )
        for name, p in wn.head_pumps()
    )
    coords = {name: (float(n.coordinates[0]), float(n.coordinates[1])) for name, n in wn.nodes()}
    return Network(
        junctions,
        tuple(reservoirs),
        pipes,
        pumps,
        "H-W",
        coords,
        "EPANET Net1 reduced to steady state (tank 2 as reservoir), Hazen-Williams",
    )


def main() -> None:
    """Write all test networks."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builds = {
        "single_loop_hw": single_loop(),
        "single_loop_dw": _with_formula(single_loop(), "D-W"),
        "two_loop_hw": two_loop(),
        "two_loop_dw": _with_formula(two_loop(), "D-W"),
        "net1_steady_hw": net1_steady(),
    }
    for name, net in builds.items():
        path = OUT_DIR / f"{name}.inp"
        write_inp(net, path, units="LPS")
        print(
            f"Wrote {path.relative_to(ROOT)}: {len(net.junctions)} junctions, "
            f"{len(net.pipes)} pipes, {len(net.pumps)} pumps"
        )


if __name__ == "__main__":
    main()
