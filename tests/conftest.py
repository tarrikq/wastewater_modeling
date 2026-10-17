"""Shared fixtures: a small WNTR network that exercises every Phase 4 analysis.

Layout (SI): reservoir R (head 60 m) feeds pump PU into J0, then a loop J1-J2-J3-J4 with a
dead-end branch J3-J5. Tank T floats on the system through pipe PT at J2.

* Pump PU runs 01:00-15:00 each day on simulation-time controls (like Net3's pump 10).
* A small gravity bypass PB (R to J4) opens when tank T is below 2 m and closes above 6 m
  (tank level controls, like Net3's Tank 1).
* Junction demands follow a 24-hour pattern with a single peak at hour 18.
"""

from pathlib import Path
from typing import Any

import pytest
import wntr
from wntr.network.base import LinkStatus
from wntr.network.controls import Control, ControlAction, SimTimeCondition, TankLevelCondition

HOUR = 3600
PATTERN = [0.6] * 6 + [1.0] * 6 + [1.1] * 6 + [1.6] + [1.2] * 5  # peak at hour 18


def build_toy() -> Any:
    wn = wntr.network.WaterNetworkModel()
    wn.options.hydraulic.headloss = "H-W"
    wn.options.time.duration = 72 * HOUR
    wn.options.time.hydraulic_timestep = HOUR
    wn.options.time.pattern_timestep = HOUR
    wn.options.time.report_timestep = HOUR
    wn.options.time.quality_timestep = 300
    wn.add_pattern("dem", PATTERN)
    wn.add_reservoir("R", base_head=60.0, coordinates=(0, 1))
    for name, elev, demand, xy in [
        ("J0", 0.0, 0.0, (0.5, 1)),
        ("J1", 10.0, 0.004, (1, 1)),
        ("J2", 12.0, 0.006, (2, 1)),
        ("J3", 11.0, 0.005, (3, 1)),
        ("J4", 9.0, 0.004, (1, 0)),
        ("J5", 13.0, 0.003, (4, 1)),
    ]:
        pattern = None if demand == 0.0 else "dem"
        wn.add_junction(
            name, base_demand=demand, elevation=elev, demand_pattern=pattern, coordinates=xy
        )
    wn.add_tank(
        "T",
        elevation=50.0,
        init_level=5.0,
        min_level=1.0,
        max_level=10.0,
        diameter=12.0,
        coordinates=(2, 2),
    )
    wn.add_curve("PC", "HEAD", [(0.04, 35.0)])
    wn.add_pump("PU", "R", "J0", "HEAD", "PC")
    for name, a, b, length, d in [
        ("P0", "J0", "J1", 100.0, 0.30),
        ("P1", "J1", "J2", 500.0, 0.20),
        ("P2", "J2", "J3", 500.0, 0.15),
        ("P3", "J3", "J4", 700.0, 0.15),
        ("P4", "J1", "J4", 500.0, 0.20),
        ("P5", "J3", "J5", 400.0, 0.10),
        ("PT", "J2", "T", 100.0, 0.20),
        ("PB", "R", "J4", 300.0, 0.10),
    ]:
        wn.add_pipe(name, a, b, length=length, diameter=d, roughness=120.0)
    wn.get_link("PB").initial_status = LinkStatus.Closed

    pump = wn.get_link("PU")
    pump.initial_status = LinkStatus.Closed
    for day in range(6):
        for hour, status in ((1, LinkStatus.Open), (15, LinkStatus.Closed)):
            cond = SimTimeCondition(wn, "=", (24 * day + hour) * HOUR)
            wn.add_control(f"pu_{day}_{hour}", Control(cond, ControlAction(pump, "status", status)))

    tank, bypass = wn.get_node("T"), wn.get_link("PB")
    wn.add_control(
        "pb_open",
        Control(
            TankLevelCondition(tank, "level", "<", 2.0),
            ControlAction(bypass, "status", LinkStatus.Open),
        ),
    )
    wn.add_control(
        "pb_close",
        Control(
            TankLevelCondition(tank, "level", ">", 6.0),
            ControlAction(bypass, "status", LinkStatus.Closed),
        ),
    )
    return wn


@pytest.fixture
def toy() -> Any:
    return build_toy()


ASSUMPTIONS_FILE = Path(__file__).resolve().parents[1] / "data" / "assumptions.yaml"
