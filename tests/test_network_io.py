"""Tests for wdtk.network.model and wdtk.network.io."""

from pathlib import Path

import numpy as np
import pytest

from wdtk import units
from wdtk.network.io import UnsupportedInpError, read_inp, write_inp
from wdtk.network.model import Junction, Network, Pipe, Pump, Reservoir

NETWORKS = Path(__file__).resolve().parents[1] / "data" / "networks"


def small_network(headloss: str = "H-W") -> Network:
    rough = 120.0 if headloss == "H-W" else 1e-4
    return Network(
        junctions=(Junction("J1", 10.0, 0.01), Junction("J2", 12.0, 0.02)),
        reservoirs=(Reservoir("R1", 60.0),),
        pipes=(
            Pipe("P1", "R1", "J1", 500.0, 0.3, rough, minor_k=2.0),
            Pipe("P2", "J1", "J2", 400.0, 0.2, rough),
            Pipe("P3", "J2", "R1", 300.0, 0.1, rough, closed=True),
        ),
        pumps=(Pump("PU", "J1", "J2", ((0.0, 40.0), (0.05, 30.0), (0.1, 0.0))),),
        headloss=headloss,  # type: ignore[arg-type]
        coordinates={"J1": (1.0, 2.0), "J2": (3.0, 4.0), "R1": (0.0, 0.0)},
        title="small",
    )


def assert_networks_close(a: Network, b: Network) -> None:
    assert a.headloss == b.headloss
    for xs, ys in ((a.junctions, b.junctions), (a.reservoirs, b.reservoirs), (a.pipes, b.pipes)):
        assert len(xs) == len(ys)
        for x, y in zip(xs, ys, strict=True):
            for field in x.__dataclass_fields__:
                vx, vy = getattr(x, field), getattr(y, field)
                if isinstance(vx, float):
                    assert vx == pytest.approx(vy, rel=1e-12, abs=1e-15), field
                else:
                    assert vx == vy, field
    for pa, pb in zip(a.pumps, b.pumps, strict=True):
        np.testing.assert_allclose(pa.curve_points, pb.curve_points, rtol=1e-12)
    assert dict(a.coordinates) == pytest.approx(dict(b.coordinates))


@pytest.mark.parametrize("units_flag", ["LPS", "GPM", "CMH", "MGD"])
@pytest.mark.parametrize("headloss", ["H-W", "D-W"])
def test_write_read_round_trip(tmp_path: Path, units_flag: str, headloss: str) -> None:
    net = small_network(headloss)
    path = tmp_path / "net.inp"
    write_inp(net, path, units=units_flag)
    assert_networks_close(net, read_inp(path))


def test_us_units_conversion(tmp_path: Path) -> None:
    path = tmp_path / "us.inp"
    path.write_text(
        "[JUNCTIONS]\nJ1 100 50\n[RESERVOIRS]\nR1 300\n"
        "[PIPES]\nP1 R1 J1 1000 12 0.5 0 Open\n[OPTIONS]\nUnits GPM\nHeadloss D-W\n[END]\n",
        encoding="utf-8",
    )
    net = read_inp(path)
    assert net.junctions[0].elevation_m == pytest.approx(units.length_to_m(100.0, "ft"))
    assert net.junctions[0].demand_m3s == pytest.approx(units.flow_to_m3s(50.0, "gpm"))
    assert net.reservoirs[0].head_m == pytest.approx(300.0 * units.FT_M)
    p = net.pipes[0]
    assert p.length_m == pytest.approx(1000.0 * units.FT_M)
    assert p.diameter_m == pytest.approx(12.0 * units.IN_M)
    assert p.roughness == pytest.approx(0.5e-3 * units.FT_M)  # D-W roughness in 1e-3 ft


def test_cosmetic_sections_skipped_and_coordinates_kept(tmp_path: Path) -> None:
    path = tmp_path / "c.inp"
    path.write_text(
        "[TITLE]\nDemo net\n[JUNCTIONS]\nJ1 0 1\n[RESERVOIRS]\nR1 50\n"
        "[PIPES]\nP1 R1 J1 100 300 120\n"
        '[OPTIONS]\nUnits LPS\n[TAGS]\nNODE J1 x\n[VERTICES]\nP1 1 1\n[LABELS]\n1 1 "a"\n'
        "[BACKDROP]\nUNITS None\n[COORDINATES]\nJ1 5 6\nR1 0 0\n[END]\n",
        encoding="utf-8",
    )
    net = read_inp(path)
    assert net.title == "Demo net"
    assert net.coordinates["J1"] == (5.0, 6.0)


@pytest.mark.parametrize(
    ("snippet", "match"),
    [
        ("[TANKS]\nT1 10 5 0 10 20 0\n", r"\[TANKS\]"),
        ("[PATTERNS]\n1 1.0 1.2\n", r"\[PATTERNS\]"),
        ("[CONTROLS]\nLINK P1 CLOSED AT TIME 1\n", r"\[CONTROLS\]"),
        ("[VALVES]\nV1 J1 R1 300 PRV 50 0\n", r"\[VALVES\]"),
    ],
)
def test_unsupported_sections_raise_with_name(tmp_path: Path, snippet: str, match: str) -> None:
    path = tmp_path / "bad.inp"
    path.write_text(
        "[JUNCTIONS]\nJ1 0 1\n[RESERVOIRS]\nR1 50\n[PIPES]\nP1 R1 J1 100 300 120\n" + snippet,
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedInpError, match=match):
        read_inp(path)


@pytest.mark.parametrize(
    "body",
    [
        "[JUNCTIONS]\nJ1 0 1 pat1\n[RESERVOIRS]\nR1 50\n[PIPES]\nP1 R1 J1 100 300 120\n",
        "[JUNCTIONS]\nJ1 0 1\n[RESERVOIRS]\nR1 50\n[PIPES]\nP1 R1 J1 100 300 120 0 CV\n",
        "[JUNCTIONS]\nJ1 0 1\n[RESERVOIRS]\nR1 50\n[PIPES]\nP1 R1 J1 100 300 120\n"
        "[OPTIONS]\nUnits LPS\nTrials 50\n",
        "[JUNCTIONS]\nJ1 0 1\n[RESERVOIRS]\nR1 50\n[PIPES]\nP1 R1 J1 100 300 120\n"
        "[OPTIONS]\nUnits AFD\n",
        "[JUNCTIONS]\nJ1 0 1\n[RESERVOIRS]\nR1 50\n[PIPES]\nP1 R1 J1 100 300 120\n"
        "[OPTIONS]\nHeadloss C-M\n",
        "[JUNCTIONS]\nJ1 0 1\nJ2 0 0\n[RESERVOIRS]\nR1 50\n[PIPES]\nP1 R1 J1 100 300 120\n"
        "[PUMPS]\nPU J1 J2 POWER 10\n",
        "[JUNCTIONS]\nJ1 0 1\nJ2 0 0\n[RESERVOIRS]\nR1 50\n[PIPES]\nP1 R1 J1 100 300 120\n"
        "[PUMPS]\nPU J1 J2 HEAD missing\n",
        "J1 0 1\n[JUNCTIONS]\n",
    ],
)
def test_unsupported_features_raise(tmp_path: Path, body: str) -> None:
    path = tmp_path / "bad.inp"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(UnsupportedInpError):
        read_inp(path)


def test_write_rejects_unknown_units(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedInpError):
        write_inp(small_network(), tmp_path / "x.inp", units="AFD")


@pytest.mark.parametrize("path", sorted(NETWORKS.glob("*.inp")), ids=lambda p: p.stem)
def test_committed_networks_read(path: Path) -> None:
    net = read_inp(path)
    assert net.junctions and net.pipes


# --- Model validation and structure ----------------------------------------------------------


def test_incidence_matrices_follow_sign_convention() -> None:
    net = small_network()
    a12, a10 = net.incidence
    # Links: open pipes P1, P2 (P3 closed), then pump PU.
    assert [lk.id for lk in net.links] == ["P1", "P2", "PU"]
    np.testing.assert_array_equal(a12.toarray(), [[1, 0], [-1, 1], [-1, 1]])
    np.testing.assert_array_equal(a10.toarray(), [[-1], [0], [0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reservoirs": ()},
        {"pipes": (Pipe("P1", "R1", "J1", 1.0, 0.1, 100.0), Pipe("P1", "J1", "J2", 1, 0.1, 100))},
        {"pipes": (Pipe("P1", "R1", "JX", 1.0, 0.1, 100.0),)},
        {"pipes": (Pipe("P1", "J1", "J1", 1.0, 0.1, 100.0),)},
        {"pipes": (Pipe("P1", "R1", "J1", 1.0, 0.1, 100.0),)},  # J2 disconnected
        {"headloss": "C-M"},
    ],
)
def test_network_validation(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "junctions": (Junction("J1", 0.0), Junction("J2", 0.0)),
        "reservoirs": (Reservoir("R1", 10.0),),
        "pipes": (Pipe("P1", "R1", "J1", 1.0, 0.1, 100.0), Pipe("P2", "J1", "J2", 1, 0.1, 100)),
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        Network(**base)  # type: ignore[arg-type]


def test_pump_curve_point_rules() -> None:
    assert Pump("P", "a", "b", ((0.1, 50.0),)).curve.c == 2.0
    with pytest.raises(ValueError):
        _ = Pump("P", "a", "b", ((0.05, 40.0), (0.1, 30.0))).curve
