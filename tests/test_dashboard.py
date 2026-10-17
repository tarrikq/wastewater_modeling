"""Tests for wdtk.dashboard, the Streamlit app (headless), and the reproduce script."""

import sys
from pathlib import Path

import numpy as np
import pytest
import wntr
from conftest import ASSUMPTIONS_FILE, build_toy
from streamlit.testing.v1 import AppTest
from test_run_study import _fast_assumptions

from wdtk import dashboard, units
from wdtk import pumps as pm
from wdtk.analysis.assumptions import load_assumptions

ROOT = Path(__file__).resolve().parents[1]
A = load_assumptions(ASSUMPTIONS_FILE)


@pytest.fixture(scope="module")
def net1() -> dashboard.NetworkResults:
    return dashboard.compute_results(dashboard.NETWORKS["Net1"], A)


def test_results_tables(net1: dashboard.NetworkResults) -> None:
    assert set(net1.nodes.columns) == {"min_pressure_psi", "max_age_h", "serves_demand"}
    assert list(net1.links.index) == list(net1.wn.pipe_name_list)
    assert (net1.fire["available_gpm"] >= 0).all()


@pytest.mark.parametrize("layer", list(dashboard.LAYERS))
def test_map_layers(net1: dashboard.NetworkResults, layer: str) -> None:
    fig = dashboard.network_map_figure(net1, layer)
    kind, column, _ = dashboard.LAYERS[layer]
    hover = [t for t in fig.data if t.hovertemplate and layer in t.hovertemplate]
    assert len(hover) == 1
    if kind == "node":
        assert len(hover[0].x) == int(net1.nodes["serves_demand"].sum())
    else:
        valued = net1.links[column].dropna()
        binned = [t for t in fig.data if t.legendgroup == "bins"]
        assert len(binned) == dashboard.N_COLOR_BINS
        # Every valued pipe is drawn in exactly one class (3 coordinates per segment).
        assert sum(sum(x is not None for x in t.x) // 2 for t in binned) == len(valued)


def test_bin_edges_span_values() -> None:
    import pandas as pd

    edges = dashboard.bin_edges(pd.Series([1.0, 2.0, np.nan, 5.0]))
    assert edges[0] == 1.0 and edges[-1] == 5.0 and len(edges) == dashboard.N_COLOR_BINS + 1
    flat = dashboard.bin_edges(pd.Series([3.0, 3.0]))
    assert flat[-1] > flat[0]


def test_pump_explorer_matches_operating_point() -> None:
    curve = pm.fit_three_point(60.0, 0.10, 45.0, 0.22)
    fig, point = dashboard.pump_explorer(curve, 0.9, 50.0, 5000.0, 12.0, 120.0, 0.75)
    scaled = pm.scale_curve_speed(curve, 0.9)
    q, h = pm.operating_point(
        lambda x: pm.pump_head_m(scaled, x),
        lambda x: pm.system_head_hw_m(
            x,
            units.length_to_m(50.0, "ft"),
            units.length_to_m(5000.0, "ft"),
            units.length_to_m(12.0, "in"),
            120.0,
        ),
    )
    assert point.flow_gpm == pytest.approx(units.flow_from_m3s(q, "gpm"))
    assert point.head_ft == pytest.approx(units.length_from_m(h, "ft"))
    assert point.input_hp == pytest.approx(point.hydraulic_hp / 0.75)
    assert any(t.name == "Operating point" for t in fig.data)
    _, none = dashboard.pump_explorer(curve, 0.5, 500.0, 5000.0, 12.0, 120.0, 0.75)
    assert none.flow_gpm is None


def test_pump_curves_include_network_pumps(net1: dashboard.NetworkResults) -> None:
    curves = dashboard.pump_curves(net1)
    assert "Pump 9" in curves and dashboard.ILLUSTRATIVE_PUMP in curves


def test_fire_lookup_and_assumptions(net1: dashboard.NetworkResults) -> None:
    j = net1.fire.index[0]
    assert dashboard.fire_lookup(net1, j)["available_gpm"] == net1.fire.loc[j, "available_gpm"]
    table = dashboard.assumptions_table(A)
    assert len(table) == len(A.entries) and {"value", "units", "basis", "status"} <= set(
        table.columns
    )


def test_results_on_network_without_large_pipes(tmp_path: Path) -> None:
    inp = tmp_path / "toy.inp"
    wntr.network.write_inpfile(build_toy(), str(inp))
    res = dashboard.compute_results(inp, A)
    assert res.links["unserved_pct"].isna().all()  # no pipe reaches the screening diameter
    fig = dashboard.network_map_figure(res, "Minimum pressure (psi)")
    assert fig.data


# --- Streamlit app (headless) ----------------------------------------------------------------


def test_app_runs_every_tab_and_network() -> None:
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception
    assert len(at.tabs) == 4
    for layer in at.radio(key="layer").options:
        at.radio(key="layer").set_value(layer).run()
        assert not at.exception
    at.slider(key="speed").set_value(0.5).run()
    assert not at.exception
    at.selectbox(key="network").set_value("Net1").run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"Available", "Required", "Deficit"} <= set(labels)


# --- Reproduce script --------------------------------------------------------------------------


def test_reproduce_all_writes_every_output(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import reproduce_all

    reports, networks = tmp_path / "reports", tmp_path / "networks"
    code = reproduce_all.main(
        [
            "--reports",
            str(reports),
            "--networks",
            str(networks),
            "--assumptions",
            str(_fast_assumptions(tmp_path)),
        ]
    )
    assert code == 0
    expected = [
        "phase1/hw_dw_headloss_ratio.png",
        "phase1/equivalent_c_factor.png",
        "phase2/pump_system_vfd.png",
        "phase3/convergence.png",
        "phase3/validation_summary.csv",
        "phase4/4f_map_criticality.png",
        "key_results.json",
    ]
    assert all((reports / e).exists() for e in expected)
    assert len(list(networks.glob("*.inp"))) == 5


def test_readme_images_exist() -> None:
    import re

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    missing = [p for p in images if not (ROOT / p).exists()]
    assert not missing, missing
