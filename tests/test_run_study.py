"""Tests for the Phase 4 study runner (dry run, draft guard, end-to-end smoke run)."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import yaml
from conftest import ASSUMPTIONS_FILE

from wdtk import plotting, run_study


def _with_draft(tmp_path: Path) -> Path:
    """The real file with one entry set back to draft."""
    data: dict[str, Any] = yaml.safe_load(ASSUMPTIONS_FILE.read_text(encoding="utf-8"))
    data["service"]["min_pressure"]["status"] = "draft"
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_dry_run_lists_drafts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_study.main(["--dry-run", "--assumptions", str(_with_draft(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "4f criticality" in out and "service.min_pressure" in out


def test_full_run_refused_while_drafts_remain(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert run_study.main(["--assumptions", str(_with_draft(tmp_path)), "--out", str(out)]) == 2
    assert not out.exists()


def _fast_assumptions(tmp_path: Path) -> Path:
    """The real file with every status 'supplied' and settings shrunk for a quick run."""
    data: dict[str, Any] = yaml.safe_load(ASSUMPTIONS_FILE.read_text(encoding="utf-8"))
    for group in data.values():
        for entry in group.values():
            if entry["status"] == "draft":
                entry["status"] = "supplied"
    data["sizing"]["max_iterations"]["value"] = 1
    data["sizing"]["candidate_hops"]["value"] = 1
    data["criticality"]["min_diameter"]["value"] = 60
    data["water_age"]["simulation_duration"]["value"] = 48
    path = tmp_path / "fast.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_end_to_end_smoke(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert (
        run_study.main(["--assumptions", str(_fast_assumptions(tmp_path)), "--out", str(out)]) == 0
    )
    expected = [
        "4a_junction_compliance.csv",
        "4a_pipe_compliance.csv",
        "4a_tank_levels_ft.csv",
        "4a_map_min_pressure.png",
        "4a_map_max_velocity.png",
        "4a_tank_levels.png",
        "4b_fire_flow.csv",
        "4b_map_fire_flow.png",
        "4c_recommended_improvements.csv",
        "4c_sizing_history.csv",
        "4c_map_improvements.png",
        "4d_water_age_junctions.csv",
        "4d_tank_turnover.csv",
        "4d_map_water_age.png",
        "4e_energy_summary.csv",
        "4e_pump_power.png",
        "4f_criticality.csv",
        "4f_map_criticality.png",
        "summary.json",
    ]
    missing = [f for f in expected if not (out / f).exists()]
    assert not missing
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) >= {
        "4a_baseline",
        "4b_fire_flow",
        "4c_sizing",
        "4d_water_age",
        "4e_energy",
        "4f_criticality",
        "assumptions",
    }
    fire = pd.read_csv(out / "4b_fire_flow.csv")
    assert {"available_gpm", "required_gpm", "governing_limit"} <= set(fire.columns)


def test_network_map_draws(toy: Any) -> None:
    fig, ax = plt.subplots()
    plotting.network_map(
        ax,
        toy,
        node_values=pd.Series({"J1": 1.0, "J5": 3.0}),
        link_values=pd.Series({"P1": 0.5, "P5": 2.0}),
        label="x",
        highlight_links=["P2"],
        highlight_label="upsized",
    )
    assert len(ax.collections) >= 4
    plt.close(fig)
