"""Tests for wdtk.analysis on the toy network (structure and physical sense, no Net3 results)."""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from conftest import ASSUMPTIONS_FILE, HOUR, PATTERN

from wdtk import units
from wdtk.analysis import criteria, criticality, energy, fire_flow, sizing, water_age
from wdtk.analysis.assumptions import (
    CriticalitySettings,
    EnergySettings,
    FireFlowSettings,
    PddSettings,
    ServiceCriteria,
    SizingSettings,
    WaterAgeSettings,
    load_assumptions,
)
from wdtk.analysis.simulate import (
    EpanetSession,
    run_eps,
    set_pressure_dependent,
    steady_max_day,
)

SERVICE = ServiceCriteria(min_pressure_m=30.0, max_pressure_m=90.0, max_velocity_ms=1.0)
PDD = PddSettings(minimum_pressure_m=0.0, required_pressure_m=14.0)
FIRE = FireFlowSettings(
    residual_pressure_m=14.0, required_flow_m3s=0.03, upper_bound_m3s=0.2, tolerance_m3s=5e-4
)


def fire_snapshot(wn: Any) -> Any:
    snap = steady_max_day(wn, 1.5)
    set_pressure_dependent(snap, PDD)
    return snap


# --- Assumptions -----------------------------------------------------------------------------


def test_assumptions_file_loads_and_converts() -> None:
    a = load_assumptions(ASSUMPTIONS_FILE)
    entry = a.entries["service.min_pressure"]
    assert a.service.min_pressure_m == pytest.approx(
        units.pressure_to_head(float(entry.value), "psi", "m")
    )
    assert a.fire_flow.required_flow_m3s == pytest.approx(
        units.flow_to_m3s(float(a.entries["fire_flow.required_flow"].value), "gpm")
    )
    assert a.atmospheric_pressure_kpa == pytest.approx(units.pressure_to_kpa(14.4, "psi"))
    assert set(a.sizing.unit_cost_per_m) == set(a.sizing.diameters_m)
    assert all(a.entries[d].status == "draft" for d in a.drafts())


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "a.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d["service"].pop("min_pressure"), "missing"),
        (lambda d: d["service"]["min_pressure"].update(units="kPa"), "units"),
        (lambda d: d["service"]["min_pressure"].update(status="maybe"), "status"),
        (lambda d: d["service"]["min_pressure"].update(value=150), "Invalid"),
        (lambda d: d["sizing"]["unit_cost"]["value"].pop(6), "unit_cost"),
        (lambda d: d["service"].update(min_pressure=35), "value, units"),
    ],
)
def test_assumptions_validation(tmp_path: Path, mutate: Any, match: str) -> None:
    data = yaml.safe_load(ASSUMPTIONS_FILE.read_text(encoding="utf-8"))
    mutate(data)
    with pytest.raises(ValueError, match=match):
        load_assumptions(_write(tmp_path, data))


# --- Simulation helpers and 4a ---------------------------------------------------------------


def test_session_matches_epanet_simulator(toy: Any) -> None:
    snap = steady_max_day(toy, 1.0)
    res = run_eps(snap, 0)
    with EpanetSession(snap) as s:
        state = s.solve_steady()
    expected = res.node["pressure"].loc[0, state.index]
    np.testing.assert_allclose(state["pressure_m"], expected, atol=1e-3)


def test_baseline_compliance_tables(toy: Any) -> None:
    res = run_eps(toy, 72 * HOUR)
    j = criteria.junction_compliance(res, toy, SERVICE)
    assert not j.loc["J0", "serves_demand"] and j.loc["J1", "serves_demand"]
    assert j.loc["J0", "steps_below_min"] == 0  # not a service point
    strict = ServiceCriteria(30.0, 1.0, 1.0)  # every pressure exceeds the max
    js = criteria.junction_compliance(res, toy, strict)
    assert js.loc["J0", "steps_above_max"] == 0 and js.loc["J1", "steps_above_max"] > 0
    p = criteria.pipe_compliance(res, toy, SERVICE)
    assert (p["max_velocity_ms"] >= 0).all() and len(p) == len(toy.pipe_name_list)
    levels = criteria.tank_levels(res, toy)
    tank = toy.get_node("T")
    assert levels["T"].min() >= tank.min_level - 1e-6 and levels["T"].max() <= tank.max_level + 1e-6
    assert (criteria.peak_hour_s(res, toy) / HOUR) % 24 == PATTERN.index(max(PATTERN))
    s = criteria.summarize(j, p)
    assert s["junctions_checked_min"] == 5


# --- 4b Fire flow ----------------------------------------------------------------------------


def test_available_fire_flow_meets_criteria_at_result(toy: Any) -> None:
    table = fire_flow.fire_flow_table(fire_snapshot(toy), FIRE)
    assert set(table.index) == {"J1", "J2", "J3", "J4", "J5"}
    assert (table["available_m3s"] <= FIRE.upper_bound_m3s).all()
    assert (table["meets_requirement"] == (table["available_m3s"] >= FIRE.required_flow_m3s)).all()
    row = table.loc["J5"]
    assert row["governing_limit"] in {
        "residual at fire node",
        "residual elsewhere",
        "delivery at fire node",
    }


def test_bigger_main_gives_more_fire_flow(toy: Any) -> None:
    before = fire_flow.fire_flow_table(fire_snapshot(toy), FIRE, ["J5"]).loc["J5", "available_m3s"]
    bigger = copy.deepcopy(toy)
    bigger.get_link("P5").diameter = 0.2
    after = fire_flow.fire_flow_table(fire_snapshot(bigger), FIRE, ["J5"]).loc[
        "J5", "available_m3s"
    ]
    assert after > before


def test_fire_flow_infeasible_before_fire(toy: Any) -> None:
    strict = FireFlowSettings(200.0, 0.03, 0.2, 5e-4)
    table = fire_flow.fire_flow_table(fire_snapshot(toy), strict, ["J1"])
    assert table.loc["J1", "available_m3s"] == 0.0
    assert "before fire flow" in table.loc["J1", "governing_limit"]


def test_fire_flow_capped_at_search_bound(toy: Any) -> None:
    tiny = FireFlowSettings(1.0, 0.001, 0.002, 1e-4)
    table = fire_flow.fire_flow_table(fire_snapshot(toy), tiny, ["J1"])
    assert table.loc["J1", "governing_limit"] == "search upper bound"


# --- 4c Sizing -------------------------------------------------------------------------------

SIZES = tuple(units.length_to_m(d, "in") for d in (4, 6, 8, 10, 12, 16))
SIZING = SizingSettings(
    diameters_m=SIZES,
    unit_cost_per_m={d: 100.0 + 2000.0 * d for d in SIZES},
    contingency=0.3,
    candidate_hops=2,
    max_iterations=8,
)


def test_next_size_and_candidates(toy: Any) -> None:
    assert sizing.next_size_m(0.15, SIZES) == pytest.approx(units.length_to_m(8, "in"))
    assert sizing.next_size_m(units.length_to_m(6, "in"), SIZES) == pytest.approx(
        units.length_to_m(8, "in")
    )
    assert sizing.next_size_m(0.13, SIZES) == pytest.approx(units.length_to_m(6, "in"))
    assert sizing.next_size_m(units.length_to_m(16, "in"), SIZES) is None
    assert set(sizing.candidate_pipes(toy, ["J5"], 1)) == {"P5", "P2", "P3"}


def test_snapshot_reproduces_eps_instant(toy: Any) -> None:
    res = run_eps(toy, 48 * HOUR)
    t = 30 * HOUR
    snap = sizing.steady_snapshot_at(toy, res, t)
    with EpanetSession(snap) as s:
        state = s.solve_steady()
    np.testing.assert_allclose(
        state["pressure_m"], res.node["pressure"].loc[t, state.index], atol=0.01
    )


def test_greedy_upsizing_reduces_deficit(toy: Any) -> None:
    res = run_eps(toy, 48 * HOUR)
    peak = sizing.steady_snapshot_at(toy, res, criteria.peak_hour_s(res, toy))
    need = FireFlowSettings(14.0, 0.06, 0.2, 5e-4)
    snap = fire_snapshot(toy)
    table = fire_flow.fire_flow_table(snap, need)
    deficient = list(table.index[~table["meets_requirement"]])
    assert deficient, "toy network should start fire-deficient for this test"
    improvements, history = sizing.greedy_upsizing(snap, peak, deficient, need, SERVICE, SIZING)
    assert len(improvements) >= 1
    assert history["deficit_m"].is_monotonic_decreasing
    assert history["deficit_m"].iloc[-1] < history["deficit_m"].iloc[0]
    for _pipe, row in improvements.iterrows():
        assert any(np.isclose(row["proposed_diameter_m"], d) for d in SIZES)
        assert row["proposed_diameter_m"] > row["existing_diameter_m"]
        expected = row["length_m"] * SIZING.unit_cost_per_m[row["proposed_diameter_m"]] * 1.3
        assert row["cost"] == pytest.approx(expected)
    improved = sizing.apply_improvements(toy, improvements)
    assert improved.get_link(improvements.index[0]).diameter == pytest.approx(
        improvements["proposed_diameter_m"].iloc[0]
    )


# --- 4d Water age ----------------------------------------------------------------------------

AGE = WaterAgeSettings(48 * HOUR, 120 * HOUR, 24 * HOUR, 20.0)


def test_water_age_dead_end_is_older(toy: Any) -> None:
    res = run_eps(toy, AGE.duration_s, quality="AGE")
    ages = water_age.junction_age(res, toy, AGE)
    assert ages.loc["J5", "max_age_s"] > ages.loc["J1", "max_age_s"]
    assert (ages["above_threshold"] == (ages["max_age_s"] > AGE.threshold_s)).all()
    turnover = water_age.tank_turnover(res, toy, 48 * HOUR)
    assert turnover.loc["T", "turnover_days"] > 0 and turnover.loc["T", "daily_fill_m3"] > 0
    assert water_age.age_stabilization(res, toy, 24 * HOUR) >= 0.0


def test_set_tank_level_controls(toy: Any) -> None:
    alt = water_age.set_tank_level_controls(toy, "T", 1.5, 9.0)
    thresholds = sorted(alt.get_control(c).condition._threshold for c in ("pb_open", "pb_close"))
    assert thresholds == [1.5, 9.0]
    assert sorted(toy.get_control(c).condition._threshold for c in ("pb_open", "pb_close")) == [
        2.0,
        6.0,
    ]
    with pytest.raises(ValueError):
        water_age.set_tank_level_controls(toy, "R", 1.0, 2.0)


# --- 4e Energy -------------------------------------------------------------------------------

TARIFF = EnergySettings(0.20, 0.05, 12.0, 20.0, 0.7, 20.0)


def test_price_schedule() -> None:
    assert energy.price_per_kwh(13 * HOUR, TARIFF) == 0.20
    assert energy.price_per_kwh((24 + 21) * HOUR, TARIFF) == 0.05


def test_pump_energy_matches_hand_calculation(toy: Any) -> None:
    res = run_eps(toy, 48 * HOUR)
    steps = energy.pump_energy(res, toy, TARIFF)
    row = steps[(steps["pump"] == "PU") & (steps["time_s"] == 5 * HOUR)].iloc[0]
    q = res.link["flowrate"].loc[5 * HOUR, "PU"]
    h = res.node["head"].loc[5 * HOUR, "J0"] - res.node["head"].loc[5 * HOUR, "R"]
    expected_kwh = 1000.0 * 9.80665 * q * h / 0.7 / 1000.0 * 1.0
    # EPANET stores results in single precision; agreement is to ~1e-7.
    assert row["energy_kwh"] == pytest.approx(expected_kwh, rel=1e-6)
    assert row["cost"] == pytest.approx(expected_kwh * 0.05, rel=1e-6)
    summary = energy.energy_summary(steps, TARIFF)
    assert summary.loc["PU", "run_hours"] == pytest.approx(28.0)  # 14 h/day x 2 days
    assert 0.0 <= summary.loc["PU", "on_peak_fraction"] <= 1.0


def test_reschedule_to_original_window_reproduces_schedule(toy: Any) -> None:
    assert energy.timed_run_hours(toy, "PU") == pytest.approx(14.0)
    same = energy.reschedule_timed_pump(toy, "PU", 1.0, 14.0)
    a = run_eps(toy, 48 * HOUR).link["status"]["PU"]
    b = run_eps(same, 48 * HOUR).link["status"]["PU"]
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    night = energy.reschedule_timed_pump(toy, "PU", 20.0, 14.0)
    s = run_eps(night, 48 * HOUR).link["status"]["PU"]
    assert s.loc[0] == 1 and s.loc[11 * HOUR] == 0 and s.loc[21 * HOUR] == 1
    with pytest.raises(ValueError):
        energy.reschedule_timed_pump(toy, "PU", 1.0, 30.0)
    with pytest.raises(ValueError):
        energy.timed_run_hours(toy, "P1")


# --- 4f Criticality --------------------------------------------------------------------------


def test_closing_dead_end_main_loses_its_demand(toy: Any) -> None:
    pda = copy.deepcopy(toy)
    set_pressure_dependent(pda, PDD)
    crit = CriticalitySettings(min_diameter_m=0.1, duration_s=24 * HOUR)
    table, excluded = criticality.criticality_table(pda, crit, SERVICE)
    assert excluded == ["PB"]  # control target
    assert "PB" not in table.index and "PU" not in table.index
    expected = 0.003 * np.mean(PATTERN) * 24 * HOUR  # J5's full daily demand
    assert table.loc["P5", "unserved_m3"] == pytest.approx(expected, rel=0.02)
    assert table.loc["P5", "junctions_below_min"] == 1
    assert table["unserved_m3"].is_monotonic_decreasing


def test_controlled_links(toy: Any) -> None:
    assert criticality.controlled_links(toy) == {"PU", "PB"}
