"""Phase 4 design and resilience study on EPANET Net3: one command reproduces every output.

Usage::

    uv run python -m wdtk.run_study            # full study -> reports/phase4/
    uv run python -m wdtk.run_study --dry-run  # list planned runs and assumption status only

The full study refuses to run while any entry in ``data/assumptions.yaml`` still has
``status: draft``: per PLAN.md, the project author reviews every assumption before results
are reported. Tables are written in US customary units (psi, gpm, ft, in, $); the analysis
itself is in SI (see :mod:`wdtk.analysis`).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wdtk import plotting, units
from wdtk.analysis import criteria, criticality, energy, fire_flow, sizing, water_age
from wdtk.analysis.assumptions import Assumptions, load_assumptions
from wdtk.analysis.simulate import (
    load_network,
    net3_path,
    run_eps,
    set_pressure_dependent,
    steady_max_day,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSUMPTIONS = ROOT / "data" / "assumptions.yaml"
DEFAULT_OUT = ROOT / "reports" / "phase4"
HOUR = 3600.0


# --- US customary conversions for reports ------------------------------------------------------
def psi(head_m: Any) -> Any:
    """Pressure head (m) -> psi."""
    return units.head_to_pressure(head_m, "m", "psi")


def gpm(q_m3s: Any) -> Any:
    """Flow (m^3/s) -> gpm."""
    return units.flow_from_m3s(q_m3s, "gpm")


def ft(x_m: Any) -> Any:
    """Length (m) -> ft."""
    return units.length_from_m(x_m, "ft")


def kgal(v_m3: Any) -> Any:
    """Volume (m^3) -> thousands of US gallons."""
    return v_m3 / units.US_GAL_M3 / 1000.0


def inch(x_m: Any) -> Any:
    """Length (m) -> in."""
    return units.length_from_m(x_m, "in")


# --- Study steps --------------------------------------------------------------------------
def baseline(wn: Any, a: Assumptions, out: Path) -> dict[str, Any]:
    """4a: baseline EPS, compliance tables, maps, tank levels."""
    res = run_eps(wn, a.eps_duration_s)
    j = criteria.junction_compliance(res, wn, a.service)
    p = criteria.pipe_compliance(res, wn, a.service)
    levels = criteria.tank_levels(res, wn)
    pd.DataFrame(
        {
            "elevation_ft": ft(j["elevation_m"]),
            "base_demand_gpm": gpm(j["base_demand_m3s"]),
            "serves_demand": j["serves_demand"],
            "min_pressure_psi": psi(j["min_pressure_m"]),
            "max_pressure_psi": psi(j["max_pressure_m"]),
            "hours_below_min": j["steps_below_min"],
            "hours_above_max": j["steps_above_max"],
        }
    ).round(2).to_csv(out / "4a_junction_compliance.csv", index_label="junction")
    pd.DataFrame(
        {
            "diameter_in": inch(p["diameter_m"]),
            "length_ft": ft(p["length_m"]),
            "max_velocity_fps": ft(p["max_velocity_ms"]),
            "hours_above_max": p["steps_above_max"],
        }
    ).round(2).to_csv(out / "4a_pipe_compliance.csv", index_label="pipe")
    ft(levels).set_axis(levels.index / HOUR).round(2).to_csv(
        out / "4a_tank_levels_ft.csv", index_label="hour"
    )

    served = j[j["serves_demand"]]
    _map(
        out / "4a_map_min_pressure.png",
        wn,
        "4a Minimum pressure over the 72-hour EPS",
        node_values=psi(served["min_pressure_m"]),
        label="Minimum pressure (psi)",
        reverse=True,
    )
    _map(
        out / "4a_map_max_velocity.png",
        wn,
        "4a Maximum pipe velocity over the 72-hour EPS",
        link_values=ft(p["max_velocity_ms"]),
        label="Maximum velocity (ft/s)",
    )

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    for color, name in zip(plotting.CATEGORICAL, levels.columns, strict=False):
        ax.plot(levels.index / HOUR, ft(levels[name]), color=color, label=f"Tank {name}")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Water level above tank bottom (ft)")
    ax.set_title("4a Tank levels, baseline EPS", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out / "4a_tank_levels.png")

    summary = criteria.summarize(j, p)
    return {
        "results": res,
        "peak_hour_s": criteria.peak_hour_s(res, wn),
        "summary": {
            "junctions_checked_min": summary["junctions_checked_min"],
            "junctions_below_min": summary["junctions_below_min"],
            "junctions_above_max": summary["junctions_above_max"],
            "pipes_above_max_velocity": summary["pipes_above_max_velocity"],
            "lowest_service_pressure_psi": round(
                float(psi(summary["lowest_service_pressure_m"])), 1
            ),
            "highest_pressure_psi": round(float(psi(summary["highest_pressure_m"])), 1),
            "highest_velocity_fps": round(float(ft(summary["highest_velocity_ms"])), 2),
        },
    }


def fire(wn: Any, a: Assumptions, out: Path, tag: str = "4b") -> pd.DataFrame:
    """4b: available fire flow at each eligible junction (max-day PDA snapshot)."""
    snap = steady_max_day(wn, a.max_day_factor)
    set_pressure_dependent(snap, a.pdd)
    table = fire_flow.fire_flow_table(snap, a.fire_flow)
    _fire_csv(table, out / f"{tag}_fire_flow.csv")
    _map(
        out / f"{tag}_map_fire_flow.png",
        wn,
        f"{tag} Available fire flow at {psi(a.fire_flow.residual_pressure_m):.0f} psi residual "
        f"(requirement {gpm(a.fire_flow.required_flow_m3s):,.0f} gpm)",
        node_values=gpm(table["available_m3s"]),
        label="Available fire flow (gpm; top of scale = search limit)",
        reverse=True,
    )
    return table


def _fire_csv(table: pd.DataFrame, path: Path) -> None:
    pd.DataFrame(
        {
            "available_gpm": gpm(table["available_m3s"]),
            "required_gpm": gpm(table["required_m3s"]),
            "deficit_gpm": gpm(table["deficit_m3s"]),
            "meets_requirement": table["meets_requirement"],
            "governing_limit": table["governing_limit"],
            "limiting_junction": table["limiting_junction"],
        }
    ).round(0).to_csv(path, index_label="junction")


def size_pipes(
    wn: Any, a: Assumptions, out: Path, fire_table: pd.DataFrame, base: dict[str, Any]
) -> dict[str, Any]:
    """4c: greedy upsizing for fire flow and peak-hour pressure deficits."""
    deficient = list(fire_table.index[~fire_table["meets_requirement"]])
    fire_snap = steady_max_day(wn, a.max_day_factor)
    set_pressure_dependent(fire_snap, a.pdd)
    peak_snap = sizing.steady_snapshot_at(wn, base["results"], base["peak_hour_s"])
    improvements, history = sizing.greedy_upsizing(
        fire_snap, peak_snap, deficient, a.fire_flow, a.service, a.sizing
    )
    pd.DataFrame(
        {
            "length_ft": ft(improvements["length_m"]),
            "existing_diameter_in": inch(improvements["existing_diameter_m"]),
            "proposed_diameter_in": inch(improvements["proposed_diameter_m"]),
            "planning_cost_usd": improvements["cost"],
        }
    ).round(1).to_csv(out / "4c_recommended_improvements.csv", index_label="pipe")
    history.assign(deficit_ft=ft(history["deficit_m"])).drop(columns="deficit_m").round(3).to_csv(
        out / "4c_sizing_history.csv", index=False
    )
    improved = sizing.apply_improvements(wn, improvements)
    after = fire(improved, a, out, tag="4c_after") if len(improvements) else fire_table
    _map(
        out / "4c_map_improvements.png",
        wn,
        "4c Recommended improvements (greedy heuristic, not a global optimum)",
        highlight_links=list(improvements.index),
        highlight_label="Upsized pipe",
        link_labels={
            p: (
                f"{p}: {inch(r['existing_diameter_m']):.0f} to "
                f"{inch(r['proposed_diameter_m']):.0f} in"
            )
            for p, r in improvements.iterrows()
        },
    )
    return {
        "fire_deficient_before": len(deficient),
        "fire_deficient_after": int((~after.loc[deficient, "meets_requirement"]).sum())
        if deficient
        else 0,
        "pipes_upsized": len(improvements),
        "total_cost_usd": round(float(improvements["cost"].sum()), 0),
        "final_deficit_ft": round(float(ft(history["deficit_m"].iloc[-1])), 3),
        "iterations": len(history) - 1,
    }


def age(wn: Any, a: Assumptions, out: Path) -> dict[str, Any]:
    """4d: water age and tank turnover, baseline vs alternative pump 10 schedule."""
    w = a.water_age
    alt_wn = energy.reschedule_timed_pump(
        wn, "10", w.alternative_pump10_start_h, energy.timed_run_hours(wn, "10")
    )
    results = {}
    for name, model in (("baseline", wn), ("alternative", alt_wn)):
        res = run_eps(model, w.duration_s, quality="AGE")
        jt = water_age.junction_age(res, model, w)
        tt = water_age.tank_turnover(res, model, 3 * 86400.0)
        results[name] = (jt, tt, water_age.age_stabilization(res, model, 86400.0))
    # Demand junctions only: zero-demand nodes (e.g. behind an idle pump) are not customers.
    service = fire_flow.eligible_junctions(wn)
    jb, ja = results["baseline"][0].loc[service], results["alternative"][0].loc[service]
    pd.DataFrame(
        {
            "max_age_h_baseline": jb["max_age_s"] / HOUR,
            "max_age_h_alternative": ja["max_age_s"] / HOUR,
            "change_h": (ja["max_age_s"] - jb["max_age_s"]) / HOUR,
            "above_threshold_baseline": jb["above_threshold"],
            "above_threshold_alternative": ja["above_threshold"],
        }
    ).round(1).to_csv(out / "4d_water_age_junctions.csv", index_label="junction")
    rows = []
    for name in ("baseline", "alternative"):
        tt = results[name][1]
        rows.append(
            pd.DataFrame(
                {
                    "scenario": name,
                    "mean_volume_kgal": kgal(tt["mean_volume_m3"]),
                    "turnover_days": tt["turnover_days"],
                    "level_range_ft": ft(tt["level_range_m"]),
                    "max_age_h": tt["max_age_s"] / HOUR,
                }
            )
        )
    pd.concat(rows).round(2).to_csv(out / "4d_tank_turnover.csv", index_label="tank")
    _map(
        out / "4d_map_water_age.png",
        wn,
        f"4d Maximum water age, final {w.report_window_s / HOUR:.0f} h (baseline)",
        node_values=jb["max_age_s"] / HOUR,
        label="Maximum water age (h)",
    )
    return {
        "junctions_above_threshold_baseline": int(jb["above_threshold"].sum()),
        "junctions_above_threshold_alternative": int(ja["above_threshold"].sum()),
        "max_age_h_baseline": round(float(jb["max_age_s"].max() / HOUR), 1),
        "max_age_h_alternative": round(float(ja["max_age_s"].max() / HOUR), 1),
        "junctions_older_by_1h_or_more": int(((ja["max_age_s"] - jb["max_age_s"]) >= HOUR).sum()),
        "junctions_younger_by_1h_or_more": int(((jb["max_age_s"] - ja["max_age_s"]) >= HOUR).sum()),
        "tank_max_age_h": {
            name: {t: round(float(v) / HOUR, 1) for t, v in results[name][1]["max_age_s"].items()}
            for name in ("baseline", "alternative")
        },
        "age_stabilization": {
            name: round(results[name][2], 4) for name in ("baseline", "alternative")
        },
    }


def pump_energy(wn: Any, a: Assumptions, out: Path, base: dict[str, Any]) -> dict[str, Any]:
    """4e: energy and cost, baseline vs alternative pump 10 schedule, with feasibility."""
    run_h = energy.timed_run_hours(wn, "10")
    alt_wn = energy.reschedule_timed_pump(wn, "10", a.energy.alternative_pump10_start_h, run_h)
    alt_res = run_eps(alt_wn, a.eps_duration_s)
    sums, steps = {}, {}
    for name, model, res in (("baseline", wn, base["results"]), ("alternative", alt_wn, alt_res)):
        steps[name] = energy.pump_energy(res, model, a.energy)
        sums[name] = energy.energy_summary(steps[name], a.energy)
    pd.concat(sums, names=["scenario", "pump"]).round(3).to_csv(out / "4e_energy_summary.csv")

    feas = {}
    for name, model, res in (("baseline", wn, base["results"]), ("alternative", alt_wn, alt_res)):
        j = criteria.junction_compliance(res, model, a.service)
        lv = criteria.tank_levels(res, model)
        mins = {t: model.get_node(t).min_level for t in lv.columns}
        feas[name] = {
            "junctions_below_min": int((j[j["serves_demand"]]["steps_below_min"] > 0).sum()),
            "tank_hours_at_min_level": int(
                sum((lv[t] <= mins[t] + 0.01).sum() for t in lv.columns)
            ),
        }
    feasible = (
        feas["alternative"]["junctions_below_min"] <= feas["baseline"]["junctions_below_min"]
        and feas["alternative"]["tank_hours_at_min_level"]
        <= feas["baseline"]["tank_hours_at_min_level"]
    )

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for color, name in zip(plotting.CATEGORICAL, ("baseline", "alternative"), strict=False):
        s = steps[name].groupby("time_s")["power_kw"].sum()
        ax.step(s.index / HOUR, s.values, where="post", color=color, label=name.capitalize())
    days = int(np.ceil(a.eps_duration_s / 86400.0))
    for d in range(days):
        ax.axvspan(
            24 * d + a.energy.on_peak_start_h,
            24 * d + a.energy.on_peak_end_h,
            color=plotting.GRID,
            alpha=0.6,
            lw=0,
            label="On-peak hours" if d == 0 else None,
        )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Total pump input power (kW)")
    ax.set_title("4e Pump power, baseline vs alternative pump 10 schedule", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, out / "4e_pump_power.png")

    cost_b = float(sums["baseline"]["cost"].sum())
    cost_a = float(sums["alternative"]["cost"].sum())
    return {
        "baseline_cost_usd": round(cost_b, 2),
        "alternative_cost_usd": round(cost_a, 2),
        "saving_pct": round(100 * (cost_b - cost_a) / cost_b, 1) if cost_b else 0.0,
        "baseline_energy_kwh": round(float(sums["baseline"]["energy_kwh"].sum()), 0),
        "alternative_energy_kwh": round(float(sums["alternative"]["energy_kwh"].sum()), 0),
        "feasibility": feas,
        "alternative_feasible": feasible,
        "period_h": a.eps_duration_s / HOUR,
    }


def critical_pipes(wn: Any, a: Assumptions, out: Path) -> dict[str, Any]:
    """4f: rank large pipes by demand not served when closed (PDA)."""
    pda = copy.deepcopy(wn)
    set_pressure_dependent(pda, a.pdd)
    table, excluded = criticality.criticality_table(pda, a.criticality, a.service)
    pd.DataFrame(
        {
            "diameter_in": inch(table["diameter_m"]),
            "length_ft": ft(table["length_m"]),
            "unserved_kgal": kgal(table["unserved_m3"]),
            "unserved_pct": table["unserved_pct"],
            "junctions_newly_below_min": table["junctions_below_min"],
        }
    ).round(2).to_csv(out / "4f_criticality.csv", index_label="pipe")
    top = list(table.index[:10])
    # Rank numbers on the map (short labels stay legible in dense areas); details in a table.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    plotting.network_map(
        ax,
        wn,
        highlight_links=top,
        highlight_label="Ten most critical (number = rank)",
        link_labels={k: str(i + 1) for i, k in enumerate(top)},
    )
    rows = ["Rank  Pipe  Dia (in)  Not served"]
    for i, k in enumerate(top):
        rows.append(
            f"{i + 1:>4}  {k:>4}  {inch(table.loc[k, 'diameter_m']):>8.0f}  "
            f"{table.loc[k, 'unserved_pct']:>9.1f}%"
        )
    ax.text(
        0.01,
        0.07,
        "\n".join(rows),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        family="monospace",
        color=plotting.INK_SECONDARY,
        zorder=7,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": plotting.SURFACE,
            "edgecolor": plotting.GRID,
        },
    )
    ax.set_title(
        "4f Ten most critical pipes: % of 24-h demand not served when closed (PDA)",
        fontsize=10,
        loc="left",
    )
    _save(fig, out / "4f_map_criticality.png")
    return {"pipes_screened": len(table), "excluded_control_targets": excluded, "top10": top}


# --- Output helpers ---------------------------------------------------------------------------
def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _map(path: Path, wn: Any, title: str, **kwargs: Any) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    plotting.network_map(ax, wn, **kwargs)
    ax.set_title(title, fontsize=10, loc="left")
    _save(fig, path)


def _planned_runs(wn: Any, a: Assumptions) -> list[str]:
    n_fire = len(fire_flow.eligible_junctions(wn))
    bisect = int(np.ceil(np.log2(a.fire_flow.upper_bound_m3s / a.fire_flow.tolerance_m3s))) + 2
    n_crit = sum(1 for _, p in wn.pipes() if p.diameter >= a.criticality.min_diameter_m)
    return [
        f"4a baseline EPS: 1 run, {a.eps_duration_s / HOUR:.0f} h",
        f"4b fire flow: {n_fire} junctions x up to {bisect} steady solves",
        f"4c sizing: up to {a.sizing.max_iterations} iterations x candidates x deficient junctions",
        f"4d water age: 2 AGE runs, {a.water_age.duration_s / HOUR:.0f} h each",
        f"4e energy: 1 extra EPS, {a.eps_duration_s / HOUR:.0f} h",
        f"4f criticality: up to {n_crit} pipe closures x "
        f"{a.criticality.duration_s / HOUR:.0f} h PDA",
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the study (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--assumptions", type=Path, default=DEFAULT_ASSUMPTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="list planned runs only")
    args = parser.parse_args(argv)

    a = load_assumptions(args.assumptions)
    wn = load_network(net3_path())
    drafts = a.drafts()
    print(
        f"Network: {net3_path().name} ({len(wn.junction_name_list)} junctions, "
        f"{len(wn.pipe_name_list)} pipes)"
    )
    print(f"Assumptions: {args.assumptions} ({len(a.entries)} entries, {len(drafts)} draft)")
    if args.dry_run:
        for line in _planned_runs(wn, a):
            print("  " + line)
        if drafts:
            print("Draft entries awaiting review:")
            for d in drafts:
                e = a.entries[d]
                print(f"  {d} = {e.value} {e.units}  [{e.basis}]")
        return 0
    if drafts:
        print(
            "Refusing to run: these assumptions are still drafts (review them first):",
            file=sys.stderr,
        )
        for d in drafts:
            print(f"  {d}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    plotting.apply_style()
    timings: dict[str, float] = {}
    summary: dict[str, Any] = {}

    def step(key: str, fn: Any, *fargs: Any) -> Any:
        t0 = time.perf_counter()
        value = fn(*fargs)
        timings[key] = round(time.perf_counter() - t0, 1)
        print(f"  {key}: done in {timings[key]} s")
        return value

    base = step("4a_baseline", baseline, wn, a, args.out)
    summary["4a_baseline"] = base["summary"]
    summary["4a_baseline"]["peak_hour_h"] = base["peak_hour_s"] / HOUR
    fire_table = step("4b_fire_flow", fire, wn, a, args.out)
    summary["4b_fire_flow"] = {
        "junctions_tested": len(fire_table),
        "junctions_meeting_requirement": int(fire_table["meets_requirement"].sum()),
    }
    summary["4c_sizing"] = step("4c_sizing", size_pipes, wn, a, args.out, fire_table, base)
    summary["4d_water_age"] = step("4d_water_age", age, wn, a, args.out)
    summary["4e_energy"] = step("4e_energy", pump_energy, wn, a, args.out, base)
    summary["4f_criticality"] = step("4f_criticality", critical_pipes, wn, a, args.out)
    summary["runtime_s"] = timings
    summary["assumptions"] = {
        k: {"value": e.value, "units": e.units, "status": e.status} for k, e in a.entries.items()
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
