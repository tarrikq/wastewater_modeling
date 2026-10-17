"""Phase 3 validation: wdtk GGA and Hardy Cross vs EPANET 2.2 (through WNTR).

Writes to ``reports/phase3/``:

* ``validation_heads.csv`` and ``validation_flows.csv``: per junction / link, wdtk GGA vs
  EPANET, with differences and pass/fail against the acceptance tolerances.
* ``validation_summary.csv``: worst-case differences and iteration counts per network.
* ``convergence.png``: Hardy Cross vs GGA convergence histories on the looped networks.

Usage: ``uv run python scripts/phase3_validation.py`` (after ``scripts/build_test_networks.py``)
"""

import csv
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import wntr

from wdtk import plotting
from wdtk.network.gga import solve_gga
from wdtk.network.hardy_cross import solve_hardy_cross
from wdtk.network.io import read_inp

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ROOT / "data" / "networks"
OUT = ROOT / "reports" / "phase3"
NAMES = ["single_loop_hw", "two_loop_hw", "net1_steady_hw", "single_loop_dw", "two_loop_dw"]
LOOPED = ["single_loop_hw", "two_loop_hw", "single_loop_dw", "two_loop_dw"]
HEAD_TOL_M, FLOW_REL_TOL, FLOW_ABS_TOL_M3S = 0.01, 1e-3, 1e-6


def epanet(inp: Path, workdir: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Run EPANET (accuracy 1e-6, demand-driven) and return heads (m) and flows (m^3/s)."""
    wn = wntr.network.WaterNetworkModel(str(inp))
    wn.options.hydraulic.accuracy = 1e-6
    wn.options.hydraulic.demand_model = "DD"
    wn.options.time.duration = 0
    res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(workdir / inp.stem))
    return res.node["head"].loc[0].to_dict(), res.link["flowrate"].loc[0].to_dict()


def main() -> None:
    """Run the comparisons and write tables and the figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    head_rows, flow_rows, summary = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name in NAMES:
            inp = NETWORKS / f"{name}.inp"
            ours = solve_gga(read_inp(inp))
            heads, flows = epanet(inp, Path(tmp))
            max_dh = max_dq = max_dq_rel = 0.0
            for j in ours.junction_ids:
                dh = ours.head(j) - heads[j]
                max_dh = max(max_dh, abs(dh))
                head_rows.append(
                    [
                        name,
                        j,
                        f"{ours.head(j):.6f}",
                        f"{heads[j]:.6f}",
                        f"{dh:+.2e}",
                        abs(dh) <= HEAD_TOL_M,
                    ]
                )
            for lk in ours.link_ids:
                dq = ours.flow(lk) - flows[lk]
                ok = abs(dq) <= max(FLOW_REL_TOL * abs(flows[lk]), FLOW_ABS_TOL_M3S)
                rel = abs(dq) / abs(flows[lk]) if flows[lk] else 0.0
                max_dq, max_dq_rel = max(max_dq, abs(dq)), max(max_dq_rel, rel)
                flow_rows.append(
                    [
                        name,
                        lk,
                        f"{ours.flow(lk) * 1000:.6f}",
                        f"{flows[lk] * 1000:.6f}",
                        f"{dq * 1000:+.2e}",
                        f"{rel:.2e}",
                        ok,
                    ]
                )
            hc_iters = solve_hardy_cross(read_inp(inp)).iterations if name in LOOPED else ""
            summary.append(
                [
                    name,
                    f"{max_dh:.2e}",
                    f"{max_dq * 1000:.2e}",
                    f"{max_dq_rel:.2e}",
                    ours.iterations,
                    hc_iters,
                ]
            )

    def write(fname: str, header: list[str], rows: list[list[object]]) -> None:
        with (OUT / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write(
        "validation_heads.csv",
        ["network", "junction", "head_wdtk_m", "head_epanet_m", "diff_m", "pass"],
        head_rows,
    )
    write(
        "validation_flows.csv",
        ["network", "link", "flow_wdtk_Ls", "flow_epanet_Ls", "diff_Ls", "rel_diff", "pass"],
        flow_rows,
    )
    write(
        "validation_summary.csv",
        [
            "network",
            "max_abs_head_diff_m",
            "max_abs_flow_diff_Ls",
            "max_rel_flow_diff",
            "gga_iterations",
            "hardy_cross_sweeps",
        ],
        summary,
    )

    print(
        f"{'network':16s} {'max|dH| m':>10s} {'max|dQ| L/s':>12s} {'max rel dQ':>11s} "
        f"{'GGA it':>7s} {'HC sweeps':>9s}"
    )
    for row in summary:
        print(
            f"{row[0]:16s} {row[1]:>10s} {row[2]:>12s} {row[3]:>11s} {row[4]!s:>7s} {row[5]!s:>9s}"
        )
    n_fail = sum(1 for r in head_rows + flow_rows if not r[-1])
    print(f"Elements outside tolerance: {n_fail} of {len(head_rows) + len(flow_rows)}")

    plotting.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharey=True)
    titles = {
        "single_loop_hw": "Single loop, Hazen-Williams",
        "two_loop_hw": "Two loop, Hazen-Williams",
        "single_loop_dw": "Single loop, Darcy-Weisbach",
        "two_loop_dw": "Two loop, Darcy-Weisbach",
    }
    for ax, name in zip(axes.flat, LOOPED, strict=True):
        net = read_inp(NETWORKS / f"{name}.inp")
        results = [solve_gga(net, accuracy=1e-12), solve_hardy_cross(net, accuracy=1e-12)]
        plotting.plot_convergence(ax, results, titles[name])
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    fig.suptitle(
        "Convergence of GGA (Newton, all loops at once) vs Hardy Cross (loop by loop)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png")
    print(f"Wrote tables and figure to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
