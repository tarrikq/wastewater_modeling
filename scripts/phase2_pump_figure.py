"""Phase 2 figure: pump curve family at variable speed, system curve, operating points.

All inputs are illustrative (not a specific pump or site); they are stated on the figure.

Usage: ``uv run python scripts/phase2_pump_figure.py``
"""

from pathlib import Path

import matplotlib.pyplot as plt

from wdtk import plotting, validation
from wdtk import pumps as pm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase2" / "pump_system_vfd.png"

# Illustrative inputs
SHUTOFF_HEAD_M, DESIGN_FLOW_M3S, DESIGN_HEAD_M, MAX_FLOW_M3S = 60.0, 0.10, 45.0, 0.22
STATIC_HEAD_M, LENGTH_M, DIAMETER_M, C_FACTOR, MINOR_K = 20.0, 2000.0, 0.3, 120.0, 5.0


def main(out_dir: Path | None = None) -> list[Path]:
    """Build and save the figure; return the written path(s)."""
    out = (out_dir or OUT.parent) / OUT.name
    plotting.apply_style()
    curve = pm.fit_three_point(SHUTOFF_HEAD_M, DESIGN_FLOW_M3S, DESIGN_HEAD_M, MAX_FLOW_M3S)

    def system(q: float) -> float:
        return pm.system_head_hw_m(q, STATIC_HEAD_M, LENGTH_M, DIAMETER_M, C_FACTOR, MINOR_K)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    points = plotting.plot_pump_system(ax, curve, system)
    ax.set_title(
        "Pump and system curves with variable-speed operating points (illustrative)\n"
        f"Pump: {SHUTOFF_HEAD_M:g} m shutoff, {DESIGN_FLOW_M3S * 1000:g} L/s at "
        f"{DESIGN_HEAD_M:g} m. System: {STATIC_HEAD_M:g} m static, {LENGTH_M:g} m of "
        f"{DIAMETER_M * 1000:g} mm pipe, C = {C_FACTOR:g}, K = {MINOR_K:g}",
        fontsize=9,
        loc="left",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    for q, h in points:
        print(f"Q = {q * 1000:7.2f} L/s, H = {h:6.2f} m")
    print(f"Wrote {out}")
    table = out.parent / "pump_validation_epanet.csv"
    validation.pump_validation_table().round(4).to_csv(table, index=False)
    print(f"Wrote {table}")
    return [out, table]


if __name__ == "__main__":
    main()
