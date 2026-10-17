"""Phase 1 figures: Hazen-Williams vs Darcy-Weisbach.

Inputs: water at 20 degC (viscosity from the sourced property table), absolute roughness
0.1 mm (cement-lined ductile iron, project author), HW calibrated to DW at D = 300 mm,
V = 1 m/s. Writes ``reports/phase1/hw_dw_headloss_ratio.png`` and
``reports/phase1/equivalent_c_factor.png``.

Usage: ``uv run python scripts/phase1_headloss_figures.py``
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wdtk import fluid, plotting
from wdtk import headloss as hl
from wdtk.headloss_comparison import equivalent_c, ratio_grid

ROOT = Path(__file__).resolve().parents[1]
TEMPERATURE_C = 20.0
EPS_M = 0.1e-3
CALIBRATION = (0.300, 1.0)  # D (m), V (m/s)


def main(out_dir: Path | None = None) -> list[Path]:
    """Write the Phase 1 figures and return their paths."""
    out = out_dir or ROOT / "reports" / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    plotting.apply_style()
    nu = fluid.kinematic_viscosity_m2s(TEMPERATURE_C)

    grid = ratio_grid(
        np.geomspace(0.05, 1.2, 90), np.linspace(0.3, 3.0, 90), EPS_M, nu, CALIBRATION
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    plotting.plot_hw_dw_ratio(
        ax,
        grid,
        CALIBRATION,
        f"C = {grid.c_calibrated:.0f} calibrated at the marked point; "
        f"ε = {EPS_M * 1000:g} mm, {TEMPERATURE_C:g} °C water",
    )
    fig.tight_layout()
    ratio_path = out / "hw_dw_headloss_ratio.png"
    fig.savefig(ratio_path)
    plt.close(fig)

    v = np.linspace(0.3, 3.0, 60)
    by_d = {f"D = {d * 1000:.0f} mm": equivalent_c(v, d, EPS_M, nu) for d in (0.1, 0.3, 0.9)}
    by_eps = {
        label: equivalent_c(v, CALIBRATION[0], e, nu)
        for label, e in (("ε = 0 (smooth)", 0.0), ("ε = 0.1 mm", 1e-4), ("ε = 1 mm", 1e-3))
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    plotting.plot_equivalent_c(
        axes,
        v,
        [
            (f"(a) Varying diameter, ε = {EPS_M * 1000:g} mm", by_d),
            (f"(b) Varying roughness, D = {CALIBRATION[0] * 1000:.0f} mm", by_eps),
        ],
    )
    fig.suptitle(
        f"C that makes Hazen-Williams reproduce Darcy-Weisbach, {TEMPERATURE_C:g} °C water",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    c_path = out / "equivalent_c_factor.png"
    fig.savefig(c_path)
    plt.close(fig)
    # Swamee-Jain vs Colebrook over the range used in the Phase 1 acceptance test.
    re, rr = np.meshgrid(np.logspace(np.log10(5e3), 8, 120), np.logspace(-6, -2, 120))
    f_cw = hl.friction_factor_colebrook(re, rr)
    err = np.abs(hl.friction_factor_swamee_jain(re, rr) - f_cw) / f_cw
    i = np.unravel_index(np.argmax(err), err.shape)
    sj_path = out / "swamee_jain_accuracy.csv"
    sj_path.write_text(
        "max_rel_error_pct,at_reynolds,at_rel_roughness,fraction_of_grid_over_1pct\n"
        f"{100 * err.max():.3f},{re[i]:.4g},{rr[i]:.4g},{(err > 0.01).mean():.4f}\n",
        encoding="utf-8",
    )
    return [ratio_path, c_path, sj_path]


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
