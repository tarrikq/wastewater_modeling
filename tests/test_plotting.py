"""Smoke tests for wdtk.plotting (renders off-screen with the Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from wdtk import plotting
from wdtk import pumps as pm


def test_plot_pump_system_returns_operating_points() -> None:
    plotting.apply_style()
    curve = pm.fit_three_point(60.0, 0.10, 45.0, 0.22)

    def system(q: float) -> float:
        return pm.system_head_hw_m(q, 20.0, 2000.0, 0.3, 120.0)

    fig, ax = plt.subplots()
    points = plotting.plot_pump_system(ax, curve, system, speeds=(1.0, 0.8))
    plt.close(fig)
    for (q, h), s in zip(points, (1.0, 0.8), strict=True):
        c_s = pm.scale_curve_speed(curve, s)
        assert h == pytest.approx(pm.pump_head_m(c_s, q), rel=1e-12)
        assert h == pytest.approx(system(q), rel=1e-9)


@pytest.mark.parametrize("n", [0, 6])
def test_ordinal_colors_rejects_out_of_range(n: int) -> None:
    with pytest.raises(ValueError):
        plotting.ordinal_colors(n)


def test_ordinal_colors_spans_ramp() -> None:
    colors = plotting.ordinal_colors(3)
    assert colors[0] == plotting.BLUE_RAMP[0] and colors[-1] == plotting.BLUE_RAMP[-1]
