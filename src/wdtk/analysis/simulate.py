"""Thin helpers around WNTR and the EPANET 2.2 toolkit for the Phase 4 analyses.

Two ways of running EPANET are used:

* :func:`run_eps` runs a whole extended-period simulation through WNTR's
  ``EpanetSimulator`` (hydraulics and, optionally, water quality). Used for the few long runs.
* :class:`EpanetSession` keeps one EPANET project open and changes demands, diameters, or link
  status in place between hydraulic solves. Used for the thousands of runs in fire flow
  bisection, greedy sizing, and criticality, where writing and re-reading an input file each
  time would dominate the run time.

Both run the same EPANET 2.2 engine. All values crossing this module's boundary are SI
(m, m^3/s, m of pressure head); the session writes its input file in LPS units so that
EPANET's own units are m, L/s, and mm.
"""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wntr
from numpy.typing import NDArray
from wntr.epanet.toolkit import ENepanet
from wntr.epanet.util import EN

from wdtk.analysis.assumptions import PddSettings

WaterNetworkModel = Any  # wntr.network.WaterNetworkModel (WNTR ships without type hints)

LPS_PER_M3S = 1000.0
MM_PER_M = 1000.0


def net3_path() -> Path:
    """Path to the Net3.inp bundled with WNTR (``wntr/library/networks/Net3.inp``)."""
    return Path(wntr.__file__).parent / "library" / "networks" / "Net3.inp"


def load_network(path: str | Path | None = None) -> WaterNetworkModel:
    """Load an EPANET input file with WNTR (default: Net3).

    Parameters
    ----------
    path : str or Path, optional
        Input file; defaults to :func:`net3_path`.

    Returns
    -------
    wntr.network.WaterNetworkModel
        Model in WNTR's SI units.
    """
    return wntr.network.WaterNetworkModel(str(path or net3_path()))


def set_pressure_dependent(wn: WaterNetworkModel, pdd: PddSettings) -> None:
    """Switch a model to EPANET 2.2 pressure-dependent demand (PDA), in place.

    Delivered demand follows Wagner's relation between ``minimum_pressure_m`` (no delivery)
    and ``required_pressure_m`` (full delivery), exponent 0.5 (EPANET default).

    Parameters
    ----------
    wn : WaterNetworkModel
        Model to modify.
    pdd : PddSettings
        Pressure heads, m.
    """
    wn.options.hydraulic.demand_model = "PDA"
    wn.options.hydraulic.minimum_pressure = pdd.minimum_pressure_m
    wn.options.hydraulic.required_pressure = pdd.required_pressure_m


def steady_max_day(wn: WaterNetworkModel, max_day_factor: float) -> WaterNetworkModel:
    """Return a copy of a model as a max-day steady snapshot.

    Base demands are multiplied by ``max_day_factor`` and demand patterns removed, so each
    junction draws its average-day demand times the factor; tanks stay at their initial
    levels and links at their t = 0 status. Duration is set to zero.

    Parameters
    ----------
    wn : WaterNetworkModel
        Source model (not modified).
    max_day_factor : float
        Max-day to average-day demand ratio.

    Returns
    -------
    WaterNetworkModel
        Modified copy.
    """
    out = copy.deepcopy(wn)
    for _, junction in out.junctions():
        for ts in junction.demand_timeseries_list:
            ts.base_value = ts.base_value * max_day_factor
            ts.pattern_name = None
    out.options.time.duration = 0
    return out


def run_eps(
    wn: WaterNetworkModel,
    duration_s: float,
    quality: str | None = None,
    workdir: str | Path | None = None,
) -> Any:
    """Run an extended-period simulation of a copy of ``wn`` with WNTR's EpanetSimulator.

    Parameters
    ----------
    wn : WaterNetworkModel
        Model (not modified).
    duration_s : float
        Simulation length, s.
    quality : str, optional
        Water quality parameter, e.g. ``"AGE"`` (default: none).
    workdir : str or Path, optional
        Directory for EPANET's scratch files (default: a temporary directory).

    Returns
    -------
    wntr.sim.results.SimulationResults
        Results in SI units (heads and pressures in m, flows in m^3/s, age in s).
    """
    model = copy.deepcopy(wn)
    model.options.time.duration = int(duration_s)
    if quality is not None:
        model.options.quality.parameter = quality
    with _workdir(workdir) as d:
        sim = wntr.sim.EpanetSimulator(model)
        return sim.run_sim(file_prefix=str(Path(d) / "eps"))


@contextmanager
def _workdir(workdir: str | Path | None) -> Iterator[str]:
    if workdir is not None:
        yield str(workdir)
        return
    with tempfile.TemporaryDirectory() as d:
        yield d


class EpanetSession:
    """An open EPANET 2.2 project for many fast re-solves of one network.

    Use as a context manager. Changes are made with the setter methods and persist until
    changed back (or the session is closed). Values are SI at the interface.

    Parameters
    ----------
    wn : WaterNetworkModel
        Model to load (copied; written to a temporary LPS-units input file).
    """

    def __init__(self, wn: WaterNetworkModel) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        inp = str(Path(self._tmp.name) / "session.inp")
        wntr.network.write_inpfile(wn, inp, units="LPS", version=2.2)
        self._en = ENepanet(version=2.2)
        self._en.ENopen(inp, str(Path(self._tmp.name) / "session.rpt"), "")
        self.junction_names: list[str] = list(wn.junction_name_list)
        self.link_names: list[str] = list(wn.link_name_list)
        self._node_idx = {n: self._en.ENgetnodeindex(n) for n in wn.node_name_list}
        self._link_idx = {n: self._en.ENgetlinkindex(n) for n in self.link_names}
        self._j_idx = [self._node_idx[n] for n in self.junction_names]

    def __enter__(self) -> EpanetSession:
        """Enter the context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close EPANET and remove temporary files."""
        self.close()

    def close(self) -> None:
        """Close the EPANET project and remove the temporary directory."""
        if self._en is not None:
            self._en.ENclose()
            self._en = None
        self._tmp.cleanup()

    # --- Setters ---------------------------------------------------------------------------
    def base_demand_m3s(self, junction: str) -> float:
        """Return the base demand (first demand category) of a junction, m^3/s."""
        return float(self._en.ENgetnodevalue(self._node_idx[junction], EN.BASEDEMAND)) / (
            LPS_PER_M3S
        )

    def set_base_demand(self, junction: str, demand_m3s: float) -> None:
        """Set the base demand (first demand category) of a junction, m^3/s."""
        self._en.ENsetnodevalue(self._node_idx[junction], EN.BASEDEMAND, demand_m3s * LPS_PER_M3S)

    def diameter_m(self, link: str) -> float:
        """Return the diameter of a pipe, m."""
        return float(self._en.ENgetlinkvalue(self._link_idx[link], EN.DIAMETER)) / MM_PER_M

    def set_diameter(self, link: str, diameter_m: float) -> None:
        """Set the diameter of a pipe, m."""
        self._en.ENsetlinkvalue(self._link_idx[link], EN.DIAMETER, diameter_m * MM_PER_M)

    def set_open(self, link: str, is_open: bool) -> None:
        """Set a link's initial status open (True) or closed (False)."""
        self._en.ENsetlinkvalue(self._link_idx[link], EN.INITSTATUS, 1.0 if is_open else 0.0)

    # --- Solving ---------------------------------------------------------------------------
    def _junction_state(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        pressure = np.array([self._en.ENgetnodevalue(i, EN.PRESSURE) for i in self._j_idx])
        demand = np.array([self._en.ENgetnodevalue(i, EN.DEMAND) for i in self._j_idx])
        return pressure, demand / LPS_PER_M3S

    def solve_steady(self) -> pd.DataFrame:
        """Solve hydraulics for a single period (the model's duration should be zero).

        Returns
        -------
        DataFrame
            Index: junction names. Columns: ``pressure_m`` (pressure head) and
            ``demand_m3s`` (delivered demand; less than requested under PDA when pressure is
            below the required pressure).
        """
        self._en.ENsolveH()
        pressure, demand = self._junction_state()
        return pd.DataFrame(
            {"pressure_m": pressure, "demand_m3s": demand}, index=self.junction_names
        )

    def run_eps(
        self,
        duration_s: float,
        on_step: Callable[[float, NDArray[np.float64], NDArray[np.float64]], None],
    ) -> None:
        """Run an extended-period hydraulic simulation, calling ``on_step`` at each step.

        Parameters
        ----------
        duration_s : float
            Simulation length, s.
        on_step : callable
            ``on_step(time_s, pressure_m, demand_m3s)`` with junction arrays in
            ``junction_names`` order; called at every hydraulic time step (including
            intermediate steps triggered by controls or tank events).
        """
        self._en.ENsettimeparam(EN.DURATION, int(duration_s))
        self._en.ENopenH()
        self._en.ENinitH(0)
        try:
            while True:
                t = float(self._en.ENrunH())
                pressure, demand = self._junction_state()
                on_step(t, pressure, demand)
                if self._en.ENnextH() <= 0:
                    break
        finally:
            self._en.ENcloseH()
