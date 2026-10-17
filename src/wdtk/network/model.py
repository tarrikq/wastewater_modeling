"""Minimal data model for a steady-state pressurized pipe network.

Units are SI throughout: m, m^3/s, m of head. Elements are plain frozen dataclasses (data
records); solvers live in :mod:`wdtk.network.gga` and :mod:`wdtk.network.hardy_cross`.

Topology convention (used by both solvers). Links are ordered pipes first, then pumps. Each
link k runs from ``start`` to ``end``; positive flow is start -> end. The energy equation for
link k is

    h_k(Q_k) = H_start - H_end,

and continuity at junction i is (inflow - outflow) = demand_i. In matrix form, with
A12[k, i] = -1 if junction i is the start of link k and +1 if it is the end (A10 likewise for
reservoirs):

    h(Q) + A12 H + A10 H0 = 0,     A21 Q = q,   A21 = A12^T.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from wdtk.pumps import PumpCurve, fit_single_point, fit_three_point

HeadlossFormula = Literal["H-W", "D-W"]


@dataclass(frozen=True)
class Junction:
    """Demand node with unknown head.

    Attributes
    ----------
    id : str
        Unique node ID.
    elevation_m : float
        Elevation, m (pressure head = H - elevation).
    demand_m3s : float
        Demand withdrawn at the node, m^3/s (negative for an inflow).
    """

    id: str
    elevation_m: float
    demand_m3s: float = 0.0


@dataclass(frozen=True)
class Reservoir:
    """Fixed-head node (infinite source or sink).

    Attributes
    ----------
    id : str
        Unique node ID.
    head_m : float
        Total head (hydraulic grade), m.
    """

    id: str
    head_m: float


@dataclass(frozen=True)
class Pipe:
    """Pipe link.

    Attributes
    ----------
    id : str
        Unique link ID.
    start, end : str
        Node IDs; positive flow is start -> end.
    length_m, diameter_m : float
        Length and internal diameter, m.
    roughness : float
        Hazen-Williams C (dimensionless) for H-W networks; absolute roughness in m for D-W.
    minor_k : float
        Minor loss coefficient referenced to the pipe velocity.
    closed : bool
        True if the pipe is closed (carries no flow and is excluded from the solve).
    """

    id: str
    start: str
    end: str
    length_m: float
    diameter_m: float
    roughness: float
    minor_k: float = 0.0
    closed: bool = False


@dataclass(frozen=True)
class Pump:
    """Pump link with a head curve given by EPANET-style curve points.

    Attributes
    ----------
    id : str
        Unique link ID.
    start, end : str
        Suction and discharge node IDs.
    curve_points : tuple of (flow m^3/s, head m)
        One design point, or three points with the first at zero flow.
    """

    id: str
    start: str
    end: str
    curve_points: tuple[tuple[float, float], ...]

    @property
    def curve(self) -> PumpCurve:
        """Pump curve H = a - b Q^c fitted with EPANET's conventions (see wdtk.pumps)."""
        pts = self.curve_points
        if len(pts) == 1:
            return fit_single_point(*pts[0])
        if len(pts) == 3 and pts[0][0] == 0.0:
            (_, h1), (q2, h2), (q3, h3) = pts
            return fit_three_point(h1, q2, h2, q3, h3)
        raise ValueError(
            f"Pump {self.id}: supported curves are one point, or three points starting at Q = 0."
        )


@dataclass(frozen=True)
class Network:
    """A steady-state network: junctions, reservoirs, pipes, and pumps.

    Attributes
    ----------
    junctions, reservoirs, pipes, pumps : tuple
        Network elements.
    headloss : {"H-W", "D-W"}
        Friction formula for all pipes.
    coordinates : mapping of str to (x, y)
        Optional node coordinates for plotting (any consistent units).
    title : str
        Optional description.
    """

    junctions: tuple[Junction, ...]
    reservoirs: tuple[Reservoir, ...]
    pipes: tuple[Pipe, ...]
    pumps: tuple[Pump, ...] = ()
    headloss: HeadlossFormula = "H-W"
    coordinates: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    title: str = ""

    def __post_init__(self) -> None:
        """Validate topology and IDs."""
        node_ids = [n.id for n in self.junctions] + [n.id for n in self.reservoirs]
        link_ids = [lk.id for lk in self.pipes] + [lk.id for lk in self.pumps]
        for kind, ids in (("node", node_ids), ("link", link_ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                raise ValueError(f"Duplicate {kind} IDs: {dupes}.")
        if not self.reservoirs:
            raise ValueError("Network needs at least one reservoir (fixed-head node).")
        if self.headloss not in ("H-W", "D-W"):
            raise ValueError(f"Unknown headloss formula {self.headloss!r}.")
        nodes = set(node_ids)
        all_links: list[Pipe | Pump] = [*self.pipes, *self.pumps]
        for lk in all_links:
            for end in (lk.start, lk.end):
                if end not in nodes:
                    raise ValueError(f"Link {lk.id} references unknown node {end!r}.")
            if lk.start == lk.end:
                raise ValueError(f"Link {lk.id} starts and ends at the same node.")
        graph = self.graph()
        if not nx.is_connected(graph):
            raise ValueError("Network (open links only) is not connected.")

    # --- Derived structure (cached; the dataclass is frozen) ----------------------------------
    @functools.cached_property
    def links(self) -> tuple[Pipe | Pump, ...]:
        """Open links in solver order: open pipes, then pumps."""
        open_pipes: list[Pipe | Pump] = [p for p in self.pipes if not p.closed]
        return (*open_pipes, *self.pumps)

    @functools.cached_property
    def junction_index(self) -> dict[str, int]:
        """Junction ID -> row/column index."""
        return {j.id: i for i, j in enumerate(self.junctions)}

    @functools.cached_property
    def reservoir_index(self) -> dict[str, int]:
        """Reservoir ID -> column index in A10."""
        return {r.id: i for i, r in enumerate(self.reservoirs)}

    @property
    def n_pipes(self) -> int:
        """Number of open pipes (the first ``n_pipes`` links)."""
        return sum(1 for p in self.pipes if not p.closed)

    def graph(self) -> nx.MultiGraph:
        """Undirected multigraph of nodes and open links (link ID stored as edge key)."""
        g = nx.MultiGraph()
        g.add_nodes_from([j.id for j in self.junctions] + [r.id for r in self.reservoirs])
        for lk in self.links:
            g.add_edge(lk.start, lk.end, key=lk.id)
        return g

    @functools.cached_property
    def incidence(self) -> tuple[sp.csr_array, sp.csr_array]:
        """(A12, A10): link-junction and link-reservoir incidence, entries -1 (start), +1 (end)."""
        n_l = len(self.links)
        rows_j: list[int] = []
        cols_j: list[int] = []
        vals_j: list[float] = []
        rows_r: list[int] = []
        cols_r: list[int] = []
        vals_r: list[float] = []
        for k, lk in enumerate(self.links):
            for node, sign in ((lk.start, -1.0), (lk.end, 1.0)):
                if node in self.junction_index:
                    rows_j.append(k)
                    cols_j.append(self.junction_index[node])
                    vals_j.append(sign)
                else:
                    rows_r.append(k)
                    cols_r.append(self.reservoir_index[node])
                    vals_r.append(sign)
        a12 = sp.csr_array((vals_j, (rows_j, cols_j)), shape=(n_l, len(self.junctions)))
        a10 = sp.csr_array((vals_r, (rows_r, cols_r)), shape=(n_l, len(self.reservoirs)))
        return a12, a10

    @property
    def demands_m3s(self) -> NDArray[np.float64]:
        """Junction demands, m^3/s, in junction order."""
        return np.array([j.demand_m3s for j in self.junctions], dtype=np.float64)

    @property
    def reservoir_heads_m(self) -> NDArray[np.float64]:
        """Reservoir heads, m, in reservoir order."""
        return np.array([r.head_m for r in self.reservoirs], dtype=np.float64)

    @property
    def elevations_m(self) -> NDArray[np.float64]:
        """Junction elevations, m, in junction order."""
        return np.array([j.elevation_m for j in self.junctions], dtype=np.float64)
