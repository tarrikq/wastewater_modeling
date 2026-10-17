"""Solver results and convergence history shared by the network solvers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IterationRecord:
    """Convergence measures after one solver iteration.

    Attributes
    ----------
    iteration : int
        Iteration number, starting at 1.
    relative_flow_change : float
        sum |dQ| / sum |Q| (EPANET's convergence measure), dimensionless.
    max_flow_change_m3s : float
        max |dQ|, m^3/s.
    max_head_imbalance_m : float
        Largest energy residual |h_k(Q_k) - (H_start - H_end)| over links (Hardy Cross: over
        loops), m, evaluated with the updated flows and heads.
    max_continuity_residual_m3s : float
        Largest |inflow - outflow - demand| over junctions, m^3/s.
    """

    iteration: int
    relative_flow_change: float
    max_flow_change_m3s: float
    max_head_imbalance_m: float
    max_continuity_residual_m3s: float


@dataclass(frozen=True, eq=False)
class SolveResult:
    """Steady-state solution of a network.

    Attributes
    ----------
    method : str
        Solver name.
    link_ids : tuple of str
        Link IDs in the order of ``flows_m3s`` (open pipes, then pumps).
    junction_ids : tuple of str
        Junction IDs in the order of ``heads_m``.
    flows_m3s : ndarray
        Link flows, m^3/s (positive start -> end).
    heads_m : ndarray
        Junction heads, m.
    pressures_m : ndarray
        Junction pressure heads (head - elevation), m.
    converged : bool
        Whether the convergence criteria were met.
    history : list of IterationRecord
        Per-iteration convergence measures.
    """

    method: str
    link_ids: tuple[str, ...]
    junction_ids: tuple[str, ...]
    flows_m3s: NDArray[np.float64]
    heads_m: NDArray[np.float64]
    pressures_m: NDArray[np.float64]
    converged: bool
    history: list[IterationRecord] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        """Number of iterations performed."""
        return len(self.history)

    def flow(self, link_id: str) -> float:
        """Flow in a link, m^3/s."""
        return float(self.flows_m3s[self.link_ids.index(link_id)])

    def head(self, junction_id: str) -> float:
        """Head at a junction, m."""
        return float(self.heads_m[self.junction_ids.index(junction_id)])
