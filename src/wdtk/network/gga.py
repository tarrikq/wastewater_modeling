"""Global gradient algorithm (Todini and Pilati, 1987): the method used by EPANET.

Unknowns are the link flows Q (n_l) and junction heads H (n_j). With the topology convention
of :mod:`wdtk.network.model`, steady state requires

    F1(Q, H) = h(Q) + A12 H + A10 H0 = 0        (energy, one per link)
    F2(Q)    = A21 Q - q             = 0        (continuity, one per junction; A21 = A12^T)

Newton's method on [F1; F2] with D = diag(dh/dQ) gives the block system

    [ D    A12 ] [dQ]   [ -F1 ]
    [ A21   0  ] [dH] = [ -F2 ].

Eliminating dQ = -D^-1 (F1 + A12 dH) leaves the Schur complement for the heads,

    (A21 D^-1 A12) dH = F2 - A21 D^-1 F1,

a sparse symmetric positive-definite n_j x n_j system (D > 0 and A12 has full column rank
when every junction connects to a reservoir). Heads are solved first, then flows are updated
link by link. Convergence follows EPANET's criterion sum|dQ| / sum|Q| < accuracy; continuity
is satisfied exactly after the first iteration because F2 is linear.

Reference: Todini, E., and Pilati, S. (1987). "A gradient algorithm for the analysis of pipe
networks." In Computer Applications in Water Supply, Vol. 1, Research Studies Press, 1-20.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from wdtk import headloss as hl
from wdtk.network.hydraulics import (
    HydraulicOptions,
    continuity_residual_m3s,
    energy_residual_m,
    initial_flows_m3s,
    link_headloss,
)
from wdtk.network.model import Network, Pump
from wdtk.network.results import IterationRecord, SolveResult


def solve_gga(
    network: Network,
    *,
    accuracy: float = 1e-6,
    max_iter: int = 200,
    convention: hl.HWConvention = "epanet",
    kinematic_viscosity_m2s: float | None = None,
    turbulent: hl.TurbulentMethod = "swamee-jain",
) -> SolveResult:
    """Solve a steady-state, demand-driven network with the global gradient algorithm.

    Parameters
    ----------
    network : Network
        Network to solve (SI units).
    accuracy : float, optional
        Convergence tolerance on sum|dQ| / sum|Q| (default 1e-6; EPANET's default is 1e-3).
    max_iter : int, optional
        Maximum Newton iterations (default 200).
    convention : {"epanet", "exact"}, optional
        Head loss constants (default ``"epanet"``, matching EPANET); see
        :class:`wdtk.network.hydraulics.HydraulicOptions`.
    kinematic_viscosity_m2s : float, optional
        Kinematic viscosity for Darcy-Weisbach, m^2/s. Defaults to EPANET's value under
        the EPANET convention; required under the exact convention.
    turbulent : {"swamee-jain", "colebrook"}, optional
        Turbulent friction factor for Darcy-Weisbach (default ``"swamee-jain"``).

    Returns
    -------
    SolveResult
        Flows (m^3/s), heads and pressure heads (m), and the iteration history.

    Raises
    ------
    RuntimeError
        If the iteration does not converge, or a pump ends with reverse flow (pump closure
        logic is not modeled).
    """
    options = HydraulicOptions(convention, kinematic_viscosity_m2s, turbulent)
    a12, a10 = network.incidence
    a21 = a12.T.tocsr()
    q_dem = network.demands_m3s
    h0 = network.reservoir_heads_m

    flows = initial_flows_m3s(network)
    heads = np.zeros(len(network.junctions))
    history: list[IterationRecord] = []
    converged = False

    for it in range(1, max_iter + 1):
        h, grad = link_headloss(network, flows, options)
        f1 = h + a12 @ heads + a10 @ h0
        f2 = a21 @ flows - q_dem
        d_inv = 1.0 / grad
        schur = (a21 @ sp.diags_array(d_inv) @ a12).tocsc()
        d_heads = np.atleast_1d(spsolve(schur, f2 - a21 @ (d_inv * f1)))
        d_flows = -d_inv * (f1 + a12 @ d_heads)
        heads = heads + d_heads
        flows = flows + d_flows

        rel = float(np.sum(np.abs(d_flows)) / max(np.sum(np.abs(flows)), 1e-300))
        history.append(
            IterationRecord(
                iteration=it,
                relative_flow_change=rel,
                max_flow_change_m3s=float(np.max(np.abs(d_flows))),
                max_head_imbalance_m=float(
                    np.max(np.abs(energy_residual_m(network, flows, heads, options)))
                ),
                max_continuity_residual_m3s=float(
                    np.max(np.abs(continuity_residual_m3s(network, flows)), initial=0.0)
                ),
            )
        )
        if rel < accuracy:
            converged = True
            break

    if not converged:
        raise RuntimeError(f"GGA did not converge in {max_iter} iterations.")
    for k, lk in enumerate(network.links):
        if isinstance(lk, Pump) and flows[k] < 0.0:
            raise RuntimeError(f"Pump {lk.id} has reverse flow; pump closure is not modeled.")

    return SolveResult(
        method="GGA",
        link_ids=tuple(lk.id for lk in network.links),
        junction_ids=tuple(j.id for j in network.junctions),
        flows_m3s=flows,
        heads_m=heads,
        pressures_m=heads - network.elevations_m,
        converged=converged,
        history=history,
    )
