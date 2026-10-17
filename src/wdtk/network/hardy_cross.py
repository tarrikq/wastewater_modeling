"""Hardy Cross loop-correction method (Cross, 1936): a teaching solver.

The method keeps continuity satisfied at every step and iterates on the energy equation
around each loop. For a loop L with links k traversed in a fixed direction (s_k = +1 if the
traversal follows the link's start -> end direction, else -1), the head imbalance is

    r_L = sum_k s_k h_k(Q_k) - (H_depart - H_arrive),

where the bracketed term is nonzero only for a pseudo-loop: a path between two reservoirs,
closed through a virtual node that merges all reservoirs. A first-order (Newton) correction
of the loop flow,

    dQ_L = -r_L / sum_k dh_k/dQ_k,    Q_k <- Q_k + s_k dQ_L,

is applied loop by loop (Gauss-Seidel order) and leaves continuity unchanged because every
loop flow enters and leaves each node it visits.

Scope (a stated limit of this teaching implementation): pipes only; networks with pumps or
pipes joining two reservoirs raise ``NotImplementedError``. Use :func:`wdtk.network.gga.solve_gga`
for those.

Reference: Cross, H. (1936). "Analysis of flow in networks of conduits or conductors."
University of Illinois Engineering Experiment Station Bulletin No. 286.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from wdtk import headloss as hl
from wdtk.network.hydraulics import HydraulicOptions, continuity_residual_m3s, link_headloss
from wdtk.network.model import Network
from wdtk.network.results import IterationRecord, SolveResult

_VIRTUAL = "__reservoirs__"


@dataclass(frozen=True)
class Loop:
    """A loop (or pseudo-loop) as link indices with traversal signs.

    Attributes
    ----------
    links : tuple of int
        Link indices in ``network.links``.
    signs : tuple of float
        +1 if traversal follows the link direction, else -1.
    head_offset_m : float
        H_depart - H_arrive for a pseudo-loop through the reservoirs, m (0 for real loops).
    """

    links: tuple[int, ...]
    signs: tuple[float, ...]
    head_offset_m: float = 0.0


def _merged(network: Network, node: str) -> str:
    return _VIRTUAL if node in network.reservoir_index else node


def find_loops(network: Network) -> list[Loop]:
    """Independent loops and reservoir-to-reservoir pseudo-loops of a network.

    Reservoirs are merged into one virtual node; every link is subdivided by a link node so
    parallel links are distinct; ``networkx.cycle_basis`` then gives n_links - n_junctions
    independent cycles.

    Parameters
    ----------
    network : Network
        Pipe-only network.

    Returns
    -------
    list of Loop
        Loops with traversal signs and reservoir head offsets.
    """
    g = nx.Graph()
    for k, lk in enumerate(network.links):
        a, b = _merged(network, lk.start), _merged(network, lk.end)
        if a == b == _VIRTUAL:
            raise NotImplementedError(f"Link {lk.id} joins two reservoirs (not supported).")
        g.add_edge(a, ("link", k))
        g.add_edge(("link", k), b)

    heads = {r.id: r.head_m for r in network.reservoirs}
    loops: list[Loop] = []
    for cycle in nx.cycle_basis(g):
        links: list[int] = []
        signs: list[float] = []
        depart = arrive = None
        n = len(cycle)
        for i, item in enumerate(cycle):
            if not (isinstance(item, tuple) and item[0] == "link"):
                continue
            k = item[1]
            lk = network.links[k]
            prev, nxt = cycle[i - 1], cycle[(i + 1) % n]
            forward = _merged(network, lk.start) == prev and _merged(network, lk.end) == nxt
            links.append(k)
            signs.append(1.0 if forward else -1.0)
            # Track which actual reservoir the traversal leaves from / arrives at.
            enter_node, leave_node = (lk.start, lk.end) if forward else (lk.end, lk.start)
            if nxt == _VIRTUAL:
                arrive = leave_node
            if prev == _VIRTUAL:
                depart = enter_node
        offset = heads[depart] - heads[arrive] if depart and arrive else 0.0
        loops.append(Loop(tuple(links), tuple(signs), offset))
    return loops


def _continuity_flows(network: Network) -> NDArray[np.float64]:
    """Flows satisfying continuity: demands routed through a BFS spanning tree, chords zero."""
    flows = np.zeros(len(network.links))
    adjacency: dict[str, list[tuple[int, str]]] = {}
    for k, lk in enumerate(network.links):
        a, b = _merged(network, lk.start), _merged(network, lk.end)
        adjacency.setdefault(a, []).append((k, b))
        adjacency.setdefault(b, []).append((k, a))
    parent: dict[str, tuple[int, str]] = {}
    order = [_VIRTUAL]
    queue = deque([_VIRTUAL])
    while queue:
        node = queue.popleft()
        for k, other in adjacency.get(node, []):
            if other != _VIRTUAL and other not in parent:
                parent[other] = (k, node)
                order.append(other)
                queue.append(other)
    demand = {j.id: j.demand_m3s for j in network.junctions}
    for node in reversed(order[1:]):  # leaves first
        k, up = parent[node]
        lk = network.links[k]
        # Flow from `up` to `node` equals everything consumed at and below `node`.
        sign = 1.0 if _merged(network, lk.end) == node else -1.0
        flows[k] += sign * demand[node]
        if up != _VIRTUAL:
            demand[up] += demand[node]
    return flows


def _heads_from_flows(network: Network, h_link: NDArray[np.float64]) -> NDArray[np.float64]:
    """Junction heads by walking from the reservoirs: H_end = H_start - h (and reverse)."""
    known = {r.id: r.head_m for r in network.reservoirs}
    queue = deque(known)
    adjacency: dict[str, list[tuple[int, str, float]]] = {}
    for k, lk in enumerate(network.links):
        adjacency.setdefault(lk.start, []).append((k, lk.end, -1.0))
        adjacency.setdefault(lk.end, []).append((k, lk.start, 1.0))
    while queue:
        node = queue.popleft()
        for k, other, sign in adjacency.get(node, []):
            if other not in known:
                known[other] = known[node] + sign * h_link[k]
                queue.append(other)
    return np.array([known[j.id] for j in network.junctions])


def solve_hardy_cross(
    network: Network,
    *,
    accuracy: float = 1e-6,
    max_iter: int = 500,
    convention: hl.HWConvention = "epanet",
    kinematic_viscosity_m2s: float | None = None,
    turbulent: hl.TurbulentMethod = "swamee-jain",
) -> SolveResult:
    """Solve a pipe network with the Hardy Cross loop-correction method.

    Parameters
    ----------
    network : Network
        Pipe-only network (SI units).
    accuracy : float, optional
        Convergence tolerance on sum|dQ| / sum|Q| per sweep (default 1e-6).
    max_iter : int, optional
        Maximum sweeps over all loops (default 500).
    convention, kinematic_viscosity_m2s, turbulent
        Head loss options, as for :func:`wdtk.network.gga.solve_gga`.

    Returns
    -------
    SolveResult
        Flows, heads, pressures, and per-sweep history (head imbalance = max |r_L|).

    Raises
    ------
    NotImplementedError
        If the network has pumps or a pipe joining two reservoirs.
    RuntimeError
        If the iteration does not converge.
    """
    if network.pumps:
        raise NotImplementedError("Hardy Cross here is pipes-only; use solve_gga for pumps.")
    options = HydraulicOptions(convention, kinematic_viscosity_m2s, turbulent)
    loops = find_loops(network)
    flows = _continuity_flows(network)
    history: list[IterationRecord] = []
    converged = False

    for it in range(1, max_iter + 1):
        flows_before = flows.copy()
        for loop in loops:
            idx = np.array(loop.links)
            s = np.array(loop.signs)
            h, dh = link_headloss(network, flows, options)
            imbalance = float(np.sum(s * h[idx])) - loop.head_offset_m
            flows[idx] += s * (-imbalance / float(np.sum(dh[idx])))
        h, _ = link_headloss(network, flows, options)
        imbalances = [
            abs(float(np.sum(np.array(lp.signs) * h[np.array(lp.links)])) - lp.head_offset_m)
            for lp in loops
        ]
        d_flows = flows - flows_before
        rel = float(np.sum(np.abs(d_flows)) / max(np.sum(np.abs(flows)), 1e-300))
        history.append(
            IterationRecord(
                iteration=it,
                relative_flow_change=rel,
                max_flow_change_m3s=float(np.max(np.abs(d_flows))),
                max_head_imbalance_m=max(imbalances, default=0.0),
                max_continuity_residual_m3s=float(
                    np.max(np.abs(continuity_residual_m3s(network, flows)), initial=0.0)
                ),
            )
        )
        if rel < accuracy:
            converged = True
            break

    if not converged:
        raise RuntimeError(f"Hardy Cross did not converge in {max_iter} sweeps.")
    h, _ = link_headloss(network, flows, options)
    heads = _heads_from_flows(network, h)
    return SolveResult(
        method="Hardy Cross",
        link_ids=tuple(lk.id for lk in network.links),
        junction_ids=tuple(j.id for j in network.junctions),
        flows_m3s=flows,
        heads_m=heads,
        pressures_m=heads - network.elevations_m,
        converged=converged,
        history=history,
    )
