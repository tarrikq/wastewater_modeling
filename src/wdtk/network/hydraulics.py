"""Link head loss and its flow gradient, shared by the network solvers.

For every open link k, h_k(Q_k) is the head drop from start to end node:

* Hazen-Williams pipe:  h = r Q|Q|^(n-1) + m Q|Q|,  with r, n from
  :func:`wdtk.headloss.hazen_williams_resistance` and m = K / (2 g A^2) for minor losses.
* Darcy-Weisbach pipe:  h = (f L/D + K) Q|Q| / (2 g A^2),  f from
  :func:`wdtk.headloss.friction_factor` at Re = |V| D / nu.
* Pump:  h = -(a - b Q|Q|^(c-1))  (negative: a head gain), from the pump curve H = a - b Q^c.

Gradients dh/dQ (all positive) feed the Newton (GGA) and loop-correction (Hardy Cross)
updates. For Darcy-Weisbach the friction factor is held fixed when differentiating
(dh/dQ = 2h/Q), a standard quasi-Newton simplification that does not change the converged
solution. Gradients are evaluated at max(|Q|, ``Q_GRADIENT_FLOOR_M3S``) so they stay positive
at zero flow; this regularizes the iteration only (residuals use the exact h).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wdtk import headloss as hl
from wdtk.network.model import Network, Pipe, Pump
from wdtk.units import G_STANDARD_MS2

Q_GRADIENT_FLOOR_M3S = 1.0e-6
"""Flow magnitude below which link gradients are evaluated at this value, m^3/s."""


@dataclass(frozen=True)
class HydraulicOptions:
    """Options for evaluating link head losses.

    Attributes
    ----------
    convention : {"epanet", "exact"}
        ``"epanet"``: EPANET's rounded Hazen-Williams constants, and g = 32.2 ft/s^2 in
        Darcy-Weisbach and minor losses (``wdtk.headloss.EPANET_DW_GRAVITY_MS2``).
        ``"exact"``: constants derived from the defining equations, standard gravity.
    kinematic_viscosity_m2s : float or None
        Kinematic viscosity for Darcy-Weisbach, m^2/s. If None, the EPANET convention uses
        ``wdtk.headloss.EPANET_KINEMATIC_VISCOSITY_M2S``; the exact convention requires it.
    turbulent : {"swamee-jain", "colebrook"}
        Turbulent friction factor formula for Darcy-Weisbach.
    """

    convention: hl.HWConvention = "epanet"
    kinematic_viscosity_m2s: float | None = None
    turbulent: hl.TurbulentMethod = "swamee-jain"

    @property
    def g_ms2(self) -> float:
        """Gravitational acceleration used in Darcy-Weisbach and minor losses, m/s^2."""
        return hl.EPANET_DW_GRAVITY_MS2 if self.convention == "epanet" else G_STANDARD_MS2

    @property
    def viscosity_m2s(self) -> float:
        """Kinematic viscosity for Darcy-Weisbach, m^2/s (see class docstring)."""
        if self.kinematic_viscosity_m2s is not None:
            return self.kinematic_viscosity_m2s
        if self.convention == "epanet":
            return hl.EPANET_KINEMATIC_VISCOSITY_M2S
        raise ValueError("Darcy-Weisbach with the exact convention needs a kinematic viscosity.")


def initial_flows_m3s(network: Network, velocity_ms: float = 0.3) -> NDArray[np.float64]:
    """Return starting flows: a uniform velocity in pipes, half the maximum flow in pumps.

    Parameters
    ----------
    network : Network
        The network.
    velocity_ms : float, optional
        Initial pipe velocity, m/s (default 0.3). Any positive value works for GGA.

    Returns
    -------
    ndarray
        Flows, m^3/s, in link order.
    """
    q = np.empty(len(network.links), dtype=np.float64)
    for k, lk in enumerate(network.links):
        if isinstance(lk, Pipe):
            q[k] = velocity_ms * float(hl.area_m2(lk.diameter_m))
        else:
            q[k] = 0.5 * lk.curve.max_flow_m3s
    return q


def link_headloss(
    network: Network, flows_m3s: NDArray[np.float64], options: HydraulicOptions
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Head loss and its flow gradient for every open link.

    Parameters
    ----------
    network : Network
        The network.
    flows_m3s : ndarray
        Link flows, m^3/s, in link order (open pipes, then pumps).
    options : HydraulicOptions
        Formula options.

    Returns
    -------
    tuple of ndarray
        (h, m: head drop start -> end; dh/dQ, m per m^3/s: positive).
    """
    q = np.asarray(flows_m3s, dtype=np.float64)
    h = np.empty_like(q)
    dh = np.empty_like(q)
    n_p = network.n_pipes
    pipes = [lk for lk in network.links[:n_p] if isinstance(lk, Pipe)]
    if pipes:
        h[:n_p], dh[:n_p] = _pipe_headloss(network, pipes, q[:n_p], options)
    for k, lk in enumerate(network.links[n_p:], start=n_p):
        assert isinstance(lk, Pump)
        c = lk.curve
        qk = q[k]
        h[k] = -(c.a_m - c.b * qk * abs(qk) ** (c.c - 1.0))
        dh[k] = c.c * c.b * max(abs(qk), Q_GRADIENT_FLOOR_M3S) ** (c.c - 1.0)
    return h, dh


def _pipe_headloss(
    network: Network, pipes: list[Pipe], q: NDArray[np.float64], options: HydraulicOptions
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    length = np.array([p.length_m for p in pipes])
    diameter = np.array([p.diameter_m for p in pipes])
    rough = np.array([p.roughness for p in pipes])
    area = np.pi * diameter**2 / 4.0
    g = options.g_ms2
    minor = np.array([p.minor_k for p in pipes]) / (2.0 * g * area**2)
    q_abs = np.abs(q)
    q_grad = np.maximum(q_abs, Q_GRADIENT_FLOOR_M3S)

    if network.headloss == "H-W":
        r, n = hl.hazen_williams_resistance(length, diameter, rough, options.convention)
        h = r * q * q_abs ** (n - 1.0) + minor * q * q_abs
        dh = n * r * q_grad ** (n - 1.0) + 2.0 * minor * q_grad
        return h, dh

    nu = options.viscosity_m2s
    re = q_grad / area * diameter / nu
    f = hl.friction_factor(re, rough / diameter, options.turbulent)
    coef = (f * length / diameter) / (2.0 * g * area**2) + minor
    f_actual = np.where(q_abs > 0.0, coef, 0.0)
    # Exact head loss uses f at the actual Re; recompute where |Q| is below the floor.
    small = (q_abs > 0.0) & (q_abs < Q_GRADIENT_FLOOR_M3S)
    if np.any(small):
        re_s = q_abs[small] / area[small] * diameter[small] / nu
        f_s = hl.friction_factor(re_s, rough[small] / diameter[small], options.turbulent)
        f_actual[small] = (f_s * length[small] / diameter[small]) / (
            2.0 * g * area[small] ** 2
        ) + minor[small]
    h = f_actual * q * q_abs
    dh = 2.0 * coef * q_grad
    return h, dh


def energy_residual_m(
    network: Network,
    flows_m3s: NDArray[np.float64],
    heads_m: NDArray[np.float64],
    options: HydraulicOptions,
) -> NDArray[np.float64]:
    """Energy-equation residual per link, h(Q) + A12 H + A10 H0, m.

    Parameters
    ----------
    network : Network
        The network.
    flows_m3s : ndarray
        Link flows, m^3/s.
    heads_m : ndarray
        Junction heads, m.
    options : HydraulicOptions
        Formula options.

    Returns
    -------
    ndarray
        Residual per link, m (zero at the solution).
    """
    a12, a10 = network.incidence
    h, _ = link_headloss(network, flows_m3s, options)
    return np.asarray(h + a12 @ heads_m + a10 @ network.reservoir_heads_m, dtype=np.float64)


def continuity_residual_m3s(
    network: Network, flows_m3s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Continuity residual per junction, (inflow - outflow) - demand, m^3/s.

    Parameters
    ----------
    network : Network
        The network.
    flows_m3s : ndarray
        Link flows, m^3/s.

    Returns
    -------
    ndarray
        Residual per junction, m^3/s (zero at the solution).
    """
    a12, _ = network.incidence
    return np.asarray(a12.T @ flows_m3s - network.demands_m3s, dtype=np.float64)
