"""Read and write the subset of the EPANET .inp format used by :mod:`wdtk.network`.

Supported sections: [JUNCTIONS], [RESERVOIRS], [PIPES], [PUMPS] (HEAD curves), [CURVES],
[OPTIONS] (Units, Headloss; Viscosity and Specific Gravity only at their default of 1.0).
Cosmetic sections are skipped: [TITLE] (kept as the title), [COORDINATES] (kept), [VERTICES],
[LABELS], [TAGS], [BACKDROP], [END]. Any other section, or an unsupported keyword or column
(demand patterns, check valves, pump speed or power settings), raises
:class:`UnsupportedInpError` naming it: the reader never silently drops hydraulics.

Units follow EPANET's conventions, as implemented in WNTR 1.5.0 ``wntr/epanet/util.py``:
US flow units (CFS, GPM, MGD) use ft for lengths, elevations and heads, inches for diameters,
and 10^-3 ft for Darcy-Weisbach roughness; SI flow units (LPS, LPM, CMH, CMD, MLD) use m,
mm for diameters, and mm for Darcy-Weisbach roughness. Hazen-Williams C is dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wdtk.network.model import HeadlossFormula, Junction, Network, Pipe, Pump, Reservoir
from wdtk.units import FT_M, IN_M, flow_from_m3s, flow_to_m3s

SUPPORTED_SECTIONS = ("JUNCTIONS", "RESERVOIRS", "PIPES", "PUMPS", "CURVES", "OPTIONS")
COSMETIC_SECTIONS = ("TITLE", "COORDINATES", "VERTICES", "LABELS", "TAGS", "BACKDROP", "END")


class UnsupportedInpError(ValueError):
    """Raised when an .inp file uses a section or feature this reader does not support."""


@dataclass(frozen=True)
class _UnitSystem:
    flow_unit: str  # key in wdtk.units.FLOW_TO_M3S
    length_m: float  # lengths, elevations, heads
    diameter_m: float
    dw_roughness_m: float


_UNIT_SYSTEMS: dict[str, _UnitSystem] = {
    "CFS": _UnitSystem("cfs", FT_M, IN_M, 1e-3 * FT_M),
    "GPM": _UnitSystem("gpm", FT_M, IN_M, 1e-3 * FT_M),
    "MGD": _UnitSystem("MGD", FT_M, IN_M, 1e-3 * FT_M),
    "LPS": _UnitSystem("L/s", 1.0, 1e-3, 1e-3),
    "LPM": _UnitSystem("L/min", 1.0, 1e-3, 1e-3),
    "CMH": _UnitSystem("m3/h", 1.0, 1e-3, 1e-3),
    "CMD": _UnitSystem("m3/d", 1.0, 1e-3, 1e-3),
    "MLD": _UnitSystem("ML/d", 1.0, 1e-3, 1e-3),
}


def _split_sections(text: str) -> list[tuple[str, list[list[str]]]]:
    """Return (section name, rows of whitespace-split fields) with comments removed."""
    sections: list[tuple[str, list[list[str]]]] = []
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            name = line.strip("[]").strip().upper()
            sections.append((name, []))
        elif not sections:
            raise UnsupportedInpError(f"Data before the first section header: {raw!r}.")
        else:
            sections[-1][1].append(line.split())
    return sections


def _parse_options(rows: list[list[str]]) -> tuple[_UnitSystem, HeadlossFormula]:
    units = _UNIT_SYSTEMS["GPM"]  # EPANET default
    headloss: HeadlossFormula = "H-W"
    for row in rows:
        key = row[0].upper()
        if key == "UNITS":
            flag = row[1].upper()
            if flag not in _UNIT_SYSTEMS:
                raise UnsupportedInpError(f"[OPTIONS] Units {row[1]!r} is not supported.")
            units = _UNIT_SYSTEMS[flag]
        elif key == "HEADLOSS":
            value = row[1].upper()
            if value not in ("H-W", "D-W"):
                raise UnsupportedInpError(f"[OPTIONS] Headloss {row[1]!r} is not supported.")
            headloss = "H-W" if value == "H-W" else "D-W"
        elif key in ("VISCOSITY", "SPECIFIC") and float(row[-1]) == 1.0:
            continue  # default relative values; nothing to change
        else:
            raise UnsupportedInpError(f"[OPTIONS] {' '.join(row)!r} is not supported.")
    return units, headloss


def read_inp(path: str | Path) -> Network:
    """Read a network from an EPANET .inp file (supported subset only).

    Parameters
    ----------
    path : str or Path
        File path.

    Returns
    -------
    Network
        Network in SI units.

    Raises
    ------
    UnsupportedInpError
        If the file uses an unsupported section, option, or feature.
    """
    sections = _split_sections(Path(path).read_text(encoding="utf-8"))
    names = [name for name, _ in sections]
    for name in names:
        if name not in SUPPORTED_SECTIONS and name not in COSMETIC_SECTIONS:
            raise UnsupportedInpError(f"Section [{name}] is not supported by wdtk.network.io.")
    rows: dict[str, list[list[str]]] = {}
    for name, body in sections:
        rows.setdefault(name, []).extend(body)

    units, headloss = _parse_options(rows.get("OPTIONS", []))
    to_len = units.length_m
    rough_scale = units.dw_roughness_m if headloss == "D-W" else 1.0

    junctions = []
    for r in rows.get("JUNCTIONS", []):
        if len(r) > 3:
            raise UnsupportedInpError(f"[JUNCTIONS] demand pattern on {r[0]} is not supported.")
        demand = flow_to_m3s(float(r[2]), units.flow_unit) if len(r) > 2 else 0.0
        junctions.append(Junction(r[0], float(r[1]) * to_len, demand))

    reservoirs = []
    for r in rows.get("RESERVOIRS", []):
        if len(r) > 2:
            raise UnsupportedInpError(f"[RESERVOIRS] head pattern on {r[0]} is not supported.")
        reservoirs.append(Reservoir(r[0], float(r[1]) * to_len))

    pipes = []
    for r in rows.get("PIPES", []):
        status = r[7].upper() if len(r) > 7 else "OPEN"
        if status not in ("OPEN", "CLOSED"):
            raise UnsupportedInpError(f"[PIPES] status {r[7]!r} on {r[0]} is not supported.")
        pipes.append(
            Pipe(
                id=r[0],
                start=r[1],
                end=r[2],
                length_m=float(r[3]) * to_len,
                diameter_m=float(r[4]) * units.diameter_m,
                roughness=float(r[5]) * rough_scale,
                minor_k=float(r[6]) if len(r) > 6 else 0.0,
                closed=status == "CLOSED",
            )
        )

    curves: dict[str, list[tuple[float, float]]] = {}
    for r in rows.get("CURVES", []):
        curves.setdefault(r[0], []).append(
            (flow_to_m3s(float(r[1]), units.flow_unit), float(r[2]) * to_len)
        )

    pumps = []
    for r in rows.get("PUMPS", []):
        if len(r) != 5 or r[3].upper() != "HEAD":
            raise UnsupportedInpError(
                f"[PUMPS] {r[0]}: only 'HEAD <curve>' pumps are supported (got {r[3:]!r})."
            )
        if r[4] not in curves:
            raise UnsupportedInpError(f"[PUMPS] {r[0]}: curve {r[4]!r} not found in [CURVES].")
        pumps.append(Pump(r[0], r[1], r[2], tuple(curves[r[4]])))

    coordinates = {r[0]: (float(r[1]), float(r[2])) for r in rows.get("COORDINATES", [])}
    title = " ".join(" ".join(r) for r in rows.get("TITLE", []))
    return Network(
        junctions=tuple(junctions),
        reservoirs=tuple(reservoirs),
        pipes=tuple(pipes),
        pumps=tuple(pumps),
        headloss=headloss,
        coordinates=coordinates,
        title=title,
    )


def write_inp(network: Network, path: str | Path, units: str = "LPS") -> None:
    """Write a network to an EPANET .inp file using only the supported subset.

    The file is readable by :func:`read_inp` and by EPANET (e.g. through WNTR), so both
    solvers can be run on the same input.

    Parameters
    ----------
    network : Network
        Network to write (SI units).
    path : str or Path
        Output file path.
    units : str, optional
        EPANET flow-units flag (default ``"LPS"``); see ``_UNIT_SYSTEMS``.
    """
    if units not in _UNIT_SYSTEMS:
        raise UnsupportedInpError(f"Units {units!r} are not supported.")
    u = _UNIT_SYSTEMS[units]
    rough_scale = u.dw_roughness_m if network.headloss == "D-W" else 1.0

    def num(x: float) -> str:
        return repr(float(x))

    def flow(q: float) -> str:
        return num(flow_from_m3s(q, u.flow_unit))

    lines = ["[TITLE]", network.title or "wdtk network", "", "[JUNCTIONS]"]
    lines += [
        f"{j.id}\t{num(j.elevation_m / u.length_m)}\t{flow(j.demand_m3s)}"
        for j in network.junctions
    ]
    lines += ["", "[RESERVOIRS]"]
    lines += [f"{r.id}\t{num(r.head_m / u.length_m)}" for r in network.reservoirs]
    lines += ["", "[PIPES]"]
    lines += [
        f"{p.id}\t{p.start}\t{p.end}\t{num(p.length_m / u.length_m)}\t"
        f"{num(p.diameter_m / u.diameter_m)}\t{num(p.roughness / rough_scale)}\t"
        f"{num(p.minor_k)}\t{'Closed' if p.closed else 'Open'}"
        for p in network.pipes
    ]
    lines += ["", "[PUMPS]"]
    lines += [f"{p.id}\t{p.start}\t{p.end}\tHEAD\tC_{p.id}" for p in network.pumps]
    lines += ["", "[CURVES]"]
    for p in network.pumps:
        lines += [f"C_{p.id}\t{flow(q)}\t{num(h / u.length_m)}" for q, h in p.curve_points]
    lines += ["", "[OPTIONS]", f"Units\t{units}", f"Headloss\t{network.headloss}", ""]
    if network.coordinates:
        lines += ["[COORDINATES]"]
        lines += [f"{n}\t{num(x)}\t{num(y)}" for n, (x, y) in network.coordinates.items()]
        lines += [""]
    lines += ["[END]", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
