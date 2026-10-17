"""Load the study assumptions (``data/assumptions.yaml``) into SI-unit dataclasses.

Every entry in the YAML file has ``value``, ``units``, ``basis``, and ``status``. The loader
checks that each entry is present with the expected units, converts it to SI with
:mod:`wdtk.units`, and keeps the raw entries so reports can cite each value's basis and
status. :meth:`Assumptions.drafts` lists entries still awaiting review.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from wdtk import units

STATUSES = ("supplied", "draft", "net3")
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Entry:
    """One assumption as written in the file (original units)."""

    value: Any
    units: str
    basis: str
    status: str


@dataclass(frozen=True)
class ServiceCriteria:
    """Service level criteria: pressure head limits (m) and maximum velocity (m/s)."""

    min_pressure_m: float
    max_pressure_m: float
    max_velocity_ms: float


@dataclass(frozen=True)
class PddSettings:
    """Pressure-dependent demand: zero delivery at ``minimum``, full at ``required`` (m)."""

    minimum_pressure_m: float
    required_pressure_m: float


@dataclass(frozen=True)
class FireFlowSettings:
    """Fire flow criteria and bisection settings (m of head, m^3/s)."""

    residual_pressure_m: float
    required_flow_m3s: float
    upper_bound_m3s: float
    tolerance_m3s: float


@dataclass(frozen=True)
class SizingSettings:
    """Greedy pipe sizing settings: sizes (m), unit costs ($/m by size), and search limits."""

    diameters_m: tuple[float, ...]
    unit_cost_per_m: Mapping[float, float]
    contingency: float
    candidate_hops: int
    max_iterations: int


@dataclass(frozen=True)
class WaterAgeSettings:
    """Water age threshold, run length, report window (s); alternative pump 10 start (h)."""

    threshold_s: float
    duration_s: float
    report_window_s: float
    alternative_pump10_start_h: float


@dataclass(frozen=True)
class EnergySettings:
    """Time-of-use tariff ($/kWh, on-peak hours [start, end)), efficiency, alternative start."""

    on_peak_price_per_kwh: float
    off_peak_price_per_kwh: float
    on_peak_start_h: float
    on_peak_end_h: float
    pump_efficiency: float
    alternative_pump10_start_h: float


@dataclass(frozen=True)
class CriticalitySettings:
    """Pipe criticality screen: minimum diameter (m) and outage simulation length (s)."""

    min_diameter_m: float
    duration_s: float


@dataclass(frozen=True)
class Assumptions:
    """All study assumptions in SI units, plus the raw entries for citation."""

    service: ServiceCriteria
    pdd: PddSettings
    max_day_factor: float
    eps_duration_s: float
    fire_flow: FireFlowSettings
    sizing: SizingSettings
    water_age: WaterAgeSettings
    energy: EnergySettings
    criticality: CriticalitySettings
    atmospheric_pressure_kpa: float
    npsh_ratio_range: tuple[float, float]
    entries: Mapping[str, Entry]

    def drafts(self) -> list[str]:
        """Return the dotted names of entries whose status is still ``draft``."""
        return [name for name, e in self.entries.items() if e.status == "draft"]


def _psi_to_head_m(v: float) -> float:
    return units.pressure_to_head(float(v), "psi", "m")


def _hours(v: float) -> float:
    return float(v) * SECONDS_PER_HOUR


_CONVERTERS: dict[str, Callable[[Any], Any]] = {
    "psi": _psi_to_head_m,
    "ft/s": lambda v: units.length_to_m(float(v), "ft"),
    "gpm": lambda v: units.flow_to_m3s(float(v), "gpm"),
    "in": lambda v: units.length_to_m(float(v), "in"),
    "ft": lambda v: units.length_to_m(float(v), "ft"),
    "mm": lambda v: units.length_to_m(float(v), "mm"),
    "h": _hours,
    "-": lambda v: v,
    "$/kWh": float,
    "h of day": float,
    "h of day [start, end)": lambda v: (float(v[0]), float(v[1])),
}


class _Reader:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = data
        self.entries: dict[str, Entry] = {}

    def entry(self, path: str, expected_units: str) -> Entry:
        node: Any = self.data
        for key in path.split("."):
            if not isinstance(node, Mapping) or key not in node:
                raise ValueError(f"Assumption {path!r} is missing.")
            node = node[key]
        if not isinstance(node, Mapping) or not {"value", "units", "basis", "status"} <= set(node):
            raise ValueError(f"Assumption {path!r} needs value, units, basis, and status.")
        if node["units"] != expected_units:
            raise ValueError(
                f"Assumption {path!r} has units {node['units']!r}; expected {expected_units!r}."
            )
        if node["status"] not in STATUSES:
            raise ValueError(f"Assumption {path!r} has unknown status {node['status']!r}.")
        e = Entry(node["value"], node["units"], str(node["basis"]), node["status"])
        self.entries[path] = e
        return e

    def si(self, path: str, expected_units: str) -> Any:
        value = self.entry(path, expected_units).value
        return _CONVERTERS[expected_units](value)


def load_assumptions(path: str | Path) -> Assumptions:
    """Read and validate the assumptions file.

    Parameters
    ----------
    path : str or Path
        Path to ``assumptions.yaml``.

    Returns
    -------
    Assumptions
        Values converted to SI (m of head for pressures, m^3/s, m, s, $/m).

    Raises
    ------
    ValueError
        If an entry is missing, has unexpected units or status, or a value is out of range.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    r = _Reader(data)

    service = ServiceCriteria(
        r.si("service.min_pressure", "psi"),
        r.si("service.max_pressure", "psi"),
        r.si("service.max_velocity", "ft/s"),
    )
    pdd = PddSettings(
        r.si("pressure_dependent_demand.minimum_pressure", "psi"),
        r.si("pressure_dependent_demand.required_pressure", "psi"),
    )
    fire = FireFlowSettings(
        r.si("fire_flow.residual_pressure", "psi"),
        r.si("fire_flow.required_flow", "gpm"),
        r.si("fire_flow.search_upper_bound", "gpm"),
        r.si("fire_flow.search_tolerance", "gpm"),
    )
    diameters_in = [float(d) for d in r.entry("sizing.nominal_diameters", "in").value]
    cost_in = {float(k): float(v) for k, v in r.entry("sizing.unit_cost", "$/ft").value.items()}
    if sorted(cost_in) != sorted(diameters_in):
        raise ValueError("sizing.unit_cost must list exactly the sizing.nominal_diameters.")
    sizing = SizingSettings(
        diameters_m=tuple(units.length_to_m(d, "in") for d in sorted(diameters_in)),
        unit_cost_per_m={
            units.length_to_m(d, "in"): c / units.FT_M for d, c in sorted(cost_in.items())
        },
        contingency=float(r.si("sizing.contingency", "-")),
        candidate_hops=int(r.si("sizing.candidate_hops", "-")),
        max_iterations=int(r.si("sizing.max_iterations", "-")),
    )
    age = WaterAgeSettings(
        r.si("water_age.concern_threshold", "h"),
        r.si("water_age.simulation_duration", "h"),
        r.si("water_age.report_window", "h"),
        r.si("water_age.alternative_pump10_start", "h of day"),
    )
    peak_start, peak_end = r.si("energy.on_peak_hours", "h of day [start, end)")
    energy = EnergySettings(
        r.si("energy.on_peak_price", "$/kWh"),
        r.si("energy.off_peak_price", "$/kWh"),
        peak_start,
        peak_end,
        float(r.si("energy.pump_efficiency", "-")),
        r.si("energy.alternative_pump10_start", "h of day"),
    )
    crit = CriticalitySettings(
        r.si("criticality.min_diameter", "in"), r.si("criticality.duration", "h")
    )
    for path, unit in (
        ("network.source", "-"),
        ("demand.peak_hour", "-"),
        ("fire_flow.eligible_junctions", "-"),
        ("fire_flow.scenario", "-"),
        ("pipes.dw_roughness", "mm"),
    ):
        r.entry(path, unit)

    a = Assumptions(
        service=service,
        pdd=pdd,
        max_day_factor=float(r.si("demand.max_day_factor", "-")),
        eps_duration_s=r.si("eps.duration", "h"),
        fire_flow=fire,
        sizing=sizing,
        water_age=age,
        energy=energy,
        criticality=crit,
        atmospheric_pressure_kpa=units.pressure_to_kpa(
            float(r.entry("pumps.atmospheric_pressure", "psi").value), "psi"
        ),
        npsh_ratio_range=(
            float(r.si("pumps.npsh_ratio_min", "-")),
            float(r.si("pumps.npsh_ratio_max", "-")),
        ),
        entries=r.entries,
    )
    _validate(a)
    return a


def _validate(a: Assumptions) -> None:
    checks = [
        (a.service.min_pressure_m < a.service.max_pressure_m, "service min < max pressure"),
        (a.service.max_velocity_ms > 0, "service max velocity > 0"),
        (a.pdd.minimum_pressure_m < a.pdd.required_pressure_m, "PDD minimum < required"),
        (a.max_day_factor >= 1.0, "max-day factor >= 1"),
        (0 < a.fire_flow.tolerance_m3s < a.fire_flow.upper_bound_m3s, "fire search bounds"),
        (a.fire_flow.required_flow_m3s <= a.fire_flow.upper_bound_m3s, "fire need <= bound"),
        (a.sizing.contingency >= 0.0, "contingency >= 0"),
        (a.sizing.candidate_hops >= 1 and a.sizing.max_iterations >= 1, "sizing limits"),
        (a.water_age.report_window_s <= a.water_age.duration_s, "age window <= duration"),
        (0.0 <= a.water_age.alternative_pump10_start_h < 24.0, "age alternative start hour"),
        (0.0 <= a.energy.on_peak_start_h < a.energy.on_peak_end_h <= 24.0, "on-peak hours"),
        (0.0 < a.energy.pump_efficiency <= 1.0, "pump efficiency in (0, 1]"),
        (a.npsh_ratio_range[0] <= a.npsh_ratio_range[1], "NPSH ratio range"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        raise ValueError(f"Invalid assumptions: {failed}.")
