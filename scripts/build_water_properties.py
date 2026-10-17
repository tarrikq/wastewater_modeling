"""Build ``src/wdtk/data/water_properties.csv`` from the raw property table.

Input: ``data/raw/Properties_of_Water_at_Various_Temperatures.csv``.

Conversions (all exact):
* temperature_c = (Temp (degF) - 32) * 5/9. The raw degC column is rounded to 0.1 degC.
* density_kgm3 = Density (kg/m3) column, as given.
* kinematic_viscosity_m2s = Kinematic Viscosity (cSt) * 1e-6. The cSt column carries more
  significant figures than the raw m2/sec column.

Only rows up to the first temperature at or above the upper end of ``wdtk.fluid.VALID_RANGE_C``
are kept, so the spline covers the full valid range without extrapolation.

Usage: ``uv run python scripts/build_water_properties.py``
"""

import csv
from pathlib import Path

from wdtk.fluid import VALID_RANGE_C

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Properties_of_Water_at_Various_Temperatures.csv"
OUT = ROOT / "src" / "wdtk" / "data" / "water_properties.csv"
SOURCE = (
    "Journal of Physical and Chemical Reference Data 7, 941 (1978); "
    "https://doi.org/10.1063/1.555581 (citation supplied by project author)"
)


def main() -> None:
    """Convert the raw table and write the packaged CSV."""
    with RAW.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    # Values are kept as strings so the raw decimal digits pass through unchanged;
    # cSt -> m2/s is an exact decimal shift, written as an e-06 exponent.
    out_rows: list[tuple[float, str, str]] = []
    for r in rows:
        t_c = (float(r["Temp (°F)"]) - 32.0) * 5.0 / 9.0
        out_rows.append(
            (t_c, r["Density (kg/m3)"].strip(), f"{r['Kinematic Viscosity (cSt)'].strip()}e-06")
        )
        if t_c >= VALID_RANGE_C[1]:
            break

    lines = [
        f"# source: {SOURCE}",
        f"# derived from data/raw/{RAW.name} by scripts/{Path(__file__).name}",
        "# temperature_c = (degF - 32) * 5/9; kinematic_viscosity_m2s = cSt * 1e-6",
        "temperature_c,density_kgm3,kinematic_viscosity_m2s",
    ]
    lines += [f"{t!r},{rho},{nu}" for t, rho, nu in out_rows]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_rows)} rows to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
