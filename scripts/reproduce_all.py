"""Reproduce every figure and table in the README with one command.

Usage::

    uv sync
    uv run python scripts/reproduce_all.py

Runs, in order: Phase 1 head loss figures, the Phase 2 pump figure, the Phase 3 test networks
and EPANET validation, and the Phase 4 Net3 study. Writes a manifest of every output and
``reports/key_results.json`` (the numbers quoted in the README).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import build_test_networks  # noqa: E402
import phase1_headloss_figures  # noqa: E402
import phase2_pump_figure  # noqa: E402
import phase3_validation  # noqa: E402

from wdtk import run_study  # noqa: E402


def _phase3_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    """Run every phase's outputs; see module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--networks", type=Path, default=ROOT / "data" / "networks")
    parser.add_argument("--assumptions", type=Path, default=ROOT / "data" / "assumptions.yaml")
    args = parser.parse_args(argv)

    written: dict[str, list[Path]] = {}
    timings: dict[str, float] = {}

    def run(key: str, fn: Any, *fargs: Any) -> Any:
        t0 = time.perf_counter()
        print(f"== {key}")
        result = fn(*fargs)
        timings[key] = round(time.perf_counter() - t0, 1)
        return result

    written["phase1"] = run("phase1", phase1_headloss_figures.main, args.reports / "phase1")
    written["phase2"] = run("phase2", phase2_pump_figure.main, args.reports / "phase2")
    written["networks"] = run("networks", build_test_networks.main, args.networks)
    written["phase3"] = run(
        "phase3", phase3_validation.main, args.reports / "phase3", args.networks
    )
    code = run(
        "phase4",
        run_study.main,
        ["--assumptions", str(args.assumptions), "--out", str(args.reports / "phase4")],
    )
    if code != 0:
        print("Phase 4 did not run (see message above).", file=sys.stderr)
        return code
    written["phase4"] = sorted((args.reports / "phase4").iterdir())

    phase4 = json.loads((args.reports / "phase4" / "summary.json").read_text(encoding="utf-8"))
    key_results = {
        "phase3_validation": _phase3_table(args.reports / "phase3" / "validation_summary.csv"),
        "phase4": {k: v for k, v in phase4.items() if k.startswith("4")},
    }
    (args.reports / "key_results.json").write_text(
        json.dumps(key_results, indent=2), encoding="utf-8"
    )

    print("\nOutputs:")
    for phase, paths in written.items():
        for p in paths:
            print(f"  [{phase}] {p}")
    print(f"  [summary] {args.reports / 'key_results.json'}")
    print(
        f"Done in {sum(timings.values()):.1f} s: "
        + ", ".join(f"{k} {v} s" for k, v in timings.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
