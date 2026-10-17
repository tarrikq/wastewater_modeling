# Project Plan: Water Distribution System Modeling Toolkit

## Purpose

A portfolio project demonstrating fundamental knowledge of pressurized water distribution hydraulics and the ability to apply it in a professional design workflow. The project has two layers:

1. **Fundamentals layer:** hydraulic calculations and a network solver written from scratch, validated against EPA EPANET.
2. **Engineering layer:** a design and resilience study on a standard benchmark network using WNTR (EPA's Python package for EPANET), structured the way a consulting firm would run a distribution system master plan analysis.

A stormwater extension (SWMM via PySWMM) is scoped as an optional final phase.

The audience is hiring engineers at water resources consulting firms. Every result should be traceable to a governing equation, a stated assumption, or a validation check.

---

## Instructions for Claude Code

- Work one phase at a time. At the end of each phase, stop, summarize what was built, report test results, and wait for review before starting the next phase.
- Use plan mode at the start of each phase to propose the file layout and function signatures before writing code.
- **Do not invent reference values.** Validation targets must come from (a) EPANET results produced through WNTR in this repo, or (b) closed-form results computed in the test itself. If a textbook example value is needed, ask me for it rather than recalling one.
- Every public function gets a docstring stating the governing equation, the units of every argument and return value, and the valid range or assumptions.
- Prefer small pure functions over classes until a class is clearly warranted.
- Keep notebooks for exploration only. All reusable logic lives in the package under `src/`.
- Do not add dependencies beyond those listed below without asking.

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.11+ |
| Environment and packaging | `uv` with `pyproject.toml` |
| Numerics | `numpy`, `scipy` |
| Data | `pandas` |
| Network reference solver | `wntr` (EPANET 2.2 engine) |
| Graph utilities | `networkx` (installed with WNTR) |
| Plotting | `matplotlib`, `plotly` |
| Dashboard | `streamlit` |
| Testing | `pytest`, `pytest-cov`, `hypothesis` (property tests where useful) |
| Lint, format, types | `ruff`, `mypy` |
| CI | GitHub Actions |

---

## Units Convention

- **Internal calculations use SI:** m, s, m³/s, m of head, kPa where pressure is needed.
- **Inputs and reports support US customary**, since most US utility and consulting deliverables use gpm, ft, psi, and inches.
- All unit conversions live in one module (`units.py`) with named constants and tested round trips.
- Variable names carry units where ambiguity is possible: `flow_m3s`, `diameter_m`, `head_m`, `pressure_psi`.
- WNTR uses SI internally, which keeps the validation comparisons clean.

---

## Repository Structure

```
water-distribution-toolkit/
├── README.md                  # technical write-up (see final section)
├── PLAN.md                    # this file
├── CLAUDE.md                  # project conventions for Claude Code
├── pyproject.toml
├── .github/workflows/ci.yml
├── src/wdtk/
│   ├── __init__.py
│   ├── units.py               # conversions and constants
│   ├── fluid.py               # water properties vs temperature
│   ├── headloss.py            # Hazen-Williams, Darcy-Weisbach, minor losses
│   ├── pumps.py               # pump curves, system curves, operating point
│   ├── network/
│   │   ├── model.py           # minimal network data structure
│   │   ├── io.py              # read a subset of EPANET .inp files
│   │   ├── hardy_cross.py     # loop-based teaching solver
│   │   └── gga.py             # global gradient algorithm solver
│   ├── analysis/
│   │   ├── criteria.py        # design criteria and compliance checks
│   │   ├── fire_flow.py
│   │   ├── sizing.py          # pipe sizing heuristic
│   │   ├── water_age.py
│   │   ├── energy.py          # pump energy and cost
│   │   └── criticality.py     # pipe break resilience analysis
│   └── plotting.py
├── data/
│   ├── networks/              # .inp files (Net1, Net3, small test networks)
│   └── assumptions.yaml       # design criteria, unit costs, fire flows
├── tests/
├── notebooks/                 # exploration only
├── reports/                   # generated figures and tables
└── app/streamlit_app.py
```

---

## Phase 0: Scaffold

**Tasks**
- Initialize the repo with `uv`, `pyproject.toml`, `src/` layout, and the package `wdtk`.
- Configure `ruff`, `mypy` (strict on `src/`), and `pytest` with coverage.
- GitHub Actions workflow running lint, type check, and tests on push and PR.
- Create `CLAUDE.md` summarizing the conventions in this plan (units, docstrings, validation rules, phase-by-phase workflow).
- Implement `units.py` and `fluid.py` (density and kinematic viscosity of water as a function of temperature, 0 to 40 °C).

**Acceptance criteria**
- `uv run pytest` passes; CI is green.
- Unit conversion round trips tested for flow (gpm, MGD, cfs, L/s, m³/s), length, pressure (psi, kPa, m and ft of head).

---

## Phase 1: Pipe Hydraulics Fundamentals

**Tasks** (`headloss.py`)
- Velocity and Reynolds number.
- Hazen-Williams head loss, SI and US customary forms, with C-factor input.
- Darcy-Weisbach head loss with friction factor from:
  - Laminar flow (f = 64/Re)
  - Colebrook-White (iterative solve with `scipy.optimize.brentq`)
  - Swamee-Jain (explicit approximation)
- Minor losses from K coefficients.
- Inverse problems: required diameter for a target head loss or velocity, and flow capacity for a given available head.

**Acceptance criteria**
- Swamee-Jain agrees with Colebrook within its documented accuracy over Re 5,000 to 10⁸ and relative roughness 10⁻⁶ to 10⁻² (property test).
- Hazen-Williams SI and US forms agree after unit conversion.
- Inverse functions round-trip with forward functions.
- A short notebook comparing Hazen-Williams and Darcy-Weisbach across diameters and velocities, with a plain-language note on when each is appropriate.

---

## Phase 2: Pumps and System Curves

**Tasks** (`pumps.py`)
- Fit a pump curve from manufacturer-style points (shutoff head, design point, max flow) to H = a − bQᶜ, matching the EPANET single-point and three-point curve conventions.
- System curve: static lift plus friction and minor losses as a function of flow.
- Operating point by root finding.
- Hydraulic and brake power, wire-to-water efficiency.
- Affinity laws for variable speed operation.
- Pumps in series and in parallel (combined curves and operating points).
- NPSH available calculation and a check against a user-supplied NPSH required.

**Acceptance criteria**
- Operating point from the from-scratch solver matches a WNTR single-pump, single-pipe, two-reservoir model within 0.5% in flow and head.
- Affinity-law predictions match the WNTR result for a pump with a speed setting below 1.0.
- A figure showing pump curve, system curve, operating point, and a VFD speed family.

---

## Phase 3: Network Solver

This is the core "fundamentals" deliverable.

**Tasks**
- `network/model.py`: minimal data model for junctions (elevation, demand), reservoirs, pipes (length, diameter, roughness), and optionally a single pump.
- `network/io.py`: read the subset of the EPANET `.inp` format needed for these networks: `[JUNCTIONS]`, `[RESERVOIRS]`, `[PIPES]`, `[PUMPS]`, `[CURVES]`, `[OPTIONS]`. Fail loudly on unsupported sections rather than silently ignoring them.
- `network/hardy_cross.py`: Hardy Cross method for looped networks, as a teaching implementation.
- `network/gga.py`: the global gradient algorithm of Todini and Pilati (1987), which is the method EPANET uses. Sparse matrices via `scipy.sparse`. Steady state, demand-driven, Hazen-Williams first, then Darcy-Weisbach.
- Convergence reporting: iterations, max flow change, max head imbalance.

**Validation**
- Build three test networks: a single loop, a textbook-style two-loop network, and EPANET's Net1 with controls and tanks stripped (steady state, reservoirs only).
- For each, run the same network through WNTR's `EpanetSimulator` with matching head loss formula and demand-driven mode.

**Acceptance criteria**
- GGA heads within 0.01 m and flows within 0.1% of EPANET on all three networks.
- Hardy Cross converges to the same solution as GGA on the looped test networks.
- A figure comparing iteration counts and convergence behavior of Hardy Cross vs GGA.
- A notebook explaining the GGA formulation (energy and continuity equations, Jacobian structure) with the equations written out.

---

## Phase 4: Design and Resilience Study (WNTR)

Run on EPANET's **Net3** benchmark network (source the `.inp` from the WNTR repository's `examples/networks` folder or the EPANET distribution, and record the source in `data/networks/README.md`).

All design criteria, fire flows, and unit costs go in `data/assumptions.yaml` with a comment stating each value is an assumption for demonstration and noting the typical source type (state design standards, Ten States Standards, AWWA manuals, local fire code). I will review these values before results are reported.

**4a. Baseline extended period simulation**
- 72-hour EPS with the network's diurnal demand patterns.
- Pressure, velocity, and tank level time series.
- Compliance checks against criteria in `assumptions.yaml` (minimum and maximum service pressure, maximum velocity).
- Map of minimum pressure by junction and maximum velocity by pipe.

**4b. Fire flow analysis**
- For each eligible junction, apply a fire flow on top of max day demand using pressure-dependent demand.
- Report available fire flow at the minimum residual pressure, and flag junctions that cannot meet the required fire flow.
- Map of available fire flow.

**4c. Pipe sizing**
- Greedy upsizing heuristic: iteratively upsize the pipe with the largest benefit per unit cost from a list of commercial nominal diameters until all criteria are met.
- Output a table of recommended improvements with length, existing and proposed diameter, and planning-level cost.
- Clearly state that this is a heuristic, not a global optimum.

**4d. Water age and tank turnover**
- Run WNTR water quality in AGE mode.
- Report maximum water age by junction and tank turnover time.
- Identify areas of concern and test one operational change (tank level settings or pump schedule) to reduce age.

**4e. Pump energy**
- Energy use and cost over the EPS with a time-of-use tariff in `assumptions.yaml`.
- Compare baseline to one alternative pump schedule.

**4f. Criticality analysis**
- Close each pipe in turn (or each pipe above a diameter threshold) and measure population or demand affected below minimum pressure, using pressure-dependent demand.
- Rank the most critical pipes and map them.

**Acceptance criteria**
- Each sub-analysis is a function in `src/wdtk/analysis/` with a small test on a toy network, plus a script or notebook that runs it on Net3 and writes figures and tables to `reports/`.
- All results reproducible with one command (for example `uv run python -m wdtk.run_study`).

---

## Phase 5: Dashboard and Write-up

**Streamlit app**
- Select network and analysis.
- Interactive network map colored by pressure, velocity, water age, or criticality.
- Pump curve explorer with a speed slider.
- Fire flow lookup by junction.

**README (technical write-up)**
1. Summary of what the project does and why it matters (one paragraph).
2. Governing equations: head loss, pump curves, GGA formulation.
3. Validation results: tables and figures comparing the from-scratch solver to EPANET.
4. Design study results: key findings, recommended improvements, figures.
5. Assumptions and limitations, stated plainly.
6. How to run.

**Acceptance criteria**
- A reviewer can clone the repo, run two commands, and reproduce every figure in the README.

---

## Phase 6 (Optional): Stormwater Extension

Scope separately once Phases 0 through 5 are complete.

- Hand calculations: Rational Method, SCS Curve Number runoff, TR-55 time of concentration, using NOAA Atlas 14 precipitation frequency data for a chosen location.
- PySWMM model of a small urban catchment: subcatchments, conduits, and a detention or LID scenario.
- Compare peak flows and volumes between the hand calculations and SWMM, and between baseline and LID scenarios.

---

## Definition of Done

- All phases through Phase 5 complete with passing tests and green CI.
- From-scratch solver validated against EPANET within the stated tolerances.
- Every assumption is recorded in one place and referenced in the README.
- Figures are labeled with units and readable in print.
- The README reads as a professional technical memo, not a code tour.
