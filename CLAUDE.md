# CLAUDE.md: project conventions for `wdtk`

`PLAN.md` is the source of truth for scope and phases. This file condenses its rules.

## Workflow
- Work one phase at a time. Start each phase in plan mode by proposing the file layout and
  function signatures. End each phase by stopping, summarizing what was built, reporting test
  results, and waiting for review before starting the next phase.
- Don't add dependencies beyond the PLAN.md tech stack without asking. Add approved deps in the
  phase that first needs them.

## Validation rule (strict)
- **Never invent reference values.** Test targets come only from (a) EPANET results produced
  through WNTR in this repo, or (b) closed-form results computed in the test itself.
- If a textbook or table value is needed, ask the user for it. Don't recall one from memory.
- Tabulated data files carry a `# source:` line citing where the values came from.

## Units
- Internal calculations use SI: m, s, m³/s, m of head, kPa.
- US customary (gpm, MGD, cfs, ft, in, psi) is supported for inputs and reports only.
- All conversion factors live in `src/wdtk/units.py`. Never hard-code a factor elsewhere.
- Variable names carry units where ambiguity is possible: `flow_m3s`, `diameter_m`, `head_m`,
  `pressure_psi`, `density_kgm3`, `temperature_c`.

## Code style
- Every public function has a numpy-style docstring stating the governing equation, the units
  of every argument and return value, and the valid range or assumptions.
- Prefer small pure functions over classes until a class is clearly warranted.
- Functions accept `float` or `NDArray[np.float64]` where vectorization is natural.
- Notebooks (`notebooks/`) are for exploration only. All reusable logic lives in `src/wdtk/`.
- Generated figures and tables go in `reports/`.

## Commands
```
uv sync                   # create/update the environment
uv run pytest             # tests + coverage (fail under 90%)
uv run ruff check         # lint
uv run ruff format        # format
uv run mypy               # strict type check of src/
```
