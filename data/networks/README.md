# Network files

All `.inp` files here are written by `scripts/build_test_networks.py` using
`wdtk.network.io.write_inp`. They use only the EPANET input subset supported by
`wdtk.network.io` (SI units, `LPS` flow units), so the same file is read by the wdtk solvers and
by EPANET 2.2 through WNTR.

| File | Description | Source |
|---|---|---|
| `single_loop_hw.inp`, `single_loop_dw.inp` | 1 reservoir, feed main, 4-junction loop | Designed for this project (illustrative values) |
| `two_loop_hw.inp`, `two_loop_dw.inp` | 1 reservoir, 2 × 3 junction grid, 2 loops | Designed for this project (illustrative values) |
| `net1_steady_hw.inp` | EPANET Net1 reduced to steady state | `Net1.inp` bundled with WNTR 1.5.0 (`wntr/library/networks/Net1.inp`), originally distributed with EPANET |

## Net1 reduction

- Tank 2 is replaced by a reservoir at its initial head: elevation 850 ft + initial level 120 ft =
  970 ft (295.656 m).
- Junction demands are the base demands (pattern multipliers not applied).
- Controls, rules, patterns, energy, quality, reactions, and time settings are dropped.
- Pump 9 and its single-point head curve are kept.

## Darcy-Weisbach variants

The `_dw` files use the same layout with absolute roughness 0.1 mm, representing cement-lined
ductile iron (value supplied by the project author).

## Phase 4

EPANET Net3 for the design study will be taken from the same WNTR library folder
(`wntr/library/networks/Net3.inp`).
