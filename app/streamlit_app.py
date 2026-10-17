"""Water distribution toolkit dashboard.

Run with ``uv run streamlit run app/streamlit_app.py``. All computation lives in
:mod:`wdtk.dashboard`; this file only arranges widgets.
"""

from pathlib import Path

import streamlit as st

from wdtk import dashboard, units
from wdtk.analysis.assumptions import Assumptions, load_assumptions

ROOT = Path(__file__).resolve().parents[1]
ASSUMPTIONS = ROOT / "data" / "assumptions.yaml"

st.set_page_config(page_title="Water distribution toolkit", layout="wide")


@st.cache_resource(show_spinner="Loading assumptions...")
def get_assumptions() -> Assumptions:
    """Load the study assumptions once per session."""
    return load_assumptions(ASSUMPTIONS)


@st.cache_resource(show_spinner="Running hydraulic, water age, fire flow, and criticality runs...")
def get_results(network: str) -> dashboard.NetworkResults:
    """Run (once per network) every analysis the dashboard shows."""
    return dashboard.compute_results(dashboard.NETWORKS[network], get_assumptions())


a = get_assumptions()
st.title("Water distribution system toolkit")
st.caption(
    "EPANET 2.2 hydraulics through WNTR, with the analysis methods and assumptions of the "
    "Phase 4 design study. Values are in US customary units."
)
network = st.sidebar.selectbox("Network", list(dashboard.NETWORKS), key="network")
drafts = a.drafts()
if drafts:
    st.sidebar.warning(f"{len(drafts)} assumptions are still drafts; results are provisional.")
results = get_results(network)

tab_map, tab_pump, tab_fire, tab_assume = st.tabs(
    ["Network map", "Pump curve explorer", "Fire flow lookup", "Assumptions"]
)

with tab_map:
    layer = st.radio("Layer", list(dashboard.LAYERS), horizontal=True, key="layer")
    st.plotly_chart(dashboard.network_map_figure(results, layer), width="stretch")
    kind = dashboard.LAYERS[layer][0]
    if kind == "node":
        st.caption("Demand junctions only. Hover a junction for its value.")
    else:
        st.caption(
            "Pipes grouped into equal-width classes; hover the middle of a pipe for its value. "
            "Criticality covers pipes at or above the screening diameter."
        )

with tab_pump:
    curves = dashboard.pump_curves(results)
    left, right = st.columns([1, 2])
    with left:
        pump_name = st.selectbox("Pump", list(curves), key="pump")
        curve = curves[pump_name]
        shutoff_ft = float(units.length_from_m(curve.a_m, "ft"))
        speed = st.slider("Speed (fraction of rated)", 0.5, 1.2, 1.0, 0.05, key="speed")
        static = st.slider(
            "Static head (ft)",
            0.0,
            float(round(shutoff_ft * 1.2)),
            float(round(shutoff_ft * 0.4)),
            1.0,
            key="static",
        )
        length = st.slider("Equivalent pipe length (ft)", 100, 50000, 5000, 100)
        diameter = st.select_slider(
            "Pipe diameter (in)", [6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48], 16
        )
        c_factor = st.slider("Hazen-Williams C", 60, 150, 120)
    fig, point = dashboard.pump_explorer(
        curve, speed, static, length, diameter, c_factor, a.energy.pump_efficiency
    )
    with right:
        st.plotly_chart(fig, width="stretch")
        if point.flow_gpm is None:
            st.warning("The pump cannot overcome the static head at this speed: no flow.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Flow", f"{point.flow_gpm:,.0f} gpm")
            m2.metric("Head", f"{point.head_ft:,.1f} ft")
            m3.metric("Hydraulic power", f"{point.hydraulic_hp:,.0f} hp")
            m4.metric(f"Input power at {a.energy.pump_efficiency:.0%}", f"{point.input_hp:,.0f} hp")

with tab_fire:
    fire = results.fire
    st.caption(
        "Max-day steady state, pressure-dependent demand. Available fire flow is the largest "
        "flow that keeps the fire junction and every other demand junction at or above the "
        "residual pressure. Every demand junction is treated as a hydrant location."
    )
    junction = st.selectbox("Junction", list(fire.index), key="junction")
    row = dashboard.fire_lookup(results, junction)
    c1, c2, c3 = st.columns(3)
    c1.metric("Available", f"{row['available_gpm']:,.0f} gpm")
    c2.metric("Required", f"{row['required_gpm']:,.0f} gpm")
    c3.metric("Deficit", f"{row['deficit_gpm']:,.0f} gpm")
    limit = row["governing_limit"]
    where = row["limiting_junction"]
    st.write(f"Governing limit: **{limit}**" + (f" (junction {where})" if where else ""))
    st.dataframe(fire.round(0), width="stretch")

with tab_assume:
    st.caption(
        f"From {ASSUMPTIONS.relative_to(ROOT)}. Every value is an assumption for "
        "demonstration; status shows who supplied it."
    )
    st.dataframe(dashboard.assumptions_table(a), width="stretch")
