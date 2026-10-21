"""Water distribution toolkit Streamlit app.

Run with ``uv run streamlit run app/streamlit_app.py``. The sidebar switches between the project
write-up (the README, rendered by ``views/overview.py``) and the interactive dashboard
(``views/dashboard.py``).
"""

import streamlit as st

st.set_page_config(page_title="Water distribution toolkit", page_icon=":material/water_drop:")
page = st.navigation(
    [
        st.Page(
            "views/overview.py",
            title="Project overview",
            icon=":material/description:",
            default=True,
        ),
        st.Page("views/dashboard.py", title="Interactive dashboard", icon=":material/insights:"),
    ]
)
page.run()
