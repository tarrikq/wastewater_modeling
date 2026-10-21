"""Project overview page: the repository README with its figures."""

from pathlib import Path

import streamlit as st

from wdtk import readme

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(layout="centered")
left, right = st.columns(2)
left.page_link(
    "views/dashboard.py", label="Open the interactive dashboard", icon=":material/insights:"
)
right.page_link(readme.REPO_URL, label="Source code on GitHub", icon=":material/code:")

for block in readme.blocks((ROOT / "README.md").read_text(encoding="utf-8")):
    if block.kind == "image":
        st.image(str(ROOT / block.body), caption=block.caption)
    else:
        st.markdown(block.body)
