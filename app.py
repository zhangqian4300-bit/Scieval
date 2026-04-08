"""SciEval Demo — Scientific Data Quality Evaluation Platform."""
import sys
import os

# Add demo directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="SciEval — 科学数据质量评测",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation
pages = {
    "首页": "pages/1_home.py",
    "数据集浏览": "pages/2_dataset_explorer.py",
    "单次评测": "pages/3_single_eval.py",
    "Quality Matrix": "pages/4_quality_matrix.py",
    "交互式探索": "pages/5_interactive_explore.py",
}

st.sidebar.title("SciEval Demo")
st.sidebar.markdown("**科学数据质量评测平台**")
st.sidebar.markdown("---")

selection = st.sidebar.radio("导航", list(pages.keys()))

# Run selected page
page_path = os.path.join(os.path.dirname(__file__), pages[selection])
with open(page_path, "r") as f:
    exec(f.read())
