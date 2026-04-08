"""Page 2: Dataset Explorer — browse the 5 datasets."""
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.registry import DATASET_METADATA

st.title("数据集浏览器")
st.markdown("5 个来自 MoleculeNet 的真实分子数据集")

# Check if data is downloaded
data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")

for d_id, meta in DATASET_METADATA.items():
    with st.expander(f"**{meta['name']}** — {meta['n_samples']:,} 分子", expanded=False):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"**描述**：{meta['description']}")
            st.markdown(f"**来源**：{meta['source']}")
            st.markdown(f"**性质类型**：{', '.join(meta['property_types'])}")
            st.markdown(f"**化学空间**：{meta['chemical_space_description']}")
            st.markdown(f"**分子表示**：{', '.join(meta['molecular_representations'])}")

        with col2:
            # Try to load and show MW distribution
            csv_path = os.path.join(data_dir, f"{d_id}.csv")
            if os.path.exists(csv_path):
                try:
                    from rdkit import Chem
                    from rdkit.Chem import Descriptors

                    df = pd.read_csv(csv_path)
                    smiles = df["smiles"].dropna().head(3000).tolist()

                    mws = []
                    for smi in smiles:
                        mol = Chem.MolFromSmiles(str(smi))
                        if mol:
                            mws.append(Descriptors.MolWt(mol))

                    if mws:
                        fig = px.histogram(
                            x=mws, nbins=50,
                            labels={"x": "Molecular Weight", "y": "Count"},
                            title=f"{meta['name']} — 分子量分布",
                        )
                        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("安装 RDKit 后可查看分子量分布")
            else:
                st.warning(f"数据未下载。运行 `python data/download.py`")
