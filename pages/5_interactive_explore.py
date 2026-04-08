"""Page 5: Interactive Exploration — chemical space map + distribution comparison."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.registry import DATASET_METADATA, TASK_REGISTRY, get_task

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")

DATASET_COLORS = {
    "QM9": "#636EFA",
    "Tox21": "#EF553B",
    "BACE": "#00CC96",
    "ESOL": "#AB63FA",
    "FreeSolv": "#FFA15A",
}

DESCRIPTOR_LABELS = {
    "mw": "分子量 (MW)",
    "logp": "logP (脂水分配系数)",
    "tpsa": "TPSA (拓扑极性表面积)",
    "hbd": "氢键供体数 (HBD)",
    "hba": "氢键受体数 (HBA)",
    "rotbonds": "可旋转键数",
    "rings": "环数",
    "heavy_atoms": "重原子数",
}


@st.cache_data(show_spinner="正在用 RDKit 计算分子描述符...")
def load_all_descriptors(max_per_dataset=2000):
    """Load and compute descriptors for all datasets."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    all_data = {}
    for d_id, meta in DATASET_METADATA.items():
        csv_path = os.path.join(DATA_DIR, f"{d_id}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        smiles_list = df["smiles"].dropna().tolist()

        # Sample if too large
        if len(smiles_list) > max_per_dataset:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(smiles_list), max_per_dataset, replace=False)
            smiles_list = [smiles_list[i] for i in indices]

        records = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            records.append({
                "smiles": smi,
                "dataset": meta["name"],
                "mw": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
                "rotbonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "rings": rdMolDescriptors.CalcNumRings(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            })
        all_data[meta["name"]] = pd.DataFrame(records)

    return all_data


@st.cache_data(show_spinner="正在计算化学空间 PCA 投影...")
def compute_pca_projection(max_per_dataset=1000):
    """Compute PCA of Morgan fingerprints for chemical space visualization."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.decomposition import PCA

    all_fps = []
    all_labels = []

    for d_id, meta in DATASET_METADATA.items():
        csv_path = os.path.join(DATA_DIR, f"{d_id}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        smiles_list = df["smiles"].dropna().tolist()

        if len(smiles_list) > max_per_dataset:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(smiles_list), max_per_dataset, replace=False)
            smiles_list = [smiles_list[i] for i in indices]

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            all_fps.append(np.array(fp))
            all_labels.append(meta["name"])

    if not all_fps:
        return None

    X = np.array(all_fps)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    result = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "dataset": all_labels,
    })
    variance = pca.explained_variance_ratio_
    return result, variance


# ============ Page Start ============

st.title("交互式探索")
st.markdown("从化学空间和分子描述符两个角度，直观理解数据集之间的差异")

# Dataset selector
all_dataset_names = [meta["name"] for meta in DATASET_METADATA.values()]
selected_datasets = st.multiselect(
    "选择要对比的数据集",
    all_dataset_names,
    default=all_dataset_names,
)

if not selected_datasets:
    st.warning("请至少选择一个数据集")
    st.stop()

# Task context selector
task_options = {"无（仅看数据集分布）": None}
task_options.update({v["name"]: k for k, v in TASK_REGISTRY.items()})
selected_task_name = st.selectbox("叠加任务目标范围（可选）", list(task_options.keys()))
selected_task_id = task_options[selected_task_name]

st.markdown("---")

# ============ Tab Layout ============
tab1, tab2, tab3 = st.tabs(["化学空间地图", "分布对比", "数据集概览"])

# ============ Tab 1: Chemical Space Map ============
with tab1:
    st.markdown("### 化学空间 PCA 投影")
    st.markdown("基于 Morgan 指纹（radius=2, 1024 bits）的 PCA 降维。每个点代表一个分子，颜色代表来源数据集。")

    pca_result = compute_pca_projection()
    if pca_result is not None:
        pca_df, variance = pca_result

        # Filter to selected datasets
        plot_df = pca_df[pca_df["dataset"].isin(selected_datasets)]

        fig = go.Figure()
        for ds_name in selected_datasets:
            subset = plot_df[plot_df["dataset"] == ds_name]
            fig.add_trace(go.Scattergl(
                x=subset["PC1"],
                y=subset["PC2"],
                mode="markers",
                name=ds_name,
                marker=dict(
                    size=3,
                    color=DATASET_COLORS.get(ds_name, "#999"),
                    opacity=0.5,
                ),
                hovertemplate=f"{ds_name}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
            ))

        fig.update_layout(
            xaxis_title=f"PC1 ({variance[0]:.1%} variance)",
            yaxis_title=f"PC2 ({variance[1]:.1%} variance)",
            height=600,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Insights
        if "QM9" in selected_datasets and len(selected_datasets) > 1:
            st.info("注意 QM9（蓝色）形成了一个独立的聚团，几乎不与其他数据集重叠——这就是为什么它在药物相关任务上得分极低。")
    else:
        st.warning("数据未加载，请先运行 `python data/download.py`")


# ============ Tab 2: Distribution Comparison ============
with tab2:
    st.markdown("### 分子描述符分布对比")

    # Descriptor selector
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_desc = st.radio(
            "选择描述符",
            list(DESCRIPTOR_LABELS.keys()),
            format_func=lambda x: DESCRIPTOR_LABELS[x],
        )

    with col2:
        all_data = load_all_descriptors()

        fig = go.Figure()

        for ds_name in selected_datasets:
            if ds_name not in all_data:
                continue
            df = all_data[ds_name]
            if selected_desc not in df.columns:
                continue

            values = df[selected_desc].dropna()
            fig.add_trace(go.Histogram(
                x=values,
                name=ds_name,
                opacity=0.6,
                marker_color=DATASET_COLORS.get(ds_name, "#999"),
                nbinsx=60,
            ))

        # Overlay task reference range if selected
        if selected_task_id:
            task = get_task(selected_task_id)
            req = task["requirements"]
            ref = task["reference_distribution"]

            # MW and logP ranges from requirements
            range_map = {
                "mw": req.get("mw_range"),
                "logp": req.get("logp_range"),
            }

            if selected_desc in range_map and range_map[selected_desc]:
                r = range_map[selected_desc]
                fig.add_vrect(
                    x0=r[0], x1=r[1],
                    fillcolor="green", opacity=0.08,
                    line=dict(color="green", width=2, dash="dash"),
                    annotation_text=f"任务目标范围\n({selected_task_name})",
                    annotation_position="top left",
                    annotation_font_size=11,
                )

            # Reference mean line
            if selected_desc in ref:
                ref_mean = ref[selected_desc]["mean"]
                fig.add_vline(
                    x=ref_mean,
                    line=dict(color="red", width=2, dash="dot"),
                    annotation_text=f"任务参考均值 {ref_mean}",
                    annotation_position="top right",
                    annotation_font_size=11,
                )

        fig.update_layout(
            barmode="overlay",
            xaxis_title=DESCRIPTOR_LABELS[selected_desc],
            yaxis_title="Count",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Dynamic insight based on selection
        if selected_task_id and selected_desc == "mw":
            task = get_task(selected_task_id)
            mw_range = task["requirements"].get("mw_range", (0, 1000))
            st.markdown(f"**任务「{selected_task_name}」要求分子量在 {mw_range[0]}-{mw_range[1]} 范围内。**"
                        "绿色阴影区域是目标范围，落在范围外的分子对该任务没有价值。")

    # Stats table
    st.markdown("#### 描述符统计对比")
    stats_rows = []
    for ds_name in selected_datasets:
        if ds_name not in all_data:
            continue
        df = all_data[ds_name]
        if selected_desc not in df.columns:
            continue
        vals = df[selected_desc]
        stats_rows.append({
            "数据集": ds_name,
            "均值": f"{vals.mean():.1f}",
            "标准差": f"{vals.std():.1f}",
            "最小值": f"{vals.min():.1f}",
            "中位数": f"{vals.median():.1f}",
            "最大值": f"{vals.max():.1f}",
        })
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)


# ============ Tab 3: Dataset Overview ============
with tab3:
    st.markdown("### 数据集基础信息对比")

    all_data = load_all_descriptors()

    # Build comparison table
    overview_rows = []
    for ds_name in selected_datasets:
        if ds_name not in all_data:
            continue
        df = all_data[ds_name]
        # Find metadata
        d_id = [k for k, v in DATASET_METADATA.items() if v["name"] == ds_name][0]
        meta = DATASET_METADATA[d_id]

        overview_rows.append({
            "数据集": ds_name,
            "总样本数": f"{meta['n_samples']:,}",
            "性质类型": ", ".join(meta["property_types"]),
            "MW 均值": f"{df['mw'].mean():.0f}",
            "MW 范围": f"{df['mw'].min():.0f}-{df['mw'].max():.0f}",
            "logP 均值": f"{df['logp'].mean():.1f}",
            "平均重原子数": f"{df['heavy_atoms'].mean():.1f}",
            "平均环数": f"{df['rings'].mean():.1f}",
        })

    if overview_rows:
        st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

    # Radar chart comparing all selected datasets on normalized descriptors
    st.markdown("### 多维描述符雷达图")

    desc_keys = ["mw", "logp", "tpsa", "hbd", "hba", "rotbonds", "rings"]
    desc_labels = [DESCRIPTOR_LABELS[k] for k in desc_keys]

    # Normalize each descriptor to 0-100 across all datasets
    all_means = {}
    for ds_name in selected_datasets:
        if ds_name not in all_data:
            continue
        df = all_data[ds_name]
        all_means[ds_name] = {k: df[k].mean() for k in desc_keys if k in df.columns}

    if all_means:
        # Get global min/max for normalization
        global_stats = {}
        for k in desc_keys:
            vals = [m[k] for m in all_means.values() if k in m]
            if vals:
                global_stats[k] = (min(vals), max(vals))

        fig = go.Figure()
        for ds_name in selected_datasets:
            if ds_name not in all_means:
                continue
            means = all_means[ds_name]
            normalized = []
            for k in desc_keys:
                if k in means and k in global_stats:
                    lo, hi = global_stats[k]
                    if hi > lo:
                        normalized.append((means[k] - lo) / (hi - lo) * 100)
                    else:
                        normalized.append(50)
                else:
                    normalized.append(0)
            normalized.append(normalized[0])  # close polygon

            fig.add_trace(go.Scatterpolar(
                r=normalized,
                theta=desc_labels + [desc_labels[0]],
                name=ds_name,
                fill="toself",
                opacity=0.3,
                line=dict(color=DATASET_COLORS.get(ds_name, "#999")),
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=500,
            legend=dict(yanchor="top", y=1.1, xanchor="center", x=0.5, orientation="h"),
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("*数值已归一化到 0-100（跨数据集相对比较）。形状差异越大，说明数据集的化学空间特征越不同。*")
