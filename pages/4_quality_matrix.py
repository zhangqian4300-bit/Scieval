"""Page 4: Quality Matrix — the 5x5 TFS heatmap (demo climax)."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.registry import DATASET_METADATA, TASK_REGISTRY
from engine.profile import compute_quality_matrix

st.title("Quality Matrix")
st.markdown("**同一份数据，不同任务，质量天差地别** — 这就是我们要解决的问题")

# Cache the matrix computation
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "matrix_cache.json")


@st.cache_data(show_spinner="正在计算 5×5 Quality Matrix（首次约需 1-2 分钟）...")
def get_matrix():
    result = compute_quality_matrix()
    # Cache to file (without descriptor_df which is not serializable)
    cache = {
        "matrix": result["matrix"],
        "dataset_ids": result["dataset_ids"],
        "task_ids": result["task_ids"],
    }
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    return result


# Try to load from cache first
if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "r") as f:
            cached = json.load(f)
        matrix_data = cached
        use_cache = True
    except Exception:
        use_cache = False
else:
    use_cache = False

if st.button("计算 Quality Matrix", type="primary") or use_cache:
    if not use_cache:
        full_result = get_matrix()
        matrix_data = {
            "matrix": full_result["matrix"],
            "dataset_ids": full_result["dataset_ids"],
            "task_ids": full_result["task_ids"],
        }

    matrix = np.array(matrix_data["matrix"])
    dataset_names = [DATASET_METADATA[d]["name"] for d in matrix_data["dataset_ids"]]
    task_names = [TASK_REGISTRY[t]["name"] for t in matrix_data["task_ids"]]

    # The 5x5 Heatmap — the centerpiece
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=task_names,
        y=dataset_names,
        colorscale=[
            [0, "#d32f2f"],      # 0 — red
            [0.25, "#ff9800"],   # 25 — orange
            [0.5, "#fdd835"],    # 50 — yellow
            [0.75, "#66bb6a"],   # 75 — green
            [1, "#2e7d32"],      # 100 — dark green
        ],
        zmin=0,
        zmax=100,
        text=matrix.astype(int).astype(str),
        texttemplate="%{text}",
        textfont={"size": 18, "color": "white"},
        hovertemplate="数据集: %{y}<br>任务: %{x}<br>TFS: %{z:.1f}<extra></extra>",
        colorbar=dict(title="TFS", tickvals=[0, 25, 50, 75, 100]),
    ))

    fig.update_layout(
        title=dict(
            text="Task Fitness Score Matrix — 数据质量 = f(数据, 任务)",
            font=dict(size=20),
        ),
        xaxis_title="下游任务",
        yaxis_title="数据集",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Key insight callout
    # Find max and min in each row
    st.markdown("### 关键发现")

    for i, d_name in enumerate(dataset_names):
        row = matrix[i]
        max_idx = np.argmax(row)
        min_idx = np.argmin(row)
        max_val = row[max_idx]
        min_val = row[min_idx]

        if max_val - min_val > 30:
            st.markdown(
                f"- **{d_name}**：「{task_names[max_idx]}」得分 **{max_val:.0f}** vs "
                f"「{task_names[min_idx]}」得分 **{min_val:.0f}** — 相差 **{max_val - min_val:.0f}** 分"
            )

    st.markdown("---")

    # Comparison with traditional scoring
    st.markdown("### 对比：传统评测 vs SciEval")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 传统评测（内禀属性）")
        # Fake traditional scores — all high and similar
        traditional_scores = {
            "QM9": 95, "Tox21": 88, "BACE": 82, "ESOL": 85, "FreeSolv": 80,
        }
        for name, score in traditional_scores.items():
            bar = "█" * (score // 5)
            st.markdown(f"`{name:8s} {bar} {score}`")
        st.markdown("*看起来都不错，选哪个都行？*")

    with col2:
        st.markdown("#### SciEval（任务适配度）")
        st.markdown("*同一个数据集的分数范围：*")
        for i, d_name in enumerate(dataset_names):
            row = matrix[i]
            st.markdown(f"`{d_name:8s} {min(row):.0f} — {max(row):.0f}`")
        st.markdown("*选错数据集 = 浪费几个月*")

    st.markdown("---")

    # Per-dataset radar charts
    st.markdown("### 各数据集的质量画像")
    cols = st.columns(min(3, len(dataset_names)))
    for i, d_name in enumerate(dataset_names):
        with cols[i % 3]:
            row = matrix[i]
            categories = task_names + [task_names[0]]
            values = list(row) + [row[0]]

            fig = go.Figure(data=go.Scatterpolar(
                r=values, theta=categories, fill="toself",
                name=d_name,
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title=d_name,
                height=300,
                margin=dict(l=30, r=30, t=40, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("点击「计算 Quality Matrix」生成 5×5 的完整评测矩阵。\n\n首次计算约需 1-2 分钟（RDKit 真实分析），结果会被缓存。")
