"""Page 3: Single Evaluation — the core interactive evaluation page."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.registry import DATASET_METADATA, TASK_REGISTRY, get_task
from engine.profile import compute_tfs
from agent.trace import generate_trace


st.title("单次评测")
st.markdown("选择一个数据集和一个任务，查看完整的评测过程")

# Selection
col1, col2 = st.columns(2)
with col1:
    dataset_options = {v["name"]: k for k, v in DATASET_METADATA.items()}
    selected_dataset_name = st.selectbox("选择数据集", list(dataset_options.keys()))
    dataset_id = dataset_options[selected_dataset_name]

with col2:
    task_options = {v["name"]: k for k, v in TASK_REGISTRY.items()}
    selected_task_name = st.selectbox("选择下游任务", list(task_options.keys()))
    task_id = task_options[selected_task_name]

st.markdown("---")

# Run evaluation
if st.button("开始评测", type="primary", use_container_width=True):
    # Compute TFS
    with st.spinner("正在计算..."):
        result = compute_tfs(dataset_id, task_id)
        trace_steps = generate_trace(result)

    # Display TFS prominently
    tfs = result["final_tfs"]
    if tfs >= 70:
        color = "green"
    elif tfs >= 40:
        color = "orange"
    else:
        color = "red"

    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 10px; margin: 10px 0;">
        <h1 style="color: {color}; font-size: 72px; margin: 0;">{tfs}</h1>
        <p style="color: #ccc; font-size: 18px;">Task Fitness Score — {result['grade']}</p>
        <p style="color: #999; font-size: 14px;">{selected_dataset_name} × {selected_task_name}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**{result['summary']}**")

    # Agent Trace
    st.markdown("### Agent 评测过程")

    for step in trace_steps:
        status_icon = {"pass": "complete", "warn": "running", "fail": "error", "info": "running"}
        icon_map = {"pass": ":white_check_mark:", "warn": ":warning:", "fail": ":x:", "info": ":mag:"}

        with st.expander(
            f"{icon_map.get(step['status'], '')} 阶段{step['phase']}：{step['title']}",
            expanded=True
        ):
            st.markdown(step["content"])

    # Visualization: Distribution plots (only if L1 has data)
    l1 = result.get("l1_result", {})
    desc_df = l1.get("descriptor_df")

    if desc_df is not None and len(desc_df) > 0:
        st.markdown("### 分布分析可视化")

        task = get_task(task_id)
        ref_dist = task["reference_distribution"]

        # MW distribution comparison
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()

            # Dataset distribution
            fig.add_trace(go.Histogram(
                x=desc_df["mw"], nbinsx=50, name=f"{selected_dataset_name}",
                opacity=0.7, marker_color="#636EFA"
            ))

            # Task reference range
            mw_range = task["requirements"]["mw_range"]
            fig.add_vrect(
                x0=mw_range[0], x1=mw_range[1],
                fillcolor="green", opacity=0.1,
                annotation_text="任务目标范围",
                annotation_position="top left"
            )

            fig.update_layout(
                title="分子量 (MW) 分布",
                xaxis_title="Molecular Weight",
                yaxis_title="Count",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=desc_df["logp"], nbinsx=50, name=f"{selected_dataset_name}",
                opacity=0.7, marker_color="#EF553B"
            ))

            logp_range = task["requirements"]["logp_range"]
            fig.add_vrect(
                x0=logp_range[0], x1=logp_range[1],
                fillcolor="green", opacity=0.1,
                annotation_text="任务目标范围",
                annotation_position="top left"
            )

            fig.update_layout(
                title="logP 分布",
                xaxis_title="logP",
                yaxis_title="Count",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Radar chart of L1 sub-scores
        if "distribution_score" in l1:
            categories = ["分布匹配", "空间覆盖度", "Scaffold多样性"]
            values = [
                l1["distribution_score"],
                l1["coverage_score"],
                l1["diversity_score"],
            ]
            values.append(values[0])  # close the polygon
            categories.append(categories[0])

            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name="L1 Sub-scores"
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="L1 评测子维度",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

elif "last_result" not in st.session_state:
    st.info("点击「开始评测」运行真实的分子分析。\n\n推荐体验：先选 **QM9 + 量子性质预测**，再切换到 **QM9 + 药物-靶点结合预测**，观察分数变化。")
