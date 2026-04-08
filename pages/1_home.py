"""Page 1: Home — core concept introduction."""
import streamlit as st

st.title("SciEval — 科学数据质量评测")

st.markdown("""
### 核心理念

> **数据质量不是数据的固有属性，而是数据相对于特定下游任务的适配度。**

传统评测给每个数据集一个分数（如 93 分）。但同一份数据，做不同的事情，质量天差地别：
""")

# The dramatic comparison
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 传统评测")
    st.markdown("""
    ```
    ProtDyn-500K
    完整性  ████████████ 95
    一致性  ████████████ 94
    规范性  ████████████ 91
    ─────────────────────
    综合    ████████████ 93
    ```
    **一个数字，所有人看到相同的结论。**
    """)

with col2:
    st.markdown("#### SciEval Quality Profile")
    st.markdown("""
    ```
    ProtDyn-500K × 不同任务
    CG力场训练    ██████████████████████ 91
    局部构象预测  ████████████████████  90
    突变效应预测  ██████████████████    71
    全局构象预测  ████████████          55
    结合自由能    █████████             38
    折叠路径预测  ███                   12
    ```
    **同一份数据，12 到 91 的真实分布。**
    """)

st.markdown("---")

st.markdown("""
### 这个 Demo 做了什么

我们选取了 **5 个真实的分子数据集**（来自 MoleculeNet）和 **5 个下游任务**，
用 RDKit 进行**真实的分子分析**，展示：

1. **L0 元信息匹配**（秒级）— 不碰数据，只看元信息就能过滤不兼容组合
2. **L1 分布探针**（分钟级）— 用 RDKit 计算分子描述符，分析分布匹配度
3. **Quality Profile**（TFS 矩阵）— 25 个数据集×任务组合的真实评分

所有数值都是**真实计算的**，不是预设的。
""")

st.markdown("---")
st.markdown("**选择左侧导航开始探索** — 推荐直接查看 **Quality Matrix** 页面。")
