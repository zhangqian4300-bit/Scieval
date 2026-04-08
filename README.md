# SciEval — 科学数据质量评测平台 Demo

> **数据质量不是数据的固有属性，而是数据相对于特定下游任务的适配度。**

同一份分子数据集，做量子性质预测得 81 分，做药物靶点预测只得 20 分。传统评测给的 93 分掩盖了这一切。

## 核心概念

```
数据质量 = f(数据, 下游任务)
```

传统数据质量评估围绕完整性、一致性等内禀属性打分，脱离使用场景。我们的评测基于**下游任务适配度** — 输出的不是一个分数，而是一张 **Quality Profile（质量画像）**：每个数据集在每个任务上的 Task Fitness Score（TFS）。

## Demo 做了什么

用 **5 个真实的 MoleculeNet 数据集** 和 **5 个下游任务**，通过 RDKit 真实分子分析，生成 5×5 的质量矩阵：

```
             量子性质   毒性预测  药物-靶点  溶解度/ADMET  分子生成
QM9           80.6     20.0     20.0      61.1       61.8
Tox21         20.0     76.1     60.7      71.7       68.9
BACE          20.0     64.8     83.3      64.3       69.4
ESOL          56.5     41.9     20.0      74.5       52.1
FreeSolv      48.4     28.1     18.4      54.2       42.6
```

每一行是同一份数据，分数从 18 到 83 — 传统评测看不到这个。

## 评测方法

两级评测漏斗，从轻到重：

- **L0 元信息匹配**（秒级）— 不碰数据，检查模态、性质类型、规模、化学空间兼容性
- **L1 分布探针**（分钟级）— 用 RDKit 计算分子描述符（MW, logP, TPSA, Morgan FP 等），分析分布匹配度、化学空间覆盖度、scaffold 多样性

## 数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| QM9 | 134K | 极小分子（≤9重原子），量子化学性质 |
| Tox21 | 8K | 多样类药分子，12种毒性标签 |
| BACE | 1.5K | BACE-1 酶抑制剂，窄靶点 |
| ESOL | 1.1K | 水溶性数据 |
| FreeSolv | 642 | 水合自由能 |

## 页面

| 页面 | 功能 |
|------|------|
| 首页 | 核心理念介绍 |
| 数据集浏览 | 5 个数据集的基本信息和分子量分布 |
| 单次评测 | 选数据集 × 任务，查看 Agent 评测思考链和分布图 |
| Quality Matrix | 5×5 TFS 热力图 + 雷达图（Demo 核心页面） |
| 交互式探索 | 化学空间 PCA 地图、描述符分布对比、多维雷达图 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt
# 注意：rdkit-pypi 需要 numpy<2，如有冲突先 pip install "numpy<2"

# 下载数据集
python data/download.py

# 启动
streamlit run app.py
```

浏览器打开 http://localhost:8501

## 推荐体验路径

1. 打开 **Quality Matrix** 页面，看 5×5 热力图，感受同一行颜色差异
2. 进入 **单次评测**，选 **QM9 + 量子性质预测** → TFS=81，再切到 **药物-靶点预测** → TFS=20
3. 打开 **交互式探索**，在化学空间地图中看 QM9 和其他数据集完全不重叠

## 技术栈

- **Streamlit** — Web 界面
- **RDKit** — 分子描述符计算、Morgan 指纹、Murcko scaffold
- **Plotly** — 交互式可视化
- **scikit-learn** — PCA 降维

## 项目结构

```
├── app.py                    # Streamlit 主入口
├── requirements.txt
├── data/
│   └── download.py           # MoleculeNet 数据集下载
├── tasks/
│   └── registry.py           # 5 个任务定义 + 5 个数据集元信息
├── engine/
│   ├── l0_metadata.py        # L0 元信息匹配引擎
│   ├── l1_distribution.py    # L1 分布探针引擎（RDKit）
│   └── profile.py            # TFS 计算与矩阵生成
├── agent/
│   └── trace.py              # Agent 评测思考链生成
└── pages/
    ├── 1_home.py
    ├── 2_dataset_explorer.py
    ├── 3_single_eval.py
    ├── 4_quality_matrix.py
    └── 5_interactive_explore.py
```

## 说明

这是一个概念验证 Demo。数据集和分子分析是真实的，评分框架（任务参考分布、权重、阈值）是设计出来的，尚未用下游任务的真实训练结果校准。Demo 证明的是机制可行性，不是评分的绝对准确性。
