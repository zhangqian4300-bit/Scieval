"""Task Registry: defines downstream tasks and their requirements."""

TASK_REGISTRY = {
    "quantum_property": {
        "name": "量子性质预测",
        "name_en": "Quantum Property Prediction",
        "description": "预测分子的量子化学性质（HOMO、LUMO、能隙、偶极矩等）",
        "requirements": {
            "property_type": "quantum",
            "min_samples": 10000,
            "chemical_space": "small_molecule",  # MW < 300
            "mw_range": (0, 300),
            "logp_range": (-5, 5),
            "required_features": ["quantum_properties"],
            "scaffold_diversity_min": 0.1,  # low bar — small molecules are fine
        },
        "reference_distribution": {
            # Reference: QM9-like small molecule space
            "mw": {"mean": 120, "std": 30, "range": (30, 250)},
            "logp": {"mean": 0.5, "std": 1.5, "range": (-4, 4)},
            "tpsa": {"mean": 40, "std": 25, "range": (0, 150)},
            "hbd": {"mean": 1.0, "std": 1.0, "range": (0, 5)},
            "hba": {"mean": 2.0, "std": 1.5, "range": (0, 8)},
        },
    },
    "toxicity": {
        "name": "毒性预测",
        "name_en": "Toxicity Prediction",
        "description": "预测分子对生物体的毒性效应（急性毒性、器官毒性、致突变性等）",
        "requirements": {
            "property_type": "toxicity",
            "min_samples": 5000,
            "chemical_space": "drug_like",  # MW 200-600, drug-like
            "mw_range": (150, 700),
            "logp_range": (-2, 7),
            "required_features": ["toxicity_labels"],
            "scaffold_diversity_min": 0.3,  # need diverse scaffolds
        },
        "reference_distribution": {
            # Reference: drug-like molecule space
            "mw": {"mean": 350, "std": 100, "range": (150, 700)},
            "logp": {"mean": 2.5, "std": 1.8, "range": (-2, 7)},
            "tpsa": {"mean": 75, "std": 35, "range": (0, 200)},
            "hbd": {"mean": 2.0, "std": 1.5, "range": (0, 8)},
            "hba": {"mean": 4.5, "std": 2.5, "range": (0, 12)},
        },
    },
    "binding_affinity": {
        "name": "药物-靶点结合预测",
        "name_en": "Drug-Target Binding Prediction",
        "description": "预测小分子药物与蛋白质靶点的结合亲和力",
        "requirements": {
            "property_type": "binding",
            "min_samples": 1000,
            "chemical_space": "drug_like",
            "mw_range": (200, 800),
            "logp_range": (-1, 6),
            "required_features": ["binding_labels"],
            "scaffold_diversity_min": 0.2,
        },
        "reference_distribution": {
            "mw": {"mean": 420, "std": 120, "range": (200, 800)},
            "logp": {"mean": 3.0, "std": 1.5, "range": (-1, 6)},
            "tpsa": {"mean": 80, "std": 40, "range": (0, 250)},
            "hbd": {"mean": 2.0, "std": 1.5, "range": (0, 8)},
            "hba": {"mean": 5.0, "std": 2.5, "range": (0, 12)},
        },
    },
    "solubility_admet": {
        "name": "溶解度/ADMET 预测",
        "name_en": "Solubility / ADMET Prediction",
        "description": "预测分子的溶解度、吸收、分布、代谢、排泄和毒性等药代动力学性质",
        "requirements": {
            "property_type": "physicochemical",
            "min_samples": 500,
            "chemical_space": "drug_like",
            "mw_range": (100, 600),
            "logp_range": (-3, 7),
            "required_features": ["physicochemical_labels"],
            "scaffold_diversity_min": 0.15,
        },
        "reference_distribution": {
            "mw": {"mean": 300, "std": 110, "range": (100, 600)},
            "logp": {"mean": 2.0, "std": 2.0, "range": (-3, 7)},
            "tpsa": {"mean": 70, "std": 35, "range": (0, 200)},
            "hbd": {"mean": 2.0, "std": 1.5, "range": (0, 8)},
            "hba": {"mean": 4.0, "std": 2.5, "range": (0, 12)},
        },
    },
    "molecular_generation": {
        "name": "分子生成",
        "name_en": "Molecular Generation",
        "description": "训练生成模型，生成具有特定性质的新分子结构",
        "requirements": {
            "property_type": "generative",
            "min_samples": 50000,
            "chemical_space": "drug_like",
            "mw_range": (100, 700),
            "logp_range": (-3, 8),
            "required_features": [],  # no specific labels needed
            "scaffold_diversity_min": 0.4,  # high diversity critical
        },
        "reference_distribution": {
            "mw": {"mean": 380, "std": 120, "range": (100, 700)},
            "logp": {"mean": 2.5, "std": 2.0, "range": (-3, 8)},
            "tpsa": {"mean": 70, "std": 40, "range": (0, 250)},
            "hbd": {"mean": 2.0, "std": 1.5, "range": (0, 10)},
            "hba": {"mean": 5.0, "std": 3.0, "range": (0, 15)},
        },
    },
}

# Dataset metadata (pre-defined, not computed from data)
DATASET_METADATA = {
    "qm9": {
        "name": "QM9",
        "description": "~134K 小分子（≤9个重原子）的量子化学性质数据集，包含 DFT 计算的 HOMO、LUMO、能隙等 12 个量子性质",
        "n_samples": 133885,
        "property_types": ["quantum"],
        "available_features": ["quantum_properties"],
        "molecular_representations": ["smiles"],
        "source": "Ramakrishnan et al., 2014",
        "chemical_space_description": "极小分子（C, H, O, N, F），≤9个重原子",
    },
    "tox21": {
        "name": "Tox21",
        "description": "~8K 分子在 12 种毒性测定中的活性数据，来自 Tox21 挑战赛",
        "n_samples": 7831,
        "property_types": ["toxicity"],
        "available_features": ["toxicity_labels"],
        "molecular_representations": ["smiles"],
        "source": "Tox21 Challenge, NIH",
        "chemical_space_description": "多样的类药分子，覆盖环境化学品和药物分子",
    },
    "bace": {
        "name": "BACE",
        "description": "~1.5K 个 BACE-1 酶抑制剂的活性数据，二分类（活性/非活性）",
        "n_samples": 1513,
        "property_types": ["binding"],
        "available_features": ["binding_labels"],
        "molecular_representations": ["smiles"],
        "source": "Subramanian et al., 2016",
        "chemical_space_description": "针对 BACE-1 靶点的化合物库，结构相似度较高",
    },
    "esol": {
        "name": "ESOL",
        "description": "~1.1K 个分子的水溶性数据（log solubility in mol/L）",
        "n_samples": 1128,
        "property_types": ["physicochemical"],
        "available_features": ["physicochemical_labels"],
        "molecular_representations": ["smiles"],
        "source": "Delaney, 2004",
        "chemical_space_description": "多样的有机小分子，覆盖从极性到非极性",
    },
    "freesolv": {
        "name": "FreeSolv",
        "description": "~642 个分子的水合自由能数据（实验值 + 计算值）",
        "n_samples": 642,
        "property_types": ["physicochemical"],
        "available_features": ["physicochemical_labels"],
        "molecular_representations": ["smiles"],
        "source": "Mobley & Guthrie, 2014",
        "chemical_space_description": "有机小分子，分子量和复杂度范围广",
    },
}


def get_task(task_id: str) -> dict:
    return TASK_REGISTRY[task_id]


def get_dataset_meta(dataset_id: str) -> dict:
    return DATASET_METADATA[dataset_id]


def list_tasks() -> list:
    return [(k, v["name"]) for k, v in TASK_REGISTRY.items()]


def list_datasets() -> list:
    return [(k, v["name"]) for k, v in DATASET_METADATA.items()]
