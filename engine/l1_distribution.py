"""L1: Distribution Probe Engine — minutes, light compute.

Uses RDKit to compute molecular descriptors and analyze
distribution alignment between dataset and task requirements.
"""
import os
import numpy as np
import pandas as pd
from functools import lru_cache

from tasks.registry import get_task, get_dataset_meta

# Lazy import RDKit to allow importing this module even if rdkit is not installed
_rdkit_available = None


def _check_rdkit():
    global _rdkit_available
    if _rdkit_available is None:
        try:
            from rdkit import Chem
            _rdkit_available = True
        except ImportError:
            _rdkit_available = False
    return _rdkit_available


def _compute_descriptors(smiles_list: list) -> pd.DataFrame:
    """Compute molecular descriptors for a list of SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        records.append({
            "smiles": smi,
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotbonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
        })
    return pd.DataFrame(records)


def _compute_fingerprints(smiles_list: list, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprints (radius=2) for a list of SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        fps.append(np.array(fp))
    return np.array(fps) if fps else np.array([]).reshape(0, n_bits)


def _compute_scaffolds(smiles_list: list) -> list:
    """Extract Murcko scaffolds."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(core))
        except Exception:
            scaffolds.append("")
    return scaffolds


def _kl_divergence(p_values: np.ndarray, ref_mean: float, ref_std: float,
                   n_bins: int = 50) -> float:
    """Compute KL divergence between empirical distribution and reference Gaussian."""
    if len(p_values) < 10 or ref_std <= 0:
        return 10.0  # high divergence as fallback

    # Create histogram of data
    p_min = min(p_values.min(), ref_mean - 4 * ref_std)
    p_max = max(p_values.max(), ref_mean + 4 * ref_std)
    bins = np.linspace(p_min, p_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    hist, _ = np.histogram(p_values, bins=bins, density=True)
    hist = hist + 1e-10  # avoid zeros
    hist = hist / hist.sum()

    # Reference Gaussian
    ref = np.exp(-0.5 * ((bin_centers - ref_mean) / ref_std) ** 2)
    ref = ref + 1e-10
    ref = ref / ref.sum()

    # KL divergence
    kl = np.sum(hist * np.log(hist / ref))
    return max(0, kl)


def _coverage_score(dataset_fps: np.ndarray, ref_mean: float, ref_std: float,
                    mw_values: np.ndarray, mw_range: tuple) -> float:
    """Estimate how well dataset covers the task's target chemical space."""
    if len(mw_values) == 0:
        return 0.0

    # What fraction of dataset falls within the task's MW range
    in_range = np.sum((mw_values >= mw_range[0]) & (mw_values <= mw_range[1]))
    coverage = in_range / len(mw_values)
    return coverage


def _load_dataset_smiles(dataset_id: str, max_samples: int = 5000) -> list:
    """Load SMILES from a downloaded dataset CSV."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "datasets")
    csv_path = os.path.join(data_dir, f"{dataset_id}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}. Run data/download.py first.")

    df = pd.read_csv(csv_path)
    smiles = df["smiles"].dropna().tolist()

    # Sample if too large (QM9 has 134K)
    if len(smiles) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(smiles), max_samples, replace=False)
        smiles = [smiles[i] for i in indices]

    return smiles


def run_l1(dataset_id: str, task_id: str) -> dict:
    """Run L1 distribution probes.

    Returns:
        dict with: score, descriptor_stats, kl_divergences, coverage,
                   scaffold_diversity, descriptor_df (for visualization)
    """
    if not _check_rdkit():
        return {
            "dataset_id": dataset_id,
            "task_id": task_id,
            "score": 50,
            "error": "RDKit not installed, using fallback scores",
        }

    task = get_task(task_id)
    meta = get_dataset_meta(dataset_id)
    ref_dist = task["reference_distribution"]
    req = task["requirements"]

    # Load data
    smiles = _load_dataset_smiles(dataset_id)

    # 1. Compute descriptors
    desc_df = _compute_descriptors(smiles)
    if len(desc_df) < 10:
        return {"dataset_id": dataset_id, "task_id": task_id, "score": 5,
                "error": "Too few valid molecules"}

    # 2. KL divergences for each descriptor
    kl_results = {}
    descriptor_keys = ["mw", "logp", "tpsa", "hbd", "hba"]
    for key in descriptor_keys:
        if key in ref_dist and key in desc_df.columns:
            kl = _kl_divergence(
                desc_df[key].values,
                ref_dist[key]["mean"],
                ref_dist[key]["std"],
            )
            kl_results[key] = round(kl, 4)

    # 3. Distribution match score (lower KL = better match)
    # Normalize KL to 0-100 score. KL < 0.1 = excellent, KL > 2.0 = terrible
    kl_scores = []
    for key, kl in kl_results.items():
        if kl < 0.05:
            s = 100
        elif kl < 0.2:
            s = 90 - (kl - 0.05) * 200
        elif kl < 0.5:
            s = 60 - (kl - 0.2) * 100
        elif kl < 1.0:
            s = 30 - (kl - 0.5) * 40
        elif kl < 2.0:
            s = 10 - (kl - 1.0) * 8
        else:
            s = max(0, 2)
        kl_scores.append(max(0, min(100, s)))

    distribution_score = np.mean(kl_scores) if kl_scores else 50

    # 4. Chemical space coverage
    mw_range = req.get("mw_range", (0, 1000))
    coverage = _coverage_score(None, 0, 0, desc_df["mw"].values, mw_range)
    coverage_score = coverage * 100

    # 5. Scaffold diversity
    scaffolds = _compute_scaffolds(smiles[:2000])  # limit for speed
    unique_scaffolds = len(set(s for s in scaffolds if s))
    total = max(len(scaffolds), 1)
    scaffold_ratio = unique_scaffolds / total

    diversity_req = req.get("scaffold_diversity_min", 0.1)
    if scaffold_ratio >= diversity_req:
        diversity_score = min(100, (scaffold_ratio / diversity_req) * 70)
    else:
        diversity_score = (scaffold_ratio / diversity_req) * 50

    # 6. Composite L1 score
    l1_score = (
        distribution_score * 0.40 +
        coverage_score * 0.30 +
        diversity_score * 0.30
    )
    l1_score = max(0, min(100, l1_score))

    # Descriptor statistics for visualization
    desc_stats = {}
    for key in descriptor_keys:
        if key in desc_df.columns:
            vals = desc_df[key]
            desc_stats[key] = {
                "mean": round(vals.mean(), 2),
                "std": round(vals.std(), 2),
                "min": round(vals.min(), 2),
                "max": round(vals.max(), 2),
                "median": round(vals.median(), 2),
            }

    return {
        "dataset_id": dataset_id,
        "task_id": task_id,
        "score": round(l1_score, 1),
        "distribution_score": round(distribution_score, 1),
        "coverage_score": round(coverage_score, 1),
        "coverage_ratio": round(coverage, 3),
        "diversity_score": round(diversity_score, 1),
        "scaffold_ratio": round(scaffold_ratio, 4),
        "unique_scaffolds": unique_scaffolds,
        "total_molecules_analyzed": len(desc_df),
        "kl_divergences": kl_results,
        "descriptor_stats": desc_stats,
        "descriptor_df": desc_df,  # for plotting
    }
