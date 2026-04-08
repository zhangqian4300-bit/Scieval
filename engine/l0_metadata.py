"""L0: Metadata Matching Engine — seconds, zero compute.

Checks basic compatibility between a dataset and a task
without touching the actual molecular data.
"""
from tasks.registry import get_task, get_dataset_meta


def _score_modality(meta: dict, task: dict) -> tuple:
    """Check if molecular representations match task needs."""
    # All our datasets have SMILES, all tasks accept SMILES
    # In a real system this would check for 3D coords, graphs, etc.
    if "smiles" in meta["molecular_representations"]:
        return 20, "SMILES 表示可用"
    return 5, "缺少所需分子表示"


def _score_property_type(meta: dict, task: dict) -> tuple:
    """Check if dataset property types match task requirements.
    This is the hard gate — mismatch caps total score.
    """
    req_type = task["requirements"]["property_type"]
    dataset_types = meta["property_types"]

    # Direct match
    if req_type in dataset_types:
        return 20, f"性质类型匹配：数据集包含 {req_type} 类型数据"

    # Generative tasks don't need specific property types
    if req_type == "generative":
        return 15, "生成任务不强制要求特定性质标签"

    # Partial compatibility matrix between property types
    # (from_type, to_type) -> (score, reason)
    partial_compat = {
        ("quantum", "physicochemical"): (10, "量子性质与物化性质有一定关联"),
        ("physicochemical", "quantum"): (8, "物化性质对量子计算任务参考价值有限"),
        ("physicochemical", "toxicity"): (7, "物化性质与毒性有间接关联（Lipinski 规则）"),
        ("toxicity", "binding"): (6, "毒性数据含部分靶点活性信息"),
        ("binding", "toxicity"): (6, "结合数据含部分毒理信息"),
        ("toxicity", "physicochemical"): (5, "毒性数据包含部分物化性质"),
        ("quantum", "toxicity"): (2, "量子性质与毒性关联极弱"),
        ("quantum", "binding"): (2, "量子性质与结合亲和力关联极弱"),
        ("binding", "physicochemical"): (5, "结合数据含部分物化信息"),
        ("physicochemical", "binding"): (4, "物化性质对结合预测参考价值有限"),
        ("binding", "quantum"): (2, "结合数据对量子性质预测无用"),
        ("toxicity", "quantum"): (2, "毒性数据对量子性质预测无用"),
    }

    for dt in dataset_types:
        key = (dt, req_type)
        if key in partial_compat:
            score, note = partial_compat[key]
            return score, f"部分兼容：{note}（数据集为 {dt}，任务需要 {req_type}）"

    # Complete mismatch
    return 3, f"性质类型不匹配：数据集为 {', '.join(dataset_types)}，任务需要 {req_type}"


def _score_scale(meta: dict, task: dict) -> tuple:
    """Check if dataset size meets task minimum."""
    n = meta["n_samples"]
    req = task["requirements"]["min_samples"]

    if n >= req:
        return 20, f"规模充足：{n:,} 样本 >= 任务要求 {req:,}"
    elif n >= req * 0.5:
        ratio = n / req
        score = int(20 * ratio)
        return score, f"规模偏小：{n:,} 样本，任务要求 {req:,}（满足 {ratio:.0%}）"
    else:
        ratio = n / req
        score = max(2, int(20 * ratio))
        return score, f"规模严重不足：{n:,} 样本，任务要求 {req:,}（仅 {ratio:.0%}）"


def _score_chemical_space(meta: dict, task: dict) -> tuple:
    """Check chemical space compatibility based on metadata descriptions."""
    req_space = task["requirements"]["chemical_space"]
    desc = meta["chemical_space_description"]

    # Simple heuristic based on metadata
    if req_space == "small_molecule":
        if "小分子" in desc or "≤9" in desc:
            return 20, "化学空间匹配：数据集覆盖小分子空间"
        elif "类药" in desc or "药物" in desc:
            return 10, "部分兼容：数据集偏向类药分子，任务目标是小分子空间"
        return 8, "化学空间部分重叠"

    if req_space == "drug_like":
        if "类药" in desc or "药物" in desc or "化合物库" in desc:
            return 20, "化学空间匹配：数据集覆盖类药分子空间"
        if "多样" in desc or "有机" in desc:
            return 14, "部分兼容：数据集含有机分子，与类药空间有重叠"
        if "小分子" in desc or "≤9" in desc:
            return 3, "化学空间不匹配：数据集为极小分子，任务需要类药分子"
        return 8, "化学空间兼容性待验证"

    return 10, "化学空间兼容性待验证"


def _score_features(meta: dict, task: dict) -> tuple:
    """Check if required features are available."""
    required = set(task["requirements"]["required_features"])
    available = set(meta["available_features"])

    if not required:
        return 20, "该任务不要求特定特征标签"

    overlap = required & available
    if overlap == required:
        return 20, f"所需特征全部可用：{', '.join(required)}"
    elif overlap:
        return 10, f"部分特征可用：有 {', '.join(overlap)}，缺 {', '.join(required - overlap)}"
    else:
        return 2, f"缺少所需特征：需要 {', '.join(required)}"


def run_l0(dataset_id: str, task_id: str) -> dict:
    """Run L0 metadata matching.

    Returns:
        dict with keys: score, verdict, checks (list of check results)
    """
    meta = get_dataset_meta(dataset_id)
    task = get_task(task_id)

    checks = []
    check_funcs = [
        ("模态匹配", _score_modality),
        ("性质类型匹配", _score_property_type),
        ("规模匹配", _score_scale),
        ("化学空间匹配", _score_chemical_space),
        ("特征可用性", _score_features),
    ]

    total_score = 0
    property_type_score = 0

    for name, func in check_funcs:
        score, reason = func(meta, task)
        checks.append({"name": name, "score": score, "max_score": 20, "reason": reason})
        total_score += score
        if name == "性质类型匹配":
            property_type_score = score

    # Hard gate: only truly incompatible property types cap the score
    if property_type_score < 5:
        total_score = min(total_score, 25)

    # Determine verdict
    if total_score >= 70:
        verdict = "兼容"
    elif total_score >= 40:
        verdict = "部分兼容"
    else:
        verdict = "不兼容"

    return {
        "dataset_id": dataset_id,
        "task_id": task_id,
        "score": total_score,
        "verdict": verdict,
        "checks": checks,
    }
