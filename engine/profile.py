"""Quality Profile generator — combines L0 and L1 into final TFS."""
from engine.l0_metadata import run_l0
from engine.l1_distribution import run_l1
from tasks.registry import TASK_REGISTRY, DATASET_METADATA


def compute_tfs(dataset_id: str, task_id: str) -> dict:
    """Compute Task Fitness Score for a dataset-task pair.

    Returns:
        dict with l0_result, l1_result, final_tfs, and summary
    """
    # Run L0
    l0 = run_l0(dataset_id, task_id)

    # Run L1
    l1 = run_l1(dataset_id, task_id)

    l0_score = l0["score"]
    l1_score = l1.get("score", 50)

    # Composite TFS with graduated L0 gating
    if l0_score < 30:
        # Incompatible: L0 severely limits final score
        # But allow L1 to create some variation (cap at L0_score * 0.8)
        raw = 0.3 * l0_score + 0.7 * l1_score
        cap = max(5, l0_score * 0.8)
        final_tfs = min(cap, raw)
    elif l0_score < 50:
        # Partially compatible: moderate penalty
        final_tfs = 0.35 * l0_score + 0.65 * l1_score
    else:
        final_tfs = 0.3 * l0_score + 0.7 * l1_score

    final_tfs = round(max(0, min(100, final_tfs)), 1)

    # Generate summary
    if final_tfs >= 80:
        grade = "优秀"
        summary = "该数据集非常适合此任务，可直接使用"
    elif final_tfs >= 60:
        grade = "良好"
        summary = "该数据集基本适合此任务，但存在一些不足"
    elif final_tfs >= 40:
        grade = "中等"
        summary = "该数据集与此任务有一定适配度，但存在明显短板"
    elif final_tfs >= 20:
        grade = "较差"
        summary = "该数据集不太适合此任务，需谨慎使用或补充数据"
    else:
        grade = "不适用"
        summary = "该数据集与此任务存在根本性不匹配，不建议使用"

    return {
        "dataset_id": dataset_id,
        "task_id": task_id,
        "l0_result": l0,
        "l1_result": l1,
        "final_tfs": final_tfs,
        "grade": grade,
        "summary": summary,
    }


def compute_quality_matrix() -> dict:
    """Compute the full TFS matrix for all datasets x tasks.

    Returns:
        dict with: matrix (2D list), dataset_ids, task_ids, results (full details)
    """
    dataset_ids = list(DATASET_METADATA.keys())
    task_ids = list(TASK_REGISTRY.keys())

    matrix = []
    results = {}

    for d_id in dataset_ids:
        row = []
        for t_id in task_ids:
            result = compute_tfs(d_id, t_id)
            row.append(result["final_tfs"])
            results[(d_id, t_id)] = result
        matrix.append(row)

    return {
        "matrix": matrix,
        "dataset_ids": dataset_ids,
        "task_ids": task_ids,
        "results": results,
    }
