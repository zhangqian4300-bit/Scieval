"""Agent Trace Generator — deterministic templates filled with real computed values."""
from tasks.registry import get_task, get_dataset_meta


def generate_trace(tfs_result: dict) -> list:
    """Generate a step-by-step evaluation trace from TFS computation results.

    Returns a list of trace steps, each with:
        phase, title, content, status (pass/warn/fail)
    """
    d_id = tfs_result["dataset_id"]
    t_id = tfs_result["task_id"]
    task = get_task(t_id)
    meta = get_dataset_meta(d_id)
    l0 = tfs_result["l0_result"]
    l1 = tfs_result["l1_result"]
    final_tfs = tfs_result["final_tfs"]

    steps = []

    # Phase 1: Understand task
    req = task["requirements"]
    steps.append({
        "phase": 1,
        "title": "理解任务",
        "content": (
            f"**目标任务**：{task['name']}（{task['name_en']}）\n\n"
            f"{task['description']}\n\n"
            f"**关键需求**：\n"
            f"- 性质类型：{req['property_type']}\n"
            f"- 化学空间：{req['chemical_space']}（MW {req['mw_range'][0]}-{req['mw_range'][1]}）\n"
            f"- 最低样本数：{req['min_samples']:,}\n"
            f"- Scaffold 多样性 ≥ {req['scaffold_diversity_min']}"
        ),
        "status": "info",
    })

    # Phase 2: L0 Metadata Matching
    l0_lines = []
    for check in l0["checks"]:
        icon = "pass" if check["score"] >= 15 else ("warn" if check["score"] >= 8 else "fail")
        l0_lines.append(f"- **{check['name']}**：{check['score']}/{check['max_score']} — {check['reason']}")

    l0_status = "pass" if l0["score"] >= 70 else ("warn" if l0["score"] >= 40 else "fail")
    steps.append({
        "phase": 2,
        "title": "L0 元信息匹配",
        "content": (
            f"数据集 **{meta['name']}**：{meta['description'][:60]}...\n\n"
            + "\n".join(l0_lines) +
            f"\n\n**L0 判定**：{l0['verdict']}（{l0['score']}/100）"
        ),
        "status": l0_status,
    })

    # If L0 fails hard, short-circuit
    if l0["score"] < 30:
        steps.append({
            "phase": 3,
            "title": "L1 分布分析（跳过）",
            "content": (
                f"L0 判定为**不兼容**（{l0['score']}/100），数据集与任务存在根本性不匹配。\n\n"
                "跳过 L1 分布分析，不值得消耗计算资源。"
            ),
            "status": "fail",
        })
    elif "error" in l1:
        steps.append({
            "phase": 3,
            "title": "L1 分布分析",
            "content": f"分析异常：{l1['error']}",
            "status": "warn",
        })
    else:
        # Phase 3: L1 Distribution Analysis
        kl = l1.get("kl_divergences", {})
        stats = l1.get("descriptor_stats", {})

        l1_lines = []

        # MW distribution
        if "mw" in stats:
            mw_stat = stats["mw"]
            kl_mw = kl.get("mw", "N/A")
            mw_range = req.get("mw_range", (0, 1000))
            l1_lines.append(
                f"- **分子量分布**：均值 {mw_stat['mean']:.0f}，范围 [{mw_stat['min']:.0f}, {mw_stat['max']:.0f}]，"
                f"任务目标 [{mw_range[0]}, {mw_range[1]}]，KL散度 = {kl_mw}"
            )

        # logP
        if "logp" in stats:
            logp_stat = stats["logp"]
            kl_logp = kl.get("logp", "N/A")
            l1_lines.append(
                f"- **logP 分布**：均值 {logp_stat['mean']:.1f}±{logp_stat['std']:.1f}，KL散度 = {kl_logp}"
            )

        # Coverage
        coverage_pct = l1.get("coverage_ratio", 0) * 100
        l1_lines.append(
            f"- **化学空间覆盖度**：{coverage_pct:.1f}% 的分子落在任务目标 MW 范围内"
        )

        # Scaffold diversity
        scaffold_ratio = l1.get("scaffold_ratio", 0)
        n_scaffolds = l1.get("unique_scaffolds", 0)
        l1_lines.append(
            f"- **Scaffold 多样性**：{n_scaffolds} 个独立骨架，多样性比 = {scaffold_ratio:.3f}（要求 ≥ {req['scaffold_diversity_min']}）"
        )

        l1_status = "pass" if l1["score"] >= 70 else ("warn" if l1["score"] >= 40 else "fail")
        steps.append({
            "phase": 3,
            "title": "L1 分布分析",
            "content": (
                f"分析了 **{l1.get('total_molecules_analyzed', '?')}** 个有效分子：\n\n"
                + "\n".join(l1_lines) +
                f"\n\n**分布匹配**：{l1.get('distribution_score', '?')}/100 | "
                f"**覆盖度**：{l1.get('coverage_score', '?')}/100 | "
                f"**多样性**：{l1.get('diversity_score', '?')}/100\n\n"
                f"**L1 综合**：{l1['score']}/100"
            ),
            "status": l1_status,
        })

    # Phase 4: Final verdict
    tfs_status = "pass" if final_tfs >= 70 else ("warn" if final_tfs >= 40 else "fail")
    steps.append({
        "phase": 4,
        "title": "综合研判",
        "content": (
            f"**最终 Task Fitness Score = {final_tfs}**\n\n"
            f"评级：**{tfs_result['grade']}**\n\n"
            f"{tfs_result['summary']}\n\n"
            + _generate_suggestion(tfs_result)
        ),
        "status": tfs_status,
    })

    return steps


def _generate_suggestion(tfs_result: dict) -> str:
    """Generate actionable suggestions based on evaluation results."""
    l0 = tfs_result["l0_result"]
    l1 = tfs_result["l1_result"]
    tfs = tfs_result["final_tfs"]
    task = get_task(tfs_result["task_id"])
    meta = get_dataset_meta(tfs_result["dataset_id"])

    suggestions = []

    if tfs < 20:
        suggestions.append(f"此数据集与「{task['name']}」任务存在根本性错配，建议寻找其他数据集")

    # Check specific issues
    for check in l0["checks"]:
        if check["score"] < 8:
            if "性质类型" in check["name"]:
                suggestions.append(f"数据集缺少 {task['requirements']['property_type']} 类型的标签数据")
            elif "规模" in check["name"]:
                suggestions.append(f"数据量不足，建议补充至 {task['requirements']['min_samples']:,}+ 样本")
            elif "化学空间" in check["name"]:
                suggestions.append("数据集的化学空间与任务目标域不匹配")

    if "error" not in l1:
        coverage = l1.get("coverage_ratio", 1.0)
        if coverage < 0.5:
            suggestions.append(f"仅 {coverage*100:.0f}% 的分子落在目标化学空间内，覆盖度不足")

        scaffold_ratio = l1.get("scaffold_ratio", 1.0)
        if scaffold_ratio < task["requirements"].get("scaffold_diversity_min", 0.1):
            suggestions.append("Scaffold 多样性不足，可能导致模型泛化能力差")

    if not suggestions:
        suggestions.append("该数据集与任务匹配良好，可直接用于模型训练")

    return "**建议**：\n" + "\n".join(f"- {s}" for s in suggestions)
