"""Summarize the fixed five-task TIES experiment; --partial writes current CSVs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


RUN = Path(__file__).resolve().parent
PREVIOUS = RUN.parent / "20261116_urs_finetune_seen5"
TASKS = ["cvrp", "cvrpb", "cvrptw", "ocvrp", "ocvrptw"]
DEFAULT = "ties_k20_l1p0"
PAPER = dict(zip(TASKS, [1.81, 1.46, 6.13, 3.24, 5.07]))
MAIN_SCALES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_record(path, config, job):
    payload = read_json(path)
    rows = payload["rows"]
    assert not payload["missing"] and [row["task"] for row in rows] == TASKS, path
    return {
        "config_id": config["id"], "method": config["method"].replace("_", "-"),
        "keep_percent": config.get("keep_percent"), "scale": config.get("scale"),
        "job_id": job["id"], "phase": job["phase"],
        "node": job.get("node", ""), "gpu": job.get("gpu", ""),
        "gpu_name": job.get("gpu_name", ""),
        "checkpoint": payload["checkpoint"], "source_json": str(path.resolve()),
        "gaps": {row["task"]: row["aug_gap"] for row in rows}, "rows": rows,
    }


def collect_records(configs, partial):
    metadata = {}
    plan_path = RUN / "evaluation_plan.json"
    if plan_path.exists():
        metadata = {job["id"]: job for job in read_json(plan_path)}
    elif not partial:
        raise SystemExit(f"Missing {plan_path}")
    for path in sorted(RUN.glob("state_*.json")):
        for job in read_json(path).get("jobs", []):
            metadata[job["id"]] = {**metadata.get(job["id"], {}), **job}

    old_jobs = {job["name"]: job for job in read_json(PREVIOUS / "queue_state.json")["jobs"]}
    records = []
    for task in ["base"] + TASKS:
        filename = "unified_checkpoint_500.json" if task == "base" else "checkpoint_30.json"
        config = {"id": "base" if task == "base" else f"expert_{task}",
                  "method": "base" if task == "base" else "expert"}
        job = {**old_jobs[task], "id": f"previous_{task}", "phase": "previous",
               "node": "gpu05", "gpu_name": "NVIDIA GeForce RTX 4090"}
        records.append(load_record(PREVIOUS / "evaluation" / task / filename, config, job))

    config_map = {config["id"]: config for config in configs}
    for path in sorted((RUN / "results").glob("*.json")):
        job = metadata[path.stem]
        if job.get("status") != "completed":
            continue
        config_id = job.get("config_id", path.stem)
        if path.stem.startswith("ties_default__"):
            config_id = DEFAULT
        elif path.stem.startswith("base__"):
            config_id = "base"
        config = {"id": "base", "method": "base"} if config_id == "base" else config_map[config_id]
        record = load_record(path, config, job)
        assert record["checkpoint"] == job["checkpoint"], path
        records.append(record)
    return records


def select_records(records):
    selected = {}
    for record in records:
        # Keep calibration and scan runs in the long table. The formal row uses
        # a confirmation if present, otherwise the scan or 4090 calibration.
        priority = {"previous": 0, "calibration": 1, "scan": 2, "confirm": 4}[record["phase"]]
        if record["job_id"].endswith("__4090"):
            priority = 3
        config_id = record["config_id"]
        if config_id not in selected or priority > selected[config_id][0]:
            selected[config_id] = (priority, record)
    return {key: value[1] for key, value in selected.items()}


def summarize(record, base):
    gaps = record["gaps"]
    deltas = {task: gaps[task] - base["gaps"][task] for task in TASKS}
    worst = max(TASKS, key=lambda task: deltas[task])
    summary = {key: value for key, value in record.items() if key not in ["gaps", "rows"]}
    summary.update(mean_gap=mean(gaps.values()), delta_vs_base_pp=mean(deltas.values()),
                   improved_tasks=sum(delta < 0 for delta in deltas.values()),
                   worsened_tasks=sum(delta > 0 for delta in deltas.values()),
                   max_regression_pp=max(0.0, deltas[worst]), worst_task=worst,
                   worst_delta_pp=deltas[worst])
    summary.update({f"{task}_gap": gaps[task] for task in TASKS})
    return summary


def export_csv(records, selected, configs):
    long_rows = []
    for record in records:
        metadata = {key: value for key, value in record.items() if key not in ["gaps", "rows"]}
        for row in record["rows"]:
            long_rows.append({**metadata, "selected": selected[record["config_id"]] is record,
                              "task": row["task"], "instance_size": row["scale"],
                              **{key: value for key, value in row.items() if key not in ["task", "scale"]}})
    write_csv(RUN / "all_results.csv", long_rows)
    summaries = [summarize(selected[config["id"]], selected["base"])
                 for config in configs if config["id"] in selected]
    write_csv(RUN / "summary.csv", summaries)
    return summaries


def plot_results(summaries, base_mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ties = [row for row in summaries if row["method"] == "ties"]
    grid = {(row["keep_percent"], row["scale"]): row["mean_gap"] for row in ties}
    keeps = sorted({row["keep_percent"] for row in ties})
    values = np.array([[grid[keep, scale] for scale in MAIN_SCALES] for keep in keeps])
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 4.7), constrained_layout=True)
    image = left.imshow(values, aspect="auto", cmap="viridis_r")
    left.set(xticks=range(len(MAIN_SCALES)), xticklabels=MAIN_SCALES,
             yticks=range(len(keeps)), yticklabels=keeps,
             xlabel="Scale (lambda)", ylabel="Nominal keep K (%)",
             title="TIES main grid: mean gap (%)")
    midpoint = (values.min() + values.max()) / 2
    for i in range(len(keeps)):
        for j in range(len(MAIN_SCALES)):
            left.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center",
                      color="white" if values[i, j] > midpoint else "black", fontsize=8)
    fig.colorbar(image, ax=left, label="Mean gap (%) - lower is better", shrink=0.85)
    k20 = sorted((row for row in ties if row["keep_percent"] == 20), key=lambda row: row["scale"])
    right.plot([row["scale"] for row in k20], [row["mean_gap"] for row in k20],
               marker="o", markersize=4, label="TIES K=20%")
    right.axhline(base_mean, linestyle="--", color="gray", label="URS base")
    right.axvline(1.0, linestyle=":", color="gray", label="Default lambda=1")
    right.set(xlabel="Scale (lambda)", ylabel="Mean gap (%)",
              title="K=20%: all evaluated scales")
    right.grid(alpha=0.25)
    right.legend()
    for extension in ["png", "pdf"]:
        fig.savefig(RUN / f"heatmap.{extension}", dpi=180)
    plt.close(fig)


def markdown_table(rows, base, include_metrics=True):
    header = ["模型"] + [task.upper() for task in TASKS] + ["均值"]
    if include_metrics:
        header += ["相对 base（百分点）", "改善任务数"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for label, gaps, own_task in rows:
        if label.startswith("论文"):
            cells = [label] + [f"{gaps[task]:.2f}" for task in TASKS]
            cells.append(f"{mean(gaps.values()):.3f}")
            if include_metrics:
                cells += ["—", "—"]
            lines.append("| " + " | ".join(cells) + " |")
            continue
        cells = [label] + [f"**{gaps[task]:.6f}**" if task == own_task else f"{gaps[task]:.6f}" for task in TASKS]
        cells.append(f"{mean(gaps.values()):.6f}")
        if include_metrics:
            deltas = [gaps[task] - base["gaps"][task] for task in TASKS]
            cells += [f"{mean(deltas):+.6f}", f"{sum(delta < 0 for delta in deltas)}/5"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(records, selected, summaries):
    base = selected["base"]
    by_id = {row["config_id"]: row for row in summaries}
    best_ties = min((row for row in summaries if row["method"] == "ties"),
                    key=lambda row: (row["mean_gap"], row["config_id"]))
    best_all_tasks_ties = min(
        (row for row in summaries if row["method"] == "ties" and row["improved_tasks"] == 5),
        key=lambda row: (row["mean_gap"], row["config_id"]), default=None)
    best_ta = min((row for row in summaries if row["method"] in ["task-arithmetic", "average"]),
                  key=lambda row: (row["mean_gap"], row["config_id"]))
    chosen = {"default": by_id[DEFAULT], "best_ties": best_ties,
              "average": by_id["average"], "best_ta": best_ta}
    if best_all_tasks_ties is not None:
        chosen["best_all_tasks_ties"] = best_all_tasks_ties
    diagonal = {task: selected[f"expert_{task}"]["gaps"][task] for task in TASKS}
    previous_base = next(record for record in records if record["job_id"] == "previous_base")
    rows = [("论文 URS（四舍五入）", PAPER, None),
            ("本地 base（微调实验时）", previous_base["gaps"], None),
            ("本地 base（本轮）", base["gaps"], None),
            ("默认 TIES：K=20%，λ=1", selected[DEFAULT]["gaps"], None),
            (f"最佳 TIES：K={best_ties['keep_percent']}%，λ={best_ties['scale']:g}",
             selected[best_ties["config_id"]]["gaps"], None),
            ("五专家等权平均", selected["average"]["gaps"], None),
            (f"最佳 TA：λ={0.2 if best_ta['method'] == 'average' else best_ta['scale']:g}",
             selected[best_ta["config_id"]]["gaps"], None),
            ("对应任务专家（五模型参考）", diagonal, None)]
    all_tasks_note = "本轮扫描未找到五项任务均优于 base 的 TIES 配置。"
    if best_all_tasks_ties is not None:
        candidate = best_all_tasks_ties
        if candidate["config_id"] != best_ties["config_id"]:
            rows.insert(5, (f"五项均改善的最佳 TIES：K={candidate['keep_percent']}%，λ={candidate['scale']:g}",
                            selected[candidate["config_id"]]["gaps"], None))
        all_tasks_note = (
            f"补充展示：五项任务均优于 base 的 TIES 配置中，按平均 gap 最小、平分按配置 ID 选择，"
            f"得到 K={candidate['keep_percent']}%、λ={candidate['scale']:g}，平均 gap 为 "
            f"{candidate['mean_gap']:.6f}%，相对 base 为 {candidate['delta_vs_base_pp']:+.6f} 个百分点。"
            "这仍是同一数据上的探索性结果，不替换预先固定的平均 gap 主选择标准。")
    experts = [(f"{task.upper()} 微调专家", selected[f"expert_{task}"]["gaps"], task) for task in TASKS]
    default = chosen["default"]
    ties_rows = [row for row in summaries if row["method"] == "ties"]
    better_mean = sum(row["delta_vs_base_pp"] < 0 for row in ties_rows)
    better_all = sum(row["improved_tasks"] == 5 for row in ties_rows)
    best_scales = {min((row for row in ties_rows if row["keep_percent"] == keep),
                      key=lambda row: (row["mean_gap"], row["config_id"]))["scale"]
                   for keep in {row["keep_percent"] for row in ties_rows}}
    grid_note = (f"完整网格中，{better_mean}/{len(ties_rows)} 个 TIES 配置的平均 gap 优于 base，"
                 f"其中 {better_all} 个配置五项任务均改善。")
    if len(best_scales) == 1:
        grid_note += f"在本次网格内，六种名义保留比例各自的最佳 λ 均为 {next(iter(best_scales)):g}。"
    text = ["# 五个 URS 专家的 TIES-Merging 结果", "",
            "72 个合并配置均已完成五任务评测：66 个 TIES、1 个等权平均、5 个额外 TA。"
            "所有 gap 单位为 %，越低越好；均值是五任务等权平均，差值单位为百分点。", "",
            markdown_table(rows, base), "", grid_note, "",
            f"默认 TIES 平均 gap 为 {default['mean_gap']:.6f}%，相对本轮 base 为 "
            f"{default['delta_vs_base_pp']:+.6f} 个百分点，改善 {default['improved_tasks']}/5 项任务；"
            f"最坏退化为 {default['max_regression_pp']:.6f} 个百分点。", "",
            f"探索性最佳 TIES 平均 gap 为 {best_ties['mean_gap']:.6f}%，相对 base 为 "
            f"{best_ties['delta_vs_base_pp']:+.6f} 个百分点，改善 {best_ties['improved_tasks']}/5 项任务；"
            f"最坏退化为 {best_ties['max_regression_pp']:.6f} 个百分点。"
            f"相对等权平均的均值差为 {best_ties['mean_gap'] - chosen['average']['mean_gap']:+.6f} 个百分点。", "",
            all_tasks_note, "",
            "五个专家的完整交叉评测如下，粗体为其微调任务。专家对角线来自五个不同模型，不能视为一个多任务模型的结果。", "",
            markdown_table(experts, base, include_metrics=False), "",
            "默认 TIES 是预先固定的免调参设置。最佳 TIES 与最佳 TA 都按同一原始评测数据上的五任务平均 gap 最小选择，"
            "平分时按配置 ID 排序；它们是探索性结果，不是独立测试集上的泛化结论，也不宣称统计显著。"
            "每个合并模型五任务共用同一份权重和参数。", "",
            "沿用 URS 原数据各前 1000 例、N=100、seed=3407、greedy、8 倍增强、batch=125；"
            "CVRPB 保持原 80 个 POMO 起点，其余任务 100 个。"
            "正式表优先采用 4090 confirm；未经 confirm 的配置保留其扫描结果。"
            "所有校准、扫描和复核数值均保留在长表中。", "",
            "TA 的候选包含 λ=0.2 对应的等权平均；二者数学等价，但 FP32 逐项平均与 base+任务增量之和的计算顺序可能导致舍入差异。", "",
            "| 代表结果 | 配置 ID | 采用结果 | GPU |", "| --- | --- | --- | --- |"]
    for label, record in [("base", summarize(base, base))] + list(chosen.items()):
        text.append(f"| {label} | {record['config_id']} | {record['job_id']} | {record['gpu_name']} / {record['gpu']} |")
    text += ["", "![参数扫描](heatmap.png)", ""]
    diagnostics = read_json(RUN / "merge_diagnostics.json")
    k90 = next(row for row in diagnostics["by_keep_percent"] if row["keep_percent"] == 90)
    if all(value == 1.0 for value in k90["actual_threshold_mask_fraction_by_expert"]):
        text += ["参数图中的 K 是名义保留比例：本轮 K=90% 时因阈值为 0 且存在并列，"
                 "官方 `>=` 规则得到的五专家阈值掩码均为 100%（原本为 0 的增量仍为 0），"
                 "因此不能解读为严格保留 90%；详见 [merge_diagnostics.json](merge_diagnostics.json)。", ""]
    verification_path = RUN / "final_verification.json"
    if verification_path.exists():
        verification = read_json(verification_path)
        scanned = read_json(RUN / "scan_summary.json")["best_ties"]
        text += [
            "这里的最佳配置指本次预设网格中按扫描结果选出、再经 4090 复核的配置；"
            "其余候选保留各自扫描硬件的数值。",
            f"主配置扫描均值为 {scanned['mean_gap']:.9f}%，4090 复核为 {best_ties['mean_gap']:.9f}%。"
            f"四个代表配置复核中，单任务 gap 最大差异为 {verification['max_confirmation_gap_difference_pp']:.12f} 个百分点，"
            "来自 K=30%、λ=0.4 的 OCVRPTW；其余代表配置的任务 gap 一致。"
            "最终数值、协议与模型哈希核验见 [final_verification.json](final_verification.json)。", ""]
    text += [
             "完整配置与指标：[summary.csv](summary.csv)；全部原始结果及来源：[all_results.csv](all_results.csv)；"
             "代表配置、模型路径及数值：[summary.json](summary.json)；[PDF 图](heatmap.pdf)。", ""]
    (RUN / "RESULTS.md").write_text("\n".join(text))
    output = {"completed_configs": len(summaries), "selection_metric": "five_task_mean_aug_gap",
              "tie_break": "config_id", "selection_scope": "same original evaluation data; exploratory",
              "base": summarize(base, base), **chosen,
              "best_all_tasks_ties": best_all_tasks_ties, "expert_diagonal_gaps": diagonal}
    (RUN / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    plot_results(summaries, mean(base["gaps"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    configs = read_json(RUN / "configs.json")
    assert len(configs) == 72
    records = collect_records(configs, args.partial)
    selected = select_records(records)
    missing = [config["id"] for config in configs if config["id"] not in selected]
    if not args.partial:
        if missing:
            raise SystemExit(f"Incomplete scan ({72 - len(missing)}/72): {', '.join(missing)}")
        if selected["base"]["job_id"] != "base__4090":
            raise SystemExit("The base__4090 result is required for the full report")
    summaries = export_csv(records, selected, configs)
    print(f"Completed configurations: {len(summaries)}/72; raw model evaluations: {len(records)}")
    if not args.partial:
        write_report(records, selected, summaries)
        print(f"Wrote {RUN / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
