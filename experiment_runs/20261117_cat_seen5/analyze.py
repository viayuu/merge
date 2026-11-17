"""Summarize CAT runs and repeated exemplar selections against the old URS base."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from run_queue import TASKS, validate_result

RUN = Path(__file__).resolve().parent
TIES = RUN.parent / '20261117_ties_seen5'


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_id(job):
    value = job.get('config_id', job.get('config', job['id']))
    return json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value)


def load_record(job, path):
    payload = validate_result(path)
    assert Path(payload['checkpoint']).resolve() == Path(job['checkpoint']).resolve(), path
    return {**job, 'config_id': group_id(job), 'source_json': str(path),
            'gaps': {row['task']: row['aug_gap'] for row in payload['rows']},
            'rows': payload['rows']}


def summarize(records, base, expected):
    first = records[0]
    gaps = {task: mean(record['gaps'][task] for record in records) for task in TASKS}
    deltas = {task: gaps[task] - base['gaps'][task] for task in TASKS}
    run_means = [mean(record['gaps'].values()) for record in records]
    row = {
        'config_id': first['config_id'], 'family': first.get('family', 'unknown'),
        'label': first.get('label', first['config_id']),
        'is_default': first.get('is_default', False),
        'completed_repeats': len(records), 'planned_repeats': expected,
        'seeds': ','.join(str(record.get('seed', '')) for record in records),
        'mean_gap': mean(run_means), 'std_gap': pstdev(run_means),
        'delta_vs_base_pp': mean(deltas.values()),
        'improved_tasks': sum(delta < 0 for delta in deltas.values()),
        'max_regression_pp': max(0.0, max(deltas.values())),
    }
    for task in TASKS:
        row[f'{task}_gap'] = gaps[task]
        row[f'{task}_std'] = pstdev(record['gaps'][task] for record in records)
        row[f'{task}_delta_pp'] = deltas[task]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    plan = json.loads((RUN / 'evaluation_plan.json').read_text())
    state_path = RUN / 'queue_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs': []}
    statuses = {job['id']: job for job in state['jobs']}
    records, pending = [], []
    for entry in plan:
        job = {**entry, **statuses.get(entry['id'], {})}
        path = RUN / 'results' / f"{entry['id']}.json"
        if job.get('status') == 'completed':
            records.append(load_record(job, path))
        else:
            pending.append(entry['id'])
    if pending and not args.partial:
        raise SystemExit(f'{len(pending)} jobs unfinished; use --partial for interim results.')

    long_rows = []
    for record in records:
        for row in record['rows']:
            long_rows.append({
                'job_id': record['id'], 'config_id': record['config_id'],
                'family': record.get('family', 'unknown'), 'seed': record.get('seed', ''),
                'gpu': record.get('gpu', ''), 'gpu_name': record.get('gpu_name', ''),
                'checkpoint': record['checkpoint'], 'source_json': record['source_json'],
                'task': row['task'], 'episodes': row['episodes'],
                'aug_gap': row['aug_gap'], 'no_aug_gap': row['no_aug_gap'],
                'aug_score': row['aug_score'], 'oracle_score': row['oracle_score'],
            })
    write_csv(RUN / 'all_results.csv', long_rows)
    base = next((record for record in records if record['id'] == 'base'), None)
    if base is None:
        (RUN / 'summary.json').write_text(json.dumps({
            'completed': len(records), 'total': len(plan), 'pending': pending,
            'note': 'Waiting for the current RTX 3090 base evaluation.'}, indent=2) + '\n')
        print(f'{len(records)}/{len(plan)} complete; waiting for base.')
        return

    expected = defaultdict(int)
    for job in plan:
        expected[group_id(job)] += 1
    groups = defaultdict(list)
    for record in records:
        groups[record['config_id']].append(record)
    summaries = [summarize(group, base, expected[key]) for key, group in groups.items()]
    summaries.sort(key=lambda row: (row['family'], row['mean_gap'], row['config_id']))
    write_csv(RUN / 'summary.csv', summaries)

    historical = []
    for name, filename in [
        ('历史等权平均', 'confirm_average.json'),
        ('历史 TIES 默认 K20, λ1', 'confirm_ties_k20_l1p0.json'),
        ('历史 TIES 探索最优 K30, λ0.4', 'confirm_ties_k30_l0p4.json'),
    ]:
        path = TIES / 'results' / filename
        if path.exists():
            payload = validate_result(path)
            historical.append({'label': name, 'source_json': str(path),
                               'gaps': {row['task']: row['aug_gap'] for row in payload['rows']}})

    selection = {}
    for row in summaries:
        family = row['family']
        if family in ['base', 'control', 'task-arithmetic', 'average']:
            continue
        if row['completed_repeats'] != row['planned_repeats']:
            continue
        if family not in selection or row['mean_gap'] < selection[family]['mean_gap']:
            selection[family] = row
    result = {
        'completed': len(records), 'total': len(plan), 'pending': pending,
        'base': {'checkpoint': base['checkpoint'], 'source_json': base['source_json'],
                 'gaps': base['gaps'], 'mean_gap': mean(base['gaps'].values())},
        'groups': summaries, 'exploratory_best_by_family': selection,
        'historical_references': historical,
        'standard_deviation': 'population std across exemplar-selection repeats (ddof=0)',
        'selection_note': 'All configurations use the same original evaluation instances; best settings are exploratory, not held-out estimates.',
        'runs': [{key: value for key, value in record.items() if key != 'rows'} for record in records],
    }
    (RUN / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')

    lines = ['# URS 五任务 CAT Merging 实验结果', '',
             f"完成 **{len(records)}/{len(plan)}** 个评测。主对照为原始第 500 epoch URS 多任务模型，未使用续训模型。", '',
             '指标为原 URS 测试集的增强后 gap（%，越低越好），五任务等权平均。每任务原 1000 个 N100 实例，greedy、POMO 多起点、8 倍增强、batch 125。', '',
             '同一配置的重复实验只改变 CAT 特征取样；表中为重复均值与总体标准差（ddof=0），不会挑选最有利的随机种子。', '',
             '| 配置 | 重复数 | CVRP | CVRPB | CVRPTW | OCVRP | OCVRPTW | 平均 gap ± std | 较 base（百分点） | 改善任务数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    ordered = sorted(summaries, key=lambda row: (
        row['config_id'] != group_id(base), not row['is_default'], row['family'], row['mean_gap']))
    for row in ordered:
        label = row['label'] + ('（预设配置）' if row['is_default'] else '')
        values = ' | '.join(f"{row[f'{task}_gap']:.6f}" for task in TASKS)
        lines.append(f"| {label} | {row['completed_repeats']}/{row['planned_repeats']} | {values} | "
                     f"{row['mean_gap']:.6f} ± {row['std_gap']:.6f} | {row['delta_vs_base_pp']:+.6f} | {row['improved_tasks']}/5 |")
    lines += ['', '## 同数据探索性选择', '',
              '以下按同一配置全部重复的平均 gap 选择，并仅考虑重复已完成的配置。原数据同时用于比较配置，最优值属于同数据探索性结果。', '']
    for family, row in selection.items():
        lines.append(f"- {family}：`{row['config_id']}`，平均 gap **{row['mean_gap']:.6f}%**，较 base **{row['delta_vs_base_pp']:+.6f}** 个百分点。")
    best = selection.get('paper_full')
    if best:
        lines += ['', '论文公式完整 CAT 最优配置的三个固定模型（对应三次取样；未挑选最优seed）：', '']
        for record in records:
            if record['config_id'] == best['config_id']:
                checkpoint = Path(record['checkpoint']).relative_to(RUN)
                lines.append(f"- seed {record['seed']}：[模型]({checkpoint})，"
                             f"单次平均 gap {mean(record['gaps'].values()):.6f}%。")
    if historical:
        lines += ['', '## 历史结果参考', '',
                  '以下是此前同一数据/评测协议的 4090 复核结果，本轮没有重新评测这些模型；本轮 CAT 与 base 均在 3090 运行。', '',
                  '| 模型 | CVRP | CVRPB | CVRPTW | OCVRP | OCVRPTW | 平均 gap |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for item in historical:
            values = ' | '.join(f"{item['gaps'][task]:.6f}" for task in TASKS)
            lines.append(f"| {item['label']} | {values} | {mean(item['gaps'].values()):.6f} |")
    lines += ['', '## 图表与复现', '',
              '[合并系数曲线](alpha_sensitivity.png) · [预设配置的逐任务变化](default_task_deltas.png)。图在全量评测后由 plot_results.py 生成，另有同名PDF。', '',
              '官方发布代码版本与论文对齐版本的实现差异、必要 URS 适配和预设参数见本目录 PLAN.md。', '',
              '原始五任务结果位于 results/；all_results.csv 保存逐次逐任务结果；summary.csv 保存跨重复汇总；summary.json 保存模型路径、来源和选择信息。', '']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines))
    print(f'{len(records)}/{len(plan)} complete; wrote {len(summaries)} configuration summaries.')


if __name__ == '__main__':
    main()
