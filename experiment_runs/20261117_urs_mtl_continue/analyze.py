"""Compare completed URS continuation runs with the existing merge results."""

import csv
import json
from pathlib import Path
from statistics import mean

RUN = Path(__file__).resolve().parent
TIES_RUN = RUN.parent / '20261117_ties_seen5'
TASKS = ['cvrp', 'cvrpb', 'cvrptw', 'ocvrp', 'ocvrptw']


def main():
    references = json.loads((TIES_RUN / 'summary.json').read_text())
    records = []
    for key, label in [
        ('base', '原URS base'), ('average', '五专家等权平均'),
        ('default', '默认TIES K20 / λ1'),
        ('best_ties', '扫描最佳TIES K30 / λ0.4'),
        ('best_all_tasks_ties', '五项均改善TIES K90 / λ0.4'),
    ]:
        ref = references[key]
        records.append({'id': key, 'label': label,
                        'checkpoint': ref['checkpoint'],
                        'source_json': ref['source_json'],
                        'rows': json.loads(Path(ref['source_json']).read_text())['rows']})
    state_path = RUN / 'queue_state.json'
    jobs = json.loads(state_path.read_text())['jobs'] if state_path.exists() else []
    labels = {'total_100k': 'URS续训30轮：每轮总10万',
              'per_task_100k': 'URS续训30轮：每任务每轮10万'}
    completed = []
    for job in jobs:
        if job['status'] != 'completed':
            continue
        result = Path(job['result'])
        payload = json.loads(result.read_text())
        assert not payload['missing']
        assert [row['task'] for row in payload['rows']] == TASKS
        assert all(row['episodes'] == 1000 and row['aug_factor'] == 8
                   for row in payload['rows'])
        completed.append(job['name'])
        records.append({'id': job['name'], 'label': labels[job['name']],
                        'checkpoint': payload['checkpoint'],
                        'source_json': str(result), 'rows': payload['rows']})
    base = {row['task']: row['aug_gap'] for row in records[0]['rows']}
    summaries, long_rows = [], []
    for record in records:
        gaps = {row['task']: row['aug_gap'] for row in record['rows']}
        summary = {key: value for key, value in record.items() if key != 'rows'}
        summary.update({f'{task}_gap': gaps[task] for task in TASKS})
        summary.update(mean_gap=mean(gaps.values()),
                       delta_vs_base_pp=mean(gaps[t] - base[t] for t in TASKS),
                       improved_tasks=sum(gaps[t] < base[t] for t in TASKS))
        summaries.append(summary)
        for row in record['rows']:
            long_rows.append({**{k: v for k, v in record.items() if k != 'rows'},
                              **row, 'delta_vs_base_pp': row['aug_gap'] - base[row['task']]})
    for filename, rows in [('summary.csv', summaries), ('results.csv', long_rows)]:
        with (RUN / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = ['# URS 多任务续训对照', '',
            f'两组续训及完整评测已完成 {len(completed)}/2。gap单位为%，越低越好；差值单位为百分点。', '',
            '| 模型 | CVRP | CVRPB | CVRPTW | OCVRP | OCVRPTW | 平均 | 相对base | 改善任务 |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in summaries:
        cells = [row['label']] + [f"{row[task + '_gap']:.6f}" for task in TASKS]
        cells += [f"{row['mean_gap']:.6f}", f"{row['delta_vs_base_pp']:+.6f}",
                  f"{row['improved_tasks']}/5"]
        text.append('| ' + ' | '.join(cells) + ' |')
    for name, label in labels.items():
        if name not in completed:
            text.append('| ' + label + ' | ' + ' | '.join(['待完成'] * 8) + ' |')
    text += ['', '两组均从epoch-500恢复模型与AdamW状态，追加30轮至epoch530；'
             'batch128、学习率1e-5、weight decay 1e-6，保持原URS训练算法。'
             '第一组总计300万训练实例；第二组每任务300万、总计1500万。', '',
             '此前专家微调采用新建优化器和3e-5→3e-6学习率，本轮按用户要求使用原URS恢复语义；'
             '第二组匹配专家总实例量和更新数，但优化器状态及学习率并不相同。', '',
             '所有模型使用原五任务数据各1000例、N100、greedy、8倍增强，原参考值和gap。'
             '本轮续训使用固定epoch530；TIES是此前在同一评测数据上扫描选出后经4090复核的结果。', '',
             '训练方案：[PLAN.md](PLAN.md)；状态：[queue_state.json](queue_state.json)；'
             '逐任务结果：[results.csv](results.csv)；汇总：[summary.csv](summary.csv)。', '']
    (RUN / 'RESULTS.md').write_text('\n'.join(text))
    (RUN / 'summary.json').write_text(json.dumps(
        {'completed_continuations': completed, 'models': summaries}, indent=2,
        ensure_ascii=False) + '\n')
    print(f'Completed continuation evaluations: {len(completed)}/2', flush=True)


if __name__ == '__main__':
    main()
