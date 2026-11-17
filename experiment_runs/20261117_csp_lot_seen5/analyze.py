"""Compare CSP+LOT with the old URS base and matched pure LOT controls."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean, pstdev

RUN = Path(__file__).resolve().parent
TASKS = ['cvrp', 'cvrpb', 'cvrptw', 'ocvrp', 'ocvrptw']


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def std(values):
    values = list(values)
    return pstdev(values) if len(values) > 1 else None


def fmt(value, signed=False):
    return '—' if value is None else format(value, '+.6f' if signed else '.6f')


def load_record(job):
    path = Path(job.get('source_result') or job.get('result') or
                RUN / 'results' / f"{job['id']}.json")
    result = json.loads(path.read_text())
    assert not result['missing'], path
    assert [row['task'] for row in result['rows']] == TASKS, path
    assert Path(result['checkpoint']).resolve() == Path(job['checkpoint']).resolve(), path
    for row in result['rows']:
        assert row['scale'] == 100 and row['episodes'] == 1000, row
        assert row['aug_factor'] == 8 and row['reference_available'], row
        assert all(math.isfinite(row[key]) for key in
                   ['aug_gap', 'no_aug_gap', 'aug_score', 'no_aug_score']), row
    build_path = RUN / 'build_records' / f"{job['id']}.json"
    build = json.loads(build_path.read_text()) if build_path.exists() else {}
    eval_seconds = None
    if not job.get('source_result') and job.get('started') and job.get('finished'):
        eval_seconds = (datetime.fromisoformat(job['finished']) -
                        datetime.fromisoformat(job['started'])).total_seconds()
    return {**job, 'source_json': str(path), 'historical_reuse': bool(job.get('source_result')),
            'gaps': {row['task']: row['aug_gap'] for row in result['rows']},
            'rows': result['rows'], 'timing': build.get('timing', {}),
            'diagnostics': build.get('diagnostics', {}),
            'feature_path': build.get('feature_path'),
            'lot_cache_path': build.get('lot_cache_path'),
            'svd_cache_path': build.get('svd_cache_path'), 'eval_seconds': eval_seconds}


def summarize(records, base, planned, lot_controls):
    first = records[0]
    for key in ['family', 'samples', 'scale', 'k', 'rank_mode']:
        assert len({record.get(key) for record in records}) == 1, first['config_id']
    gaps = {task: mean(record['gaps'][task] for record in records) for task in TASKS}
    changes = {task: gaps[task] - base['gaps'][task] for task in TASKS}
    means = [mean(record['gaps'].values()) for record in records]
    pairs = [(record, lot_controls[(record['samples'], record['scale'], record['seed'])])
             for record in records
             if (record.get('samples'), record.get('scale'), record.get('seed')) in lot_controls]
    lot_changes = {task: mean(record['gaps'][task] - control['gaps'][task]
                             for record, control in pairs) if pairs else None for task in TASKS}
    paired_changes = [mean(record['gaps'][task] - control['gaps'][task] for task in TASKS)
                      for record, control in pairs]
    row = {
        'config_id': first['config_id'], 'family': first['family'],
        'label': first.get('label', first['config_id']),
        'samples': first.get('samples'), 'scale': first.get('scale'),
        'k': first.get('k'), 'rank_mode': first.get('rank_mode'),
        'is_default': first.get('is_default', False),
        'completed_repeats': len(records), 'planned_repeats': planned,
        'seeds': ','.join(str(record.get('seed', '')) for record in records),
        'historical_repeats': sum(record['historical_reuse'] for record in records),
        'mean_gap': mean(means), 'std_gap': std(means),
        'delta_vs_base_pp': mean(changes.values()),
        'improved_tasks': sum(value < 0 for value in changes.values()),
        'max_regression_vs_base_pp': max(0., max(changes.values())),
        'matched_lot_config_id': pairs[0][1]['config_id'] if pairs else None,
        'matched_lot_repeats': len(pairs),
        'matched_lot_mean_gap': mean(mean(control['gaps'].values()) for _, control in pairs) if pairs else None,
        'delta_vs_lot_pp': mean(paired_changes) if pairs else None,
        'std_delta_vs_lot_pp': std(paired_changes),
        'improved_tasks_vs_lot': sum(value < 0 for value in lot_changes.values()) if pairs else None,
        'max_regression_vs_lot_pp': max(0., max(lot_changes.values())) if pairs else None,
    }
    for task in TASKS:
        row[f'{task}_gap'] = gaps[task]
        row[f'{task}_std'] = std(record['gaps'][task] for record in records)
        row[f'{task}_delta_vs_base_pp'] = changes[task]
        row[f'{task}_delta_vs_lot_pp'] = lot_changes[task]
    diagnostics = [record['diagnostics'] for record in records if record['diagnostics'].get('layers')]
    full_rank_counts = [sum(layer['rank'] == layer['max_rank'] for layer in diag['layers'].values())
                        for diag in diagnostics]
    row['full_rank_matrices_mean'] = mean(full_rank_counts) if full_rank_counts else None
    ratios = [diag['retained_delta_norm_ratio'] for diag in diagnostics]
    row['retained_delta_norm_ratio_mean'] = mean(ratios) if ratios else None
    for stage in ['svd_seconds', 'lot_seconds', 'projection_seconds', 'save_seconds']:
        values = [record['timing'][stage] for record in records
                  if record['timing'].get(stage) is not None]
        row[stage + '_mean_recorded'] = mean(values) if values else None
    times = [record['eval_seconds'] for record in records if record['eval_seconds'] is not None]
    row['eval_seconds_mean'] = mean(times) if times else None
    return row


def table(rows, task_details=False):
    columns = ['配置', '完成重复', '平均gap ± std', '较base（百分点）',
               '较匹配LOT（百分点）', '改善任务：base / LOT']
    if task_details:
        columns += [task.upper() for task in TASKS]
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    for row in rows:
        label = row['label'] + ('（入口预设r=5）' if row['is_default'] else '')
        wins = f"{row['improved_tasks']}/5 / " + (
            '—' if row['improved_tasks_vs_lot'] is None else f"{row['improved_tasks_vs_lot']}/5")
        cells = [label, f"{row['completed_repeats']}/{row['planned_repeats']}",
                 f"{fmt(row['mean_gap'])} ± {fmt(row['std_gap'])}",
                 fmt(row['delta_vs_base_pp'], True), fmt(row['delta_vs_lot_pp'], True), wins]
        if task_details:
            cells += [fmt(row[f'{task}_gap']) for task in TASKS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


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
        if job.get('status') == 'completed':
            records.append(load_record(job))
        else:
            pending.append(job['id'])
    if pending and not args.partial:
        raise SystemExit(f'{len(pending)} jobs unfinished; use --partial for interim results.')
    long_rows = []
    for record in records:
        for row in record['rows']:
            long_rows.append({
                'job_id': record['id'], 'config_id': record['config_id'], 'family': record['family'],
                'samples': record.get('samples'), 'scale': record.get('scale'),
                'k': record.get('k'), 'rank_mode': record.get('rank_mode'), 'seed': record.get('seed'),
                'historical_reuse': record['historical_reuse'], 'gpu_name': record.get('gpu_name'),
                'checkpoint': record['checkpoint'], 'source_json': record['source_json'],
                'task': row['task'], 'episodes': row['episodes'], 'aug_gap': row['aug_gap'],
                'no_aug_gap': row['no_aug_gap'], 'aug_score': row['aug_score'], 'oracle_score': row['oracle_score'],
            })
    write_csv(RUN / 'all_results.csv', long_rows)
    base = next((record for record in records if record['family'] == 'base'), None)
    if base is None:
        (RUN / 'summary.json').write_text(json.dumps({'completed': len(records), 'total': len(plan),
            'pending': pending, 'note': 'Waiting for old URS base evaluation.'}, indent=2) + '\n')
        print(f'{len(records)}/{len(plan)} complete; waiting for base.')
        return
    expected, grouped = defaultdict(int), defaultdict(list)
    for entry in plan:
        expected[entry['config_id']] += 1
    for record in records:
        grouped[record['config_id']].append(record)
    lot_controls = {(record['samples'], record['scale'], record['seed']): record
                    for record in records if record['family'] == 'lot'}
    summaries = [summarize(group, base, expected[key], lot_controls) for key, group in grouped.items()]
    summaries.sort(key=lambda row: (row['family'], row['samples'] or 0, row['scale'] or 0,
                                   row['rank_mode'] or '', row['k'] or 0))
    write_csv(RUN / 'summary.csv', summaries)
    best = {}
    for row in sorted(summaries, key=lambda value: (value['mean_gap'], value['config_id'])):
        if row['family'] == 'csp_lot' and row['completed_repeats'] == row['planned_repeats']:
            key = f"n{row['samples']}_a{row['scale']}_{row['rank_mode']}"
            best.setdefault(key, row)
    fresh = [record for record in records if not record['historical_reuse']]
    eval_seconds = sum(record['eval_seconds'] or 0 for record in fresh)
    starts = [datetime.fromisoformat(record['started']) for record in fresh if record.get('started')]
    ends = [datetime.fromisoformat(record['finished']) for record in fresh if record.get('finished')]
    stages = {}
    for stage, cache in [('svd', 'svd_cache_path'), ('lot', 'lot_cache_path')]:
        actual = {record[cache] or record['id']: record['timing'].get(stage + '_seconds')
                  for record in fresh if not record['timing'].get(stage + '_reused', False)}
        stages[stage + '_seconds'] = sum(value for value in actual.values() if value is not None)
    for stage in ['projection', 'save']:
        stages[stage + '_seconds'] = sum(record['timing'].get(stage + '_seconds') or 0 for record in fresh
            if not record['timing'].get(stage + '_reused', False))
    timing = {'new_evaluations_completed': len(fresh),
              'historical_evaluations_reused': len(records) - len(fresh),
              'summed_evaluation_seconds': eval_seconds, 'evaluation_gpu_hours': eval_seconds / 3600,
              'evaluation_wall_seconds': (max(ends) - min(starts)).total_seconds() if starts and ends else None,
              'new_build_stages': stages, 'accounted_build_seconds': sum(stages.values()),
              'note': 'SVD and LOT costs deduplicated by cache path, excluding reused stages; projection/save costs sum actual stages. Shared historical feature extraction is not charged again. This is accumulated work, not parallel wall time.'}
    result = {'completed': len(records), 'total': len(plan), 'pending': pending,
              'base': {'checkpoint': base['checkpoint'], 'gaps': base['gaps'],
                       'mean_gap': mean(base['gaps'].values()), 'source_json': base['source_json']},
              'groups': summaries, 'exploratory_best_by_track_and_rank_mode': best, 'timing': timing,
              'standard_deviation': 'Population std (ddof=0) across exemplar selections; null for singleton N1000.',
              'comparison_note': 'CSP versus LOT is paired by sample count, scale and seed. N64/alpha1.4 and N1000/alpha1.2 differ in two settings; cross-track differences are not sample effects.',
              'selection_note': 'Same original evaluation inputs and benchmark; best configurations are exploratory selections, not held-out test selections.',
              'runs': [{key: value for key, value in record.items() if key != 'rows'} for record in records]}
    (RUN / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    headline = {row['config_id']: row for row in summaries if row['family'] in ['base', 'lot'] or row['is_default']}
    headline.update({row['config_id']: row for row in best.values()})
    lines = ['# URS 五任务 CSP + LOT 实验结果', '',
             f"完成 **{len(records)}/{len(plan)}** 项：本轮新评测 {len(fresh)} 项，复用历史 LOT {len(records)-len(fresh)} 项。对照为旧的、未续训 URS 多任务 base。", '',
             '原五任务各1000个N100实例，greedy、POMO多起点、8倍增强；增强gap（%）越低越好，五任务等权平均。', '',
             'N64使用三次固定seed取样；N1000使用固定原集合全量一次，std记为“—”，不表示不确定性为零。取样重复不是重训专家。', '',
             '## 对照与关键配置', '']
    lines += table(list(headline.values()), task_details=True)
    lines += ['', '“较匹配LOT”严格按相同样本量、相同合并系数和相同取样seed配对；负值表示CSP改善LOT。两轨N64/α1.4与N1000/α1.2同时改变样本量和系数，不能将跨轨差值单独归因于样本数量。部分结果配对数保存在summary.csv的matched_lot_repeats。', '',
              '## 完整固定秩扫描', '']
    for samples, scale in [(64, 1.4), (1000, 1.2)]:
        rows = [row for row in summaries if row['family'] == 'csp_lot' and
                row['samples'] == samples and row['scale'] == scale and row['rank_mode'] == 'rank']
        lines += [f'### N={samples}，α={scale}', ''] + table(rows) + ['']
    energy = [row for row in summaries if row['rank_mode'] == 'energy']
    lines += ['## 按层能量选秩（URS补充配置）', '',
              'p按发布代码的累计奇异值平方能量定义；不是声称复现论文附录未给出的p网格。', '']
    lines += table(energy)
    lines += ['', '## CSP是否优于对应LOT', '',
              '下列最优配置按全部预设重复的平均gap选择，不挑seed；只列已完成全部重复的组。同数据选参属于探索性结果。', '']
    for key, row in best.items():
        lines.append(f"- {key}：`{row['config_id']}`，gap **{fmt(row['mean_gap'])}%**；"
                     f"较base {fmt(row['delta_vs_base_pp'], True)}，较匹配LOT **{fmt(row['delta_vs_lot_pp'], True)}** 个百分点。")
    lines += ['', '## 投影强度与耗时', '',
              '秩r越大，二维补空间越小；r达到矩阵最小维度时，该二维增量理论上清零（保留官方FP32舍入）。一维norm/bias仍沿用LOT，所以不能把全秩投影解释为整个模型恢复base。', '',
              '| 配置 | 满秩投影矩阵数 | 整体增量范数保留比例 |', '|---|---:|---:|']
    for row in summaries:
        if row['family'] == 'csp_lot':
            lines.append(f"| {row['label']} | {fmt(row['full_rank_matrices_mean'])} | {fmt(row['retained_delta_norm_ratio_mean'])} |")
    lines += ['', f"正式新评测累计GPU时间 **{eval_seconds / 3600:.3f}小时**；并行评测跨度 **{(timing['evaluation_wall_seconds'] or 0) / 60:.2f}分钟**。去重后记录的新增构建阶段累计 **{timing['accounted_build_seconds']:.2f}秒**，不是墙钟时间。", '',
              '[固定秩曲线](rank_sensitivity.png) · [按层能量曲线](energy_sensitivity.png)，另有同名PDF。', '',
              '逐任务、逐seed的原始长表见all_results.csv；全部配置与配对差值见summary.csv；summary.json保留输入结果路径、模型路径、计时和逐层投影诊断。具体来源与必要适配见PLAN.md。', '']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines))
    print(f'{len(records)}/{len(plan)} complete; wrote {len(summaries)} configuration summaries.')


if __name__ == '__main__':
    main()
