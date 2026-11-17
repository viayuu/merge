"""Summarize LOT/CAT exemplar-count experiments against the original URS base."""

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
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def std(values):
    values = list(values)
    return pstdev(values) if len(values) > 1 else None


def fmt(value, signed=False):
    if value is None:
        return '—'
    return format(value, '+.6f' if signed else '.6f')


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
    timing = build.get('timing', {})
    feature_reused = build.get('feature_reused', timing.get('feature_reused', False))
    eval_seconds = None
    if not job.get('source_result') and job.get('started') and job.get('finished'):
        eval_seconds = (datetime.fromisoformat(job['finished']) -
                        datetime.fromisoformat(job['started'])).total_seconds()
    return {**job, 'source_json': str(path), 'historical_reuse': bool(job.get('source_result')),
            'gaps': {row['task']: row['aug_gap'] for row in result['rows']},
            'rows': result['rows'], 'timing': timing, 'feature_reused': feature_reused,
            'feature_path': build.get('feature_path'), 'merge_key': build.get('merge_key'),
            'eval_seconds': eval_seconds}


def summarize(records, base, planned):
    first = records[0]
    for key in ['family', 'samples', 'scale']:
        assert len({record.get(key) for record in records}) == 1, (first['config_id'], key)
    gaps = {task: mean(record['gaps'][task] for record in records) for task in TASKS}
    changes = {task: gaps[task] - base['gaps'][task] for task in TASKS}
    means = [mean(record['gaps'].values()) for record in records]
    row = {
        'config_id': first['config_id'], 'family': first['family'],
        'label': first.get('label', first['config_id']),
        'samples': first.get('samples'), 'scale': first.get('scale'),
        'is_default': first.get('is_default', False),
        'completed_repeats': len(records), 'planned_repeats': planned,
        'seeds': ','.join(str(record.get('seed', '')) for record in records),
        'historical_repeats': sum(record['historical_reuse'] for record in records),
        'mean_gap': mean(means), 'std_gap': std(means),
        'delta_vs_base_pp': mean(changes.values()),
        'improved_tasks': sum(value < 0 for value in changes.values()),
        'max_regression_pp': max(0.0, max(changes.values())),
    }
    for task in TASKS:
        row[f'{task}_gap'] = gaps[task]
        row[f'{task}_std'] = std(record['gaps'][task] for record in records)
        row[f'{task}_delta_pp'] = changes[task]
    for stage in ['feature_seconds', 'merge_seconds', 'save_seconds']:
        values = [record['timing'][stage] for record in records
                  if record['timing'].get(stage) is not None]
        row[stage + '_mean_recorded'] = mean(values) if values else None
    times = [record['eval_seconds'] for record in records if record['eval_seconds'] is not None]
    row['eval_seconds_mean'] = mean(times) if times else None
    return row


def markdown_table(rows):
    lines = ['| 配置 | 样本/任务 | 重复数 | CVRP | CVRPB | CVRPTW | OCVRP | OCVRPTW | 平均 gap ± std | 较 base（百分点） |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        label = row['label'] + ('（预设配置）' if row['is_default'] else '')
        gaps = ' | '.join(fmt(row[f'{task}_gap']) for task in TASKS)
        lines.append(f"| {label} | {row['samples'] or '—'} | "
                     f"{row['completed_repeats']}/{row['planned_repeats']} | {gaps} | "
                     f"{fmt(row['mean_gap'])} ± {fmt(row['std_gap'])} | "
                     f"{fmt(row['delta_vs_base_pp'], True)} |")
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
                'job_id': record['id'], 'config_id': record['config_id'],
                'family': record['family'], 'samples': record.get('samples'),
                'scale': record.get('scale'), 'seed': record.get('seed'),
                'historical_reuse': record['historical_reuse'],
                'gpu': record.get('gpu'), 'gpu_name': record.get('gpu_name'),
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
            'note': 'Waiting for the original URS base evaluation.'}, indent=2) + '\n')
        print(f'{len(records)}/{len(plan)} complete; waiting for base.')
        return
    expected, grouped = defaultdict(int), defaultdict(list)
    for entry in plan:
        expected[entry['config_id']] += 1
    for record in records:
        grouped[record['config_id']].append(record)
    summaries = [summarize(group, base, expected[key]) for key, group in grouped.items()]
    summaries.sort(key=lambda row: (row['family'], row['samples'] or 0, row['scale'] or 0))
    write_csv(RUN / 'summary.csv', summaries)
    complete = [row for row in summaries if row['completed_repeats'] == row['planned_repeats']]
    best = {}
    for row in sorted(complete, key=lambda row: (row['mean_gap'], row['config_id'])):
        if row['family'] != 'base':
            best.setdefault(row['family'], row)
    fresh = [record for record in records if not record['historical_reuse']]
    eval_seconds = sum(record['eval_seconds'] or 0 for record in fresh)
    starts = [datetime.fromisoformat(record['started']) for record in fresh if record.get('started')]
    ends = [datetime.fromisoformat(record['finished']) for record in fresh if record.get('finished')]
    new_features = {record['feature_path'] or record['id']: record['timing'].get('feature_seconds')
                    for record in fresh if not record['feature_reused']}
    feature_seconds = sum(seconds for seconds in new_features.values() if seconds is not None)
    merge_seconds = sum(record['timing'].get('merge_seconds') or 0 for record in fresh)
    save_seconds = sum(record['timing'].get('save_seconds') or 0 for record in fresh)
    build_seconds = feature_seconds + merge_seconds + save_seconds
    timing = {'new_evaluations_completed': len(fresh),
              'historical_evaluations_reused': len(records) - len(fresh),
              'summed_evaluation_seconds': eval_seconds,
              'evaluation_gpu_hours': eval_seconds / 3600,
              'completed_evaluation_wall_seconds': (max(ends) - min(starts)).total_seconds() if starts and ends else None,
              'new_feature_collection_seconds': feature_seconds,
              'new_merge_seconds': merge_seconds, 'new_save_seconds': save_seconds,
              'accounted_build_stage_seconds': build_seconds,
              'note': 'Feature collection is deduplicated by feature_path and excludes pre-existing caches. Per-config feature timing is nominal collection cost, even for cache hits; missing historical timing remains null. Merge/save times are actual newly incurred stages; totals are not parallel wall time.'}
    result = {
        'completed': len(records), 'total': len(plan), 'pending': pending,
        'base': {'checkpoint': base['checkpoint'], 'source_json': base['source_json'],
                 'gaps': base['gaps'], 'mean_gap': mean(base['gaps'].values())},
        'groups': summaries, 'exploratory_best_by_family': best, 'timing': timing,
        'standard_deviation': 'Population std across exemplar selections (ddof=0); null for a single run, including the fixed full 1000-instance set.',
        'selection_note': 'Same original evaluation data; configuration selection is exploratory. CAT and LOT scales act on different merged vectors and are not directly interchangeable.',
        'runs': [{key: value for key, value in record.items() if key != 'rows'} for record in records],
    }
    (RUN / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    base_rows = [row for row in summaries if row['family'] == 'base']
    lot_default = [row for row in summaries if row['family'] == 'lot' and row['samples'] == 64 and row['scale'] == 1.2]
    sample_rows = [row for row in summaries if row['family'] != 'base' and
                   (row['family'] != 'lot' or row['scale'] == 1.2)]
    lot_scales = [row for row in summaries if row['family'] == 'lot' and row['samples'] == 64]
    lines = ['# URS 五任务 LOT Merging 与 CAT 样本量实验', '',
             f"完成 **{len(records)}/{len(plan)}** 项，其中本轮新评测 {len(fresh)} 项、复用历史 CAT 结果 {len(records)-len(fresh)} 项。对照始终是未续训的 URS 第500轮模型。", '',
             '原五任务各1000个N100实例，greedy、POMO多起点、8倍增强、batch125；指标为增强gap（%，越低越好），五任务等权平均。', '',
             '2/16/64/256个样本采用三次固定seed取样；1000个样本是原集合全量，只做一次固定顺序评测，其std记为“—”，不能解释为不确定性为零。重复仅反映无标签输入取样，不是重新训练专家。', '',
             '## LOT 预设配置与旧 base', '']
    lines += markdown_table(base_rows + lot_default)
    lines += ['', '## 固定合并系数的样本量比较', '',
              'LOT固定系数1.2；CAT分别保留论文公式版0.2/0.4和发布代码适配版0.4。仅比较同一方法、同一系数随样本量的变化；两种方法的系数含义不能直接等同。', '']
    lines += markdown_table(sample_rows)
    lines += ['', '## LOT：64个样本时的系数扫描', '']
    lines += markdown_table(lot_scales)
    lines += ['', '## 同数据探索性选择', '',
              '按每配置全部预设重复的平均gap选择，不选择最有利的seed。数据同时用于评测和配置比较，最优值属于同数据探索性结果。', '']
    for family, row in best.items():
        lines.append(f"- {family}：`{row['config_id']}`，平均gap **{fmt(row['mean_gap'])}%**，"
                     f"相对base **{fmt(row['delta_vs_base_pp'], True)}** 个百分点，{row['improved_tasks']}/5任务改善。")
    if 'lot' in best:
        lines += ['', 'LOT最优配置的全部固定模型：', '']
        for record in records:
            if record['config_id'] == best['lot']['config_id']:
                lines.append(f"- seed {record.get('seed')}：[模型]({record['checkpoint']})，"
                             f"单次平均gap {mean(record['gaps'].values()):.6f}%。")
    lines += ['', '## 耗时', '',
              f"已完成新评测累计占用GPU **{eval_seconds / 3600:.3f}小时**；两卡并行评测时间跨度 **{(timing['completed_evaluation_wall_seconds'] or 0) / 60:.2f}分钟**。记录的构建阶段累计 **{build_seconds:.2f}秒**（不是并行墙钟时间，未重复计入已标记复用的特征采集）。", '',
              '下表的特征秒数是该配置所需缓存的原始采集耗时，缓存命中也保留该耗时；合并秒数则是本次新增计算，复用方向时为0。旧缓存没有采集计时则记为“—”。因此特征秒数可比较样本量成本，不能把各配置特征秒数相加当作本轮总成本。summary.json的runs保留原始timing、feature_path和merge_key。', '',
              '| 配置 | 样本/任务 | 名义特征采集秒数 | 新增合并秒数 | 评测秒数 |',
              '|---|---:|---:|---:|---:|']
    for row in sample_rows:
        lines.append(f"| {row['label']} | {row['samples']} | "
                     f"{fmt(row['feature_seconds_mean_recorded'])} | "
                     f"{fmt(row['merge_seconds_mean_recorded'])} | "
                     f"{fmt(row['eval_seconds_mean'])} |")
    lines += ['', '时间为配置内已记录重复的均值；不包含历史结果的本轮评测耗时。', '',
              '## 图表与复现', '',
              '[样本数量曲线](sample_count_sensitivity.png) · [LOT系数曲线](lot_scale_sensitivity.png)，另有同名PDF。', '',
              '输入、必要适配和样本使用方式见PLAN.md与manifest.json。all_results.csv为逐次逐任务原始长表；summary.csv为配置汇总；summary.json保存模型路径、原始结果来源及耗时。', '']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines))
    print(f'{len(records)}/{len(plan)} complete; wrote {len(summaries)} configuration summaries.')


if __name__ == '__main__':
    main()
