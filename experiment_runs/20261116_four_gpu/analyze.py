"""Summarize completed jobs from this fixed URS model-merging experiment."""

import csv
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from statistics import mean


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SETS = {
    'T2': ('cvrp', 'cvrptw'),
    'T4': ('cvrp', 'cvrpb', 'cvrptw', 'ocvrp'),
    'T11': ('tsp', 'atsp', 'cvrp', 'acvrp', 'cvrpb', 'cvrptw',
            'ocvrp', 'ocvrptw', 'op', 'pctsp', 'pdtsp'),
}
FIELDS = [
    'job_id', 'phase', 'model', 'task', 'scale', 'episodes', 'aug_factor',
    'objective', 'reference_objective', 'gap_pct', 'no_aug_gap_pct',
    'paper_urs_gap_pct', 'paper_stl_gap_pct', 'delta_vs_paper_urs_pp',
    'delta_vs_paper_stl_pp', 'local_unified_gap_pct', 'delta_vs_local_unified_pp',
    'paper_comparable', 'local_unified_comparable', 'protocol_matches',
    'reference_available', 'seed', 'eval_type', 'pomo_size',
    'configured_batch_size', 'time_seconds', 'device', 'slot', 'node', 'gpu',
    'checkpoint', 'data_path', 'solution_path', 'source_json',
]


def fmt(value):
    return '—' if value is None else f'{value:.3f}'


def table(lines, columns, rows):
    lines += ['| ' + ' | '.join(columns) + ' |',
              '|' + '|'.join('---' for _ in columns) + '|']
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    lines.append('')


def main():
    manifest = json.loads((RUN / 'manifest.json').read_text())
    protocol = manifest['protocol']
    state_path = RUN / 'queue_state.json'
    states = ({job['id']: job for job in json.loads(state_path.read_text())['jobs']}
              if state_path.exists() else {})
    with (ROOT / 'merge_lab/paper_results.csv').open(newline='') as handle:
        paper = {(r['task'], int(r['scale'])): r for r in csv.DictReader(handle)}
    rows, notes = [], []
    for job in manifest['jobs']:
        state = states.get(job['id'], {})
        if states and state.get('status') != 'completed':
            continue
        path = Path(job['output'])
        if not path.exists():
            if state.get('status') == 'completed':
                notes.append(f"{job['id']}: 状态为 completed，但结果文件不存在。")
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            notes.append(f"{job['id']}: JSON 尚不完整，暂未纳入。")
            continue
        if (payload.get('missing') or len(payload['rows']) != 1
                or payload['rows'][0]['task'] != job['task']
                or Path(payload['checkpoint']).resolve() != Path(job['checkpoint']).resolve()):
            notes.append(f"{job['id']}: 输出与 manifest 的任务/模型不一致或不完整，未纳入。")
            continue
        result = payload['rows'][0]
        task, scale = result['task'], int(result['scale'])
        reference = paper.get((task, scale))
        eval_type = payload['model_params']['eval_type']
        original_data = Path(payload['data_dir']).resolve() == (ROOT / 'URS/dataset').resolve()
        comparable = bool(reference and original_data and result['reference_available']
                          and int(result['episodes']) == int(reference['episodes'])
                          and int(result['aug_factor']) == int(reference['augmentation_factor'])
                          and eval_type == 'greedy')
        protocol_matches = (original_data and scale == protocol['scale']
                            and int(result['episodes']) == protocol['episodes']
                            and payload['seed'] == protocol['seed']
                            and eval_type == protocol['eval_type']
                            and reference is not None
                            and int(result['aug_factor']) == int(reference['augmentation_factor']))
        urs = float(reference['paper_urs_gap_pct']) if reference else None
        stl = float(reference['paper_stl_gap_pct']) if reference else None
        row = dict(
            job_id=job['id'], phase=job['phase'], model=job['model'], task=task,
            scale=scale, episodes=int(result['episodes']), aug_factor=int(result['aug_factor']),
            objective=result['aug_score'], reference_objective=result['oracle_score'],
            gap_pct=result['aug_gap'], no_aug_gap_pct=result['no_aug_gap'],
            paper_urs_gap_pct=urs, paper_stl_gap_pct=stl,
            delta_vs_paper_urs_pp=result['aug_gap'] - urs if comparable else None,
            delta_vs_paper_stl_pp=result['aug_gap'] - stl if comparable else None,
            local_unified_gap_pct=None, delta_vs_local_unified_pp=None,
            paper_comparable=comparable, local_unified_comparable=False,
            protocol_matches=protocol_matches, reference_available=result['reference_available'],
            seed=payload['seed'], eval_type=eval_type, pomo_size=int(result['pomo_size']),
            configured_batch_size=int(result['configured_batch_size']), time_seconds=result['time'],
            device=payload['device'], slot=state.get('slot', ''), node=state.get('node', ''),
            gpu=state.get('gpu', ''), checkpoint=payload['checkpoint'],
            data_path=result['data_path'], solution_path=result['solution_path'], source_json=str(path),
        )
        row['_pair_key'] = (
            scale, row['episodes'], row['aug_factor'], row['seed'], eval_type,
            row['pomo_size'], row['configured_batch_size'], row['data_path'],
            row['solution_path'], row['reference_objective'], row['reference_available'],
            json.dumps(payload['model_params'], sort_keys=True),
        )
        if not protocol_matches:
            notes.append(f"{job['id']}: 配置与本批次协议不一致，保留逐项数值，不纳入完整集合均值。")
        rows.append(row)

    by_model_task = {(r['model'], r['task']): r for r in rows}
    for row in rows:
        unified = by_model_task.get(('unified', row['task']))
        if unified:
            row['local_unified_gap_pct'] = unified['gap_pct']
            row['local_unified_comparable'] = row['_pair_key'] == unified['_pair_key']
            if row['local_unified_comparable']:
                row['delta_vs_local_unified_pp'] = row['gap_pct'] - unified['gap_pct']
            else:
                notes.append(f"{row['job_id']}: 与本地 unified 的配置/数据参考不一致，Δ本地U留空。")
    with (RUN / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    completed = {row['job_id'] for row in rows}
    statuses = Counter(states.get(job['id'], {}).get(
        'status', 'completed' if job['id'] in completed else 'pending') for job in manifest['jobs'])
    lines = [
        '# URS 模型合并实验结果', '',
        f"更新：{datetime.now().astimezone().isoformat(timespec='seconds')}。"
        f"已纳入有效结果 {len(rows)}/{len(manifest['jobs'])}。", '',
        '队列状态：' + '；'.join(f'{key}={value}' for key, value in sorted(statuses.items())) + '。', '',
        '所有 gap 沿用原 URS Tester 与原数据集参考值；数值越低越好。'
        'Δ 为“本次 gap − 对照 gap”，单位百分点（pp），负值更好。'
        '目标值为增强后的原始 objective；OP 为最大化任务，其余任务为最小化。', '',
        '论文对照来自原论文 Table 2/3（N=100、1000例、greedy、8/128倍增强）。'
        '本地U指同批次 unified checkpoint。未重建或拆分数据集。'
        '三位小数为显示精度；CSV保留原输出精度，论文值保留发表精度。', '',
        '## 执行进度', '',
    ]
    table(lines, ['阶段', '有效结果', '计划任务', '状态'], [
        [phase, sum(r['phase'] == phase for r in rows),
         sum(j['phase'] == phase for j in manifest['jobs']),
         '完成' if all(j['id'] in completed for j in manifest['jobs'] if j['phase'] == phase) else '进行中/待执行']
        for phase in dict.fromkeys(j['phase'] for j in manifest['jobs'])
    ])
    active = [state for state in states.values() if state['status'] in ('running', 'failed')]
    if active:
        table(lines, ['任务', '状态', 'GPU槽位', '节点', '错误'], [
            [s['id'], s['status'], s.get('slot', ''), s.get('node', ''),
             str(s.get('error') or '—').replace('|', '/').replace('\n', ' ')] for s in active
        ])

    lines += ['## 固定任务集合的宏均值', '',
              'T2 = CVRP、CVRPTW；T4 = CVRP、CVRPB、CVRPTW、OCVRP；'
              'T11 = 全部11项任务。每个任务等权。只对任务齐全且匹配本批次协议的集合计算均值；'
              '未完成集合仅显示完成数。论文/本地U差值要求整个集合的对应比较均可用。', '']
    planned = {}
    for job in manifest['jobs']:
        planned.setdefault(job['model'], set()).add(job['task'])
    macro_rows = []
    for model in [*planned, 'specialist_system']:
        for name, tasks in SETS.items():
            if model != 'specialist_system' and not set(tasks) <= planned[model]:
                continue
            group = [by_model_task.get((f'specialist_{task}' if model == 'specialist_system' else model, task))
                     for task in tasks]
            present = [r for r in group if r is not None]
            full = len(present) == len(tasks) and all(r['protocol_matches'] for r in present)
            paper_full = full and all(r['paper_comparable'] for r in present)
            local_full = full and all(r['local_unified_comparable'] for r in present)
            macro_rows.append([
                f'专家系统（{len(tasks)}个模型）' if model == 'specialist_system' else model,
                name, f'{len(present)}/{len(tasks)}',
                '完整' if full else ('配置不一致' if len(present) == len(tasks) else 'partial'),
                fmt(mean(r['gap_pct'] for r in present) if full else None),
                fmt(mean(float(paper[(t, protocol['scale'])]['paper_urs_gap_pct']) for t in tasks)),
                fmt(mean(float(paper[(t, protocol['scale'])]['paper_stl_gap_pct']) for t in tasks)),
                fmt(mean(r['delta_vs_paper_urs_pp'] for r in present) if paper_full else None),
                fmt(mean(r['delta_vs_paper_stl_pp'] for r in present) if paper_full else None),
                fmt(mean(r['delta_vs_local_unified_pp'] for r in present) if local_full else None),
            ])
    table(lines, ['模型/系统', '集合', '已完成', '完整性', '本次gap %', '论文URS %', '论文STL %',
                  'Δ论文URS pp', 'Δ论文STL pp', 'Δ本地U pp'], macro_rows)
    lines += ['专家系统按已知任务调用各自单任务模型；它是多模型参考，不能作为一个合并模型的成绩。', '',
              '## CVRP/CVRPTW 专家交叉矩阵', '', '单元格为原 gap %。', '']
    table(lines, ['模型', 'CVRP', 'CVRPTW'], [
        [model, *[fmt(by_model_task.get((model, task), {}).get('gap_pct')) for task in SETS['T2']]]
        for model in ('specialist_cvrp', 'specialist_cvrptw')
    ])
    lines += ['## 两专家插值曲线数据', '',
              '定义：θ(α) = (1−α) θCVRP + α θCVRPTW。端点复用原专家成绩。'
              '每个α对应同一个固定模型，在两个任务上评测。', '']
    alpha_rows = []
    for alpha, model in [(0, 'specialist_cvrp'), (0.25, 'interp_025'),
                         (0.5, 'average_2'), (0.75, 'interp_075'), (1, 'specialist_cvrptw')]:
        group = [by_model_task.get((model, task)) for task in SETS['T2']]
        full = all(r is not None and r['protocol_matches'] for r in group)
        alpha_rows.append([alpha, model, *[fmt(r['gap_pct'] if r else None) for r in group],
                           fmt(mean(r['gap_pct'] for r in group) if full else None),
                           f'{sum(r is not None for r in group)}/2'])
    table(lines, ['α', '模型', 'CVRP gap %', 'CVRPTW gap %', 'T2平均gap %', '已完成'], alpha_rows)

    lines += ['## 逐模型、逐任务结果', '',
              'GPU槽位来自调度记录；device为该子进程看到的CUDA编号。完整路径与配置见 results.csv。', '']
    table(lines, ['模型', '任务', 'Objective', 'gap %', '论文URS %', '论文STL %',
                  'Δ论文URS pp', 'Δ论文STL pp', '本地U %', 'Δ本地U pp', 'GPU槽位', 'device'], [
        [r['model'], r['task'], fmt(r['objective']), fmt(r['gap_pct']),
         fmt(r['paper_urs_gap_pct']), fmt(r['paper_stl_gap_pct']), fmt(r['delta_vs_paper_urs_pp']),
         fmt(r['delta_vs_paper_stl_pp']), fmt(r['local_unified_gap_pct']),
         fmt(r['delta_vs_local_unified_pp']), r['slot'], r['device']] for r in rows
    ])
    if notes:
        lines += ['## 需要核对的条目', '', *['- ' + note for note in notes], '']
    lines += ['所有计划任务均有有效输出。' if len(rows) == len(manifest['jobs'])
              else '实验尚未全部完成；以上为实时结果，不据此宣称整体motivation已验证。', '',
              '论文来源：[URS arXiv:2509.23413v3](https://arxiv.org/abs/2509.23413v3)。'
              '本地原始对照表：`merge_lab/paper_results.csv`。', '']
    (RUN / 'summary.md').write_text('\n'.join(lines))
    print(f'{len(rows)}/{len(manifest["jobs"])} results -> {RUN / "summary.md"}')


if __name__ == '__main__':
    main()
