"""Write the five-task evaluation matrix and differences from the same URS base."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


RUN = Path(__file__).resolve().parent
TASKS = ['cvrp', 'ocvrp', 'cvrpbp', 'cvrpl', 'cvrptw']
LABELS = ['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW']
models = {}
state = json.loads((RUN / 'queue_state.json').read_text())
completed = {job['name']: job['result'] for job in state['jobs']
             if job['status'] == 'completed' and 'result' in job}
for model in ['base', *TASKS]:
    if model not in completed:
        continue
    payload = json.loads(Path(completed[model]).read_text())
    assert not payload['missing'] and len(payload['rows']) == 5
    models[model] = {row['task']: row for row in payload['rows']}

lines = ['# URS 单任务微调：五任务评测', '',
         '第30个epoch模型；原URS数据，每任务1000例、N=100、greedy、8倍增强。',
         'Gap单位为%；越低越好。Δ为新模型减原URS，单位为百分点，负值表示改善。',
         'VRPB对应cvrpbp（先送后取），VRPL对应cvrpl；这两项不在base原11个训练任务中。', '',
         '| 模型 | ' + ' | '.join(LABELS) + ' | 五任务平均 |',
         '|---|' + '---:|' * 6]
rows = []
for model, values in models.items():
    gaps = [values[task]['aug_gap'] for task in TASKS]
    label = '原URS多任务模型' if model == 'base' else f'{LABELS[TASKS.index(model)]}微调'
    lines.append('| ' + label + ' | ' + ' | '.join(f'{gap:.6f}' for gap in gaps)
                 + f' | {mean(gaps):.6f} |')
    for task in TASKS:
        row = values[task]
        baseline = models.get('base', {}).get(task)
        if baseline is not None:
            assert row['oracle_score'] == baseline['oracle_score']
            assert row['data_path'] == baseline['data_path']
        rows.append({'model': model, 'task': task, 'gap_pct': row['aug_gap'],
                     'base_gap_pct': baseline['aug_gap'] if baseline else None,
                     'delta_pp': row['aug_gap'] - baseline['aug_gap'] if baseline else None,
                     'objective': row['aug_score'], 'reference': row['oracle_score'],
                     'episodes': row['episodes'], 'aug_factor': row['aug_factor']})

if 'base' in models:
    lines += ['', '| 微调模型 | ' + ' | '.join(LABELS) + ' | 平均Δ |',
              '|---|' + '---:|' * 6]
    for model in TASKS:
        if model not in models:
            continue
        differences = [models[model][task]['aug_gap'] - models['base'][task]['aug_gap']
                       for task in TASKS]
        lines.append('| ' + LABELS[TASKS.index(model)] + ' | '
                     + ' | '.join(f'{value:+.6f}' for value in differences)
                     + f' | {mean(differences):+.6f} |')

    available = [task for task in TASKS if task in models]
    if available:
        lines += ['', '本任务专项表现：', '',
                  '| 任务 | 原URS gap % | 本任务微调 gap % | Δ（百分点） |',
                  '|---|---:|---:|---:|']
        for task in available:
            base_gap = models['base'][task]['aug_gap']
            new_gap = models[task][task]['aug_gap']
            lines.append(f'| {LABELS[TASKS.index(task)]} | {base_gap:.6f} | {new_gap:.6f} | {new_gap-base_gap:+.6f} |')
        improved = sum(models[task][task]['aug_gap'] < models['base'][task]['aug_gap'] for task in available)
        lines += ['', f'已评测的{len(available)}个专家中，{improved}个在本任务上优于原URS。'
                  '其他列用于观察跨任务保留和退化；这些结果尚未涉及模型合并。']

if len(models) == 6:
    import torch

    checks = []
    for task in TASKS:
        model_dir = RUN / 'models' / task
        metadata = json.loads((model_dir / 'finetune_metadata.json').read_text())
        path = model_dir / 'checkpoint_30.pt'
        with torch.serialization.safe_globals([set]):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        steps = sorted({int(value['step']) for value in checkpoint['optimizer_state_dict']['state'].values()})
        assert checkpoint['epoch'] == 30 and steps == [23460]
        assert metadata['status'] == 'completed' and metadata['optimizer_state_reset']
        assert metadata['base_model_sha256'] == metadata['initial_model_sha256']
        assert all(torch.isfinite(value).all() for value in checkpoint['model_state_dict'].values())
        assert all(abs(group['lr'] - 3e-6) < 1e-12 for group in checkpoint['optimizer_state_dict']['param_groups'])
        checks.append({'task': task, 'epoch': 30, 'optimizer_steps': steps,
                       'initial_model_sha256': metadata['initial_model_sha256'],
                       'checkpoint': str(path), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'finite_weights': True})
        del checkpoint
    assert len({row['initial_model_sha256'] for row in checks}) == 1
    (RUN / 'final_verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    lines += ['', '训练产物核验：五个模型均为epoch30、23460次优化器更新、末期学习率3e-6，'
              '全部权重有限且共同初始权重SHA一致。详见final_verification.json。']

missing = [name for name in ['base', *TASKS] if name not in models]
lines += ['', '尚未完成：' + (', '.join(missing) if missing else '无；全部6个模型×5项任务已完成。'), '']
(RUN / 'RESULTS.md').write_text('\n'.join(lines))
if rows:
    with (RUN / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
print('Evaluated models:', ', '.join(models), flush=True)
