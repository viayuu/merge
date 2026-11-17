"""Fix CAT configurations and verify the six existing input checkpoints."""

from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hashlib


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    previous = json.loads((RUN.parent / '20261117_ties_seen5/manifest.json').read_text())
    for item in [previous['base']] + previous['experts']:
        path = item.get('path', item.get('checkpoint'))
        expected = item.get('sha256', item.get('checkpoint_sha256'))
        assert digest(path) == expected, path
    for task in previous['datasets'].values():
        for kind in ['data_path', 'solution_path']:
            assert digest(task[kind]['path']) == task[kind]['sha256']
    (RUN / 'models').mkdir(exist_ok=True)
    (RUN / 'features').mkdir(exist_ok=True)
    configs = [
        {'config_id': 'upstream_c4_a0p4', 'family': 'upstream',
         'label': '发布代码适配 c4 α0.4', 'is_default': True,
         'mode': 'upstream', 'linear_count': 4, 'norm_count': 0,
         'bias_count': 0, 'feature_scale': .4, 'scale': .4},
        {'config_id': 'paper_full_c4_a0p4', 'family': 'paper_full',
         'label': '论文公式完整 CAT c4 α0.4', 'is_default': True,
         'mode': 'paper', 'linear_count': 4, 'norm_count': 4,
         'bias_count': 4, 'feature_scale': 1., 'scale': .4},
        {'config_id': 'paper_linear_c4_a0p4', 'family': 'paper_linear',
         'label': '论文公式仅线性 c4 α0.4', 'is_default': False,
         'mode': 'paper', 'linear_count': 4, 'norm_count': 0,
         'bias_count': 0, 'feature_scale': 1., 'scale': .4},
    ]
    template = configs[1]
    for step in range(1, 10):
        if step != 4:
            configs.append({**template, 'config_id': f'paper_full_c4_a0p{step}',
                            'label': f'论文公式完整 CAT c4 α0.{step}',
                            'is_default': False, 'scale': step / 10})
    for count in [2, 3]:
        configs.append({**template, 'config_id': f'paper_full_c{count}_a0p4',
                        'label': f'论文公式完整 CAT c{count} α0.4',
                        'is_default': False, 'linear_count': count,
                        'norm_count': count, 'bias_count': count})
    seeds = [3407, 3408, 3409]
    plan = [{'id': 'base', 'config_id': 'base', 'family': 'base',
             'label': '原 URS base（未续训）', 'checkpoint': previous['base']['path']}]
    # Re-evaluate the untrimmed alpha=.4 control on the same two cards.
    plan.append({'id': 'ta_a0p4', 'config_id': 'ta_a0p4', 'family': 'control',
                 'label': 'Task Arithmetic α0.4', 'checkpoint': str(
                     RUN.parent / '20261117_ties_seen5/models/ta_l0p4.pt')})
    for config in configs:
        for seed in seeds:
            job_id = f"{config['config_id']}_s{seed}"
            plan.append({**config, 'id': job_id, 'seed': seed,
                         'checkpoint': str(RUN / 'models' / f'{job_id}.pt')})
    manifest = {
        'created': datetime.now().astimezone().isoformat(),
        'base': previous['base'], 'experts': previous['experts'],
        'datasets': previous['datasets'], 'task_order': previous['task_order'],
        'protocol': previous['protocol'], 'seeds': seeds, 'examples_per_task': 2,
        'linear_lambda': 5, 'norm_lambda': 10,
        'upstream_repository': 'https://github.com/SunWenJu123/model-merging',
        'upstream_commit': 'ab87731e405b9eeca1f822685ede0276d756907c',
        'feature_data': '2 uniformly selected existing test inputs per task; no labels; no new split',
        'selection': 'mean across 3 exemplar seeds; same-data exploratory selection',
    }
    for filename, value in [('configs.json', configs), ('evaluation_plan.json', plan),
                            ('manifest.json', manifest)]:
        (RUN / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    print(f'Fixed {len(configs)} CAT configs x {len(seeds)} seeds + 2 controls = {len(plan)} evaluations.')


if __name__ == '__main__':
    main()
