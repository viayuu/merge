"""Fix LOT and CAT exemplar-count experiments before any new evaluation."""

from datetime import datetime
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
CAT = RUN.parent / '20261117_cat_seen5'


def main():
    old = json.loads((CAT / 'manifest.json').read_text())
    for item in [old['base']] + old['experts']:
        path = item.get('path', item.get('checkpoint'))
        expected = item.get('sha256', item.get('checkpoint_sha256'))
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    for data in old['datasets'].values():
        for kind in ['data_path', 'solution_path']:
            assert hashlib.sha256(Path(data[kind]['path']).read_bytes()).hexdigest() == data[kind]['sha256']
    for folder in ['models', 'features', 'build_records', 'logs', 'results']:
        (RUN / folder).mkdir(parents=True, exist_ok=True)
    plan = [{'id': 'base', 'config_id': 'base', 'family': 'base', 'label': '原URS base（未续训）',
             'samples': 0, 'scale': 0, 'seed': 3407, 'is_default': False,
             'checkpoint': old['base']['path']}]

    def add(family, samples, scale):
        config_id = f'{family}_n{samples}_a{str(float(scale)).replace(".", "p")}'
        seeds = [3407] if samples == 1000 else [3407, 3408, 3409]
        title = {'lot': 'LOT', 'cat_paper': 'CAT论文公式',
                 'cat_upstream': 'CAT发布代码适配'}[family]
        for seed in seeds:
            job_id = f'{config_id}_s{seed}'
            entry = {'id': job_id, 'config_id': config_id, 'family': family,
                     'label': f'{title} N={samples} α={scale}', 'samples': samples,
                     'scale': scale, 'seed': seed,
                     'feature_scale': .4 if family == 'cat_upstream' else 1.,
                     'is_default': family == 'lot' and samples == 64 and scale == 1.2,
                     'checkpoint': str(RUN / 'models' / f'{job_id}.pt')}
            if samples == 2 and family.startswith('cat_'):
                old_config = ('upstream_c4_a0p4' if family == 'cat_upstream' else
                              f'paper_full_c4_a{str(scale).replace(".", "p")}')
                old_id = f'{old_config}_s{seed}'
                entry.update(checkpoint=str(CAT / 'models' / f'{old_id}.pt'),
                             source_result=str(CAT / 'results' / f'{old_id}.json'))
                assert Path(entry['source_result']).exists()
            plan.append(entry)

    # The source default first, followed by the sample-count comparisons.
    for samples in [64, 2, 16, 256, 1000]:
        add('lot', samples, 1.2)
    for scale in [.6, .8, 1., 1.5, 2.]:
        add('lot', 64, scale)
    for samples in [2, 16, 64, 256, 1000]:
        add('cat_paper', samples, .2)
        add('cat_paper', samples, .4)
        add('cat_upstream', samples, .4)
    manifest = {**old, 'created': datetime.now().astimezone().isoformat(),
                'method': 'LOT merging and CAT exemplar-count study',
                'sizes': [2, 16, 64, 256, 1000], 'feature_batch_size': 128,
                'seeds': [3407, 3408, 3409],
                'full_set_policy': '1000 inputs in original order; one run, std not estimable',
                'samples': 'nested subsets of original 1000 inputs; no labels or new split',
                'source_protocol_difference': 'LOT paper samples training inputs; user requests existing original data here',
                'lot_defaults': {'scale': 1.2, 'samples_per_task': 64,
                                 'pinv': 'torch.linalg.pinv default, float32'},
                'total_evaluations': len(plan),
                'reused_evaluations': sum('source_result' in x for x in plan),
                'source_snapshot': 'third_party/lot_merging',
                'upstream_commit': 'ab87731e405b9eeca1f822685ede0276d756907c'}
    for name, value in [('manifest.json', manifest), ('evaluation_plan.json', plan)]:
        (RUN / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    print(f'Fixed {len(plan)} evaluations, {manifest["reused_evaluations"]} historical CAT results reused.')


if __name__ == '__main__':
    main()
