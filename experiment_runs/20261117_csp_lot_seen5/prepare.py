"""Freeze the official CSP rank sweep and the URS full-exemplar extension."""

from datetime import datetime
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
LOT = RUN.parent / '20261117_lot_samples_seen5'


def main():
    old = json.loads((LOT / 'manifest.json').read_text())
    for item in [old['base']] + old['experts']:
        path = item.get('path', item.get('checkpoint'))
        expected = item.get('sha256', item.get('checkpoint_sha256'))
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    for data in old['datasets'].values():
        for kind in ['data_path', 'solution_path']:
            assert hashlib.sha256(Path(data[kind]['path']).read_bytes()).hexdigest() == data[kind]['sha256']
    for folder in ['models', 'cache', 'build_records', 'logs', 'results']:
        (RUN / folder).mkdir(parents=True, exist_ok=True)

    plan = [{'id': 'base', 'config_id': 'base', 'family': 'base',
             'label': '原URS base（未续训）', 'samples': 0, 'scale': 0,
             'seed': 3407, 'k': None, 'rank_mode': 'none', 'is_default': False,
             'checkpoint': old['base']['path']}]

    def add(samples, scale, k, historical=False):
        mode = 'none' if k == 0 else 'energy' if k < 1 else 'rank'
        family = 'lot' if k == 0 else 'csp_lot'
        suffix = '' if k == 0 else f'_{"r" if mode == "rank" else "p"}{str(k).replace(".", "p")}'
        config = f'{"lot" if k == 0 else "csp"}_n{samples}_a{str(scale).replace(".", "p")}{suffix}'
        for seed in ([3407] if samples == 1000 else [3407, 3408, 3409]):
            job_id = f'{config}_s{seed}'
            title = '纯LOT' if k == 0 else f'CSP+LOT {"r" if mode == "rank" else "p"}={k}'
            entry = {'id': job_id, 'config_id': config, 'family': family,
                     'label': f'{title} N={samples} α={scale}', 'samples': samples,
                     'scale': scale, 'seed': seed, 'k': k, 'rank_mode': mode,
                     'is_default': samples == 64 and scale == 1.4 and k == 5,
                     'checkpoint': str(RUN / 'models' / f'{job_id}.pt')}
            if historical:
                entry.update(checkpoint=str(LOT / 'models' / f'{job_id}.pt'),
                             source_result=str(LOT / 'results' / f'{job_id}.json'))
                assert Path(entry['source_result']).exists()
            plan.append(entry)

    # Matched LOT controls precede each CSP track. Rank 5 is the entry preset;
    # the actual upstream script sweeps every integer from 1 through 20.
    for samples, scale in [(64, 1.4), (1000, 1.2)]:
        add(samples, scale, 0)
        for k in [5] + [r for r in range(1, 21) if r != 5]:
            add(samples, scale, k)
        for p in [.1, .5, .9]:
            add(samples, scale, p)
    add(64, 1.2, 0, historical=True)

    features = {}
    for samples in [64, 1000]:
        for seed in ([3407] if samples == 1000 else [3407, 3408, 3409]):
            path = LOT / 'features' / f'n{samples}_s{seed}_scale1.0.pt'
            features[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'created': datetime.now().astimezone().isoformat(),
                'method': 'CSP after LOT, official outout projection',
                **{key: old[key] for key in ['base', 'experts', 'datasets', 'task_order']},
                'upstream_commit': 'ab87731e405b9eeca1f822685ede0276d756907c',
                'official_track': {'samples': 64, 'scale': 1.4, 'ranks': list(range(1, 21)),
                                   'seeds': [3407, 3408, 3409], 'parts': ['outout']},
                'urs_track': {'samples': 1000, 'scale': 1.2, 'ranks': list(range(1, 21)),
                              'seeds': [3407]},
                'energy_extension': {'p': [.1, .5, .9], 'definition': 'cumulative squared singular values, as released code'},
                'feature_sha256': features,
                'lot_version_note': 'Source entry uses lot_merging; memory-efficient v2 has the same formula for these shared active layers.',
                'data_protocol': 'Existing original 1000 inputs and evaluation; no new split. Same-data unlabeled adaptation and exploratory comparison.',
                'total_evaluations': len(plan), 'new_evaluations': sum('source_result' not in x for x in plan),
                'reused_evaluations': sum('source_result' in x for x in plan)}
    assert len(plan) == 100 and manifest['new_evaluations'] == 97
    for name, value in [('manifest.json', manifest), ('evaluation_plan.json', plan)]:
        (RUN / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    print('Fixed 100 evaluations: 97 new, 3 historical LOT references.')


if __name__ == '__main__':
    main()
