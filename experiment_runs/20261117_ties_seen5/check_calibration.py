"""Check the original baseline and the two GPU types before the sweep."""

import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
PREVIOUS = RUN.parent / '20261116_urs_finetune_seen5'


def rows(path):
    return json.loads(path.read_text())['rows']


def main():
    checks = []
    pairs = [
        ('base_4090_vs_previous', RUN / 'results/base__4090.json',
         PREVIOUS / 'evaluation/base/unified_checkpoint_500.json'),
        ('base_3090_vs_4090', RUN / 'results/base__3090.json',
         RUN / 'results/base__4090.json'),
        ('default_3090_vs_4090', RUN / 'results/ties_default__3090.json',
         RUN / 'results/ties_default__4090.json'),
    ]
    for label, left, right in pairs:
        differences = {}
        for a, b in zip(rows(left), rows(right)):
            for key in ['task', 'scale', 'episodes', 'aug_factor', 'pomo_size',
                        'configured_batch_size', 'data_path', 'solution_path']:
                assert a[key] == b[key], (label, key)
            differences[a['task']] = a['aug_gap'] - b['aug_gap']
        checks.append({'comparison': label, 'gap_delta_pp': differences,
                       'max_abs_delta_pp': max(map(abs, differences.values()))})
    report = {'threshold_pp': 0.0001, 'checks': checks,
              'passed': all(c['max_abs_delta_pp'] <= 0.0001 for c in checks)}
    (RUN / 'calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert report['passed'], 'Review GPU differences before launching scan'


if __name__ == '__main__':
    main()
