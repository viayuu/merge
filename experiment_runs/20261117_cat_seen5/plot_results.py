"""Plot the completed CAT alpha scan and default per-task comparisons."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(__file__).resolve().parent
TASKS = ['cvrp', 'cvrpb', 'cvrptw', 'ocvrp', 'ocvrptw']


def save(figure, name):
    for extension in ['png', 'pdf']:
        figure.savefig(RUN / f'{name}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(figure)


def main():
    result = json.loads((RUN / 'summary.json').read_text())
    assert result['completed'] == result['total'], 'Run analyze.py after all jobs finish.'
    groups = {row['config_id']: row for row in result['groups']}
    alphas = np.arange(1, 10) / 10
    scan = [groups[f'paper_full_c4_a0p{index}'] for index in range(1, 10)]
    assert all(row['completed_repeats'] == 3 for row in scan)
    values = np.array([row['mean_gap'] for row in scan])
    deviations = np.array([row['std_gap'] for row in scan])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})

    figure, axis = plt.subplots(figsize=(7.1, 4.3), constrained_layout=True)
    axis.plot(alphas, values, 'o-', color='#2878B5', label='Paper-formula CAT, c = 4')
    axis.fill_between(alphas, values - deviations, values + deviations,
                      color='#2878B5', alpha=0.18, label='±1 std over 3 exemplar selections')
    axis.axhline(result['base']['mean_gap'], color='#555555', linestyle='--',
                 label='Original URS base (epoch 500)')
    axis.scatter([0.4], [values[3]], s=95, color='#E6812B', zorder=5,
                 label='Preset scale α = 0.4')
    axis.set(xlabel='Merge scale α', ylabel='Mean gap across five tasks (%)',
             title='CAT merge-scale sensitivity — lower is better', xticks=alphas)
    axis.grid(axis='y', alpha=0.2)
    axis.legend(fontsize=8.5)
    save(figure, 'alpha_sensitivity')

    figure, axis = plt.subplots(figsize=(8.1, 4.4), constrained_layout=True)
    x = np.arange(len(TASKS))
    width = 0.36
    settings = [
        ('upstream_c4_a0p4', 'Released-code adaptation', '#E6812B'),
        ('paper_full_c4_a0p4', 'Paper-formula adaptation', '#2878B5'),
    ]
    for index, (config_id, label, color) in enumerate(settings):
        row = groups[config_id]
        assert row['completed_repeats'] == 3
        delta = [row[f'{task}_delta_pp'] for task in TASKS]
        deviation = [row[f'{task}_std'] for task in TASKS]
        axis.bar(x + (index - 0.5) * width, delta, width=width,
                 yerr=deviation, capsize=4, label=label, color=color,
                 error_kw={'elinewidth': 1})
    axis.axhline(0, color='#555555', linewidth=1)
    axis.set(xticks=x, xticklabels=[task.upper() for task in TASKS],
             ylabel='Gap change versus original URS base (percentage points)',
             title='Preset CAT (c = 4, α = 0.4): negative means improvement')
    axis.grid(axis='y', alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(fontsize=9)
    figure.supxlabel('Error bars: ±1 std over 3 exemplar selections', fontsize=9)
    save(figure, 'default_task_deltas')
    print('Saved alpha_sensitivity and default_task_deltas as PNG and PDF.')


if __name__ == '__main__':
    main()
