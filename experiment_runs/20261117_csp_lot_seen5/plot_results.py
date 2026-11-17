"""Plot CSP protection strength against matched pure LOT controls."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent


def plot_mode(result, mode, filename):
    groups = result['groups']
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    for axis, (samples, scale) in zip(axes, [(64, 1.4), (1000, 1.2)]):
        rows = sorted([row for row in groups if row['family'] == 'csp_lot' and
                       row['samples'] == samples and row['scale'] == scale and
                       row['rank_mode'] == mode], key=lambda row: row['k'])
        control = next(row for row in groups if row['family'] == 'lot' and
                       row['samples'] == samples and row['scale'] == scale)
        x = [row['k'] for row in rows]
        axis.plot(x, [row['mean_gap'] for row in rows], 'o-', color='#2878B5', label='CSP + LOT')
        repeated = [row for row in rows if row['std_gap'] is not None]
        if repeated:
            axis.errorbar([row['k'] for row in repeated], [row['mean_gap'] for row in repeated],
                          yerr=[row['std_gap'] for row in repeated], fmt='none',
                          capsize=3, color='#2878B5')
        axis.axhline(control['mean_gap'], color='#E6812B', linestyle='--', label='Matched pure LOT')
        axis.axhline(result['base']['mean_gap'], color='#555555', linestyle=':', label='Old URS base')
        axis.set(title=f'{samples} instances per task; scale {scale}',
                 xlabel='Protected rank r' if mode == 'rank' else 'Protected squared-singular-value energy p',
                 ylabel='Five-task mean gap (%) — lower is better')
        axis.set_xticks([1, 5, 10, 15, 20] if mode == 'rank' else x)
        axis.grid(axis='y', alpha=.2)
        axis.legend(fontsize=8)
    figure.supxlabel('Error bars: ±1 population std for N64 (3 selections); N1000 uses the full set once.\n'
                     'Each panel has its own matched LOT control; between-panel differences are not sample effects.', fontsize=9)
    for extension in ['png', 'pdf']:
        figure.savefig(RUN / f'{filename}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(figure)


def main():
    result = json.loads((RUN / 'summary.json').read_text())
    assert result['completed'] == result['total'], 'Complete evaluations and run analyze.py first.'
    assert all(row['completed_repeats'] == row['planned_repeats'] for row in result['groups'])
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    plot_mode(result, 'rank', 'rank_sensitivity')
    plot_mode(result, 'energy', 'energy_sensitivity')
    print('Saved rank_sensitivity and energy_sensitivity as PNG/PDF.')


if __name__ == '__main__':
    main()
