"""Plot exemplar-count effects at fixed scales and LOT's 64-sample scale scan."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent


def save(figure, name):
    for extension in ['png', 'pdf']:
        figure.savefig(RUN / f'{name}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(figure)


def curve(axis, rows, x_key, label, color):
    rows = sorted(rows, key=lambda row: row[x_key])
    x = [row[x_key] for row in rows]
    y = [row['mean_gap'] for row in rows]
    axis.plot(x, y, 'o-', label=label, color=color)
    repeated = [row for row in rows if row['std_gap'] is not None]
    if repeated:
        axis.errorbar([row[x_key] for row in repeated],
                      [row['mean_gap'] for row in repeated],
                      yerr=[row['std_gap'] for row in repeated], fmt='none',
                      capsize=4, color=color, alpha=0.85)


def main():
    result = json.loads((RUN / 'summary.json').read_text())
    assert result['completed'] == result['total'], 'Run analyze.py after all jobs finish.'
    groups = result['groups']
    assert all(row['completed_repeats'] == row['planned_repeats'] for row in groups)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    settings = [
        ('lot', 1.2, 'LOT, scale 1.2', '#2878B5'),
        ('cat_paper', 0.2, 'CAT paper formula, scale 0.2', '#2A9D8F'),
        ('cat_paper', 0.4, 'CAT paper formula, scale 0.4', '#8E6CBA'),
        ('cat_upstream', 0.4, 'CAT released-code adaptation, scale 0.4', '#E6812B'),
    ]
    figure, axis = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
    for family, scale, label, color in settings:
        rows = [row for row in groups if row['family'] == family and row['scale'] == scale]
        assert {row['samples'] for row in rows} == {2, 16, 64, 256, 1000}, (family, scale)
        curve(axis, rows, 'samples', label, color)
    axis.axhline(result['base']['mean_gap'], color='#555555', linestyle='--',
                 label='Original URS base (epoch 500)')
    axis.set_xscale('log', base=2)
    axis.set(xticks=[2, 16, 64, 256, 1000], xticklabels=['2', '16', '64', '256', '1000'],
             xlabel='Unlabeled instances per task', ylabel='Mean gap across five tasks (%)',
             title='Effect of exemplar count — lower is better')
    axis.grid(axis='y', alpha=0.2)
    axis.legend(loc='center right', bbox_to_anchor=(1, .64), fontsize=8.5)
    figure.supxlabel('Error bars: ±1 std over 3 selections; 1000 uses the full set once', fontsize=9)
    save(figure, 'sample_count_sensitivity')

    rows = [row for row in groups if row['family'] == 'lot' and row['samples'] == 64]
    figure, axis = plt.subplots(figsize=(7.1, 4.4), constrained_layout=True)
    curve(axis, rows, 'scale', 'LOT: 64 instances per task', '#2878B5')
    preset = next(row for row in rows if row['scale'] == 1.2)
    axis.scatter([1.2], [preset['mean_gap']], s=90, color='#E6812B', zorder=5,
                 label='Released ViT-B/32 preset: scale 1.2')
    axis.axhline(result['base']['mean_gap'], color='#555555', linestyle='--',
                 label='Original URS base (epoch 500)')
    axis.set(xticks=sorted(row['scale'] for row in rows), xlabel='LOT merge scale',
             ylabel='Mean gap across five tasks (%)', title='LOT scale sensitivity — lower is better')
    axis.grid(axis='y', alpha=0.2)
    axis.legend(fontsize=8.5)
    zoom = axis.inset_axes([.08, .32, .48, .42])
    curve(zoom, [row for row in rows if row['scale'] <= 1.5],
          'scale', 'LOT', '#2878B5')
    zoom.axhline(result['base']['mean_gap'], color='#555555', linestyle='--')
    zoom.scatter([1.2], [preset['mean_gap']], s=35, color='#E6812B', zorder=5)
    zoom.set(xticks=[.6, .8, 1., 1.2, 1.5], title='Zoom: scales 0.6–1.5')
    zoom.tick_params(labelsize=8)
    zoom.title.set_fontsize(9)
    zoom.grid(axis='y', alpha=.2)
    figure.supxlabel('Error bars: ±1 std over 3 exemplar selections', fontsize=9)
    save(figure, 'lot_scale_sensitivity')
    print('Saved sample_count_sensitivity and lot_scale_sensitivity as PNG and PDF.')


if __name__ == '__main__':
    main()
