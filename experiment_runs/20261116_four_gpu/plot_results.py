"""Plot this run only after all 45 declared model-task results are complete."""

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from analyze import RUN, SETS


def main():
    manifest = json.loads((RUN / 'manifest.json').read_text())
    with (RUN / 'results.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = {job['id'] for job in manifest['jobs']}
    if len(rows) != 45 or {row['job_id'] for row in rows} != expected:
        raise SystemExit(f'Waiting for all 45 results: currently {len(rows)}; no figure generated.')
    if any(row['protocol_matches'] != 'True' or row['paper_comparable'] != 'True' for row in rows):
        raise SystemExit('Protocol/paper comparison mismatch; inspect summary.md before plotting.')
    results = {(row['model'], row['task']): row for row in rows}

    font_path = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
    chinese = font_path.exists()
    if chinese:
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False, 'pdf.fonttype': 42})

    fig, (curve, bars) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.81, wspace=0.25)
    alpha = [0, 0.25, 0.5, 0.75, 1]
    models = ['specialist_cvrp', 'interp_025', 'average_2', 'interp_075', 'specialist_cvrptw']
    colors = ['#0072B2', '#D55E00']
    for task, color in zip(SETS['T2'], colors):
        values = [float(results[(model, task)]['gap_pct']) for model in models]
        baseline = float(results[('unified', task)]['gap_pct'])
        curve.plot(alpha, values, marker='o', linewidth=2, color=color, label=task.upper())
        curve.axhline(baseline, linestyle='--', color=color, linewidth=1.5,
                      label=f'{task.upper()} ' + ('本地URS' if chinese else 'local URS'))
    curve.set_xticks(alpha)
    curve.set_xlabel('α（CVRPTW专家权重）' if chinese else 'α (CVRPTW expert weight)')
    curve.set_ylabel('Gap %（越低越好）' if chinese else 'Gap % (lower is better)')
    curve.set_title('(a) 两专家插值' if chinese else '(a) Two-expert interpolation', pad=12)
    curve.legend(fontsize=9, ncol=2, loc='best')

    groups = [('T2', 'average_2'), ('T4', 'average_4'), ('T11', 'average_11')]
    paper, local, merged = [], [], []
    for name, model in groups:
        tasks = SETS[name]
        paper.append(mean(float(results[('unified', task)]['paper_urs_gap_pct']) for task in tasks))
        local.append(mean(float(results[('unified', task)]['gap_pct']) for task in tasks))
        merged.append(mean(float(results[(model, task)]['gap_pct']) for task in tasks))
    labels = ['论文URS', '本地URS', '等权合并 average_K'] if chinese else [
        'Paper URS', 'Local URS', 'Equal-weight merge average_K']
    for offset, values, color, label in zip(
            [-0.26, 0, 0.26], [paper, local, merged], ['#9CA3AF', '#0072B2', '#009E73'], labels):
        containers = bars.bar([x + offset for x in range(3)], values, width=0.24,
                              color=color, label=label)
        bars.bar_label(containers, fmt='%.2f', padding=3, fontsize=9)
    bars.set_xticks(range(3), [
        'T2 / K=2\nCVRP, CVRPTW',
        'T4 / K=4\nCVRP, CVRPB,\nCVRPTW, OCVRP',
        'T11 / K=11\n' + ('全部11任务' if chinese else 'All 11 tasks'),
    ])
    bars.set_ylabel('任务等权平均 gap %' if chinese else 'Task-macro mean gap %')
    bars.set_title('(b) 各固定任务集合内的比较' if chinese else '(b) Comparison within each fixed task set', pad=12)
    bars.legend(fontsize=9, loc='best')
    bars.margins(y=0.2)
    for axis in (curve, bars):
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=0.2)
        axis.spines[['top', 'right']].set_visible(False)

    fig.suptitle('URS单任务模型合并：原数据集评测' if chinese else
                 'Merging single-task URS models on the original datasets', fontsize=16, y=0.98)
    fig.text(0.5, 0.91, 'N=100 · 1000实例/任务 · greedy · 原8/128倍增强 · seed=3407' if chinese else
             'N=100 · 1000 instances/task · greedy · original 8/128× augmentation · seed=3407', ha='center', fontsize=10)
    fig.text(0.065, 0.105, 'θ(α) = (1−α) θCVRP + α θCVRPTW；虚线为各任务本地URS。' if chinese else
             'θ(α) = (1−α) θCVRP + α θCVRPTW; dashed lines show task-specific local URS.', fontsize=10)
    fig.text(0.065, 0.065, '右图各组任务集合不同，比较应在组内进行；average_K为K个对应专家等权合并成的单模型。' if chinese else
             'Right: each group uses a different task set; compare within groups. average_K is one model merged from K experts.', fontsize=10)
    fig.text(0.065, 0.025, 'T11: ' + ', '.join(task.upper() for task in SETS['T11']), fontsize=9)
    for suffix in ('png', 'pdf'):
        path = RUN / f'results_overview.{suffix}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
