"""Compare original URS metrics with the published N=100 tables."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).with_name('paper_results.csv')


def summarize(result_paths, out_dir, paper_path=PAPER):
    with Path(paper_path).open(newline='') as handle:
        paper = {(r['task'], int(r['scale'])): r for r in csv.DictReader(handle)}
    rows = []
    notes = []
    for path in result_paths:
        payload = json.loads(Path(path).read_text())
        model = Path(payload['checkpoint']).stem
        run = f'{Path(path).parent.name}/{Path(path).stem}'
        if payload.get('missing'):
            notes.append(f"{model} 缺少任务/规模：{payload['missing']}")
        for result in payload['rows']:
            task, scale = result['task'], int(result['scale'])
            if not result['reference_available']:
                notes.append(f'{run} {task}{scale} 缺少原外部参考解；保留原输出，论文差值留空。')
            reference = paper.get((task, scale))
            comparable = bool(
                reference
                and int(result['episodes']) == int(reference['episodes'])
                and int(result['aug_factor']) == int(reference['augmentation_factor'])
                and payload['model_params']['eval_type'] == 'greedy'
                and Path(payload['data_dir']).resolve() == (ROOT / 'URS/dataset').resolve()
                and result['reference_available']
            )
            urs_gap = float(reference['paper_urs_gap_pct']) if reference else None
            stl_gap = float(reference['paper_stl_gap_pct']) if reference else None
            rows.append({
                'run': run, 'model': model, 'task': task, 'scale': scale,
                'episodes': int(result['episodes']), 'aug_factor': int(result['aug_factor']),
                'objective': result['aug_score'], 'reference_objective': result['oracle_score'],
                'gap_pct': result['aug_gap'], 'no_aug_gap_pct': result['no_aug_gap'],
                'paper_urs_gap_pct': urs_gap, 'paper_stl_gap_pct': stl_gap,
                'delta_vs_paper_urs_pp': result['aug_gap'] - urs_gap if comparable else None,
                'delta_vs_paper_stl_pp': result['aug_gap'] - stl_gap if comparable else None,
                'paper_comparable': comparable,
                'reference_available': result['reference_available'],
                'time_seconds': result['time'], 'seed': payload['seed'],
                'source_json': str(Path(path).resolve()),
            })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_dir / 'comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def fmt(value):
        return '—' if value is None else f'{value:.3f}'

    lines = [
        '# URS 原数据集结果对比', '',
        'Gap 沿用原 DataReader/Tester 的参考值及公式；差值单位为百分点，负值更好。',
        '仅原 dataset、N=100、1000例、greedy 与论文增强倍数匹配的行计算论文差值。',
        '小样本/无增强仅用于检查通路，不视为论文成绩复现。论文数值保留原表精度。', '',
        '| 运行/模型 | 任务 | N | 实例数 | 增强 | 本次 gap % | 论文 URS % | 论文 STL % | ΔURS pp | ΔSTL pp |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['task']} | {r['scale']} | {r['episodes']} | {r['aug_factor']} | "
            f"{fmt(r['gap_pct'])} | {fmt(r['paper_urs_gap_pct'])} | {fmt(r['paper_stl_gap_pct'])} | "
            f"{fmt(r['delta_vs_paper_urs_pp'])} | {fmt(r['delta_vs_paper_stl_pp'])} |"
        )
    lines += ['', '## 同一次运行、同一规模内的任务等权均值', '',
              '不同运行分别显示；比较模型时应使用相同任务集合。', '',
              '| 运行/模型 | N | 任务集合 | 本次平均 gap % | 可与论文比较数 | 平均 ΔURS pp |',
              '|---|---:|---|---:|---:|---:|']
    for source, scale in dict.fromkeys((r['source_json'], r['scale']) for r in rows):
        group = [r for r in rows if (r['source_json'], r['scale']) == (source, scale)]
        comparable = [r for r in group if r['paper_comparable']]
        delta = mean(r['delta_vs_paper_urs_pp'] for r in comparable) if comparable else None
        tasks = ', '.join(r['task'] for r in group)
        lines.append(f"| {group[0]['run']} | {scale} | {tasks} | {mean(r['gap_pct'] for r in group):.3f} | {len(comparable)} | {fmt(delta)} |")
    if notes:
        lines += ['', '## 未完成项', '', *notes]
    lines += ['', '论文来源：arXiv:2509.23413v3，Table 2/3；详见 merge_lab/paper_results.md。', '']
    (out_dir / 'comparison.md').write_text('\n'.join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path, help='JSON files from URS/test.py --output_json')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--paper', type=Path, default=PAPER)
    args = parser.parse_args()
    summarize(args.results, args.out_dir, args.paper)


if __name__ == '__main__':
    main()
