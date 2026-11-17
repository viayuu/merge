"""Print and save the current progress of the two URS continuation runs."""

import argparse
from datetime import datetime
import json
from pathlib import Path


RUN = Path(__file__).resolve().parent
TASKS = ['cvrp', 'cvrpb', 'cvrptw', 'ocvrp', 'ocvrptw']


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    directory = RUN / 'smoke' if args.smoke else RUN
    state = read_json(directory / 'queue_state.json')
    lines = ['# URS 两组多任务续训进度', '',
             '更新时间：' + datetime.now().astimezone().isoformat(timespec='seconds'), '',
             '| 组别 | 状态 | GPU | 已完成训练轮 | 当前训练轮 | 当前轮实例 | 学习率 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    task_lines = ['', '| 组别 | CVRP | CVRPB | CVRPTW | OCVRP | OCVRPTW |',
                  '| --- | --- | --- | --- | --- | --- |']
    errors = []
    for job in state.get('jobs', []):
        model_dir = directory / 'models' / job['name']
        metadata = read_json(model_dir / 'metadata.json')
        progress = read_json(model_dir / 'progress.json')
        counts_path = model_dir / 'training_counts.jsonl'
        counts = [json.loads(line) for line in counts_path.read_text().split('\n')[:-1]
                  if line] if counts_path.exists() else []
        status = job['status']
        if status == 'running':
            status = job.get('phase', status)
        epochs = metadata.get('additional_epochs', job['epochs'])
        current = str(progress.get('additional_epoch', '—'))
        absolute = progress.get('absolute_epoch')
        if absolute is not None:
            current += f'（绝对{absolute}）'
        seen, target = progress.get('completed_instances', 0), progress.get('target_instances', '—')
        lr = progress.get('lr', metadata.get('initial_lr', '—'))
        lines.append(f"| {job['name']} | {status} | 4090/{job['gpu']} | "
                     f"{len(counts)}/{epochs} | {current} | {seen}/{target} | {lr} |")
        per_task = progress.get('task_instances', {})
        task_lines.append('| ' + job['name'] + ' | ' +
                          ' | '.join(str(per_task.get(task, 0)) for task in TASKS) + ' |')
        if job.get('error'):
            errors += ['', f"错误（{job['name']}）：{job['error']}"]
    lines += ['', '当前轮各任务训练实例数：'] + task_lines + errors
    lines += ['', '训练轮完成数来自 training_counts.jsonl，不包含随后验证和保存所需的时间。',
              '两组都从 epoch 500 续训至 530；训练完成后自动按原协议评测五项任务。',
              '训练期间验证使用无增强结果，最终比较使用原8倍增强结果。', '']
    if args.smoke:
        lines[-3] = '当前为小规模启动检查：从 epoch 500 续训至 501，不计入正式结果。'
    output = '\n'.join(lines)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'PROGRESS.md').write_text(output)
    print(output)


if __name__ == '__main__':
    main()
