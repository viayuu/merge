"""Refresh a compact progress page from queue state and existing training logs."""

from datetime import datetime
import json
from pathlib import Path
import re


RUN = Path(__file__).resolve().parent
state = json.loads((RUN / 'queue_state.json').read_text())
lines = ['# URS 五任务微调进度', '',
         '更新时间：' + datetime.now().astimezone().isoformat(timespec='seconds'), '',
         '| 任务 | 状态 | GPU | 最近训练进度 | 已完成epoch | 本任务最新验证gap（无增强） |',
         '|---|---|---:|---|---:|---:|']
for job in state['jobs']:
    log_path = RUN / 'logs' / f"{job['name']}_train.log"
    log = log_path.read_text() if log_path.exists() else ''
    batches = re.findall(r'Epoch (\d+): instances (\d+)/(\d+), batch (\d+)', log)
    validation = re.findall(r'Epoch\s+(\d+): Average gap: ([\d.eE+-]+)%', log)
    progress = f'epoch {batches[-1][0]}，{batches[-1][1]}/{batches[-1][2]}例' if batches else '—'
    epoch, gap = validation[-1] if validation else ('0', '—')
    status = job['status']
    if status == 'running' and job.get('phase') == 'evaluation':
        status = 'evaluating'
    lines.append(f"| {job['name']} | {status} | {job.get('gpu', '—')} | {progress} | {epoch} | {gap} |")
    if job.get('error'):
        lines.append(f"\n错误（{job['name']}）：{job['error']}\n")
lines += ['', '每分钟更新。训练验证值无增强，不与最终8倍增强成绩直接比较。',
          '正式五任务结果见同目录RESULTS.md；每个专家完成30个epoch后自动交叉评测。', '']
(RUN / 'PROGRESS.md').write_text('\n'.join(lines))
