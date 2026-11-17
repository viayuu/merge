"""Train two pending experts early; the existing 4090 queue handles evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import torch


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.device_count() == 2
    directory = RUN / 'smoke_3090' if args.smoke else RUN
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / '3090_state.json'
    state = {'allocation': 1465, 'node': 'gpu03', 'jobs': [
        {'name': task, 'gpu': gpu, 'status': 'pending'}
        for gpu, task in enumerate(['cvrp', 'ocvrp'])]}
    lock = threading.Lock()

    def update(job, **fields):
        with lock:
            job.update(fields)
            state['updated'] = datetime.now().astimezone().isoformat(timespec='seconds')
            temp = state_path.with_suffix('.tmp')
            temp.write_text(json.dumps(state, indent=2) + '\n')
            temp.replace(state_path)

    def worker(job):
        task = job['name']
        command = [sys.executable, str(ROOT / 'merge_lab/finetune.py'),
                   '--task', task, '--output-dir', str(directory / 'models' / task),
                   '--epochs', '1' if args.smoke else '30',
                   '--instances-per-epoch', '160' if args.smoke else '100000',
                   '--batch-size', '128', '--lr', '3e-5', '--seed', '3407',
                   '--validation-episodes', '16' if args.smoke else '1000',
                   '--validation-batch-size', '125']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(job['gpu']),
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
        log_path = directory / 'logs' / f'{task}_train_3090.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        update(job, status='running', started=datetime.now().astimezone().isoformat(timespec='seconds'))
        print('START', task, '3090 GPU', job['gpu'], flush=True)
        with log_path.open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        update(job, status='completed' if result.returncode == 0 else 'failed',
               returncode=result.returncode, finished=datetime.now().astimezone().isoformat(timespec='seconds'))
        print(job['status'].upper(), task, flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(worker, job) for job in state['jobs']]:
            future.result()
    if any(job['status'] != 'completed' for job in state['jobs']):
        raise SystemExit('3090 training incomplete; inspect 3090_state.json and logs.')


if __name__ == '__main__':
    main()
