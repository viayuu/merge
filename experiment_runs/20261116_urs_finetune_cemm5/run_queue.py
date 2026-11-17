"""Run the five independent URS fine-tunes on two GPUs in one Slurm step."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import torch


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
TASKS = ['cvrp', 'ocvrp', 'cvrpbp', 'cvrpl', 'cvrptw']
BASE = ROOT / 'URS/pretrained/unified_checkpoint_500.pt'


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    print('GPUs:', [torch.cuda.get_device_name(i) for i in range(2)], flush=True)
    directory = RUN / 'smoke' if args.smoke else RUN
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'queue_state.json'
    jobs = ([{'name': 'base', 'kind': 'evaluate'}] if not args.smoke else [])
    # Start the longer constrained tasks early so the two workers stay busy.
    jobs += [{'name': task, 'kind': 'finetune'} for task in
             ['cvrptw', 'cvrpl', 'cvrpbp', 'cvrp', 'ocvrp']]
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'started': now(), 'smoke': args.smoke,
        'jobs': [{**job, 'status': 'pending'} for job in jobs],
    }
    pending = queue.Queue()
    for job in state['jobs']:
        if job['status'] == 'pending':
            pending.put(job)
        elif job['status'] != 'completed':
            raise SystemExit(f"Inspect interrupted/failed job before restarting: {job['name']}")
    lock = threading.Lock()

    def update(job, **fields):
        with lock:
            job.update(fields)
            state['updated'] = now()
            temporary = state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(state_path)

    def execute(command, log_path, env):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as handle:
            subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)

    def worker(gpu):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            name = job['name']
            update(job, status='running', gpu=gpu, started=now())
            print(now(), 'START', name, 'GPU', gpu, flush=True)
            try:
                checkpoint = BASE
                if job['kind'] == 'finetune':
                    model_dir = directory / 'models' / name
                    epochs = 1 if args.smoke else 30
                    command = [sys.executable, str(ROOT / 'merge_lab/finetune.py'),
                               '--task', name, '--base', str(BASE),
                               '--output-dir', str(model_dir), '--epochs', str(epochs),
                               '--instances-per-epoch', '160' if args.smoke else '100000',
                               '--batch-size', '128', '--lr', '3e-5', '--seed', '3407',
                               '--validation-episodes', '16' if args.smoke else '1000',
                               '--validation-batch-size', '125']
                    execute(command, directory / 'logs' / f'{name}_train.log', env)
                    checkpoint = model_dir / f'checkpoint_{epochs}.pt'
                    assert checkpoint.is_file(), checkpoint
                    update(job, training_finished=now(), checkpoint=str(checkpoint))
                if not args.smoke:
                    update(job, phase='evaluation')
                    command = [sys.executable, str(ROOT / 'merge_lab/evaluate.py'),
                               '--checkpoints', str(checkpoint),
                               '--out-dir', str(RUN / 'evaluation' / name),
                               '--problem_set', ','.join(TASKS), '--cuda', '0',
                               '--seed', '3407', '--test_scale_list', '100',
                               '--test_episodes', '1000', '--aug_factor', '8', '128',
                               '--test_batch_size_small', '125', '125', '50', '8']
                    execute(command, RUN / 'logs' / f'{name}_eval.log', env)
                    result = RUN / 'evaluation' / name / (checkpoint.stem + '.json')
                    payload = json.loads(result.read_text())
                    assert not payload['missing']
                    assert [row['task'] for row in payload['rows']] == TASKS
                    assert all(row['episodes'] == 1000 for row in payload['rows'])
                    update(job, result=str(result))
                update(job, status='completed', finished=now())
                print(now(), 'DONE', name, flush=True)
            except Exception as error:
                update(job, status='failed', finished=now(), error=str(error))
                print(now(), 'FAILED', name, repr(error), flush=True)
                return
            if not args.smoke:
                with lock:
                    subprocess.run([sys.executable, str(RUN / 'analyze.py')], check=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(worker, gpu) for gpu in range(2)]:
            future.result()
    if any(job['status'] != 'completed' for job in state['jobs']):
        raise SystemExit('Queue incomplete; inspect queue_state.json and logs.')
    print(now(), 'ALL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
