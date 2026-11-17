"""Run two URS continuation budgets, one per allocated RTX 4090."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

import torch


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
BASE = ROOT / 'URS/pretrained/unified_checkpoint_500.pt'
TASKS = ['cvrp', 'cvrpb', 'cvrptw', 'ocvrp', 'ocvrptw']


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    devices = [torch.cuda.get_device_name(i) for i in range(2)]
    assert all('4090' in name for name in devices), devices
    print('GPUs:', devices, flush=True)

    directory = RUN / 'smoke' if args.smoke else RUN
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'logs').mkdir(exist_ok=True)
    epochs = 1 if args.smoke else 30
    final_epoch = 500 + epochs
    state_path = directory / 'queue_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'started': now(), 'smoke': args.smoke,
        'jobs': [
            {'name': name, 'budget': budget, 'gpu': gpu, 'gpu_name': devices[gpu],
             'epochs': epochs, 'instances': 160 if args.smoke else 100000,
             'status': 'pending', 'phase': 'queued', 'returncode': None,
             'checkpoint': str(directory / 'models' / name / f'checkpoint_{final_epoch}.pt'),
             'result': None}
            for gpu, (name, budget) in enumerate([
                ('total_100k', 'total'), ('per_task_100k', 'per-task')])],
    }
    assert state['smoke'] == args.smoke
    for job in state['jobs']:
        if job['status'] not in ['pending', 'completed']:
            raise SystemExit(f"Inspect interrupted/failed job before restart: {job['name']}")
    lock = threading.Lock()
    analysis_lock = threading.Lock()

    def update(job=None, **fields):
        with lock:
            if job is not None:
                job.update(fields)
            state['updated'] = now()
            temporary = state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(state_path)

    def execute(command, log_path, env, cwd=ROOT):
        with log_path.open('w') as handle:
            subprocess.run(command, cwd=cwd, env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)

    def refresh_progress():
        command = [sys.executable, str(RUN / 'progress.py')]
        if args.smoke:
            command.append('--smoke')
        subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL, check=True)

    def worker(job):
        if job['status'] == 'completed':
            return
        name, gpu = job['name'], job['gpu']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
        update(job, status='running', phase='training', started=now())
        print(now(), 'START', name, 'GPU', gpu, flush=True)
        try:
            command = [sys.executable, str(ROOT / 'merge_lab/continue_multitask.py'),
                       '--budget', job['budget'], '--instances', str(job['instances']),
                       '--epochs', str(epochs), '--base', str(BASE), '--cuda', '0',
                       '--output-dir', str(directory / 'models' / name),
                       '--seed', '3407',
                       '--validation-episodes', '16' if args.smoke else '1000',
                       '--validation-batch-size', '16' if args.smoke else '1000']
            execute(command, directory / 'logs' / f'{name}_train.log', env)
            assert Path(job['checkpoint']).is_file(), job['checkpoint']
            update(job, training_finished=now())
            if not args.smoke:
                result_path = directory / 'evaluation' / name / f'checkpoint_{final_epoch}.json'
                result_path.parent.mkdir(parents=True, exist_ok=True)
                update(job, phase='evaluation')
                command = [sys.executable, str(ROOT / 'URS/test.py'),
                           '--data_dir', str(ROOT / 'URS/dataset'),
                           '--model_load', job['checkpoint'], '--output_json', str(result_path),
                           '--problem_set', ','.join(TASKS), '--cuda', '0',
                           '--seed', '3407', '--eval_type', 'greedy',
                           '--test_scale_list', '100', '--test_episodes', '1000',
                           '--aug_factor', '8', '128',
                           '--test_batch_size_small', '125', '125', '50', '8']
                execute(command, directory / 'logs' / f'{name}_eval.log', env, ROOT / 'URS')
                payload = json.loads(result_path.read_text())
                assert not payload['missing']
                assert [row['task'] for row in payload['rows']] == TASKS
                for row in payload['rows']:
                    assert row['episodes'] == 1000 and row['scale'] == 100
                    assert row['aug_factor'] == 8 and row['reference_available']
                    assert math.isfinite(row['aug_gap'])
                update(job, result=str(result_path))
            update(job, status='completed', phase='completed', returncode=0, finished=now())
            print(now(), 'DONE', name, flush=True)
            if not args.smoke:
                with analysis_lock:
                    subprocess.run([sys.executable, str(RUN / 'analyze.py')],
                                   cwd=ROOT, check=True)
        except Exception as error:
            update(job, status='failed', returncode=getattr(error, 'returncode', None),
                   finished=now(), error=str(error))
            print(now(), 'FAILED', name, repr(error), flush=True)

    update()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, job) for job in state['jobs']]
        pending = set(futures)
        while pending:
            _, pending = wait(pending, timeout=60)
            refresh_progress()
        for future in futures:
            future.result()
    if any(job['status'] != 'completed' for job in state['jobs']):
        raise SystemExit(f'Queue incomplete; inspect {state_path} and logs.')
    print(now(), 'ALL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
