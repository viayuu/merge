"""Evaluate a fixed phase of the TIES experiment on two allocated GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
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
TASKS = ['cvrp', 'cvrpb', 'cvrptw', 'ocvrp', 'ocvrptw']


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def validate_result(path):
    payload = json.loads(path.read_text())
    assert not payload['missing'], path
    assert [row['task'] for row in payload['rows']] == TASKS, path
    for row in payload['rows']:
        assert row['scale'] == 100 and row['episodes'] == 1000, row
        assert row['aug_factor'] == 8 and row['reference_available'], row
        assert all(math.isfinite(row[key]) for key in
                   ['aug_gap', 'no_aug_gap', 'aug_score', 'no_aug_score']), row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True,
                        choices=['calibration', 'scan', 'confirm'])
    parser.add_argument('--node', required=True, choices=['gpu05', 'gpu03'])
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    gpu_names = [torch.cuda.get_device_name(gpu) for gpu in range(2)]
    print('GPUs:', gpu_names, flush=True)

    state_path = RUN / f'state_{args.phase}_{args.node}.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    previous_jobs = {job['id']: job for job in previous.get('jobs', [])}
    jobs = [job for job in json.loads((RUN / 'evaluation_plan.json').read_text())
            if job['phase'] == args.phase and job['node'] == args.node]
    state = {'phase': args.phase, 'node': args.node,
             'started': previous.get('started', now()), 'jobs': []}
    for index, entry in enumerate(jobs):
        gpu = entry.get('gpu', index % 2)
        assert gpu in [0, 1], entry
        job = {**entry, 'gpu': gpu, 'gpu_name': gpu_names[gpu],
               'status': 'pending', 'returncode': None,
               'result': str(RUN / 'results' / f"{entry['id']}.json")}
        if entry['id'] in previous_jobs:
            old = previous_jobs[entry['id']]
            assert old['checkpoint'] == entry['checkpoint'], entry
            if old['status'] == 'completed':
                validate_result(Path(old['result']))
                job = old
            elif old['status'] != 'pending':
                raise SystemExit(f"Inspect interrupted/failed job before restart: {entry['id']}")
        state['jobs'].append(job)
    (RUN / 'results').mkdir(exist_ok=True)
    (RUN / 'logs').mkdir(exist_ok=True)
    lock = threading.Lock()

    def update(job=None, **fields):
        with lock:
            if job is not None:
                job.update(fields)
            state['updated'] = now()
            temporary = state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(state_path)

    def announce(event, job):
        with lock:
            print(now(), event, job['id'], 'GPU', job['gpu'], flush=True)

    def worker(gpu):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
        for job in state['jobs']:
            if job['gpu'] != gpu or job['status'] == 'completed':
                continue
            log_path = RUN / 'logs' / f"{job['id']}.log"
            update(job, status='running', started=now(), log=str(log_path))
            announce('START', job)
            returncode = None
            try:
                command = [sys.executable, str(ROOT / 'URS/test.py'),
                           '--data_dir', str(ROOT / 'URS/dataset'),
                           '--problem_set', ','.join(TASKS),
                           '--test_scale_list', '100', '--test_episodes', '1000',
                           '--seed', '3407', '--eval_type', 'greedy', '--cuda', '0',
                           '--aug_factor', '8', '128',
                           '--test_batch_size_small', '125', '125', '50', '8',
                           '--model_load', job['checkpoint'],
                           '--output_json', job['result']]
                with log_path.open('w') as handle:
                    process = subprocess.run(command, cwd=ROOT / 'URS', env=env,
                                             stdout=handle, stderr=subprocess.STDOUT)
                returncode = process.returncode
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
                validate_result(Path(job['result']))
                update(job, status='completed', returncode=returncode, finished=now())
                announce('DONE', job)
            except Exception as error:
                update(job, status='failed', returncode=returncode,
                       finished=now(), error=str(error))
                announce('FAILED', job)

    update()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(worker, gpu) for gpu in range(2)]:
            future.result()
    if any(job['status'] != 'completed' for job in state['jobs']):
        raise SystemExit(f'Queue incomplete; inspect {state_path} and logs.')
    print(now(), args.node, args.phase, 'ALL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
