"""Evaluate CSP+LOT rank experiments with the original URS protocol."""

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


def validate_result(path, episodes=1000):
    result = json.loads(path.read_text())
    assert not result['missing'], path
    assert [row['task'] for row in result['rows']] == TASKS, path
    for row in result['rows']:
        assert row['scale'] == 100 and row['episodes'] == episodes, row
        assert row['aug_factor'] == 8 and row['reference_available'], row
        assert all(math.isfinite(row[key]) for key in
                   ['aug_gap', 'no_aug_gap', 'aug_score', 'no_aug_score']), row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', help='Comma-separated job IDs; default: all jobs')
    parser.add_argument('--episodes', type=int, default=1000)
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    gpu_names = [torch.cuda.get_device_name(gpu) for gpu in range(2)]
    print('GPUs:', gpu_names, flush=True)
    output = RUN if args.episodes == 1000 else RUN / 'smoke' / f'e{args.episodes}'
    output.mkdir(parents=True, exist_ok=True)
    for folder in ['results', 'logs']:
        (output / folder).mkdir(exist_ok=True)
    state_path = output / 'queue_state.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    previous_jobs = {job['id']: job for job in previous.get('jobs', [])}
    plan = json.loads((RUN / 'evaluation_plan.json').read_text())
    selected = set(args.ids.split(',')) if args.ids else {job['id'] for job in plan}
    assert selected <= {job['id'] for job in plan}, selected
    assert len({job['id'] for job in plan}) == len(plan)
    state = {'started': previous.get('started', now()), 'episodes': args.episodes,
             'host': os.uname().nodename, 'gpu_names': gpu_names, 'jobs': []}
    for index, entry in enumerate(plan):
        gpu = entry.get('gpu', index % 2)
        job = {**entry, 'gpu': gpu, 'gpu_name': gpu_names[gpu],
               'status': 'pending', 'returncode': None,
               'result': str(output / 'results' / f"{entry['id']}.json")}
        old = previous_jobs.get(entry['id'])
        if old:
            assert old['checkpoint'] == entry['checkpoint'], entry
            if old['status'] == 'completed':
                validate_result(Path(old['result']), args.episodes)
                job = {**job, **old}
        if 'source_result' in entry and args.episodes == 1000:
            result = validate_result(Path(entry['source_result']))
            assert Path(result['checkpoint']).resolve() == Path(entry['checkpoint']).resolve()
            job.update(status='completed', returncode=0, result=entry['source_result'],
                       reused=True, gpu=None, gpu_name='historical RTX 3090')
        state['jobs'].append(job)
    lock = threading.Lock()

    def update(job=None, **fields):
        with lock:
            if job is not None:
                job.update(fields)
            state['updated'] = now()
            temporary = state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(state_path)

    def worker(gpu):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
        for job in state['jobs']:
            if (job['gpu'] != gpu or job['id'] not in selected
                    or job['status'] == 'completed'):
                continue
            log_path = output / 'logs' / f"{job['id']}.log"
            update(job, status='running', started=now(), log=str(log_path))
            print(now(), 'START', job['id'], 'GPU', gpu, flush=True)
            command = [sys.executable, str(ROOT / 'URS/test.py'),
                       '--data_dir', str(ROOT / 'URS/dataset'),
                       '--problem_set', ','.join(TASKS),
                       '--test_scale_list', '100', '--test_episodes', str(args.episodes),
                       '--seed', '3407', '--eval_type', 'greedy', '--cuda', '0',
                       '--aug_factor', '8', '128',
                       '--test_batch_size_small', '125', '125', '50', '8',
                       '--model_load', job['checkpoint'], '--output_json', job['result']]
            returncode = None
            try:
                with log_path.open('w') as handle:
                    process = subprocess.run(command, cwd=ROOT / 'URS', env=env,
                                             stdout=handle, stderr=subprocess.STDOUT)
                returncode = process.returncode
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
                result = validate_result(Path(job['result']), args.episodes)
                assert Path(result['checkpoint']).resolve() == Path(job['checkpoint']).resolve()
                update(job, status='completed', returncode=0, finished=now())
                print(now(), 'DONE', job['id'], 'GPU', gpu, flush=True)
            except Exception as error:
                update(job, status='failed', returncode=returncode,
                       finished=now(), error=str(error))
                print(now(), 'FAILED', job['id'], str(error), flush=True)

    update()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(worker, gpu) for gpu in range(2)]:
            future.result()
    if any(job['status'] != 'completed' for job in state['jobs'] if job['id'] in selected):
        raise SystemExit(f'Queue incomplete; inspect {state_path} and logs.')
    print(now(), 'SELECTED JOBS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
