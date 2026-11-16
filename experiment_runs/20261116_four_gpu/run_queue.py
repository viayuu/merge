"""Fixed URS experiment queue: one Slurm step per node, two local GPU workers."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

RUN = Path(__file__).resolve().parent
manifest = json.loads((RUN / 'manifest.json').read_text())
lock = threading.Lock()


def now():
    return datetime.now().astimezone().isoformat()


def save(path, value):
    value['updated'] = now()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def node_worker(node, phase):
    path = RUN / f'{node}_{phase}_state.json'
    local = json.loads(path.read_text())
    pending = queue.Queue()
    for job in local['jobs']:
        pending.put(job)

    def worker(slot):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            with lock:
                job.update(status='running', slot=slot['name'], node=node, gpu=slot['gpu'],
                           allocation=slot['allocation'], started=now())
                save(path, local)
                print('START', job['id'], slot['name'], flush=True)
            protocol = manifest['protocol']
            command = [
                sys.executable, str(RUN / 'gpu_test.py'),
                '--data_dir', protocol['data_dir'], '--problem_set', job['task'],
                '--test_scale_list', str(protocol['scale']), '--test_episodes', str(protocol['episodes']),
                '--seed', str(protocol['seed']), '--eval_type', protocol['eval_type'], '--cuda', '0',
                '--aug_factor', *map(str, protocol['aug_factor']),
                '--test_batch_size_small', *map(str, protocol['test_batch_size_small']),
                '--model_load', job['checkpoint'], '--output_json', job['output'],
            ]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(slot['gpu']),
                   'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
            with open(job['log'], 'w') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            error = None
            if result.returncode == 0:
                try:
                    payload = json.loads(Path(job['output']).read_text())
                    assert not payload['missing'] and len(payload['rows']) == 1
                    row = payload['rows'][0]
                    assert row['task'] == job['task'] and row['episodes'] == protocol['episodes']
                    assert payload['device'].startswith('cuda')
                    assert all(math.isfinite(row[key]) for key in ['aug_gap', 'aug_score', 'no_aug_gap', 'no_aug_score'])
                except (OSError, ValueError, AssertionError) as exc:
                    error = str(exc) or 'Incomplete or unexpected result JSON'
            else:
                error = f'Python exit {result.returncode}; inspect {job["log"]}'
            with lock:
                job.update(status='completed' if error is None else 'failed',
                           returncode=result.returncode, finished=now(), error=error)
                save(path, local)
                print(job['status'].upper(), job['id'], slot['name'], flush=True)

    slots = [slot for slot in manifest['slots'] if slot['node'] == node]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(worker, slot) for slot in slots]:
            future.result()
    return int(any(job['status'] != 'completed' for job in local['jobs']))


def main():
    state_path = RUN / 'queue_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'run_id': manifest['run_id'],
        'jobs': [{**job, 'status': 'pending'} for job in manifest['jobs']],
    }
    if any(job['status'] == 'running' for job in state['jobs']):
        raise SystemExit('State contains running jobs. Do not launch a second queue.')
    if '--retry-failed' in sys.argv:
        for job in state['jobs']:
            if job['status'] == 'failed':
                job['status'] = 'pending'
    save(state_path, state)
    nodes = list(dict.fromkeys(slot['node'] for slot in manifest['slots']))
    by_id = {job['id']: job for job in state['jobs']}
    for phase in ['B1', 'B2', 'B3']:
        jobs = [job for job in state['jobs'] if job['phase'] == phase]
        if any(job['status'] == 'failed' for job in jobs):
            raise SystemExit(f'{phase} has failed jobs; inspect logs before --retry-failed')
        pending = sorted([job for job in jobs if job['status'] == 'pending'],
                         key=lambda job: not job['task'].startswith('a'))
        processes = []
        for index, node in enumerate(nodes):
            assigned = pending[index::len(nodes)]
            if not assigned:
                continue
            node_path = RUN / f'{node}_{phase}_state.json'
            save(node_path, {'jobs': assigned})
            slot = next(slot for slot in manifest['slots'] if slot['node'] == node)
            log = open(RUN / 'logs' / f'{node}_{phase}_worker.log', 'w')
            command = ['srun', '--jobid=' + slot['allocation'], '--overlap', '-N1', '-n1',
                       '-c4', '--immediate=10', sys.executable, str(Path(__file__).resolve()),
                       '--node', node, phase]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append((process, log, node_path, assigned))
        while processes:
            remaining = []
            for process, log, node_path, assigned in processes:
                code = process.poll()
                local = json.loads(node_path.read_text())
                for job in local['jobs']:
                    by_id[job['id']].update(job)
                if code is None:
                    remaining.append((process, log, node_path, assigned))
                else:
                    log.close()
                    for job in assigned:
                        if by_id[job['id']]['status'] != 'completed':
                            by_id[job['id']].update(status='failed', error=f'Node worker exit {code}; inspect node/job logs')
            save(state_path, state)
            processes = remaining
            if processes:
                time.sleep(5)
        if any(job['status'] != 'completed' for job in jobs):
            raise SystemExit(f'{phase} incomplete; inspect logs before continuing')
        print('PHASE COMPLETE', phase, flush=True)
        subprocess.run([sys.executable, str(RUN / 'analyze.py')], check=True)
    print('ALL 45 JOBS COMPLETE', flush=True)


if __name__ == '__main__':
    if '--node' in sys.argv:
        raise SystemExit(node_worker(sys.argv[2], sys.argv[3]))
    main()
