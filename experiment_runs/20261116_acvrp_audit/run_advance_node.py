"""Two GPUs per existing Slurm allocation; run only targeted reproduction controls."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

run = Path(__file__).resolve().parent
root = run.parents[1]
node = sys.argv[1]
models = ['unified_checkpoint_500', 'acvrp_checkpoint_300']

def worker(gpu):
    model = models[gpu]
    prefix = 'unified' if gpu == 0 else 'specialist'
    # 4090: hardware comparison and same-process task order.
    # 3090: exact repetition of the previous formal baseline.
    cases = [('rng_advanced', 'acvrp')]
    for case, problems in cases:
        identity = f'{prefix}_{node}_{case}'
        command = [sys.executable, str(run / 'advance_test.py'),
                   '--problem_set', problems, '--model_load', str(root/'URS/pretrained'/f'{model}.pt'),
                   '--data_dir', str(root/'URS/dataset'),
                   '--test_scale_list', '100', '--test_episodes', '1000',
                   '--seed', '3407', '--eval_type', 'greedy', '--cuda', '0',
                   '--aug_factor', '8', '128', '--test_batch_size_small', '1000', '125', '50', '8',
                   '--output_json', str(run/'results'/f'{identity}.json')]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
        print('START', identity, flush=True)
        with (run/'logs'/f'{identity}.log').open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        print('FINISHED', identity, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        data = json.loads((run/'results'/f'{identity}.json').read_text())
        print(identity, [(r['task'],r['aug_score'],r['aug_gap']) for r in data['rows']], flush=True)

with ThreadPoolExecutor(max_workers=2) as pool:
    for future in [pool.submit(worker, gpu) for gpu in range(2)]:
        future.result()
