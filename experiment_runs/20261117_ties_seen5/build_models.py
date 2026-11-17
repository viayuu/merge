"""Build the preregistered TIES sweep and two simple merge baselines."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PREVIOUS = ROOT / 'experiment_runs/20261116_urs_finetune_seen5'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'URS'))
from merge_lab.ties import (prepare_vectors, ties_merging, vector_to_state_dict,
                            load_state, state_dict_to_vector,
                            UPSTREAM_COMMIT, UPSTREAM_REPOSITORY)
from model.Model import Model


def digest(path):
    with Path(path).open('rb') as handle:
        sha = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    torch.set_num_threads(4)
    previous = json.loads((PREVIOUS / 'manifest.json').read_text())
    experts = json.loads((PREVIOUS / 'final_verification.json').read_text())
    base_path = Path(previous['base']['path'])
    assert digest(base_path) == previous['base']['sha256']
    for expert in experts:
        assert digest(expert['checkpoint']) == expert['checkpoint_sha256']
        metadata = json.loads((Path(expert['checkpoint']).parent /
                               'finetune_metadata.json').read_text())
        assert metadata['base_file_sha256'] == previous['base']['sha256']
        assert metadata['initial_model_sha256'] == expert['initial_model_sha256']
    for task in previous['tasks'].values():
        for kind in ['data_path', 'solution_path']:
            assert digest(task[kind]['path']) == task[kind]['sha256']

    base_state, base_vector, deltas = prepare_vectors(
        base_path, [expert['checkpoint'] for expert in experts])
    state_sha = hashlib.sha256()
    for name, tensor in base_state.items():
        state_sha.update(name.encode())
        state_sha.update(tensor.numpy().tobytes())
    assert all(expert['initial_model_sha256'] == state_sha.hexdigest()
               for expert in experts)
    params = json.loads((PREVIOUS / 'models/cvrp/finetune_metadata.json')
                        .read_text())['model_params']
    model = Model(**params)
    configs = json.loads((RUN / 'configs.json').read_text())
    models_dir = RUN / 'models'
    models_dir.mkdir(exist_ok=True)
    updates = {keep: ties_merging(deltas, reset_thresh=keep / 100,
                                 merge_func='dis-mean')
               for keep in sorted({c['keep_percent'] for c in configs
                                   if c['method'] == 'ties'})}
    average = torch.stack([state_dict_to_vector(load_state(e['checkpoint']))
                           for e in experts]).mean(dim=0)
    summed = deltas.sum(dim=0)
    manifest = {
        'created': datetime.now().astimezone().isoformat(),
        'base': previous['base'], 'experts': experts,
        'base_state_sha256': state_sha.hexdigest(),
        'task_order': previous['task_order'], 'datasets': previous['tasks'],
        'upstream_repository': UPSTREAM_REPOSITORY,
        'upstream_commit': UPSTREAM_COMMIT,
        'dtype': 'float32', 'parameter_count': base_vector.numel(),
        'tensor_count': len(base_state),
        'config_sha256': digest(RUN / 'configs.json'),
        'sources': {}, 'models': [],
        'previous_evaluation': str(PREVIOUS / 'evaluation'),
        'protocol': {'seed': 3407, 'scale': 100, 'episodes': 1000,
                     'augmentation': 8, 'batch_size': 125,
                     'eval_type': 'greedy', 'task_order': previous['task_order'],
                     'pomo_sizes': [100, 80, 100, 100, 100]},
        'torch': torch.__version__,
    }
    sources = [ROOT / 'merge_lab/ties.py', Path(__file__),
               RUN / 'run_evaluations.py', ROOT / 'URS/test.py',
               ROOT / 'URS/Tester.py', ROOT / 'URS/model/Model.py',
               ROOT / 'URS/model/Model_LIB.py']
    sources += sorted((ROOT / 'third_party/ties_merging').rglob('*.py'))
    sources += [ROOT / 'third_party/ties_merging/src/ties_minimal.ipynb']
    for path in sources:
        manifest['sources'][str(path.relative_to(ROOT))] = digest(path)
    # Previously evaluated expert results remain comparable to the current code.
    for source in ['URS/Tester.py', 'URS/test.py', 'URS/model/Model.py',
                   'URS/model/Model_LIB.py']:
        assert manifest['sources'][source] == previous['source_files'][source]
    for config in configs:
        if config['method'] == 'ties':
            vector = base_vector + config['scale'] * updates[config['keep_percent']]
        elif config['method'] == 'average':
            vector = average
        else:
            assert config['method'] == 'task-arithmetic', config
            vector = base_vector + config['scale'] * summed
        assert torch.isfinite(vector).all(), config
        state = vector_to_state_dict(vector, base_state)
        model.load_state_dict(state, strict=True)
        path = models_dir / (config['id'] + '.pt')
        info = {**config, 'upstream_commit': UPSTREAM_COMMIT,
                'base_sha256': previous['base']['sha256'],
                'expert_sha256': [e['checkpoint_sha256'] for e in experts],
                'dtype': 'float32'}
        torch.save({'model_state_dict': state, 'merge_info': info}, path)
        manifest['models'].append({**config, 'checkpoint': str(path),
                                   'sha256': digest(path),
                                   'finite': True, 'strict_load': True})
        print('Built', config['id'], flush=True)
    (RUN / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    default = str(models_dir / 'ties_k20_l1p0.pt')
    plan = []
    for node, label in [('gpu05', '4090'), ('gpu03', '3090')]:
        plan.extend([
            {'id': 'base__' + label, 'phase': 'calibration', 'node': node,
             'gpu': 0, 'checkpoint': str(base_path)},
            {'id': 'ties_default__' + label, 'phase': 'calibration', 'node': node,
             'gpu': 1, 'config_id': 'ties_k20_l1p0', 'checkpoint': default}])
    remaining = [c for c in configs if c['id'] != 'ties_k20_l1p0']
    for i, config in enumerate(remaining):
        slot = i % 4
        plan.append({'id': config['id'], 'config_id': config['id'],
                     'phase': 'scan', 'node': 'gpu05' if slot < 2 else 'gpu03',
                     'gpu': slot % 2,
                     'checkpoint': str(models_dir / (config['id'] + '.pt'))})
    (RUN / 'evaluation_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Built {len(configs)} models; calibration ready.', flush=True)


if __name__ == '__main__':
    main()
