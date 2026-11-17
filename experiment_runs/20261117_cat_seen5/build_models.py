"""Build CAT checkpoints for one fixed exemplar seed, with no training."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
from merge_lab.ties import load_state
from merge_lab.cat_features import collect_features
from merge_lab.cat import trim_vectors, merge_vectors
from model.Model import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--configs', help='Optional comma-separated config IDs')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    manifest = json.loads((RUN / 'manifest.json').read_text())
    configs = json.loads((RUN / 'configs.json').read_text())
    if args.configs:
        configs = [c for c in configs if c['config_id'] in args.configs.split(',')]
    base = load_state(manifest['base']['path'])
    experts = [load_state(item['checkpoint']) for item in manifest['experts']]
    deltas = [{name: state[name] - value for name, value in base.items()}
              for state in experts]
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    indices = {task: torch.randperm(1000, generator=generator)[:2].tolist()
               for task in manifest['task_order']}
    features = {}
    for scale in sorted({c['feature_scale'] for c in configs}):
        path = RUN / 'features' / f'seed{args.seed}_scale{scale}.pt'
        if path.exists():
            features[scale] = torch.load(path, map_location='cpu', weights_only=True)
            assert features[scale]['sample_indices'] == indices
        else:
            features[scale] = collect_features(base, experts, manifest['task_order'],
                                               indices, scale)
            torch.save(features[scale], path)
    params = json.loads((RUN.parent / '20261116_urs_finetune_seen5/models/cvrp/'
                        'finetune_metadata.json').read_text())['model_params']
    model = Model(**params)
    cache, builds = {}, []
    previous_path = RUN / f'build_seed{args.seed}.json'
    if previous_path.exists():
        builds = json.loads(previous_path.read_text())['models']
    for config in configs:
        job_id = f"{config['config_id']}_s{args.seed}"
        if any(item['id'] == job_id for item in builds):
            continue
        key = (config['mode'], config['linear_count'], config['norm_count'],
               config['bias_count'], config['feature_scale'])
        started = time.monotonic()
        if key not in cache:
            cache[key] = trim_vectors(
                deltas, features[config['feature_scale']],
                linear_count=config['linear_count'], linear_lambda=5,
                norm_count=config['norm_count'], norm_lambda=10,
                bias_count=config['bias_count'], upstream=config['mode'] == 'upstream')
        vectors, diagnostics = cache[key]
        state = merge_vectors(base, vectors, config['scale'])
        assert all(torch.isfinite(value).all() for value in state.values())
        model.load_state_dict(state, strict=True)
        path = RUN / 'models' / f'{job_id}.pt'
        info = {**config, 'seed': args.seed, 'sample_indices': indices,
                'base_checkpoint': manifest['base']['path'],
                'expert_checkpoints': [item['checkpoint'] for item in manifest['experts']],
                'upstream_commit': manifest['upstream_commit']}
        torch.save({'model_state_dict': state, 'merge_info': info}, path)
        builds.append({**info, 'id': job_id, 'checkpoint': str(path),
                       'seconds': time.monotonic() - started,
                       'diagnostics': diagnostics, 'finite': True, 'strict_load': True})
        previous_path.write_text(json.dumps({
            'updated': datetime.now().astimezone().isoformat(),
            'seed': args.seed, 'gpu': torch.cuda.get_device_name(),
            'sample_indices': indices, 'models': builds}, indent=2) + '\n')
        print(f'Built {job_id}: {builds[-1]["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
