"""Build LOT/CAT checkpoints with nested original-data exemplar subsets."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
CAT = RUN.parent / '20261117_cat_seen5'
sys.path.insert(0, str(ROOT))
from merge_lab.ties import load_state
from merge_lab.cat_features import collect_features
from merge_lab.cat import trim_vectors, merge_vectors
from merge_lab.lot import lot_vector
from model.Model import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--ids', help='Optional comma-separated job IDs')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    manifest = json.loads((RUN / 'manifest.json').read_text())
    jobs = [x for x in json.loads((RUN / 'evaluation_plan.json').read_text())
            if x.get('family') != 'base' and x['seed'] == args.seed and 'source_result' not in x]
    if args.ids:
        jobs = [x for x in jobs if x['id'] in args.ids.split(',')]
    base = load_state(manifest['base']['path'])
    experts = [load_state(x['checkpoint']) for x in manifest['experts']]
    deltas = [{name: state[name] - value for name, value in base.items()} for state in experts]
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    permutations = {task: torch.randperm(1000, generator=generator).tolist()
                    for task in manifest['task_order']}
    params = json.loads((RUN.parent / '20261116_urs_finetune_seen5/models/cvrp/'
                        'finetune_metadata.json').read_text())['model_params']
    model = Model(**params)
    feature_cache, vector_cache = {}, {}

    for job in jobs:
        record_path = RUN / 'build_records' / f"{job['id']}.json"
        if record_path.exists():
            continue
        size, feature_scale = job['samples'], job['feature_scale']
        indices = {task: (list(range(1000)) if size == 1000 else order[:size])
                   for task, order in permutations.items()}
        feature_key = (size, feature_scale)
        feature_path = (CAT / 'features' / f'seed{args.seed}_scale{feature_scale}.pt'
                        if size == 2 else RUN / 'features' /
                        f'n{size}_s{args.seed}_scale{feature_scale}.pt')
        feature_reused = feature_key in feature_cache or feature_path.exists()
        if feature_key not in feature_cache:
            if feature_path.exists():
                feats = torch.load(feature_path, map_location='cpu', weights_only=True)
                assert feats['sample_indices'] == indices
            else:
                started = time.monotonic()
                feats = collect_features(base, experts, manifest['task_order'], indices,
                                         feature_scale, feature_batch_size=128)
                torch.cuda.synchronize()
                feats['collection_seconds'] = time.monotonic() - started
                torch.save(feats, feature_path)
            feature_cache[feature_key] = feats
        feats = feature_cache[feature_key]
        merge_key = (job['family'], size, feature_scale)
        merge_reused = merge_key in vector_cache
        seconds = 0.
        if not merge_reused:
            started = time.monotonic()
            if job['family'] == 'lot':
                vectors, diagnostics = lot_vector(deltas, feats)
            else:
                upstream = job['family'] == 'cat_upstream'
                vectors, diagnostics = trim_vectors(
                    deltas, feats, linear_count=4, linear_lambda=5,
                    norm_count=0 if upstream else 4, norm_lambda=10,
                    bias_count=0 if upstream else 4, upstream=upstream)
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            vector_cache[merge_key] = (vectors, diagnostics)
        vectors, diagnostics = vector_cache[merge_key]
        started = time.monotonic()
        state = ({name: value + job['scale'] * vectors[name] for name, value in base.items()}
                 if job['family'] == 'lot' else merge_vectors(base, vectors, job['scale']))
        assert all(torch.isfinite(v).all() for v in state.values())
        model.load_state_dict(state, strict=True)
        info = {**job, 'sample_indices': indices, 'base_checkpoint': manifest['base']['path'],
                'expert_checkpoints': [x['checkpoint'] for x in manifest['experts']],
                'upstream_commit': manifest['upstream_commit']}
        torch.save({'model_state_dict': state, 'merge_info': info}, job['checkpoint'])
        timing = {'feature_seconds': feats.get('collection_seconds'),
                  'feature_reused': feature_reused, 'merge_seconds': seconds,
                  'merge_reused': merge_reused, 'save_seconds': time.monotonic() - started}
        record = {**info, 'feature_path': str(feature_path), 'merge_key': list(merge_key),
                  'timing': timing, 'diagnostics': diagnostics,
                  'finite': True, 'strict_load': True,
                  'gpu': torch.cuda.get_device_name(),
                  'finished': datetime.now().astimezone().isoformat()}
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
        print(f"Built {job['id']}: merge={seconds:.2f}s, features={timing['feature_seconds']}", flush=True)


if __name__ == '__main__':
    main()
